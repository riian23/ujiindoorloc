from __future__ import annotations

import argparse
import logging
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import torch

from common.config import PROJECT_ROOT, load_config
from common.data import is_cache_valid, load_rp_coord
from common.metrics import METRIC_ORDER, evaluate
from common.report import append_rows, print_table
from mlp.predict import predict
from mlp.train import get_device, load_encoded, train_merged, train_run
from run_tune import apply_trial, report_row, runs_for

logger = logging.getLogger(__name__)

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EXPERIMENTS_CSV = OUTPUTS_DIR / "experiments_final.csv"
TRUE_KEYS = ("building", "floor", "coord")

CSV_COLUMNS = (
    "timestamp",
    "row",
    "training_set",
    "seed",
    "hidden_dims",
    "dropout",
    "learning_rate",
    "weight_decay",
    "best_epochs",
) + METRIC_ORDER


def hyperparameters(config: Dict[str, Any], row: str) -> Dict[str, Any]:
    return dict(config["mlp"]["tuned"][row])


def score_seed(
    config: Dict[str, Any], row: str, seed: int, test: Dict[str, np.ndarray]
) -> Dict[str, Any]:
    seed_config = apply_trial(config, hyperparameters(config, row), seed)
    device = get_device()
    batch_size = seed_config["mlp"]["training"]["batch_size"]
    rp_coord = load_rp_coord(seed_config)
    true_test = {name: test[name] for name in TRUE_KEYS}
    spec = seed_config["mlp"]["report"][row]

    predictions: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {"split": {}, "merged": {}}
    best_epochs = {}

    for run_name in runs_for(seed_config, row):
        model, info = train_run(seed_config, run_name)
        best_epochs[run_name] = info["best_epoch"]
        predictions["split"][run_name] = predict(
            model, test["rssi"], rp_coord, info["coord_mean"], info["coord_std"], device, batch_size
        )

        model, info = train_merged(seed_config, run_name, best_epochs[run_name])
        predictions["merged"][run_name] = predict(
            model, test["rssi"], rp_coord, info["coord_mean"], info["coord_std"], device, batch_size
        )

    return {
        "metrics": {name: report_row(runs, true_test, spec) for name, runs in predictions.items()},
        "best_epochs": best_epochs,
    }


def summarise(rows: List[Dict[str, float]]) -> Dict[str, float]:
    return {metric: float(np.mean([row[metric] for row in rows])) for metric in METRIC_ORDER}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the tuned MLP over several seeds and score it on test")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--row", default="mlp_regression")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    timestamp = datetime.now().isoformat(timespec="seconds")

    if not is_cache_valid(config):
        raise SystemExit("Cache is missing or stale. Run: python -m common.data --config config.yaml")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    settings = hyperparameters(config, args.row)
    test = load_encoded(config, "test")
    logger.info(f"{args.row}: runs {runs_for(config, args.row)}, seeds {args.seeds}, hyperparameters {settings}")

    experiments: List[Dict[str, Any]] = []
    collected: Dict[str, List[Dict[str, float]]] = {"split": [], "merged": []}

    for seed in args.seeds:
        result = score_seed(config, args.row, seed, test)
        for training_set, metrics in result["metrics"].items():
            collected[training_set].append(metrics)
            experiments.append(
                {
                    "timestamp": timestamp,
                    "row": args.row,
                    "training_set": training_set,
                    "seed": seed,
                    "best_epochs": "|".join(f"{r}={e}" for r, e in result["best_epochs"].items()),
                    **settings,
                    **metrics,
                }
            )
        logger.info(
            f"seed {seed}: test evaal split={result['metrics']['split']['evaal_3d_error_m']:.4f} "
            f"merged={result['metrics']['merged']['evaal_3d_error_m']:.4f}"
        )

    append_rows(EXPERIMENTS_CSV, CSV_COLUMNS, experiments)

    print_table({name: summarise(rows) for name, rows in collected.items()})
    print(f"\n{len(args.seeds)} seeds, test evaal_3d_error_m")
    for name, rows in collected.items():
        values = [row["evaal_3d_error_m"] for row in rows]
        print(f"  {name:>7}: {np.mean(values):.4f} +- {np.std(values):.4f} m   {[round(v, 3) for v in values]}")
    print(f"\nexperiment log appended: {EXPERIMENTS_CSV}")


if __name__ == "__main__":
    main()
