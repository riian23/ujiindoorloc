from __future__ import annotations

import argparse
import csv
import logging
from datetime import datetime
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from common.config import PROJECT_ROOT, load_config
from common.data import is_cache_valid, load_rp_coord
from common.metrics import METRIC_ORDER, evaluate
from common.report import append_rows, print_table
from mlp.plot import save_curve
from mlp.predict import predict
from mlp.train import get_device, load_encoded, train_run

logger = logging.getLogger(__name__)

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EXPERIMENTS_CSV = OUTPUTS_DIR / "experiments.csv"
LABEL_KEYS = ("building", "floor")

CONFIG_COLUMNS = (
    "seed",
    "encoder",
    "hidden_dims",
    "activation",
    "dropout",
    "batch_norm",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "patience",
    "max_epochs",
)
CSV_COLUMNS = (
    "timestamp",
    "row",
    "labels_run",
    "coord_run",
    "heads",
    "best_epochs",
) + CONFIG_COLUMNS + METRIC_ORDER


def config_columns(config: Dict[str, Any]) -> Dict[str, Any]:
    backbone = config["mlp"]["backbone"]
    training = config["mlp"]["training"]
    return {
        "seed": config["seed"],
        "encoder": config["mlp"]["encoder"],
        "hidden_dims": backbone["hidden_dims"],
        "activation": backbone["activation"],
        "dropout": backbone["dropout"],
        "batch_norm": backbone["batch_norm"],
        "batch_size": training["batch_size"],
        "learning_rate": training["learning_rate"],
        "weight_decay": training["weight_decay"],
        "patience": training["patience"],
        "max_epochs": training["epochs"],
    }


def source_columns(
    config: Dict[str, Any], spec: Dict[str, str], info_by_run: Dict[str, Dict[str, Any]]
) -> Tuple[str, str]:
    runs = [spec["labels"]]
    if spec["coord"] != spec["labels"]:
        runs.append(spec["coord"])
    heads = "|".join(
        run + "=" + ";".join(
            f"{name}:{weight}" for name, weight in config["mlp"]["runs"][run]["heads"].items()
        )
        for run in runs
    )
    best_epochs = "|".join(f"{run}={info_by_run[run]['best_epoch']}" for run in runs)
    return heads, best_epochs


def save_history(run_name: str, history: Sequence[Dict[str, float]]) -> None:
    with open(OUTPUTS_DIR / f"history_{run_name}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the MLP runs and score them on test")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    timestamp = datetime.now().isoformat(timespec="seconds")

    if not is_cache_valid(config):
        raise SystemExit("Cache is missing or stale. Run: python -m common.data --config config.yaml")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    batch_size = config["mlp"]["training"]["batch_size"]
    test_data = load_encoded(config, "test")
    rp_coord = load_rp_coord(config)

    predictions: Dict[str, Dict[str, np.ndarray]] = {}
    info_by_run: Dict[str, Dict[str, Any]] = {}

    for run_name in config["mlp"]["runs"]:
        model, info = train_run(config, run_name)
        save_curve(
            run_name,
            info["history"],
            config["mlp"]["runs"][run_name]["select"],
            info["best_epoch"],
            OUTPUTS_DIR / f"curve_{run_name}.png",
        )
        save_history(run_name, info["history"])
        predictions[run_name] = predict(
            model, test_data["rssi"], rp_coord, info["coord_mean"], info["coord_std"], device, batch_size
        )
        info_by_run[run_name] = info
        torch.save(model.state_dict(), OUTPUTS_DIR / f"{run_name}.pt")

    true_test = {name: test_data[name] for name in ("building", "floor", "coord")}
    rows: Dict[str, Dict[str, float]] = {}
    settings = config_columns(config)
    experiments: List[Dict[str, Any]] = []

    for row_name, spec in config["mlp"]["report"].items():
        prediction = {key: predictions[spec["labels"]][key] for key in LABEL_KEYS}
        prediction["coord"] = predictions[spec["coord"]]["coord"]
        rows[row_name] = evaluate(prediction, true_test)
        np.savez(OUTPUTS_DIR / f"{row_name}_test.npz", **prediction)

        heads, best_epochs = source_columns(config, spec, info_by_run)
        experiments.append(
            {
                "timestamp": timestamp,
                "row": row_name,
                "labels_run": spec["labels"],
                "coord_run": spec["coord"],
                "heads": heads,
                "best_epochs": best_epochs,
                **settings,
                **rows[row_name],
            }
        )

    append_rows(EXPERIMENTS_CSV, CSV_COLUMNS, experiments)

    print("\nbest epoch per run")
    for run_name, info in info_by_run.items():
        print(f"  {run_name}: {info['best_epoch']}")

    print_table(rows)
    print(f"\npredictions, curves and weights under {OUTPUTS_DIR}")
    print(f"experiment log appended: {EXPERIMENTS_CSV}")


if __name__ == "__main__":
    main()
