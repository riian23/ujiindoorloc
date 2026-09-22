from __future__ import annotations

import argparse
import copy
import itertools
import logging
from datetime import datetime
from statistics import mean
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from common.config import PROJECT_ROOT, load_config
from common.data import is_cache_valid, load_rp_coord
from common.metrics import METRIC_ORDER, evaluate
from common.report import append_rows
from mlp.predict import predict
from mlp.train import get_device, load_encoded, train_run

logger = logging.getLogger(__name__)

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EXPERIMENTS_CSV = OUTPUTS_DIR / "experiments_tune.csv"

LABEL_KEYS = ("building", "floor")
OBJECTIVE = "evaal_3d_error_m"

SEARCH_SPACE = {
    "hidden_dims": ([256, 128, 64], [512, 256, 128], [512, 256, 128, 64], [256, 128]),
    "dropout": (0.1, 0.2, 0.3, 0.5),
    "learning_rate": (0.0003, 0.001, 0.003),
    "weight_decay": (0.0, 0.00001, 0.0001),
}
BASELINE = {
    "hidden_dims": [256, 128, 64],
    "dropout": 0.3,
    "learning_rate": 0.001,
    "weight_decay": 0.0,
}
BACKBONE_KEYS = ("hidden_dims", "dropout")

CSV_COLUMNS = (
    "timestamp",
    "row",
    "stage",
    "trial",
    "seed",
    "hidden_dims",
    "dropout",
    "learning_rate",
    "weight_decay",
    "best_epochs",
) + METRIC_ORDER


def sample_trials(count: int, seed: int) -> List[Dict[str, Any]]:
    grid = [
        dict(zip(SEARCH_SPACE, values))
        for values in itertools.product(*SEARCH_SPACE.values())
    ]
    others = [point for point in grid if point != BASELINE]
    picked = np.random.default_rng(seed).choice(len(others), size=count - 1, replace=False)
    return [BASELINE] + [others[i] for i in sorted(picked)]


def runs_for(config: Dict[str, Any], row: str) -> Tuple[str, ...]:
    spec = config["mlp"]["report"][row]
    runs = [spec["labels"]]
    if spec["coord"] != spec["labels"]:
        runs.append(spec["coord"])
    return tuple(runs)


def report_row(
    predictions: Dict[str, Dict[str, np.ndarray]], true: Dict[str, np.ndarray], spec: Dict[str, str]
) -> Dict[str, float]:
    combined = {key: predictions[spec["labels"]][key] for key in LABEL_KEYS}
    combined["coord"] = predictions[spec["coord"]]["coord"]
    return evaluate(combined, true)


def apply_trial(config: Dict[str, Any], trial: Dict[str, Any], seed: int) -> Dict[str, Any]:
    trial_config = copy.deepcopy(config)
    trial_config["seed"] = seed
    for key, value in trial.items():
        section = "backbone" if key in BACKBONE_KEYS else "training"
        trial_config["mlp"][section][key] = value
    return trial_config


def score_trial(
    config: Dict[str, Any], row: str, trial: Dict[str, Any], seed: int
) -> Tuple[Dict[str, float], Dict[str, int]]:
    trial_config = apply_trial(config, trial, seed)
    device = get_device()
    batch_size = trial_config["mlp"]["training"]["batch_size"]

    val = load_encoded(trial_config, "val")
    rp_coord = load_rp_coord(trial_config)

    predictions, best_epochs = {}, {}
    for run_name in runs_for(trial_config, row):
        model, info = train_run(trial_config, run_name)
        predictions[run_name] = predict(
            model, val["rssi"], rp_coord, info["coord_mean"], info["coord_std"], device, batch_size
        )
        best_epochs[run_name] = info["best_epoch"]

    metrics = report_row(
        predictions,
        {name: val[name] for name in ("building", "floor", "coord")},
        trial_config["mlp"]["report"][row],
    )
    return metrics, best_epochs


def run_stage(
    config: Dict[str, Any],
    row: str,
    trials: Sequence[Tuple[int, Dict[str, Any]]],
    seeds: Sequence[int],
    stage: str,
    timestamp: str,
) -> Dict[int, List[float]]:
    scores: Dict[int, List[float]] = {index: [] for index, _ in trials}
    rows: List[Dict[str, Any]] = []

    for index, trial in trials:
        for seed in seeds:
            metrics, best_epochs = score_trial(config, row, trial, seed)
            scores[index].append(metrics[OBJECTIVE])
            rows.append(
                {
                    "timestamp": timestamp,
                    "row": row,
                    "stage": stage,
                    "trial": index,
                    "seed": seed,
                    "best_epochs": "|".join(f"{r}={e}" for r, e in best_epochs.items()),
                    **trial,
                    **metrics,
                }
            )
            logger.info(
                f"[{row}/{stage}] trial {index} seed {seed}: val {OBJECTIVE}={metrics[OBJECTIVE]:.4f} {trial}"
            )

    append_rows(EXPERIMENTS_CSV, CSV_COLUMNS, rows)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Search MLP hyperparameters on validation")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--row", default="mlp_regression")
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--search-seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument("--confirm-seeds", type=int, nargs="+", default=[44, 45, 46])
    parser.add_argument("--finalists", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    timestamp = datetime.now().isoformat(timespec="seconds")

    if not is_cache_valid(config):
        raise SystemExit("Cache is missing or stale. Run: python -m common.data --config config.yaml")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    trials = list(enumerate(sample_trials(args.trials, config["seed"])))
    logger.info(f"{args.row}: runs {runs_for(config, args.row)}, {len(trials)} trials x {len(args.search_seeds)} seeds, objective val {OBJECTIVE}")

    searched = run_stage(config, args.row, trials, args.search_seeds, "search", timestamp)
    ranked = sorted(searched, key=lambda index: mean(searched[index]))

    print(f"\nsearch: val {OBJECTIVE}, mean of {len(args.search_seeds)} seeds")
    print("rank  trial  mean     hyperparameters")
    for rank, index in enumerate(ranked, 1):
        print(f"{rank:>4}  {index:>5}  {mean(searched[index]):.4f}   {dict(trials[index][1])}")

    finalists = [trials[index] for index in ranked[: args.finalists]]
    confirmed = run_stage(config, args.row, finalists, args.confirm_seeds, "confirm", timestamp)

    print(f"\nconfirm: val {OBJECTIVE} over {len(args.search_seeds) + len(args.confirm_seeds)} seeds")
    print("trial  mean     std      hyperparameters")
    best_index, best_mean = None, None
    for index, _ in finalists:
        values = searched[index] + confirmed[index]
        combined_mean = float(np.mean(values))
        print(f"{index:>5}  {combined_mean:.4f}  {float(np.std(values)):.4f}   {dict(trials[index][1])}")
        if best_mean is None or combined_mean < best_mean:
            best_index, best_mean = index, combined_mean

    print(f"\nselected trial {best_index}: {dict(trials[best_index][1])}")
    print(f"experiment log appended: {EXPERIMENTS_CSV}")


if __name__ == "__main__":
    main()
