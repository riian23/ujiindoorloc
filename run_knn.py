from __future__ import annotations

import argparse
import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np

from common.config import PROJECT_ROOT, load_config
from common.data import is_cache_valid, load_split
from common.encode import encode
from common.metrics import METRIC_ORDER, evaluate
from common.report import append_rows, print_table
from knn.knn import nearest, predict

logger = logging.getLogger(__name__)

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EXPERIMENTS_CSV = OUTPUTS_DIR / "experiments_knn.csv"
TRUE_KEYS = ("building", "floor", "coord")

CSV_COLUMNS = (
    "timestamp",
    "split",
    "weighting",
    "k",
    "k_candidates",
    "distance",
    "encoder",
    "seed",
    "select_metric",
) + METRIC_ORDER


def load_encoded(config: Dict[str, Any], split: str) -> Dict[str, np.ndarray]:
    data = load_split(config, split)
    data["rssi"] = encode(
        config["knn"]["encoder"],
        data["rssi"],
        config["data"]["rssi_min"],
        config["data"]["rssi_max"],
    )
    return data


def sweep(
    config: Dict[str, Any], db: Dict[str, np.ndarray], val: Dict[str, np.ndarray]
) -> Dict[Tuple[str, int], Dict[str, float]]:
    knn_cfg = config["knn"]
    candidates = knn_cfg["k_candidates"]

    indices, distances = nearest(
        val["rssi"], db["rssi"], max(candidates), knn_cfg["distance"]
    )
    true_val = {name: val[name] for name in TRUE_KEYS}

    return {
        (weighting, k): evaluate(predict(indices, distances, db, k, weighting), true_val)
        for weighting in knn_cfg["weightings"]
        for k in candidates
    }


def best_combination(
    scored: Dict[Tuple[str, int], Dict[str, float]], select: Dict[str, str]
) -> Tuple[str, int]:
    pick = max if select["mode"] == "max" else min
    return pick(scored, key=lambda combo: scored[combo][select["metric"]])


def print_sweep(
    config: Dict[str, Any], scored: Dict[Tuple[str, int], Dict[str, float]]
) -> None:
    knn_cfg = config["knn"]
    metric = knn_cfg["select"]["metric"]
    weightings = knn_cfg["weightings"]
    width = max(len(w) for w in weightings) + 2

    print(f"\nval {metric}")
    print("   k".ljust(6) + "".join(w.ljust(width) for w in weightings))
    for k in knn_cfg["k_candidates"]:
        cells = [f"{scored[(w, k)][metric]:.4f}".ljust(width) for w in weightings]
        print(f"{k:>4}  " + "".join(cells))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit weighting and k on validation, score KNN on test")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    timestamp = datetime.now().isoformat(timespec="seconds")

    if not is_cache_valid(config):
        raise SystemExit("Cache is missing or stale. Run: python -m common.data --config config.yaml")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    knn_cfg = config["knn"]
    select = knn_cfg["select"]

    db = load_encoded(config, "train")
    logger.info(f"database: {len(db['rssi'])} training samples, {knn_cfg['distance']} distance")

    scored = sweep(config, db, load_encoded(config, "val"))
    weighting, k = best_combination(scored, select)
    print_sweep(config, scored)
    logger.info(f"selected {weighting} k={k}, val {select['metric']}={scored[(weighting, k)][select['metric']]:.4f}")

    test = load_encoded(config, "test")
    indices, distances = nearest(test["rssi"], db["rssi"], k, knn_cfg["distance"])
    prediction = predict(indices, distances, db, k, weighting)
    metrics = evaluate(prediction, {name: test[name] for name in TRUE_KEYS})
    np.savez(OUTPUTS_DIR / "knn_test.npz", **prediction)

    settings = {
        "k_candidates": knn_cfg["k_candidates"],
        "distance": knn_cfg["distance"],
        "encoder": knn_cfg["encoder"],
        "seed": config["seed"],
        "select_metric": select["metric"],
    }
    experiments: List[Dict[str, Any]] = [
        {
            "timestamp": timestamp,
            "split": "val",
            "weighting": combo_weighting,
            "k": combo_k,
            **settings,
            **combo_metrics,
        }
        for (combo_weighting, combo_k), combo_metrics in scored.items()
    ]
    experiments.append(
        {
            "timestamp": timestamp,
            "split": "test",
            "weighting": weighting,
            "k": k,
            **settings,
            **metrics,
        }
    )
    append_rows(EXPERIMENTS_CSV, CSV_COLUMNS, experiments)

    print_table({f"knn_{weighting}_k{k}": metrics})
    print(f"\npredictions under {OUTPUTS_DIR}")
    print(f"experiment log appended: {EXPERIMENTS_CSV}")


if __name__ == "__main__":
    main()
