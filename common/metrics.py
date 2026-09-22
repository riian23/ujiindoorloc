from __future__ import annotations

from typing import Dict

import numpy as np

FLOOR_PENALTY_M = 4.0
BUILDING_PENALTY_M = 50.0

METRIC_ORDER = (
    "building_accuracy",
    "floor_accuracy",
    "combined_hit_rate",
    "mean_2d_error_m",
    "median_2d_error_m",
    "mean_floor_distance",
    "evaal_3d_error_m",
)


def evaluate(
    pred: Dict[str, np.ndarray], true: Dict[str, np.ndarray]
) -> Dict[str, float]:
    def present(*keys: str) -> bool:
        return all(key in pred and key in true for key in keys)

    metrics: Dict[str, float] = {}

    if present("building"):
        building_hit = pred["building"] == true["building"]
        metrics["building_accuracy"] = float(building_hit.mean())

    if present("floor"):
        floor_distance = np.abs(pred["floor"] - true["floor"])
        metrics["floor_accuracy"] = float((floor_distance == 0).mean())
        metrics["mean_floor_distance"] = float(floor_distance.mean())

    if present("building", "floor"):
        metrics["combined_hit_rate"] = float((building_hit & (floor_distance == 0)).mean())

    if present("coord"):
        error_2d = np.linalg.norm(pred["coord"] - true["coord"], axis=1)
        metrics["mean_2d_error_m"] = float(error_2d.mean())
        metrics["median_2d_error_m"] = float(np.median(error_2d))

    if present("building", "floor", "coord"):
        penalty = (
            FLOOR_PENALTY_M * floor_distance + BUILDING_PENALTY_M * (~building_hit)
        )
        metrics["evaal_3d_error_m"] = float((error_2d + penalty).mean())

    return metrics
