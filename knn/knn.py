from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def euclidean(query: np.ndarray, db: np.ndarray) -> np.ndarray:
    query_sq = np.einsum("ij,ij->i", query, query)[:, None]
    db_sq = np.einsum("ij,ij->i", db, db)[None, :]
    squared = query_sq + db_sq - 2.0 * (query @ db.T)
    np.maximum(squared, 0.0, out=squared)
    return np.sqrt(squared)


DISTANCES = {
    "euclidean": euclidean,
}


def uniform(distances: np.ndarray) -> np.ndarray:
    return np.ones_like(distances)


def inverse(distances: np.ndarray) -> np.ndarray:
    return 1.0 / distances


def inverse_squared(distances: np.ndarray) -> np.ndarray:
    return 1.0 / np.square(distances)


def dudani(distances: np.ndarray) -> np.ndarray:
    nearest_distance = distances[:, :1]
    farthest_distance = distances[:, -1:]
    span = farthest_distance - nearest_distance
    return np.where(
        span > 0.0,
        (farthest_distance - distances) / np.where(span > 0.0, span, 1.0),
        1.0,
    )


WEIGHTS = {
    "uniform": uniform,
    "inverse": inverse,
    "inverse_squared": inverse_squared,
    "dudani": dudani,
}


def nearest(
    query: np.ndarray, db: np.ndarray, k_max: int, distance: str, chunk: int = 1024
) -> Tuple[np.ndarray, np.ndarray]:
    if distance not in DISTANCES:
        raise ValueError(f"Unknown distance: {distance!r} (registered: {sorted(DISTANCES)})")
    if k_max > len(db):
        raise ValueError(f"k_max {k_max} exceeds the database size {len(db)}")

    measure = DISTANCES[distance]
    indices = np.empty((len(query), k_max), dtype=np.int64)
    distances = np.empty((len(query), k_max), dtype=np.float64)

    for start in range(0, len(query), chunk):
        block = measure(query[start : start + chunk], db)
        rows = np.arange(len(block))[:, None]
        closest = np.argpartition(block, k_max - 1, axis=1)[:, :k_max]
        order = np.argsort(block[rows, closest], axis=1)
        picked = closest[rows, order]
        indices[start : start + chunk] = picked
        distances[start : start + chunk] = block[rows, picked]

    return indices, distances


def weighted_vote(labels: np.ndarray, weights: np.ndarray, n_classes: int) -> np.ndarray:
    totals = np.zeros((len(labels), n_classes), dtype=np.float64)
    np.add.at(totals, (np.arange(len(labels))[:, None], labels), weights)
    return totals.argmax(axis=1)


def predict(
    indices: np.ndarray,
    distances: np.ndarray,
    db: Dict[str, np.ndarray],
    k: int,
    weighting: str,
) -> Dict[str, np.ndarray]:
    if weighting not in WEIGHTS:
        raise ValueError(f"Unknown weighting: {weighting!r} (registered: {sorted(WEIGHTS)})")

    picked = indices[:, :k]
    weights = WEIGHTS[weighting](distances[:, :k])

    return {
        "building": weighted_vote(db["building"][picked], weights, int(db["building"].max()) + 1),
        "floor": weighted_vote(db["floor"][picked], weights, int(db["floor"].max()) + 1),
        "coord": (db["coord"][picked] * weights[:, :, None]).sum(axis=1)
        / weights.sum(axis=1, keepdims=True),
    }
