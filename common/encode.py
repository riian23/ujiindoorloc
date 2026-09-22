from __future__ import annotations

import numpy as np


def linear(rssi_dbm: np.ndarray, rssi_min: int = -105, rssi_max: int = 0) -> np.ndarray:
    return ((rssi_dbm - rssi_min) / (rssi_max - rssi_min)).astype(np.float32)


ENCODERS = {
    "linear": linear,
}


def encode(name: str, rssi_dbm: np.ndarray, rssi_min: int = -105, rssi_max: int = 0) -> np.ndarray:
    if name not in ENCODERS:
        raise ValueError(f"Unknown encoder: {name!r} (registered: {sorted(ENCODERS)})")
    return ENCODERS[name](rssi_dbm, rssi_min, rssi_max)
