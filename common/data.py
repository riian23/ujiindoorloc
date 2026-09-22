from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from common.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

WAP_PREFIX = "WAP"
BUILDING_COL = "BUILDINGID"
FLOOR_COL = "FLOOR"
LONGITUDE_COL = "LONGITUDE"
LATITUDE_COL = "LATITUDE"

GROUP_COLS = [BUILDING_COL, FLOOR_COL, "SPACEID", "RELATIVEPOSITION"]
RP_COLS = [BUILDING_COL, FLOOR_COL, LONGITUDE_COL, LATITUDE_COL]
RP_ID_COL = "rp"
RP_COORD_DECIMALS = 6

COORD_DTYPE = np.float64

SPLITS = ("train", "val", "test")
ARRAYS = ("rssi", "building", "floor", "coord", "rp")
RP_COORD_FILENAME = "rp_coord.npy"
META_FILENAME = "meta.json"


def cache_dir(config: Dict[str, Any]) -> Path:
    return PROJECT_ROOT / config["data"]["cache_dir"]


def load_raw(path: str | Path) -> pd.DataFrame:
    path = PROJECT_ROOT / path if not Path(path).is_absolute() else Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")
    return pd.read_csv(path)


def get_wap_columns(df: pd.DataFrame) -> List[str]:
    wap_cols = [c for c in df.columns if c.startswith(WAP_PREFIX)]
    if not wap_cols:
        raise ValueError(f"No columns starting with '{WAP_PREFIX}' in dataframe")
    return wap_cols


def reference_points(df: pd.DataFrame) -> pd.Series:
    return df[GROUP_COLS].astype(str).agg("|".join, axis=1)


def drop_all_missing_rows(
    df: pd.DataFrame, wap_cols: List[str], missing_value: int
) -> pd.DataFrame:
    keep = ~(df[wap_cols] == missing_value).all(axis=1)
    dropped = int((~keep).sum())
    if dropped:
        logger.info(f"dropped {dropped} rows with no detected AP")
    return df[keep].reset_index(drop=True)


def drop_duplicate_rows(df: pd.DataFrame, wap_cols: List[str]) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=wap_cols).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        logger.info(f"dropped {dropped} duplicate rows ({before} -> {len(df)})")
    return df


def split_train_val(
    df: pd.DataFrame, train_ratio: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(df, groups=reference_points(df)))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    n_train_rp = reference_points(train_df).nunique()
    n_val_rp = reference_points(val_df).nunique()
    logger.info(
        f"group split: train {len(train_df)} rows / {n_train_rp} RPs, "
        f"val {len(val_df)} rows / {n_val_rp} RPs"
    )
    return train_df, val_df


def rp_keys(df: pd.DataFrame) -> pd.DataFrame:
    keys = df[RP_COLS].copy()
    coord_cols = [LONGITUDE_COL, LATITUDE_COL]
    keys[coord_cols] = keys[coord_cols].round(RP_COORD_DECIMALS)
    return keys


def build_rp_table(df: pd.DataFrame) -> pd.DataFrame:
    table = (
        rp_keys(df)
        .drop_duplicates()
        .sort_values(RP_COLS)
        .reset_index(drop=True)
        .reset_index(names=RP_ID_COL)
    )
    logger.info(f"{len(table)} reference points from {RP_COLS}")
    return table


def assign_rp(df: pd.DataFrame, table: pd.DataFrame) -> np.ndarray:
    merged = rp_keys(df).merge(table, on=RP_COLS, how="left")
    return merged[RP_ID_COL].fillna(-1).to_numpy(dtype=np.int64)


def fingerprint(config: Dict[str, Any]) -> str:
    payload = {"data": config["data"], "seed": config["seed"]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def build_cache(config: Dict[str, Any]) -> None:
    data_cfg = config["data"]
    seed = config["seed"]
    out_dir = cache_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_raw(data_cfg["raw_train"])
    test_df = load_raw(data_cfg["raw_test"])
    wap_cols = get_wap_columns(train_df)
    logger.info(f"raw: train {len(train_df)} rows, test {len(test_df)} rows, {len(wap_cols)} APs")

    if data_cfg["drop_all_missing"]:
        train_df = drop_all_missing_rows(train_df, wap_cols, data_cfg["missing_rssi_value"])
    if data_cfg["drop_duplicates"]:
        train_df = drop_duplicate_rows(train_df, wap_cols)

    rp_table = build_rp_table(train_df)
    np.save(
        out_dir / RP_COORD_FILENAME,
        rp_table[[LONGITUDE_COL, LATITUDE_COL]].to_numpy(dtype=COORD_DTYPE),
    )

    train_split, val_split = split_train_val(train_df, data_cfg["train_val_split"], seed)

    for name, split_df in {"train": train_split, "val": val_split, "test": test_df}.items():
        rssi = split_df[wap_cols].to_numpy(dtype=np.float32)
        rssi[rssi == data_cfg["missing_rssi_value"]] = data_cfg["rssi_min"]

        rp = assign_rp(split_df, rp_table)
        np.save(out_dir / f"rp_{name}.npy", rp)

        np.save(out_dir / f"rssi_{name}.npy", rssi)
        np.save(out_dir / f"building_{name}.npy", split_df[BUILDING_COL].to_numpy(dtype=np.int64))
        np.save(out_dir / f"floor_{name}.npy", split_df[FLOOR_COL].to_numpy(dtype=np.int64))
        np.save(
            out_dir / f"coord_{name}.npy",
            split_df[[LONGITUDE_COL, LATITUDE_COL]].to_numpy(dtype=COORD_DTYPE),
        )
        logger.info(
            f"{name}: rssi {rssi.shape}, dBm range [{rssi.min():.1f}, {rssi.max():.1f}], "
            f"{int((rp < 0).sum())} rows without a reference point"
        )

    with open(out_dir / META_FILENAME, "w", encoding="utf-8") as f:
        json.dump({"fingerprint": fingerprint(config)}, f, indent=2)

    logger.info(f"cached under {out_dir}")


def is_cache_valid(config: Dict[str, Any]) -> bool:
    out_dir = cache_dir(config)
    expected = [f"{array}_{split}.npy" for split in SPLITS for array in ARRAYS]
    expected += [RP_COORD_FILENAME, META_FILENAME]
    if not all((out_dir / fname).exists() for fname in expected):
        return False

    with open(out_dir / META_FILENAME, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta.get("fingerprint") == fingerprint(config)


def load_split(config: Dict[str, Any], split: str) -> Dict[str, np.ndarray]:
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split!r} (expected one of {SPLITS})")
    out_dir = cache_dir(config)
    return {array: np.load(out_dir / f"{array}_{split}.npy") for array in ARRAYS}


def load_rp_coord(config: Dict[str, Any]) -> np.ndarray:
    return np.load(cache_dir(config) / RP_COORD_FILENAME)


if __name__ == "__main__":
    import argparse

    from common.config import load_config

    parser = argparse.ArgumentParser(description="Build the UJIIndoorLoc cache")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_cache(load_config(args.config))
