from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Sequence

from common.metrics import METRIC_ORDER


def format_value(name: str, value: float) -> str:
    if name.endswith("_m"):
        return f"{value:.2f} m"
    if name == "mean_floor_distance":
        return f"{value:.3f}"
    return f"{100 * value:.2f}%"


def print_table(rows: Dict[str, Dict[str, float]]) -> None:
    names = list(rows)
    width = max(len(n) for n in names + ["mean_floor_distance"]) + 2
    print("\n" + "metric".ljust(21) + "".join(n.ljust(width) for n in names))
    for metric in METRIC_ORDER:
        cells = [format_value(metric, rows[n][metric]).ljust(width) for n in names]
        print(metric.ljust(21) + "".join(cells))


def append_rows(path: Path, columns: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    exists = path.exists()
    if exists:
        with open(path, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), None)
        if header != list(columns):
            raise SystemExit(
                f"{path} has different columns than this version writes. Move it aside and rerun."
            )
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
