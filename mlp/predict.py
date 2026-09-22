from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from mlp.model import MLP


@torch.no_grad()
def predict(
    model: MLP,
    rssi_encoded: np.ndarray,
    rp_coord: np.ndarray,
    coord_mean: np.ndarray,
    coord_std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    model.eval()
    inputs = torch.from_numpy(rssi_encoded)
    collected: Dict[str, list] = {name: [] for name in model.heads}

    for start in range(0, len(inputs), batch_size):
        outputs = model(inputs[start : start + batch_size].to(device))
        for name, value in outputs.items():
            collected[name].append(value.cpu().numpy())

    logits = {name: np.concatenate(parts) for name, parts in collected.items()}

    predictions: Dict[str, np.ndarray] = {}
    for name in ("building", "floor"):
        if name in logits:
            predictions[name] = logits[name].argmax(axis=1)

    if "coord" in logits:
        predictions["coord"] = logits["coord"] * coord_std + coord_mean
    elif "rp" in logits:
        predictions["coord"] = rp_coord[logits["rp"].argmax(axis=1)]

    return predictions
