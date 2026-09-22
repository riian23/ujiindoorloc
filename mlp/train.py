from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from common.data import load_rp_coord, load_split
from common.encode import encode
from common.metrics import evaluate
from mlp.model import MLP
from mlp.predict import predict

logger = logging.getLogger(__name__)

TARGET_ORDER = ("building", "floor", "rp", "coord")


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def coord_stats(train_coord: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return train_coord.mean(axis=0), train_coord.std(axis=0)


def head_dims(train: Dict[str, np.ndarray], rp_coord: np.ndarray) -> Dict[str, int]:
    return {
        "building": int(train["building"].max()) + 1,
        "floor": int(train["floor"].max()) + 1,
        "rp": len(rp_coord),
        "coord": 2,
    }


def load_encoded(config: Dict[str, Any], split: str) -> Dict[str, np.ndarray]:
    data = load_split(config, split)
    data["rssi"] = encode(
        config["mlp"]["encoder"],
        data["rssi"],
        config["data"]["rssi_min"],
        config["data"]["rssi_max"],
    )
    return data


def build_dataset(
    data: Dict[str, np.ndarray], coord_mean: np.ndarray, coord_std: np.ndarray
) -> TensorDataset:
    standardised = (data["coord"] - coord_mean) / coord_std
    return TensorDataset(
        torch.from_numpy(data["rssi"]),
        torch.from_numpy(data["building"]),
        torch.from_numpy(data["floor"]),
        torch.from_numpy(data["rp"]),
        torch.from_numpy(standardised.astype(np.float32)),
    )


def batch_loss(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    weights: Dict[str, float],
) -> torch.Tensor:
    total = torch.zeros((), device=next(iter(outputs.values())).device)
    for name, logits in outputs.items():
        if name == "coord":
            loss = nn.functional.mse_loss(logits, targets[name])
        else:
            loss = nn.functional.cross_entropy(logits, targets[name])
        total = total + weights[name] * loss
    return total


def train_run(config: Dict[str, Any], run_name: str) -> Tuple[MLP, Dict[str, Any]]:
    mlp_cfg = config["mlp"]
    run_cfg = mlp_cfg["runs"][run_name]
    train_cfg = mlp_cfg["training"]
    weights = run_cfg["heads"]
    select = run_cfg["select"]

    set_seed(config["seed"])
    device = get_device()

    train_data = load_encoded(config, "train")
    val_data = load_encoded(config, "val")
    rp_coord = load_rp_coord(config)
    coord_mean, coord_std = coord_stats(train_data["coord"])

    heads = {name: head_dims(train_data, rp_coord)[name] for name in weights}
    model = MLP(
        input_dim=train_data["rssi"].shape[1],
        heads=heads,
        **mlp_cfg["backbone"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    loader = DataLoader(
        build_dataset(train_data, coord_mean, coord_std),
        batch_size=train_cfg["batch_size"],
        shuffle=True,
    )
    true_val = {name: val_data[name] for name in ("building", "floor", "coord")}

    logger.info(f"{run_name}: heads {heads} on {device}")

    best: Dict[str, Any] = {"score": None, "epoch": 0, "metrics": {}, "state": None}
    history: list[Dict[str, float]] = []
    waited = 0

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        running_loss = 0.0
        for batch in loader:
            inputs = batch[0].to(device)
            targets = {
                name: tensor.to(device)
                for name, tensor in zip(TARGET_ORDER, batch[1:])
                if name in heads
            }
            optimizer.zero_grad()
            loss = batch_loss(model(inputs), targets, weights)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(inputs)

        metrics = evaluate(
            predict(
                model,
                val_data["rssi"],
                rp_coord,
                coord_mean,
                coord_std,
                device,
                train_cfg["batch_size"],
            ),
            true_val,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / len(loader.dataset),
                **metrics,
            }
        )
        score = metrics[select["metric"]]
        improved = best["score"] is None or (
            score > best["score"] if select["mode"] == "max" else score < best["score"]
        )

        if improved:
            best = {
                "score": score,
                "epoch": epoch,
                "metrics": metrics,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
            waited = 0
            logger.info(f"{run_name}: epoch {epoch} {select['metric']}={score:.4f}")
        else:
            waited += 1
            if waited >= train_cfg["patience"]:
                logger.info(f"{run_name}: early stop at epoch {epoch}")
                break

    model.load_state_dict(best["state"])
    info = {
        "best_epoch": best["epoch"],
        "val_metrics": best["metrics"],
        "coord_mean": coord_mean,
        "coord_std": coord_std,
        "heads": heads,
        "history": history,
    }
    logger.info(
        f"{run_name}: best epoch {best['epoch']}, "
        f"val {select['metric']}={best['score']:.4f}"
    )
    return model, info


def load_merged(config: Dict[str, Any]) -> Dict[str, np.ndarray]:
    train = load_encoded(config, "train")
    val = load_encoded(config, "val")
    return {name: np.concatenate([train[name], val[name]]) for name in train}


def train_merged(
    config: Dict[str, Any], run_name: str, epochs: int
) -> Tuple[MLP, Dict[str, Any]]:
    mlp_cfg = config["mlp"]
    run_cfg = mlp_cfg["runs"][run_name]
    train_cfg = mlp_cfg["training"]
    weights = run_cfg["heads"]

    set_seed(config["seed"])
    device = get_device()

    data = load_merged(config)
    rp_coord = load_rp_coord(config)
    coord_mean, coord_std = coord_stats(data["coord"])

    heads = {name: head_dims(data, rp_coord)[name] for name in weights}
    model = MLP(
        input_dim=data["rssi"].shape[1],
        heads=heads,
        **mlp_cfg["backbone"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    loader = DataLoader(
        build_dataset(data, coord_mean, coord_std),
        batch_size=train_cfg["batch_size"],
        shuffle=True,
    )

    logger.info(f"{run_name}: merged {len(data['rssi'])} rows, {epochs} epochs, heads {heads}")

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in loader:
            inputs = batch[0].to(device)
            targets = {
                name: tensor.to(device)
                for name, tensor in zip(TARGET_ORDER, batch[1:])
                if name in heads
            }
            optimizer.zero_grad()
            loss = batch_loss(model(inputs), targets, weights)
            loss.backward()
            optimizer.step()

    return model, {
        "coord_mean": coord_mean,
        "coord_std": coord_std,
        "heads": heads,
        "epochs": epochs,
    }
