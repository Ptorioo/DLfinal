from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .metrics import binary_metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        labels = batch["label"]
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)["logits"]
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        all_logits.append(logits.detach())
        all_labels.append(labels.detach())

    metrics = binary_metrics(torch.cat(all_logits), torch.cat(all_labels))
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        labels = batch["label"]
        logits = model(batch)["logits"]
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        all_logits.append(logits)
        all_labels.append(labels)

    metrics = binary_metrics(torch.cat(all_logits), torch.cat(all_labels))
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


@torch.no_grad()
def evaluate_by_generator(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, dict[str, float]]:
    model.eval()
    grouped_logits: dict[str, list[torch.Tensor]] = defaultdict(list)
    grouped_labels: dict[str, list[torch.Tensor]] = defaultdict(list)

    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        labels = batch["label"]
        logits = model(batch)["logits"]
        for index, generator in enumerate(batch["generator"]):
            grouped_logits[generator].append(logits[index].detach().cpu().unsqueeze(0))
            grouped_labels[generator].append(labels[index].detach().cpu().unsqueeze(0))

    return {
        generator: binary_metrics(torch.cat(grouped_logits[generator]), torch.cat(grouped_labels[generator]))
        for generator in sorted(grouped_logits)
    }


def save_checkpoint(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metrics: dict[str, float]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def load_model_weights(path: str | Path, model: nn.Module, device: torch.device) -> dict[str, object]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    return checkpoint


def _move_batch_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
