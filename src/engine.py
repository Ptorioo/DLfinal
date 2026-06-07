from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from .metrics import binary_metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    epoch: int | None = None,
    total_epochs: int | None = None,
    progress_every: int = 25,
) -> dict[str, float]:
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    progress = _ProgressPrinter("train", loader, epoch, total_epochs, progress_every)

    for batch_index, batch in enumerate(loader, start=1):
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
        progress.update(batch_index, labels.size(0), total_loss)

    progress.finish(total_loss)

    metrics = binary_metrics(torch.cat(all_logits), torch.cat(all_labels))
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    epoch: int | None = None,
    total_epochs: int | None = None,
    progress_every: int = 25,
) -> dict[str, float]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    progress = _ProgressPrinter("val", loader, epoch, total_epochs, progress_every)

    for batch_index, batch in enumerate(loader, start=1):
        batch = _move_batch_to_device(batch, device)
        labels = batch["label"]
        logits = model(batch)["logits"]
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        all_logits.append(logits)
        all_labels.append(labels)
        progress.update(batch_index, labels.size(0), total_loss)

    progress.finish(total_loss)

    metrics = binary_metrics(torch.cat(all_logits), torch.cat(all_labels))
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


@torch.no_grad()
def evaluate_by_generator(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    progress_every: int = 25,
) -> dict[str, dict[str, float]]:
    model.eval()
    grouped_logits: dict[str, list[torch.Tensor]] = defaultdict(list)
    grouped_labels: dict[str, list[torch.Tensor]] = defaultdict(list)
    progress = _ProgressPrinter("by-generator", loader, None, None, progress_every, show_loss=False)

    for batch_index, batch in enumerate(loader, start=1):
        batch = _move_batch_to_device(batch, device)
        labels = batch["label"]
        logits = model(batch)["logits"]
        for index, generator in enumerate(batch["generator"]):
            grouped_logits[generator].append(logits[index].detach().cpu().unsqueeze(0))
            grouped_labels[generator].append(labels[index].detach().cpu().unsqueeze(0))
        progress.update(batch_index, labels.size(0), 0.0)

    progress.finish(0.0)

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


class _ProgressPrinter:
    def __init__(
        self,
        phase: str,
        loader: DataLoader,
        epoch: int | None,
        total_epochs: int | None,
        progress_every: int,
        show_loss: bool = True,
    ) -> None:
        self.phase = phase
        self.total_batches = len(loader)
        self.total_samples = len(loader.dataset)
        self.epoch = epoch
        self.total_epochs = total_epochs
        self.progress_every = progress_every
        self.start_time = time.perf_counter()
        self.seen_samples = 0
        self.last_printed_batch = 0
        self.show_loss = show_loss

    def update(self, batch_index: int, batch_size: int, total_loss: float) -> None:
        self.seen_samples += batch_size
        if self.progress_every <= 0:
            return
        if batch_index != 1 and batch_index % self.progress_every != 0 and batch_index != self.total_batches:
            return
        self._print(batch_index, total_loss, done=False)
        self.last_printed_batch = batch_index

    def finish(self, total_loss: float) -> None:
        if self.progress_every > 0 and self.last_printed_batch != self.total_batches:
            self._print(self.total_batches, total_loss, done=True)
            self.last_printed_batch = self.total_batches
        elif self.progress_every > 0:
            print(flush=True)

    def _print(self, batch_index: int, total_loss: float, done: bool) -> None:
        elapsed = max(time.perf_counter() - self.start_time, 1e-8)
        batches_per_second = batch_index / elapsed
        samples_per_second = self.seen_samples / elapsed
        remaining_batches = max(self.total_batches - batch_index, 0)
        eta_seconds = remaining_batches / batches_per_second if batches_per_second > 0 else 0.0
        avg_loss = total_loss / max(self.seen_samples, 1)
        epoch_text = ""
        if self.epoch is not None and self.total_epochs is not None:
            epoch_text = f" epoch {self.epoch}/{self.total_epochs}"
        percent = 100.0 * batch_index / max(self.total_batches, 1)
        bar = _progress_bar(batch_index, self.total_batches)
        end = "\n" if done else "\r"
        loss_text = f" loss={avg_loss:.4f}" if self.show_loss else ""
        print(
            f"[{self.phase}{epoch_text}] {bar} "
            f"{batch_index}/{self.total_batches} batches "
            f"({percent:5.1f}%){loss_text} "
            f"{samples_per_second:.1f} img/s eta={_format_duration(eta_seconds)}",
            end=end,
            flush=True,
        )


def _progress_bar(current: int, total: int, width: int = 24) -> str:
    filled = int(width * current / max(total, 1))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _format_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m"
    return f"{minutes:d}m{seconds:02d}s"
