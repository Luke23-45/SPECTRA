"""Trajectory plotting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt


def plot_loss_trajectory(history: Sequence[Mapping[str, object]], output_path: str | Path, title: str) -> str:
    output_path = str(output_path)
    epochs = [int(epoch["epoch"]) for epoch in history]
    train_losses = [float(epoch["train_total_loss"]) for epoch in history]
    val_losses = [float(epoch["val_total_loss"]) for epoch in history]
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_losses, label="train")
    plt.plot(epochs, val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path

