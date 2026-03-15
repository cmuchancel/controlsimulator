from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_training_history(
    history: pd.DataFrame,
    title: str,
    output_path: str | Path,
    value_columns: tuple[str, str],
) -> None:
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(history["epoch"], history[value_columns[0]], label=value_columns[0])
    axis.plot(history["epoch"], history[value_columns[1]], label=value_columns[1])
    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Value")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_response_overlays(
    time_grid: np.ndarray,
    cases: list[dict[str, np.ndarray | str]],
    output_path: str | Path,
    title: str,
) -> None:
    columns = 2
    rows = int(np.ceil(len(cases) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(12, 3.8 * rows), squeeze=False)
    figure.suptitle(title)
    for axis, case in zip(axes.flat, cases, strict=False):
        axis.plot(time_grid, case["truth"], label="simulation", linewidth=2.2)
        axis.plot(time_grid, case["prediction"], label="surrogate", linewidth=2.0, linestyle="--")
        baseline = case.get("baseline")
        if baseline is not None:
            axis.plot(time_grid, baseline, label="mean baseline", linewidth=1.5, linestyle=":")
        axis.set_title(str(case["label"]))
        axis.set_xlabel("Time [s]")
        axis.set_ylabel("Output y(t)")
        axis.grid(alpha=0.3)
        axis.legend()
    for axis in axes.flat[len(cases) :]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_confusion_matrix(
    matrix: np.ndarray,
    labels: list[str],
    output_path: str | Path,
    title: str,
) -> None:
    figure, axis = plt.subplots(figsize=(4.5, 4.0))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_title(title)
    axis.set_xticks(range(len(labels)), labels=labels)
    axis.set_yticks(range(len(labels)), labels=labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, int(matrix[row, column]), ha="center", va="center")
    figure.colorbar(image, ax=axis, shrink=0.9)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
