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


def plot_dataset_family_stability(
    samples: pd.DataFrame,
    output_path: str | Path,
    title: str,
) -> None:
    summary = samples.groupby("plant_family", dropna=False)["stable"].mean().mul(100.0)
    summary = summary.sort_values(ascending=False)
    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.bar(summary.index, summary.values, color="#4C72B0")
    axis.set_title(title)
    axis.set_ylabel("Stable Fraction [%]")
    axis.set_ylim(0, 100)
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_gain_distributions(
    samples: pd.DataFrame,
    output_path: str | Path,
    title: str,
) -> None:
    sampled = (
        samples.sample(min(120_000, len(samples)), random_state=42)
        if len(samples) > 120_000
        else samples
    )
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for axis, column, label in zip(
        axes,
        ["kp", "ki", "kd"],
        ["Kp", "Ki", "Kd"],
        strict=False,
    ):
        stable_values = sampled.loc[sampled["stable"], column]
        unstable_values = sampled.loc[~sampled["stable"], column]
        bins = np.logspace(
            np.log10(max(sampled[column].min(), 1e-6)),
            np.log10(max(sampled[column].max(), 1e-5)),
            50,
        )
        axis.hist(stable_values, bins=bins, alpha=0.7, label="stable", color="#55A868")
        axis.hist(unstable_values, bins=bins, alpha=0.55, label="unstable", color="#C44E52")
        axis.set_xscale("log")
        axis.set_xlabel(label)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Count")
    axes[0].legend()
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_trajectory_amplitudes(
    peak_values: np.ndarray,
    output_path: str | Path,
    title: str,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 4.2))
    if peak_values.size:
        axis.hist(peak_values, bins=40, color="#4C72B0", alpha=0.85)
    axis.set_title(title)
    axis.set_xlabel("Peak |y(t)|")
    axis.set_ylabel("Count")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_error_distributions(
    sample_errors: pd.DataFrame,
    output_path: str | Path,
    title: str,
) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 4.2))
    for split, color in [("test", "#4C72B0"), ("ood_test", "#C44E52")]:
        split_errors = sample_errors.loc[
            sample_errors["split"] == split,
            "trajectory_rmse",
        ].to_numpy(dtype=float)
        if split_errors.size == 0:
            continue
        axis.hist(split_errors, bins=45, alpha=0.5, label=split, density=True, color=color)
    axis.set_title(title)
    axis.set_xlabel("Per-sample trajectory RMSE")
    axis.set_ylabel("Density")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_family_performance(
    family_metrics: pd.DataFrame,
    output_path: str | Path,
    title: str,
    metric_column: str,
    ylabel: str,
) -> None:
    ordered = family_metrics.sort_values(metric_column)
    figure, axis = plt.subplots(figsize=(10, 4.5))
    colors = ["#55A868" if split == "test" else "#C44E52" for split in ordered["split"]]
    labels = [
        f"{family}\n({split})"
        for family, split in zip(ordered["plant_family"], ordered["split"], strict=False)
    ]
    axis.bar(labels, ordered[metric_column], color=colors)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_stability_boundary_slice(
    kp_values: np.ndarray,
    ki_values: np.ndarray,
    true_stability: np.ndarray,
    predicted_probability: np.ndarray,
    output_path: str | Path,
    title: str,
    fixed_kd: float,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    truth_plot = axes[0].imshow(
        true_stability,
        origin="lower",
        aspect="auto",
        extent=[kp_values.min(), kp_values.max(), ki_values.min(), ki_values.max()],
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
    )
    axes[0].set_title("Ground-Truth Stability")
    axes[0].set_xlabel("Kp")
    axes[0].set_ylabel("Ki")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")

    pred_plot = axes[1].imshow(
        predicted_probability,
        origin="lower",
        aspect="auto",
        extent=[kp_values.min(), kp_values.max(), ki_values.min(), ki_values.max()],
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    axes[1].set_title("Classifier P(stable)")
    axes[1].set_xlabel("Kp")
    axes[1].set_ylabel("Ki")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")

    figure.suptitle(f"{title} | fixed Kd={fixed_kd:.3g}")
    figure.colorbar(truth_plot, ax=axes[0], shrink=0.88)
    figure.colorbar(pred_plot, ax=axes[1], shrink=0.88)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
