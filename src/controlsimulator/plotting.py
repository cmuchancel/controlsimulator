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


def plot_training_curves(
    classifier_history: pd.DataFrame,
    regressor_history: pd.DataFrame,
    output_path: str | Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(classifier_history["epoch"], classifier_history["train_loss"], label="train_loss")
    axes[0].plot(classifier_history["epoch"], classifier_history["val_loss"], label="val_loss")
    axes[0].set_title("Classifier")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(regressor_history["epoch"], regressor_history["train_loss"], label="train_loss")
    axes[1].plot(regressor_history["epoch"], regressor_history["val_loss"], label="val_loss")
    axes[1].set_title("Regressor")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    figure.suptitle("Training Curves")
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
        knn_baseline = case.get("knn_baseline")
        if knn_baseline is not None:
            axis.plot(
                time_grid,
                knn_baseline,
                label="k-NN baseline",
                linewidth=1.2,
                linestyle="-.",
            )
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
    samples_or_summary: pd.DataFrame,
    output_path: str | Path,
    title: str,
) -> None:
    if {"plant_family", "stable_fraction_pct"}.issubset(samples_or_summary.columns):
        summary = (
            samples_or_summary[["plant_family", "stable_fraction_pct"]]
            .drop_duplicates()
            .sort_values("stable_fraction_pct", ascending=False)
        )
        x_values = summary["plant_family"]
        y_values = summary["stable_fraction_pct"]
    else:
        summary = (
            samples_or_summary.groupby("plant_family", dropna=False)["stable"]
            .mean()
            .mul(100.0)
        )
        summary = summary.sort_values(ascending=False)
        x_values = summary.index
        y_values = summary.values

    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.bar(x_values, y_values, color="#4C72B0")
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


def plot_class_balance(
    stable_count: int,
    unstable_count: int,
    output_path: str | Path,
    title: str,
) -> None:
    figure, axis = plt.subplots(figsize=(5.5, 4.2))
    axis.bar(["stable", "unstable"], [stable_count, unstable_count], color=["#55A868", "#C44E52"])
    axis.set_title(title)
    axis.set_ylabel("Count")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_pole_distribution(
    plants: pd.DataFrame,
    output_path: str | Path,
    title: str,
) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 5.0))
    for pole_index in range(3):
        real_column = f"pole_{pole_index}_real"
        imag_column = f"pole_{pole_index}_imag"
        if real_column not in plants.columns or imag_column not in plants.columns:
            continue
        axis.scatter(
            plants[real_column],
            plants[imag_column],
            s=10,
            alpha=0.25,
            label=f"pole {pole_index}",
        )
    axis.axvline(0.0, color="#444444", linestyle="--", linewidth=1.0)
    axis.set_title(title)
    axis.set_xlabel("Real part")
    axis.set_ylabel("Imaginary part")
    axis.grid(alpha=0.25)
    axis.legend()
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


def plot_oscillation_frequency_distribution(
    oscillation_hz: np.ndarray,
    output_path: str | Path,
    title: str,
) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    finite_values = oscillation_hz[np.isfinite(oscillation_hz)]
    if finite_values.size:
        axis.hist(finite_values, bins=50, color="#DD8452", alpha=0.85)
    axis.set_title(title)
    axis.set_xlabel("Dominant oscillation frequency [Hz]")
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


def plot_error_vs_continuous_feature(
    sample_errors: pd.DataFrame,
    feature_column: str,
    output_path: str | Path,
    title: str,
    xlabel: str,
) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    for split, color in [("test", "#4C72B0"), ("ood_test", "#C44E52")]:
        split_frame = sample_errors.loc[sample_errors["split"] == split]
        if split_frame.empty:
            continue
        if split_frame.shape[0] > 80_000:
            split_frame = split_frame.sample(80_000, random_state=42)
        axis.scatter(
            split_frame[feature_column],
            split_frame["trajectory_rmse"],
            s=7,
            alpha=0.18,
            label=split,
            color=color,
        )
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Trajectory RMSE")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_error_vs_order(
    sample_errors: pd.DataFrame,
    output_path: str | Path,
    title: str,
) -> None:
    grouped = (
        sample_errors.groupby(["split", "plant_order"], dropna=False)["trajectory_rmse"]
        .mean()
        .reset_index()
    )
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    for split, color in [("test", "#4C72B0"), ("ood_test", "#C44E52")]:
        split_frame = grouped.loc[grouped["split"] == split]
        if split_frame.empty:
            continue
        axis.plot(
            split_frame["plant_order"],
            split_frame["trajectory_rmse"],
            marker="o",
            linewidth=2.0,
            color=color,
            label=split,
        )
    axis.set_title(title)
    axis.set_xlabel("Plant order")
    axis.set_ylabel("Mean trajectory RMSE")
    axis.grid(alpha=0.25)
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
