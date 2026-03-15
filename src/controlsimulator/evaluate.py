from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from controlsimulator.config import EvaluationConfig
from controlsimulator.dataset import load_dataset
from controlsimulator.features import FEATURE_COLUMNS, Standardizer, feature_matrix
from controlsimulator.metrics import (
    extract_response_metrics,
    metric_mae,
    trajectory_mae,
    trajectory_rmse,
)
from controlsimulator.models import StabilityClassifier, TrajectoryRegressor
from controlsimulator.plants import plant_from_sample_row
from controlsimulator.plotting import (
    plot_confusion_matrix,
    plot_error_distributions,
    plot_family_performance,
    plot_response_overlays,
    plot_stability_boundary_slice,
)
from controlsimulator.simulate import closed_loop_is_stable
from controlsimulator.train import load_classifier_checkpoint, load_regressor_checkpoint
from controlsimulator.utils import dump_json, ensure_dir, resolve_path


@dataclass(slots=True)
class LoadedModels:
    classifier: StabilityClassifier
    regressor: TrajectoryRegressor
    feature_scaler: Standardizer
    trajectory_scaler: Standardizer
    mean_trajectory: np.ndarray


def load_models(run_dir: str | Path) -> LoadedModels:
    resolved = resolve_path(run_dir)
    classifier_payload = load_classifier_checkpoint(resolved / "classifier.pt")
    regressor_payload = load_regressor_checkpoint(resolved / "regressor.pt")

    classifier = StabilityClassifier(
        input_dim=int(classifier_payload["input_dim"]),
        hidden_sizes=list(classifier_payload["hidden_sizes"]),
        dropout=float(classifier_payload["dropout"]),
    )
    classifier.load_state_dict(classifier_payload["model_state"])
    classifier.eval()

    regressor = TrajectoryRegressor(
        input_dim=int(regressor_payload["input_dim"]),
        output_dim=int(regressor_payload["output_dim"]),
        hidden_sizes=list(regressor_payload["hidden_sizes"]),
        dropout=float(regressor_payload["dropout"]),
    )
    regressor.load_state_dict(regressor_payload["model_state"])
    regressor.eval()

    return LoadedModels(
        classifier=classifier,
        regressor=regressor,
        feature_scaler=Standardizer.from_dict(classifier_payload["feature_scaler"]),
        trajectory_scaler=Standardizer.from_dict(regressor_payload["trajectory_scaler"]),
        mean_trajectory=np.load(resolved / "mean_trajectory.npy"),
    )


def evaluate_models(config: EvaluationConfig) -> Path:
    dataset = load_dataset(config.dataset_dir)
    models = load_models(config.run_dir)
    report_dir = ensure_dir(resolve_path(config.report_dir()))
    plots_dir = ensure_dir(report_dir / "plots")
    dataset_root = resolve_path(config.dataset_dir).parent.parent
    artifact_plots_dir = ensure_dir(dataset_root / "plots" / config.name)

    raw_features = feature_matrix(dataset.samples)
    scaled_features = models.feature_scaler.transform(raw_features).astype(np.float32)
    stability_probabilities = predict_stability_probabilities(models.classifier, scaled_features)
    stability_predictions = (stability_probabilities >= 0.5).astype(int)

    samples = dataset.samples.copy()
    samples["stability_probability"] = stability_probabilities
    samples["stability_prediction"] = stability_predictions
    train_stable_majority = int(samples.loc[samples["split"] == "train", "stable"].mean() >= 0.5)
    samples["majority_baseline_prediction"] = train_stable_majority

    evaluation: dict[str, Any] = {
        "feature_columns": FEATURE_COLUMNS,
        "classifier": {},
        "regressor": {},
    }
    sample_error_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []

    for split in ["test", "ood_test"]:
        split_frame = samples.loc[samples["split"] == split].copy()
        if split_frame.empty:
            continue

        true_labels = split_frame["stable"].to_numpy(dtype=int)
        predicted_labels = split_frame["stability_prediction"].to_numpy(dtype=int)
        baseline_labels = split_frame["majority_baseline_prediction"].to_numpy(dtype=int)
        evaluation["classifier"][split] = _classifier_metrics(
            true_labels,
            predicted_labels,
            baseline_labels,
        )
        family_rows.extend(_family_classifier_rows(split_frame, split))

        matrix = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])
        confusion_path = plots_dir / f"{split}_stability_confusion.png"
        plot_confusion_matrix(
            matrix,
            labels=["unstable", "stable"],
            output_path=confusion_path,
            title=f"Stability Confusion: {split}",
        )
        shutil.copy2(confusion_path, artifact_plots_dir / confusion_path.name)

        stable_split_indices = split_frame.index[split_frame["stable"]].to_numpy(dtype=int)
        if stable_split_indices.size == 0:
            continue
        truth = dataset.trajectories[stable_split_indices]
        stable_features = scaled_features[stable_split_indices]
        predictions = predict_trajectories(
            models.regressor,
            stable_features,
            models.trajectory_scaler,
        )
        baseline = np.repeat(models.mean_trajectory[None, :], stable_split_indices.size, axis=0)

        split_metrics, sample_rows = _regression_metrics(
            dataset.time_grid,
            truth,
            predictions,
            baseline,
            split=split,
            sample_ids=split_frame.loc[split_frame["stable"], "sample_id"].to_numpy(dtype=int),
            plant_families=split_frame.loc[split_frame["stable"], "plant_family"].to_numpy(
                dtype=str
            ),
        )
        evaluation["regressor"][split] = split_metrics
        sample_error_rows.extend(sample_rows)
        family_rows.extend(
            _family_regression_rows(
                split=split,
                time_grid=dataset.time_grid,
                truth=truth,
                predictions=predictions,
                baseline=baseline,
                plant_families=split_frame.loc[
                    split_frame["stable"],
                    "plant_family",
                ].to_numpy(dtype=str),
            )
        )

        cases = _representative_cases(
            dataset.time_grid,
            truth,
            predictions,
            baseline,
            split_frame.loc[split_frame["stable"], "sample_id"].to_numpy(dtype=int),
        )
        response_path = plots_dir / f"{split}_response_examples.png"
        plot_response_overlays(
            dataset.time_grid,
            cases,
            output_path=response_path,
            title=f"Trajectory Prediction Examples: {split}",
        )
        shutil.copy2(response_path, artifact_plots_dir / response_path.name)

        _plot_boundary_slice_for_split(
            split=split,
            split_frame=split_frame,
            models=models,
            plots_dir=plots_dir,
            artifact_plots_dir=artifact_plots_dir,
            tau_d=float(split_frame["tau_d"].iloc[0]),
        )

    sample_errors = pd.DataFrame(sample_error_rows)
    family_metrics = pd.DataFrame(family_rows)
    sample_errors.to_csv(report_dir / "sample_errors.csv", index=False)
    family_metrics.to_csv(report_dir / "family_metrics.csv", index=False)

    error_path = plots_dir / "error_distributions.png"
    plot_error_distributions(
        sample_errors,
        output_path=error_path,
        title="Trajectory Error Distributions",
    )
    shutil.copy2(error_path, artifact_plots_dir / error_path.name)

    classifier_family_path = plots_dir / "family_classifier_accuracy.png"
    plot_family_performance(
        family_metrics.loc[family_metrics["task"] == "classifier"],
        output_path=classifier_family_path,
        title="Classifier Accuracy By Family",
        metric_column="accuracy",
        ylabel="Accuracy",
    )
    shutil.copy2(classifier_family_path, artifact_plots_dir / classifier_family_path.name)

    regressor_family_path = plots_dir / "family_trajectory_rmse.png"
    plot_family_performance(
        family_metrics.loc[family_metrics["task"] == "regressor"],
        output_path=regressor_family_path,
        title="Trajectory RMSE By Family",
        metric_column="trajectory_rmse",
        ylabel="RMSE",
    )
    shutil.copy2(regressor_family_path, artifact_plots_dir / regressor_family_path.name)

    evaluation["artifacts_plot_dir"] = str(artifact_plots_dir)
    evaluation["family_metrics_path"] = "family_metrics.csv"
    dump_json(evaluation, report_dir / "evaluation_summary.json")
    _write_markdown_report(evaluation, report_dir / "evaluation_summary.md")
    return report_dir


def predict_stability_probabilities(
    model: StabilityClassifier,
    inputs: np.ndarray,
    batch_size: int = 2048,
) -> np.ndarray:
    tensor_inputs = torch.tensor(inputs, dtype=torch.float32)
    probabilities = []
    with torch.no_grad():
        for start in range(0, tensor_inputs.shape[0], batch_size):
            batch = tensor_inputs[start : start + batch_size]
            logits = model(batch)
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probabilities, axis=0)


def predict_trajectories(
    model: TrajectoryRegressor,
    inputs: np.ndarray,
    trajectory_scaler: Standardizer,
    batch_size: int = 2048,
) -> np.ndarray:
    tensor_inputs = torch.tensor(inputs, dtype=torch.float32)
    outputs = []
    with torch.no_grad():
        for start in range(0, tensor_inputs.shape[0], batch_size):
            batch = tensor_inputs[start : start + batch_size]
            outputs.append(model(batch).cpu().numpy())
    scaled = np.concatenate(outputs, axis=0)
    return trajectory_scaler.inverse_transform(scaled)


def _classifier_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    baseline_labels: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "precision": float(precision_score(true_labels, predicted_labels, zero_division=0)),
        "recall": float(recall_score(true_labels, predicted_labels, zero_division=0)),
        "f1": float(f1_score(true_labels, predicted_labels, zero_division=0)),
        "majority_accuracy": float(accuracy_score(true_labels, baseline_labels)),
        "majority_f1": float(f1_score(true_labels, baseline_labels, zero_division=0)),
    }


def _family_classifier_rows(split_frame: pd.DataFrame, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, family_frame in split_frame.groupby("plant_family", dropna=False):
        true_labels = family_frame["stable"].to_numpy(dtype=int)
        predictions = family_frame["stability_prediction"].to_numpy(dtype=int)
        rows.append(
            {
                "split": split,
                "plant_family": family,
                "task": "classifier",
                "n_samples": int(family_frame.shape[0]),
                "accuracy": float(accuracy_score(true_labels, predictions)),
                "f1": float(f1_score(true_labels, predictions, zero_division=0)),
                "trajectory_rmse": np.nan,
            }
        )
    return rows


def _family_regression_rows(
    split: str,
    time_grid: np.ndarray,
    truth: np.ndarray,
    predictions: np.ndarray,
    baseline: np.ndarray,
    plant_families: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    family_frame = pd.DataFrame({"plant_family": plant_families})
    grouped_indices = family_frame.groupby("plant_family", dropna=False).indices
    for family, family_indices in grouped_indices.items():
        indices = np.asarray(list(family_indices), dtype=int)
        truth_family = truth[indices]
        prediction_family = predictions[indices]
        baseline_family = baseline[indices]
        family_summary, _ = _regression_metrics(
            time_grid=time_grid,
            truth=truth_family,
            predictions=prediction_family,
            baseline=baseline_family,
            split=split,
            sample_ids=np.arange(indices.shape[0], dtype=int),
            plant_families=np.full(indices.shape[0], family, dtype=object),
        )
        rows.append(
            {
                "split": split,
                "plant_family": family,
                "task": "regressor",
                "n_samples": int(indices.shape[0]),
                "accuracy": np.nan,
                "f1": np.nan,
                "trajectory_rmse": family_summary["trajectory_rmse"],
                "trajectory_mae": family_summary["trajectory_mae"],
                "baseline_trajectory_rmse": family_summary["baseline_trajectory_rmse"],
            }
        )
    return rows


def _regression_metrics(
    time_grid: np.ndarray,
    truth: np.ndarray,
    predictions: np.ndarray,
    baseline: np.ndarray,
    split: str,
    sample_ids: np.ndarray,
    plant_families: np.ndarray,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    truth_metrics = [_metrics_to_dict(time_grid, sample) for sample in truth]
    prediction_metrics = [_metrics_to_dict(time_grid, sample) for sample in predictions]
    baseline_metrics = [_metrics_to_dict(time_grid, sample) for sample in baseline]

    metric_names = ["overshoot_pct", "rise_time", "settling_time", "steady_state_error"]
    summary = {
        "n_stable_samples": int(truth.shape[0]),
        "trajectory_rmse": trajectory_rmse(truth, predictions),
        "trajectory_mae": trajectory_mae(truth, predictions),
        "baseline_trajectory_rmse": trajectory_rmse(truth, baseline),
        "baseline_trajectory_mae": trajectory_mae(truth, baseline),
    }
    for metric_name in metric_names:
        truth_values = np.asarray([row[metric_name] for row in truth_metrics], dtype=float)
        predicted_values = np.asarray([row[metric_name] for row in prediction_metrics], dtype=float)
        baseline_values = np.asarray([row[metric_name] for row in baseline_metrics], dtype=float)
        summary[f"{metric_name}_mae"] = metric_mae(truth_values, predicted_values)
        summary[f"baseline_{metric_name}_mae"] = metric_mae(truth_values, baseline_values)
        summary[f"{metric_name}_defined_fraction"] = float(np.isfinite(predicted_values).mean())

    sample_rmse = np.sqrt(np.mean((predictions - truth) ** 2, axis=1))
    sample_rows = [
        {
            "split": split,
            "sample_id": int(sample_id),
            "plant_family": str(plant_family),
            "trajectory_rmse": float(error),
        }
        for sample_id, plant_family, error in zip(
            sample_ids,
            plant_families,
            sample_rmse,
            strict=False,
        )
    ]
    return summary, sample_rows


def _metrics_to_dict(time_grid: np.ndarray, trajectory: np.ndarray) -> dict[str, float]:
    return extract_response_metrics(time_grid, trajectory).to_dict()


def _representative_cases(
    time_grid: np.ndarray,
    truth: np.ndarray,
    predictions: np.ndarray,
    baseline: np.ndarray,
    sample_ids: np.ndarray,
) -> list[dict[str, np.ndarray | str]]:
    sample_rmse = np.sqrt(np.mean((predictions - truth) ** 2, axis=1))
    ordered = np.argsort(sample_rmse)
    candidate_positions = [
        0,
        int(0.5 * (len(ordered) - 1)),
        int(0.85 * (len(ordered) - 1)),
        len(ordered) - 1,
    ]
    selected = []
    seen = set()
    labels = ["easy", "median", "hard", "failure"]
    for position, label in zip(candidate_positions, labels, strict=False):
        index = ordered[position]
        if int(index) in seen:
            continue
        seen.add(int(index))
        selected.append(
            {
                "label": (
                    f"{label} | sample {int(sample_ids[index])} | RMSE={sample_rmse[index]:.3f}"
                ),
                "truth": truth[index],
                "prediction": predictions[index],
                "baseline": baseline[index],
            }
        )
    return selected


def _plot_boundary_slice_for_split(
    split: str,
    split_frame: pd.DataFrame,
    models: LoadedModels,
    plots_dir: Path,
    artifact_plots_dir: Path,
    tau_d: float,
) -> None:
    plant_id = int(split_frame["plant_id"].mode().iloc[0])
    plant_rows = split_frame.loc[split_frame["plant_id"] == plant_id].copy()
    anchor_row = plant_rows.iloc[0]
    plant = plant_from_sample_row(anchor_row)
    fixed_kd = float(plant_rows["kd"].median())
    kp_center = float(max(plant_rows["kp"].median(), 1e-5))
    ki_center = float(max(plant_rows["ki"].median(), 1e-5))
    kp_values = np.geomspace(kp_center / 8.0, kp_center * 8.0, 60)
    ki_values = np.geomspace(ki_center / 8.0, ki_center * 8.0, 60)

    true_stability = np.zeros((ki_values.shape[0], kp_values.shape[0]), dtype=float)
    grid_rows: list[dict[str, Any]] = []
    for row_index, ki in enumerate(ki_values):
        for column_index, kp in enumerate(kp_values):
            true_stability[row_index, column_index] = float(
                closed_loop_is_stable(plant, float(kp), float(ki), fixed_kd, tau_d)
            )
            grid_rows.append(_plant_feature_row(anchor_row, float(kp), float(ki), fixed_kd))

    grid_frame = pd.DataFrame(grid_rows)
    grid_features = models.feature_scaler.transform(feature_matrix(grid_frame)).astype(np.float32)
    predicted_probability = predict_stability_probabilities(
        models.classifier,
        grid_features,
    ).reshape(ki_values.shape[0], kp_values.shape[0])

    boundary_path = plots_dir / f"{split}_stability_boundary.png"
    plot_stability_boundary_slice(
        kp_values=kp_values,
        ki_values=ki_values,
        true_stability=true_stability,
        predicted_probability=predicted_probability,
        output_path=boundary_path,
        title=f"PID Stability Slice: {split} plant_id={plant_id}",
        fixed_kd=fixed_kd,
    )
    shutil.copy2(boundary_path, artifact_plots_dir / boundary_path.name)


def _plant_feature_row(
    anchor_row: pd.Series,
    kp: float,
    ki: float,
    kd: float,
) -> dict[str, Any]:
    row = {
        "kp": kp,
        "ki": ki,
        "kd": kd,
        "dc_gain": float(anchor_row["dc_gain"]),
        "dominant_pole_mag": float(anchor_row["dominant_pole_mag"]),
        "mean_pole_mag": float(anchor_row["mean_pole_mag"]),
        "plant_order": int(anchor_row["plant_order"]),
    }
    for prefix in ["num", "den"]:
        width = 2 if prefix == "num" else 4
        for index in range(width):
            row[f"{prefix}_{index}"] = float(anchor_row[f"{prefix}_{index}"])
    return row


def _write_markdown_report(evaluation: dict[str, Any], path: str | Path) -> None:
    lines = ["# Evaluation Summary", ""]
    for section_name in ["classifier", "regressor"]:
        lines.append(f"## {section_name.capitalize()}")
        lines.append("")
        for split, metrics in evaluation[section_name].items():
            lines.append(f"### {split}")
            for key, value in metrics.items():
                if isinstance(value, float):
                    lines.append(f"- {key}: {value:.4f}")
                else:
                    lines.append(f"- {key}: {value}")
            lines.append("")
    lines.append(f"- family_metrics_path: {evaluation['family_metrics_path']}")
    lines.append(f"- artifacts_plot_dir: {evaluation['artifacts_plot_dir']}")
    Path(path).write_text("\n".join(lines), encoding="utf-8")
