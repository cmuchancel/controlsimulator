from __future__ import annotations

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
from controlsimulator.plotting import plot_confusion_matrix, plot_response_overlays
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

    raw_features = feature_matrix(dataset.samples)
    scaled_features = models.feature_scaler.transform(raw_features).astype(np.float32)
    stability_probabilities = predict_stability_probabilities(models.classifier, scaled_features)
    stability_predictions = (stability_probabilities >= 0.5).astype(int)

    samples = dataset.samples.copy()
    samples["stability_probability"] = stability_probabilities
    samples["stability_prediction"] = stability_predictions
    train_stable_majority = int(samples.loc[samples["split"] == "train", "stable"].mean() >= 0.5)
    samples["majority_baseline_prediction"] = train_stable_majority

    stable_mask = samples["stable"].to_numpy(dtype=bool)
    stable_indices = np.where(stable_mask)[0]
    stable_features = scaled_features[stable_mask]
    stable_predictions = predict_trajectories(
        models.regressor,
        stable_features,
        models.trajectory_scaler,
    )
    all_regressor_predictions = np.full_like(dataset.trajectories, np.nan, dtype=np.float32)
    all_regressor_predictions[stable_indices] = stable_predictions.astype(np.float32)

    evaluation = {
        "feature_columns": FEATURE_COLUMNS,
        "classifier": {},
        "regressor": {},
    }

    sample_error_rows: list[dict[str, Any]] = []
    for split in ["test", "ood_test"]:
        split_frame = samples.loc[samples["split"] == split].copy()
        if split_frame.empty:
            continue

        true_labels = split_frame["stable"].to_numpy(dtype=int)
        pred_labels = split_frame["stability_prediction"].to_numpy(dtype=int)
        baseline_labels = split_frame["majority_baseline_prediction"].to_numpy(dtype=int)
        evaluation["classifier"][split] = _classifier_metrics(
            true_labels,
            pred_labels,
            baseline_labels,
        )

        matrix = confusion_matrix(true_labels, pred_labels, labels=[0, 1])
        plot_confusion_matrix(
            matrix,
            labels=["unstable", "stable"],
            output_path=plots_dir / f"{split}_stability_confusion.png",
            title=f"Stability Confusion: {split}",
        )

        stable_split_indices = split_frame.index[split_frame["stable"]].to_numpy(dtype=int)
        if stable_split_indices.size == 0:
            continue
        truth = dataset.trajectories[stable_split_indices]
        predictions = all_regressor_predictions[stable_split_indices]
        baseline = np.repeat(models.mean_trajectory[None, :], stable_split_indices.size, axis=0)
        split_metrics, sample_rows = _regression_metrics(
            dataset.time_grid,
            truth,
            predictions,
            baseline,
            split=split,
            sample_ids=split_frame.loc[split_frame["stable"], "sample_id"].to_numpy(dtype=int),
        )
        evaluation["regressor"][split] = split_metrics
        sample_error_rows.extend(sample_rows)

        cases = _representative_cases(
            dataset.time_grid,
            truth,
            predictions,
            baseline,
            split_frame.loc[split_frame["stable"], "sample_id"].to_numpy(dtype=int),
        )
        plot_response_overlays(
            dataset.time_grid,
            cases,
            output_path=plots_dir / f"{split}_response_examples.png",
            title=f"Trajectory Prediction Examples: {split}",
        )

    pd.DataFrame(sample_error_rows).to_csv(report_dir / "sample_errors.csv", index=False)
    dump_json(evaluation, report_dir / "evaluation_summary.json")
    _write_markdown_report(evaluation, report_dir / "evaluation_summary.md")
    return report_dir


def predict_stability_probabilities(
    model: StabilityClassifier,
    inputs: np.ndarray,
    batch_size: int = 1024,
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
    batch_size: int = 512,
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


def _regression_metrics(
    time_grid: np.ndarray,
    truth: np.ndarray,
    predictions: np.ndarray,
    baseline: np.ndarray,
    split: str,
    sample_ids: np.ndarray,
) -> tuple[dict[str, float], list[dict[str, float]]]:
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
            "trajectory_rmse": float(error),
        }
        for sample_id, error in zip(sample_ids, sample_rmse, strict=False)
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
    Path(path).write_text("\n".join(lines), encoding="utf-8")
