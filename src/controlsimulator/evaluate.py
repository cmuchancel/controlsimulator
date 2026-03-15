from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import NearestNeighbors

from controlsimulator.config import EvaluationConfig
from controlsimulator.dataset import dataset_metadata, dataset_time_grid, iter_dataset_chunks
from controlsimulator.features import (
    FEATURE_COLUMNS,
    GAIN_FEATURE_COLUMNS,
    PLANT_FEATURE_COLUMNS,
    Standardizer,
    feature_matrix,
)
from controlsimulator.metrics import extract_response_metrics, metric_mae
from controlsimulator.models import StabilityClassifier, TrajectoryRegressor
from controlsimulator.plants import MAX_DEN_ORDER, MAX_NUM_ORDER, plant_from_sample_row
from controlsimulator.plotting import (
    plot_confusion_matrix,
    plot_error_distributions,
    plot_error_vs_continuous_feature,
    plot_error_vs_order,
    plot_family_performance,
    plot_response_overlays,
    plot_stability_boundary_slice,
)
from controlsimulator.simulate import closed_loop_is_stable
from controlsimulator.train import (
    load_classifier_checkpoint,
    load_regressor_checkpoint,
    predict_stability_probabilities,
    predict_trajectories,
)
from controlsimulator.utils import dump_json, ensure_dir, pick_device, resolve_path

RAW_FEATURE_COLUMNS = [
    *PLANT_FEATURE_COLUMNS,
    *[column for column in GAIN_FEATURE_COLUMNS if not column.startswith("log10_")],
]


@dataclass(slots=True)
class LoadedModels:
    classifier: StabilityClassifier
    regressor: TrajectoryRegressor
    feature_scaler: Standardizer
    trajectory_scaler: Standardizer
    mean_trajectory: np.ndarray
    device: str
    feature_columns: list[str]


@dataclass(slots=True)
class KNNBaseline:
    neighbors: NearestNeighbors
    trajectories: np.ndarray


@dataclass(slots=True)
class RegressionAccumulator:
    n_samples: int = 0
    value_count: int = 0
    model_squared_error: float = 0.0
    model_absolute_error: float = 0.0
    mean_squared_error: float = 0.0
    mean_absolute_error: float = 0.0
    knn_squared_error: float = 0.0
    knn_absolute_error: float = 0.0
    metric_values: dict[str, dict[str, list[float]]] = field(
        default_factory=lambda: {
            name: {"truth": [], "prediction": [], "mean": [], "knn": []}
            for name in ["overshoot_pct", "rise_time", "settling_time", "steady_state_error"]
        }
    )

    def update(
        self,
        time_grid: np.ndarray,
        truth: np.ndarray,
        predictions: np.ndarray,
        mean_baseline: np.ndarray,
        knn_baseline: np.ndarray,
    ) -> None:
        self.n_samples += int(truth.shape[0])
        self.value_count += int(truth.size)
        self.model_squared_error += float(np.square(predictions - truth).sum())
        self.model_absolute_error += float(np.abs(predictions - truth).sum())
        self.mean_squared_error += float(np.square(mean_baseline - truth).sum())
        self.mean_absolute_error += float(np.abs(mean_baseline - truth).sum())
        self.knn_squared_error += float(np.square(knn_baseline - truth).sum())
        self.knn_absolute_error += float(np.abs(knn_baseline - truth).sum())

        truth_metrics = [_metrics_to_dict(time_grid, trajectory) for trajectory in truth]
        prediction_metrics = [_metrics_to_dict(time_grid, trajectory) for trajectory in predictions]
        mean_metrics = [_metrics_to_dict(time_grid, trajectory) for trajectory in mean_baseline]
        knn_metrics = [_metrics_to_dict(time_grid, trajectory) for trajectory in knn_baseline]
        for metric_name, values in self.metric_values.items():
            values["truth"].extend([row[metric_name] for row in truth_metrics])
            values["prediction"].extend([row[metric_name] for row in prediction_metrics])
            values["mean"].extend([row[metric_name] for row in mean_metrics])
            values["knn"].extend([row[metric_name] for row in knn_metrics])

    def summary(self) -> dict[str, float]:
        if self.n_samples == 0:
            return {
                "n_stable_samples": 0,
                "trajectory_rmse": float("nan"),
                "trajectory_mae": float("nan"),
                "mean_baseline_trajectory_rmse": float("nan"),
                "mean_baseline_trajectory_mae": float("nan"),
                "knn_baseline_trajectory_rmse": float("nan"),
                "knn_baseline_trajectory_mae": float("nan"),
            }
        summary = {
            "n_stable_samples": int(self.n_samples),
            "trajectory_rmse": float(np.sqrt(self.model_squared_error / max(self.value_count, 1))),
            "trajectory_mae": float(self.model_absolute_error / max(self.value_count, 1)),
            "mean_baseline_trajectory_rmse": float(
                np.sqrt(self.mean_squared_error / max(self.value_count, 1))
            ),
            "mean_baseline_trajectory_mae": float(
                self.mean_absolute_error / max(self.value_count, 1)
            ),
            "knn_baseline_trajectory_rmse": float(
                np.sqrt(self.knn_squared_error / max(self.value_count, 1))
            ),
            "knn_baseline_trajectory_mae": float(
                self.knn_absolute_error / max(self.value_count, 1)
            ),
        }
        for metric_name, values in self.metric_values.items():
            truth_values = np.asarray(values["truth"], dtype=float)
            prediction_values = np.asarray(values["prediction"], dtype=float)
            mean_values = np.asarray(values["mean"], dtype=float)
            knn_values = np.asarray(values["knn"], dtype=float)
            summary[f"{metric_name}_mae"] = metric_mae(truth_values, prediction_values)
            summary[f"mean_baseline_{metric_name}_mae"] = metric_mae(truth_values, mean_values)
            summary[f"knn_baseline_{metric_name}_mae"] = metric_mae(truth_values, knn_values)
            summary[f"{metric_name}_defined_fraction"] = float(
                np.isfinite(prediction_values).mean()
            )
        return summary


def _raw_feature_columns_for(feature_columns: list[str]) -> list[str]:
    return [column for column in feature_columns if not column.startswith("log10_")]


def _classifier_columns_for(feature_columns: list[str]) -> list[str]:
    return list(
        dict.fromkeys(
            [
                *_raw_feature_columns_for(feature_columns),
                "sample_id",
                "plant_id",
                "plant_family",
                "plant_order",
                "stable",
                "split",
                "tau_d",
                "kp",
                "ki",
                "kd",
            ]
        )
    )


def _regressor_columns_for(feature_columns: list[str]) -> list[str]:
    return list(
        dict.fromkeys(
            [
                *_classifier_columns_for(feature_columns),
                "plant_min_damping_ratio",
                "closed_loop_oscillation_hz",
            ]
        )
    )


def _log_evaluation_stage(message: str) -> None:
    print(message, flush=True)


def load_models(run_dir: str | Path, device: str = "auto") -> LoadedModels:
    resolved = resolve_path(run_dir)
    model_device = pick_device(device)
    classifier_payload = load_classifier_checkpoint(resolved / "classifier.pt")
    regressor_payload = load_regressor_checkpoint(resolved / "regressor.pt")

    classifier = StabilityClassifier(
        input_dim=int(classifier_payload["input_dim"]),
        hidden_sizes=list(classifier_payload["hidden_sizes"]),
        dropout=float(classifier_payload["dropout"]),
        activation=str(classifier_payload.get("activation", "gelu")),
    ).to(model_device)
    classifier.load_state_dict(classifier_payload["model_state"])
    classifier.eval()

    regressor = TrajectoryRegressor(
        input_dim=int(regressor_payload["input_dim"]),
        output_dim=int(regressor_payload["output_dim"]),
        hidden_sizes=list(regressor_payload["hidden_sizes"]),
        dropout=float(regressor_payload["dropout"]),
        activation=str(regressor_payload.get("activation", "gelu")),
    ).to(model_device)
    regressor.load_state_dict(regressor_payload["model_state"])
    regressor.eval()

    return LoadedModels(
        classifier=classifier,
        regressor=regressor,
        feature_scaler=Standardizer.from_dict(classifier_payload["feature_scaler"]),
        trajectory_scaler=Standardizer.from_dict(regressor_payload["trajectory_scaler"]),
        mean_trajectory=np.load(resolved / "mean_trajectory.npy"),
        device=model_device,
        feature_columns=list(classifier_payload.get("feature_columns", FEATURE_COLUMNS)),
    )


def evaluate_models(config: EvaluationConfig) -> Path:
    dataset_dir = resolve_path(config.dataset_dir)
    metadata = dataset_metadata(dataset_dir)
    time_grid = dataset_time_grid(dataset_dir)
    models = load_models(config.run_dir)
    report_dir = ensure_dir(resolve_path(config.report_dir()))
    plots_dir = ensure_dir(report_dir / "plots")
    dataset_root = dataset_dir.parent.parent
    artifact_plots_dir = ensure_dir(dataset_root / "plots" / config.name)

    majority_baseline_prediction = int(
        metadata["split_stable_fraction_pct"].get("train", 0.0) >= 50.0
    )
    _log_evaluation_stage(f"[eval] knn-baseline {config.name}")
    knn_baseline = _build_knn_baseline(
        dataset_dir=dataset_dir,
        models=models,
        config=config,
    )

    evaluation: dict[str, Any] = {
        "feature_columns": models.feature_columns,
        "classifier": {},
        "regressor": {},
    }
    classifier_rows: list[dict[str, Any]] = []
    sample_error_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []

    for split in ["test", "ood_test"]:
        _log_evaluation_stage(f"[eval] split {config.name} {split}")
        classifier_true: list[np.ndarray] = []
        classifier_pred: list[np.ndarray] = []

        split_accumulator = RegressionAccumulator()
        family_accumulators: dict[str, RegressionAccumulator] = {}
        split_plant_rows: list[pd.DataFrame] = []

        for frame, trajectories in iter_dataset_chunks(
            dataset_dir,
            include_trajectories=True,
            splits={split},
            columns=_regressor_columns_for(models.feature_columns),
            progress_desc=f"eval:{config.name}:{split}",
        ):
            features = models.feature_scaler.transform(
                feature_matrix(frame, feature_columns=models.feature_columns)
            ).astype(np.float32)
            stability_probabilities = predict_stability_probabilities(
                models.classifier,
                features,
                batch_size=config.inference_batch_size,
            )
            stability_predictions = (stability_probabilities >= 0.5).astype(int)
            true_labels = frame["stable"].to_numpy(dtype=int)
            classifier_true.append(true_labels)
            classifier_pred.append(stability_predictions)
            classifier_rows.extend(
                {
                    "split": split,
                    "plant_family": family,
                    "true": int(true_label),
                    "pred": int(prediction),
                }
                for family, true_label, prediction in zip(
                    frame["plant_family"].to_numpy(dtype=str),
                    true_labels,
                    stability_predictions,
                    strict=False,
                )
            )
            split_plant_rows.append(frame.head(64).copy())

            stable_mask = frame["stable"].to_numpy(dtype=bool)
            if not np.any(stable_mask):
                continue
            if trajectories is None:
                raise RuntimeError("Expected stable trajectories during regression evaluation.")
            stable_frame = frame.loc[stable_mask].reset_index(drop=True)
            truth = trajectories[stable_mask]
            stable_features = features[stable_mask]
            predictions = predict_trajectories(
                models.regressor,
                stable_features,
                models.trajectory_scaler,
                batch_size=config.inference_batch_size,
            ).astype(np.float32)
            mean_baseline = np.repeat(
                models.mean_trajectory[None, :],
                truth.shape[0],
                axis=0,
            ).astype(np.float32)
            knn_prediction = _predict_knn_baseline(knn_baseline, stable_features).astype(np.float32)

            split_accumulator.update(
                time_grid=time_grid,
                truth=truth,
                predictions=predictions,
                mean_baseline=mean_baseline,
                knn_baseline=knn_prediction,
            )
            sample_error_rows.extend(
                _sample_error_rows(
                    split=split,
                    frame=stable_frame,
                    truth=truth,
                    predictions=predictions,
                )
            )
            grouped_indices = stable_frame.groupby("plant_family", dropna=False).indices
            for family, indices in grouped_indices.items():
                family_indices = np.asarray(list(indices), dtype=int)
                accumulator = family_accumulators.setdefault(family, RegressionAccumulator())
                accumulator.update(
                    time_grid=time_grid,
                    truth=truth[family_indices],
                    predictions=predictions[family_indices],
                    mean_baseline=mean_baseline[family_indices],
                    knn_baseline=knn_prediction[family_indices],
                )

        true_labels = np.concatenate(classifier_true)
        predicted_labels = np.concatenate(classifier_pred)
        baseline_labels = np.full_like(true_labels, majority_baseline_prediction)
        evaluation["classifier"][split] = _classifier_metrics(
            true_labels,
            predicted_labels,
            baseline_labels,
        )
        evaluation["regressor"][split] = split_accumulator.summary()
        _log_evaluation_stage(
            "[eval] complete "
            f"{config.name} {split} "
            f"clf_f1={evaluation['classifier'][split]['f1']:.4f} "
            f"traj_rmse={evaluation['regressor'][split]['trajectory_rmse']:.4f}"
        )

        matrix = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])
        confusion_path = plots_dir / f"{split}_stability_confusion.png"
        plot_confusion_matrix(
            matrix,
            labels=["unstable", "stable"],
            output_path=confusion_path,
            title=f"Stability Confusion: {split}",
        )
        shutil.copy2(confusion_path, artifact_plots_dir / confusion_path.name)

        split_classifier_rows = pd.DataFrame(classifier_rows).query("split == @split")
        for family, rows in split_classifier_rows.groupby("plant_family"):
            family_rows.append(
                {
                    "split": split,
                    "plant_family": family,
                    "task": "classifier",
                    "n_samples": int(rows.shape[0]),
                    "accuracy": float(accuracy_score(rows["true"], rows["pred"])),
                    "f1": float(f1_score(rows["true"], rows["pred"], zero_division=0)),
                    "trajectory_rmse": np.nan,
                }
            )
        for family, accumulator in family_accumulators.items():
            summary = accumulator.summary()
            family_rows.append(
                {
                    "split": split,
                    "plant_family": family,
                    "task": "regressor",
                    "n_samples": int(summary["n_stable_samples"]),
                    "accuracy": np.nan,
                    "f1": np.nan,
                    "trajectory_rmse": summary["trajectory_rmse"],
                    "trajectory_mae": summary["trajectory_mae"],
                    "mean_baseline_trajectory_rmse": summary["mean_baseline_trajectory_rmse"],
                    "knn_baseline_trajectory_rmse": summary["knn_baseline_trajectory_rmse"],
                }
            )

        if split_plant_rows:
            split_reference = pd.concat(split_plant_rows, ignore_index=True)
        else:
            split_reference = pd.DataFrame()
        if not split_reference.empty:
            _plot_boundary_slice_for_split(
                split=split,
                reference_frame=split_reference,
                models=models,
                plots_dir=plots_dir,
                artifact_plots_dir=artifact_plots_dir,
                tau_d=float(split_reference["tau_d"].iloc[0]),
                batch_size=config.inference_batch_size,
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

    frequency_path = plots_dir / "error_vs_oscillation_frequency.png"
    plot_error_vs_continuous_feature(
        sample_errors,
        feature_column="closed_loop_oscillation_hz",
        output_path=frequency_path,
        title="RMSE vs Closed-Loop Oscillation Frequency",
        xlabel="Closed-loop oscillation frequency [Hz]",
    )
    shutil.copy2(frequency_path, artifact_plots_dir / frequency_path.name)

    damping_path = plots_dir / "error_vs_damping_ratio.png"
    plot_error_vs_continuous_feature(
        sample_errors,
        feature_column="plant_min_damping_ratio",
        output_path=damping_path,
        title="RMSE vs Plant Damping Ratio",
        xlabel="Minimum plant damping ratio",
    )
    shutil.copy2(damping_path, artifact_plots_dir / damping_path.name)

    order_path = plots_dir / "error_vs_plant_order.png"
    plot_error_vs_order(
        sample_errors,
        output_path=order_path,
        title="RMSE vs Plant Order",
    )
    shutil.copy2(order_path, artifact_plots_dir / order_path.name)

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

    if not sample_errors.empty:
        available_splits = set(sample_errors["split"].astype(str).unique())
    else:
        available_splits = set()
    for split in ["test", "ood_test"]:
        if split not in available_splits:
            continue
        cases = _representative_cases_for_split(
            dataset_dir=dataset_dir,
            split=split,
            sample_errors=sample_errors,
            models=models,
            knn_baseline=knn_baseline,
            time_grid=time_grid,
            batch_size=config.inference_batch_size,
        )
        response_path = plots_dir / f"{split}_response_examples.png"
        plot_response_overlays(
            time_grid,
            cases,
            output_path=response_path,
            title=f"Trajectory Prediction Examples: {split}",
        )
        shutil.copy2(response_path, artifact_plots_dir / response_path.name)
        shutil.copy2(response_path, artifact_plots_dir / f"{split}_hard_case_diagnostics.png")

    evaluation["artifacts_plot_dir"] = str(artifact_plots_dir)
    evaluation["family_metrics_path"] = "family_metrics.csv"
    evaluation["sample_errors_path"] = "sample_errors.csv"
    dump_json(evaluation, report_dir / "evaluation_summary.json")
    dump_json(evaluation, report_dir / "metrics.json")
    _write_markdown_report(evaluation, report_dir / "evaluation_summary.md")
    _log_evaluation_stage(f"[eval] complete {config.name}")
    return report_dir


def _build_knn_baseline(
    dataset_dir: Path,
    models: LoadedModels,
    config: EvaluationConfig,
) -> KNNBaseline:
    reservoir_features: list[np.ndarray] = []
    reservoir_trajectories: list[np.ndarray] = []
    seen = 0
    rng = np.random.default_rng(config.knn_seed)

    for frame, trajectories in iter_dataset_chunks(
        dataset_dir,
        include_trajectories=True,
        splits={"train"},
        stable_only=True,
        columns=_regressor_columns_for(models.feature_columns),
        progress_desc=f"eval-knn:{config.name}",
    ):
        if trajectories is None:
            raise RuntimeError("Expected trajectories when building the k-NN baseline.")
        scaled_features = models.feature_scaler.transform(
            feature_matrix(frame, feature_columns=models.feature_columns)
        ).astype(np.float32)
        for feature_row, trajectory in zip(scaled_features, trajectories, strict=False):
            seen += 1
            if len(reservoir_features) < config.knn_train_cap:
                reservoir_features.append(feature_row.copy())
                reservoir_trajectories.append(trajectory.copy())
                continue
            replacement_index = int(rng.integers(0, seen))
            if replacement_index < config.knn_train_cap:
                reservoir_features[replacement_index] = feature_row.copy()
                reservoir_trajectories[replacement_index] = trajectory.copy()

    if not reservoir_features:
        raise RuntimeError(
            "Unable to build the k-NN baseline because no stable train samples exist."
        )
    feature_matrix_train = np.vstack(reservoir_features).astype(np.float32)
    trajectory_matrix = np.vstack(reservoir_trajectories).astype(np.float32)
    neighbors = NearestNeighbors(n_neighbors=config.knn_neighbors, algorithm="auto")
    neighbors.fit(feature_matrix_train)
    return KNNBaseline(neighbors=neighbors, trajectories=trajectory_matrix)


def _predict_knn_baseline(knn: KNNBaseline, features: np.ndarray) -> np.ndarray:
    _, indices = knn.neighbors.kneighbors(features, return_distance=True)
    neighbor_trajectories = knn.trajectories[indices]
    return neighbor_trajectories.mean(axis=1).astype(np.float32)


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


def _sample_error_rows(
    split: str,
    frame: pd.DataFrame,
    truth: np.ndarray,
    predictions: np.ndarray,
) -> list[dict[str, Any]]:
    sample_rmse = np.sqrt(np.mean((predictions - truth) ** 2, axis=1))
    return [
        {
            "split": split,
            "sample_id": int(sample_id),
            "plant_family": str(family),
            "plant_order": int(order),
            "plant_min_damping_ratio": float(damping),
            "closed_loop_oscillation_hz": float(oscillation),
            "trajectory_rmse": float(error),
        }
        for sample_id, family, order, damping, oscillation, error in zip(
            frame["sample_id"].to_numpy(dtype=int),
            frame["plant_family"].to_numpy(dtype=str),
            frame["plant_order"].to_numpy(dtype=int),
            frame["plant_min_damping_ratio"].to_numpy(dtype=float),
            frame["closed_loop_oscillation_hz"].to_numpy(dtype=float),
            sample_rmse,
            strict=False,
        )
    ]


def _metrics_to_dict(time_grid: np.ndarray, trajectory: np.ndarray) -> dict[str, float]:
    return extract_response_metrics(time_grid, trajectory).to_dict()


def _plot_boundary_slice_for_split(
    split: str,
    reference_frame: pd.DataFrame,
    models: LoadedModels,
    plots_dir: Path,
    artifact_plots_dir: Path,
    tau_d: float,
    batch_size: int,
) -> None:
    plant_id = int(reference_frame["plant_id"].mode().iloc[0])
    plant_rows = reference_frame.loc[reference_frame["plant_id"] == plant_id].copy()
    anchor_row = plant_rows.iloc[0]
    plant = plant_from_sample_row(anchor_row)
    fixed_kd = float(max(plant_rows["kd"].median(), 1e-5))
    kp_center = float(max(plant_rows["kp"].median(), 1e-5))
    ki_center = float(max(plant_rows["ki"].median(), 1e-5))
    kp_values = np.geomspace(kp_center / 10.0, kp_center * 10.0, 60)
    ki_values = np.geomspace(ki_center / 10.0, ki_center * 10.0, 60)

    true_stability = np.zeros((ki_values.shape[0], kp_values.shape[0]), dtype=float)
    grid_rows: list[dict[str, Any]] = []
    for row_index, ki in enumerate(ki_values):
        for column_index, kp in enumerate(kp_values):
            true_stability[row_index, column_index] = float(
                closed_loop_is_stable(plant, float(kp), float(ki), fixed_kd, tau_d)
            )
            grid_rows.append(_plant_feature_row(anchor_row, float(kp), float(ki), fixed_kd))

    grid_frame = pd.DataFrame(grid_rows)
    grid_features = models.feature_scaler.transform(
        feature_matrix(grid_frame, feature_columns=models.feature_columns)
    ).astype(np.float32)
    predicted_probability = predict_stability_probabilities(
        models.classifier,
        grid_features,
        batch_size=batch_size,
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
    a2 = float(anchor_row.get("a2", anchor_row.get("den_2", 0.0)))
    a1 = float(anchor_row.get("a1", anchor_row.get("den_3", 0.0)))
    a0 = float(anchor_row.get("a0", anchor_row.get("den_4", 1.0)))
    b0 = float(anchor_row.get("b0", anchor_row.get("num_2", 1.0)))
    row = {
        "kp": kp,
        "ki": ki,
        "kd": kd,
        "b0": b0,
        "a2": a2,
        "a1": a1,
        "a0": a0,
        "dc_gain": float(anchor_row.get("dc_gain", b0 / max(abs(a0), 1e-8))),
        "dominant_pole_mag": float(anchor_row.get("dominant_pole_mag", 1.0)),
        "mean_pole_mag": float(anchor_row.get("mean_pole_mag", 1.0)),
        "plant_order": int(anchor_row.get("plant_order", 3)),
        "plant_min_damping_ratio": float(anchor_row.get("plant_min_damping_ratio", 1.0)),
        "plant_max_oscillation_hz": float(anchor_row.get("plant_max_oscillation_hz", 0.0)),
        "plant_pole_spread_log10": float(anchor_row.get("plant_pole_spread_log10", 0.0)),
        "plant_has_complex_poles": float(anchor_row.get("plant_has_complex_poles", 0.0)),
    }
    for prefix, width in [("num", MAX_NUM_ORDER + 1), ("den", MAX_DEN_ORDER + 1)]:
        for index in range(width):
            row[f"{prefix}_{index}"] = float(anchor_row.get(f"{prefix}_{index}", 0.0))
    return row


def _representative_cases_for_split(
    dataset_dir: Path,
    split: str,
    sample_errors: pd.DataFrame,
    models: LoadedModels,
    knn_baseline: KNNBaseline,
    time_grid: np.ndarray,
    batch_size: int,
) -> list[dict[str, np.ndarray | str]]:
    split_errors = sample_errors.loc[sample_errors["split"] == split].copy()
    if split_errors.empty:
        return []
    split_errors = split_errors.sort_values("trajectory_rmse").reset_index(drop=True)
    positions = [
        0,
        int(0.5 * (len(split_errors) - 1)),
        int(0.85 * (len(split_errors) - 1)),
        len(split_errors) - 1,
    ]
    labels = ["easy", "median", "hard", "failure"]
    target_ids = {
        int(split_errors.iloc[position]["sample_id"]): labels[index]
        for index, position in enumerate(positions)
    }

    collected: dict[int, dict[str, np.ndarray | str]] = {}
    for frame, trajectories in iter_dataset_chunks(
        dataset_dir,
        include_trajectories=True,
        splits={split},
        stable_only=True,
        columns=_regressor_columns_for(models.feature_columns),
        progress_desc=f"eval-cases:{dataset_dir.name}:{split}",
    ):
        if trajectories is None:
            raise RuntimeError("Expected trajectories for representative-case retrieval.")
        sample_ids = frame["sample_id"].to_numpy(dtype=int)
        mask = np.isin(sample_ids, np.array(list(target_ids.keys()), dtype=int))
        if not np.any(mask):
            continue
        selected_frame = frame.loc[mask].reset_index(drop=True)
        selected_truth = trajectories[mask]
        selected_features = models.feature_scaler.transform(
            feature_matrix(selected_frame, feature_columns=models.feature_columns)
        ).astype(np.float32)
        selected_predictions = predict_trajectories(
            models.regressor,
            selected_features,
            models.trajectory_scaler,
            batch_size=batch_size,
        ).astype(np.float32)
        knn_prediction = _predict_knn_baseline(knn_baseline, selected_features)
        mean_baseline = np.repeat(models.mean_trajectory[None, :], selected_truth.shape[0], axis=0)
        selected_rmse = np.sqrt(np.mean((selected_predictions - selected_truth) ** 2, axis=1))
        for idx, sample_id in enumerate(selected_frame["sample_id"].to_numpy(dtype=int)):
            collected[int(sample_id)] = {
                "label": (
                    f"{target_ids[int(sample_id)]} | sample {sample_id} | "
                    f"RMSE={selected_rmse[idx]:.3f}"
                ),
                "truth": selected_truth[idx],
                "prediction": selected_predictions[idx],
                "baseline": mean_baseline[idx],
                "knn_baseline": knn_prediction[idx],
            }
        if len(collected) == len(target_ids):
            break

    ordered_cases = []
    for sample_id in target_ids:
        if sample_id in collected:
            ordered_cases.append(collected[sample_id])
    return ordered_cases


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
    lines.append(f"- sample_errors_path: {evaluation['sample_errors_path']}")
    lines.append(f"- artifacts_plot_dir: {evaluation['artifacts_plot_dir']}")
    Path(path).write_text("\n".join(lines), encoding="utf-8")
