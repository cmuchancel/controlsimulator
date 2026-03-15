from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn

from controlsimulator.config import TrainingConfig, save_config_snapshot
from controlsimulator.dataset import dataset_metadata, dataset_time_grid, iter_dataset_chunks
from controlsimulator.features import (
    FEATURE_COLUMNS,
    GAIN_FEATURE_COLUMNS,
    PLANT_FEATURE_COLUMNS,
    RunningStatistics,
    Standardizer,
    feature_matrix,
)
from controlsimulator.metrics import trajectory_mae, trajectory_rmse
from controlsimulator.models import StabilityClassifier, TrajectoryRegressor
from controlsimulator.plotting import plot_training_history
from controlsimulator.utils import dump_json, ensure_dir, pick_device, resolve_path, set_global_seed

RAW_FEATURE_COLUMNS = [
    *PLANT_FEATURE_COLUMNS,
    *[column for column in GAIN_FEATURE_COLUMNS if not column.startswith("log10_")],
]
CLASSIFIER_COLUMNS = [*RAW_FEATURE_COLUMNS, "stable", "split"]
REGRESSOR_COLUMNS = [*RAW_FEATURE_COLUMNS, "stable", "split"]


@dataclass(slots=True)
class ClassifierMetrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float


@dataclass(slots=True)
class RegressorMetrics:
    loss: float
    rmse: float
    mae: float


@dataclass(slots=True)
class TrainingStatistics:
    feature_scaler: Standardizer
    trajectory_scaler: Standardizer
    positive_fraction: float
    train_samples: int
    val_samples: int
    stable_train_samples: int
    stable_val_samples: int
    mean_trajectory: np.ndarray


def train_models(config: TrainingConfig) -> Path:
    set_global_seed(config.seed)
    dataset_dir = resolve_path(config.dataset_dir)
    run_dir = ensure_dir(resolve_path(config.run_dir()))
    plots_dir = ensure_dir(run_dir / "plots")
    save_config_snapshot(config, run_dir / "config_snapshot.yaml")

    device = pick_device(config.device)
    time_grid = dataset_time_grid(dataset_dir)
    data_summary = dataset_metadata(dataset_dir)
    stats = _fit_training_statistics(dataset_dir)

    classifier_path, classifier_history, classifier_summary = _train_classifier(
        config=config,
        dataset_dir=dataset_dir,
        run_dir=run_dir,
        device=device,
        feature_scaler=stats.feature_scaler,
        positive_fraction=stats.positive_fraction,
    )
    plot_training_history(
        classifier_history,
        title="Stability Classifier",
        output_path=plots_dir / "classifier_training.png",
        value_columns=("train_loss", "val_loss"),
    )

    regressor_path, regressor_history, regressor_summary = _train_regressor(
        config=config,
        dataset_dir=dataset_dir,
        run_dir=run_dir,
        device=device,
        feature_scaler=stats.feature_scaler,
        trajectory_scaler=stats.trajectory_scaler,
        time_grid=time_grid,
    )
    plot_training_history(
        regressor_history,
        title="Trajectory Regressor",
        output_path=plots_dir / "regressor_training.png",
        value_columns=("train_loss", "val_loss"),
    )

    summary = {
        "device": device,
        "dataset_name": data_summary["name"],
        "feature_columns": FEATURE_COLUMNS,
        "classifier_checkpoint": str(classifier_path.relative_to(run_dir)),
        "regressor_checkpoint": str(regressor_path.relative_to(run_dir)),
        "train_samples": stats.train_samples,
        "val_samples": stats.val_samples,
        "stable_train_samples": stats.stable_train_samples,
        "stable_val_samples": stats.stable_val_samples,
        "classifier": classifier_summary,
        "regressor": regressor_summary,
        "mean_trajectory_path": "mean_trajectory.npy",
    }
    np.save(run_dir / "mean_trajectory.npy", stats.mean_trajectory.astype(np.float32))
    dump_json(summary, run_dir / "train_summary.json")
    return run_dir


def load_classifier_checkpoint(path: str | Path) -> dict[str, Any]:
    return torch.load(resolve_path(path), map_location="cpu", weights_only=False)


def load_regressor_checkpoint(path: str | Path) -> dict[str, Any]:
    return torch.load(resolve_path(path), map_location="cpu", weights_only=False)


def _fit_training_statistics(dataset_dir: Path) -> TrainingStatistics:
    feature_stats = RunningStatistics()
    trajectory_stats = RunningStatistics()
    stable_label_sum = 0
    train_samples = 0
    val_samples = 0
    stable_train_samples = 0
    stable_val_samples = 0
    mean_trajectory_sum: np.ndarray | None = None

    for frame, trajectories in iter_dataset_chunks(
        dataset_dir,
        include_trajectories=True,
        columns=REGRESSOR_COLUMNS,
    ):
        if trajectories is None:
            raise RuntimeError("Expected chunk trajectories when fitting training statistics.")

        splits = frame["split"].to_numpy(dtype=str)
        stable_mask = frame["stable"].to_numpy(dtype=bool)
        train_mask = splits == "train"
        val_mask = splits == "val"
        stable_train_mask = train_mask & stable_mask
        stable_val_mask = val_mask & stable_mask

        if np.any(train_mask):
            feature_stats.update(feature_matrix(frame.loc[train_mask, RAW_FEATURE_COLUMNS]))
        train_samples += int(train_mask.sum())
        val_samples += int(val_mask.sum())
        stable_train_samples += int(stable_train_mask.sum())
        stable_val_samples += int(stable_val_mask.sum())
        stable_label_sum += int(stable_mask[train_mask].sum())

        if np.any(stable_train_mask):
            stable_targets = trajectories[stable_train_mask]
            trajectory_stats.update(stable_targets)
            if mean_trajectory_sum is None:
                mean_trajectory_sum = stable_targets.sum(axis=0, dtype=np.float64)
            else:
                mean_trajectory_sum += stable_targets.sum(axis=0, dtype=np.float64)

    if mean_trajectory_sum is None or stable_train_samples == 0:
        raise RuntimeError("Training split does not contain any stable trajectories.")

    return TrainingStatistics(
        feature_scaler=feature_stats.to_standardizer(),
        trajectory_scaler=trajectory_stats.to_standardizer(),
        positive_fraction=float(stable_label_sum / max(train_samples, 1)),
        train_samples=train_samples,
        val_samples=val_samples,
        stable_train_samples=stable_train_samples,
        stable_val_samples=stable_val_samples,
        mean_trajectory=(mean_trajectory_sum / stable_train_samples).astype(np.float32),
    )


def _train_classifier(
    config: TrainingConfig,
    dataset_dir: Path,
    run_dir: Path,
    device: str,
    feature_scaler: Standardizer,
    positive_fraction: float,
) -> tuple[Path, pd.DataFrame, dict[str, float]]:
    model = StabilityClassifier(
        input_dim=len(FEATURE_COLUMNS),
        hidden_sizes=config.classifier_hidden_sizes,
        dropout=config.classifier_dropout,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [(1.0 - max(positive_fraction, 1e-4)) / max(positive_fraction, 1e-4)],
            device=device,
        )
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_state: dict[str, Any] | None = None
    best_f1 = -np.inf
    patience = 0
    history_rows: list[dict[str, float]] = []
    train_start = perf_counter()

    for epoch in range(1, config.epochs + 1):
        train_loss = _run_classifier_epoch(
            model=model,
            dataset_dir=dataset_dir,
            split="train",
            feature_scaler=feature_scaler,
            batch_size=config.batch_size,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            seed=config.seed + epoch,
        )
        val_metrics = _evaluate_classifier(
            model=model,
            dataset_dir=dataset_dir,
            split="val",
            feature_scaler=feature_scaler,
            batch_size=config.batch_size,
            device=device,
            criterion=criterion,
        )
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics.loss,
                "val_f1": val_metrics.f1,
                "val_accuracy": val_metrics.accuracy,
            }
        )
        if val_metrics.f1 > best_f1:
            best_f1 = val_metrics.f1
            patience = 0
            best_state = {
                "model_state": model.state_dict(),
                "feature_scaler": feature_scaler.to_dict(),
                "input_dim": len(FEATURE_COLUMNS),
                "hidden_sizes": config.classifier_hidden_sizes,
                "dropout": config.classifier_dropout,
            }
        else:
            patience += 1
            if patience >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("Classifier training did not produce a checkpoint.")

    checkpoint_path = run_dir / "classifier.pt"
    torch.save(best_state, checkpoint_path)
    history = pd.DataFrame(history_rows)
    history.to_csv(run_dir / "classifier_history.csv", index=False)
    best_metrics = history.sort_values("val_f1", ascending=False).iloc[0].to_dict()
    best_metrics["elapsed_seconds"] = perf_counter() - train_start
    return checkpoint_path, history, best_metrics


def _train_regressor(
    config: TrainingConfig,
    dataset_dir: Path,
    run_dir: Path,
    device: str,
    feature_scaler: Standardizer,
    trajectory_scaler: Standardizer,
    time_grid: np.ndarray,
) -> tuple[Path, pd.DataFrame, dict[str, float]]:
    model = TrajectoryRegressor(
        input_dim=len(FEATURE_COLUMNS),
        output_dim=time_grid.shape[0],
        hidden_sizes=config.hidden_sizes,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.MSELoss()

    best_state: dict[str, Any] | None = None
    best_rmse = np.inf
    patience = 0
    history_rows: list[dict[str, float]] = []
    train_start = perf_counter()

    for epoch in range(1, config.epochs + 1):
        train_loss = _run_regressor_epoch(
            model=model,
            dataset_dir=dataset_dir,
            split="train",
            feature_scaler=feature_scaler,
            trajectory_scaler=trajectory_scaler,
            batch_size=config.batch_size,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            seed=config.seed + 10_000 + epoch,
        )
        val_metrics = _evaluate_regressor(
            model=model,
            dataset_dir=dataset_dir,
            split="val",
            feature_scaler=feature_scaler,
            trajectory_scaler=trajectory_scaler,
            batch_size=config.batch_size,
            device=device,
            criterion=criterion,
        )
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics.loss,
                "val_rmse": val_metrics.rmse,
                "val_mae": val_metrics.mae,
            }
        )
        if val_metrics.rmse < best_rmse:
            best_rmse = val_metrics.rmse
            patience = 0
            best_state = {
                "model_state": model.state_dict(),
                "feature_scaler": feature_scaler.to_dict(),
                "trajectory_scaler": trajectory_scaler.to_dict(),
                "input_dim": len(FEATURE_COLUMNS),
                "output_dim": int(time_grid.shape[0]),
                "hidden_sizes": config.hidden_sizes,
                "dropout": config.dropout,
                "time_grid": time_grid.tolist(),
            }
        else:
            patience += 1
            if patience >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("Regressor training did not produce a checkpoint.")

    checkpoint_path = run_dir / "regressor.pt"
    torch.save(best_state, checkpoint_path)
    history = pd.DataFrame(history_rows)
    history.to_csv(run_dir / "regressor_history.csv", index=False)
    best_metrics = history.sort_values("val_rmse", ascending=True).iloc[0].to_dict()
    best_metrics["elapsed_seconds"] = perf_counter() - train_start
    return checkpoint_path, history, best_metrics


def _run_classifier_epoch(
    model: StabilityClassifier,
    dataset_dir: Path,
    split: str,
    feature_scaler: Standardizer,
    batch_size: int,
    device: str,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    seed: int,
) -> float:
    model.train()
    rng = np.random.default_rng(seed)
    losses = []
    for frame, _ in iter_dataset_chunks(
        dataset_dir,
        include_trajectories=False,
        splits={split},
        columns=CLASSIFIER_COLUMNS,
    ):
        features = feature_scaler.transform(feature_matrix(frame[RAW_FEATURE_COLUMNS])).astype(
            np.float32
        )
        targets = frame["stable"].to_numpy(dtype=np.float32)
        indices = rng.permutation(features.shape[0])
        features = features[indices]
        targets = targets[indices]
        for start in range(0, features.shape[0], batch_size):
            batch_inputs = torch.tensor(
                features[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            batch_targets = torch.tensor(
                targets[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            optimizer.zero_grad()
            logits = model(batch_inputs)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def _evaluate_classifier(
    model: StabilityClassifier,
    dataset_dir: Path,
    split: str,
    feature_scaler: Standardizer,
    batch_size: int,
    device: str,
    criterion: nn.Module,
) -> ClassifierMetrics:
    model.eval()
    losses: list[float] = []
    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    with torch.no_grad():
        for frame, _ in iter_dataset_chunks(
            dataset_dir,
            include_trajectories=False,
            splits={split},
            columns=CLASSIFIER_COLUMNS,
        ):
            features = feature_scaler.transform(feature_matrix(frame[RAW_FEATURE_COLUMNS])).astype(
                np.float32
            )
            targets = frame["stable"].to_numpy(dtype=np.float32)
            for start in range(0, features.shape[0], batch_size):
                batch_inputs = torch.tensor(
                    features[start : start + batch_size],
                    dtype=torch.float32,
                    device=device,
                )
                batch_targets = torch.tensor(
                    targets[start : start + batch_size],
                    dtype=torch.float32,
                    device=device,
                )
                logits = model(batch_inputs)
                loss = criterion(logits, batch_targets)
                probabilities = torch.sigmoid(logits)
                predictions = (probabilities >= 0.5).float()
                losses.append(float(loss.cpu()))
                all_targets.append(batch_targets.cpu().numpy())
                all_predictions.append(predictions.cpu().numpy())

    targets = np.concatenate(all_targets)
    predictions = np.concatenate(all_predictions)
    return ClassifierMetrics(
        loss=float(np.mean(losses)),
        accuracy=float(accuracy_score(targets, predictions)),
        precision=float(precision_score(targets, predictions, zero_division=0)),
        recall=float(recall_score(targets, predictions, zero_division=0)),
        f1=float(f1_score(targets, predictions, zero_division=0)),
    )


def _run_regressor_epoch(
    model: TrajectoryRegressor,
    dataset_dir: Path,
    split: str,
    feature_scaler: Standardizer,
    trajectory_scaler: Standardizer,
    batch_size: int,
    device: str,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    seed: int,
) -> float:
    model.train()
    rng = np.random.default_rng(seed)
    losses = []
    for frame, trajectories in iter_dataset_chunks(
        dataset_dir,
        include_trajectories=True,
        splits={split},
        stable_only=True,
        columns=REGRESSOR_COLUMNS,
    ):
        if trajectories is None:
            raise RuntimeError("Expected trajectories for regressor training.")
        features = feature_scaler.transform(feature_matrix(frame[RAW_FEATURE_COLUMNS])).astype(
            np.float32
        )
        scaled_targets = trajectory_scaler.transform(trajectories).astype(np.float32)
        indices = rng.permutation(features.shape[0])
        features = features[indices]
        scaled_targets = scaled_targets[indices]
        for start in range(0, features.shape[0], batch_size):
            batch_inputs = torch.tensor(
                features[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            batch_targets = torch.tensor(
                scaled_targets[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            optimizer.zero_grad()
            predictions = model(batch_inputs)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def _evaluate_regressor(
    model: TrajectoryRegressor,
    dataset_dir: Path,
    split: str,
    feature_scaler: Standardizer,
    trajectory_scaler: Standardizer,
    batch_size: int,
    device: str,
    criterion: nn.Module,
) -> RegressorMetrics:
    model.eval()
    losses: list[float] = []
    raw_predictions: list[np.ndarray] = []
    raw_targets: list[np.ndarray] = []
    with torch.no_grad():
        for frame, trajectories in iter_dataset_chunks(
            dataset_dir,
            include_trajectories=True,
            splits={split},
            stable_only=True,
            columns=REGRESSOR_COLUMNS,
        ):
            if trajectories is None:
                raise RuntimeError("Expected trajectories for regressor evaluation.")
            features = feature_scaler.transform(feature_matrix(frame[RAW_FEATURE_COLUMNS])).astype(
                np.float32
            )
            scaled_targets = trajectory_scaler.transform(trajectories).astype(np.float32)
            for start in range(0, features.shape[0], batch_size):
                batch_inputs = torch.tensor(
                    features[start : start + batch_size],
                    dtype=torch.float32,
                    device=device,
                )
                batch_targets = torch.tensor(
                    scaled_targets[start : start + batch_size],
                    dtype=torch.float32,
                    device=device,
                )
                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_targets)
                losses.append(float(loss.cpu()))
                raw_predictions.append(
                    trajectory_scaler.inverse_transform(outputs.cpu().numpy()).astype(np.float32)
                )
                raw_targets.append(trajectories[start : start + batch_size].astype(np.float32))

    predictions = np.concatenate(raw_predictions, axis=0)
    targets = np.concatenate(raw_targets, axis=0)
    return RegressorMetrics(
        loss=float(np.mean(losses)),
        rmse=trajectory_rmse(targets, predictions),
        mae=trajectory_mae(targets, predictions),
    )


def predict_stability_probabilities(
    model: StabilityClassifier,
    inputs: np.ndarray,
    batch_size: int = 2048,
) -> np.ndarray:
    device = next(model.parameters()).device
    tensor_inputs = torch.tensor(inputs, dtype=torch.float32)
    probabilities = []
    model.eval()
    with torch.no_grad():
        for start in range(0, tensor_inputs.shape[0], batch_size):
            batch = tensor_inputs[start : start + batch_size].to(device)
            logits = model(batch)
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probabilities, axis=0)


def predict_trajectories(
    model: TrajectoryRegressor,
    inputs: np.ndarray,
    trajectory_scaler: Standardizer,
    batch_size: int = 2048,
) -> np.ndarray:
    device = next(model.parameters()).device
    tensor_inputs = torch.tensor(inputs, dtype=torch.float32)
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, tensor_inputs.shape[0], batch_size):
            batch = tensor_inputs[start : start + batch_size].to(device)
            outputs.append(model(batch).cpu().numpy())
    scaled = np.concatenate(outputs, axis=0)
    return trajectory_scaler.inverse_transform(scaled)
