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
from torch.utils.data import DataLoader, TensorDataset

from controlsimulator.config import TrainingConfig, save_config_snapshot
from controlsimulator.dataset import load_dataset
from controlsimulator.features import FEATURE_COLUMNS, Standardizer, feature_matrix
from controlsimulator.metrics import trajectory_mae, trajectory_rmse
from controlsimulator.models import StabilityClassifier, TrajectoryRegressor
from controlsimulator.plotting import plot_training_history
from controlsimulator.utils import dump_json, ensure_dir, pick_device, resolve_path, set_global_seed


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


def train_models(config: TrainingConfig) -> Path:
    set_global_seed(config.seed)
    dataset = load_dataset(config.dataset_dir)
    run_dir = ensure_dir(resolve_path(config.run_dir()))
    plots_dir = ensure_dir(run_dir / "plots")
    save_config_snapshot(config, run_dir / "config_snapshot.yaml")

    device = pick_device(config.device)
    features = feature_matrix(dataset.samples)
    splits = dataset.samples["split"].to_numpy()
    stable = dataset.samples["stable"].to_numpy(dtype=bool)

    train_mask = splits == "train"
    val_mask = splits == "val"
    stable_train_mask = train_mask & stable
    stable_val_mask = val_mask & stable

    feature_scaler = Standardizer.fit(features[train_mask])
    features_scaled = feature_scaler.transform(features).astype(np.float32)

    classifier_path, classifier_history, classifier_summary = _train_classifier(
        config=config,
        run_dir=run_dir,
        device=device,
        inputs=features_scaled,
        targets=stable.astype(np.float32),
        train_mask=train_mask,
        val_mask=val_mask,
        feature_scaler=feature_scaler,
    )
    plot_training_history(
        classifier_history,
        title="Stability Classifier",
        output_path=plots_dir / "classifier_training.png",
        value_columns=("train_loss", "val_loss"),
    )

    trajectory_scaler = Standardizer.fit(dataset.trajectories[stable_train_mask])
    regressor_path, regressor_history, regressor_summary, mean_trajectory = _train_regressor(
        config=config,
        run_dir=run_dir,
        device=device,
        inputs=features_scaled,
        targets=dataset.trajectories,
        train_mask=stable_train_mask,
        val_mask=stable_val_mask,
        feature_scaler=feature_scaler,
        trajectory_scaler=trajectory_scaler,
        time_grid=dataset.time_grid,
    )
    plot_training_history(
        regressor_history,
        title="Trajectory Regressor",
        output_path=plots_dir / "regressor_training.png",
        value_columns=("train_loss", "val_loss"),
    )

    summary = {
        "device": device,
        "feature_columns": FEATURE_COLUMNS,
        "classifier_checkpoint": str(classifier_path.relative_to(run_dir)),
        "regressor_checkpoint": str(regressor_path.relative_to(run_dir)),
        "train_samples": int(train_mask.sum()),
        "val_samples": int(val_mask.sum()),
        "stable_train_samples": int(stable_train_mask.sum()),
        "stable_val_samples": int(stable_val_mask.sum()),
        "classifier": classifier_summary,
        "regressor": regressor_summary,
        "mean_trajectory_path": "mean_trajectory.npy",
    }
    np.save(run_dir / "mean_trajectory.npy", mean_trajectory.astype(np.float32))
    dump_json(summary, run_dir / "train_summary.json")
    return run_dir


def load_classifier_checkpoint(path: str | Path) -> dict[str, Any]:
    return torch.load(resolve_path(path), map_location="cpu", weights_only=False)


def load_regressor_checkpoint(path: str | Path) -> dict[str, Any]:
    return torch.load(resolve_path(path), map_location="cpu", weights_only=False)


def _train_classifier(
    config: TrainingConfig,
    run_dir: Path,
    device: str,
    inputs: np.ndarray,
    targets: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    feature_scaler: Standardizer,
) -> tuple[Path, pd.DataFrame, dict[str, float]]:
    model = StabilityClassifier(
        input_dim=inputs.shape[1],
        hidden_sizes=config.classifier_hidden_sizes,
        dropout=config.classifier_dropout,
    ).to(device)

    train_loader = _make_loader(
        inputs[train_mask],
        targets[train_mask],
        config.batch_size,
        shuffle=True,
    )
    val_loader = _make_loader(
        inputs[val_mask],
        targets[val_mask],
        config.batch_size,
        shuffle=False,
    )
    positive_fraction = max(float(targets[train_mask].mean()), 1e-4)
    pos_weight = torch.tensor([(1.0 - positive_fraction) / positive_fraction], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
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
        model.train()
        train_losses = []
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()
            logits = model(batch_inputs)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(train_losses))
        val_metrics = _evaluate_classifier(model, val_loader, criterion, device)
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
                "input_dim": inputs.shape[1],
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
    run_dir: Path,
    device: str,
    inputs: np.ndarray,
    targets: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    feature_scaler: Standardizer,
    trajectory_scaler: Standardizer,
    time_grid: np.ndarray,
) -> tuple[Path, pd.DataFrame, dict[str, float], np.ndarray]:
    model = TrajectoryRegressor(
        input_dim=inputs.shape[1],
        output_dim=targets.shape[1],
        hidden_sizes=config.hidden_sizes,
        dropout=config.dropout,
    ).to(device)

    mean_trajectory = targets[train_mask].mean(axis=0)
    scaled_targets = trajectory_scaler.transform(targets[train_mask]).astype(np.float32)
    scaled_val_targets = trajectory_scaler.transform(targets[val_mask]).astype(np.float32)
    train_loader = _make_loader(
        inputs[train_mask],
        scaled_targets,
        config.batch_size,
        shuffle=True,
    )
    val_loader = _make_loader(
        inputs[val_mask],
        scaled_val_targets,
        config.batch_size,
        shuffle=False,
    )
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
        model.train()
        train_losses = []
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()
            predictions = model(batch_inputs)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(train_losses))
        val_metrics = _evaluate_regressor(
            model,
            val_loader,
            criterion,
            device,
            trajectory_scaler,
            targets[val_mask],
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
                "input_dim": inputs.shape[1],
                "output_dim": targets.shape[1],
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
    return checkpoint_path, history, best_metrics, mean_trajectory


def _make_loader(
    inputs: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(inputs, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _evaluate_classifier(
    model: StabilityClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> ClassifierMetrics:
    model.eval()
    losses = []
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for batch_inputs, batch_targets in loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
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


def _evaluate_regressor(
    model: TrajectoryRegressor,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    trajectory_scaler: Standardizer,
    raw_targets: np.ndarray,
) -> RegressorMetrics:
    model.eval()
    losses = []
    predictions = []
    with torch.no_grad():
        for batch_inputs, batch_targets in loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            losses.append(float(loss.cpu()))
            predictions.append(outputs.cpu().numpy())

    scaled_predictions = np.concatenate(predictions, axis=0)
    raw_predictions = trajectory_scaler.inverse_transform(scaled_predictions)
    return RegressorMetrics(
        loss=float(np.mean(losses)),
        rmse=trajectory_rmse(raw_targets, raw_predictions),
        mae=trajectory_mae(raw_targets, raw_predictions),
    )
