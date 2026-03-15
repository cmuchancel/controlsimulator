from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from controlsimulator.config import PublicationEvaluationConfig
from controlsimulator.dataset import dataset_time_grid, iter_dataset_chunks
from controlsimulator.evaluate import load_models
from controlsimulator.features import feature_matrix
from controlsimulator.metrics import extract_response_metrics
from controlsimulator.plants import heuristic_pid_scales, plant_from_sample_row
from controlsimulator.simulate import simulate_closed_loop
from controlsimulator.train import predict_stability_probabilities, predict_trajectories
from controlsimulator.utils import dump_json, ensure_dir, resolve_path


def _log_publication_stage(message: str) -> None:
    print(message, flush=True)


@dataclass(slots=True)
class BinaryConfusion:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def update(self, truth: np.ndarray, prediction: np.ndarray) -> None:
        truth = truth.astype(bool)
        prediction = prediction.astype(bool)
        self.tp += int(np.sum(truth & prediction))
        self.fp += int(np.sum(~truth & prediction))
        self.tn += int(np.sum(~truth & ~prediction))
        self.fn += int(np.sum(truth & ~prediction))

    def f1(self) -> float:
        denominator = (2 * self.tp) + self.fp + self.fn
        if denominator == 0:
            return 0.0
        return float((2 * self.tp) / denominator)

    def sample_count(self) -> int:
        return int(self.tp + self.fp + self.tn + self.fn)


@dataclass(slots=True)
class RmseAccumulator:
    squared_error_sum: float = 0.0
    value_count: int = 0
    sample_count: int = 0

    def update(self, truth: np.ndarray, prediction: np.ndarray) -> None:
        self.squared_error_sum += float(np.square(prediction - truth).sum())
        self.value_count += int(truth.size)
        self.sample_count += int(truth.shape[0])

    def rmse(self) -> float:
        if self.value_count == 0:
            return float("nan")
        return float(np.sqrt(self.squared_error_sum / self.value_count))


def _publication_columns(feature_columns: list[str]) -> list[str]:
    raw_feature_columns = [column for column in feature_columns if not column.startswith("log10_")]
    return list(
        dict.fromkeys(
            [
                *raw_feature_columns,
                "sample_id",
                "plant_id",
                "plant_family",
                "split",
                "stable",
                "tau_d",
                "kp",
                "ki",
                "kd",
                "trajectory_peak_abs",
                "plant_has_complex_poles",
                "plant_pole_spread_log10",
                "plant_min_damping_ratio",
                "response_is_oscillatory",
                "pole_0_real",
                "pole_0_imag",
                "pole_1_real",
                "pole_1_imag",
                "pole_2_real",
                "pole_2_imag",
            ]
        )
    )


def _analysis_family_label(
    plant_family: str,
    plant_has_complex_poles: bool,
    plant_pole_spread_log10: float,
) -> str:
    if plant_family == "campaign_third_order_ood_lightly_damped":
        return "lightly_damped"
    if plant_has_complex_poles:
        return "oscillatory"
    if plant_pole_spread_log10 < 0.12:
        return "critically_damped"
    return "overdamped"


def _trajectory_case_label(
    row: pd.Series,
) -> str | None:
    if bool(row["stable"]):
        if str(row["plant_family"]) == "campaign_third_order_ood_lightly_damped":
            return "lightly_damped_ood"
        if not bool(row["plant_has_complex_poles"]):
            return "overdamped"
        damping = float(row["plant_min_damping_ratio"])
        if 0.12 <= damping < 0.75:
            return "underdamped"
    return None


def _natural_frequency_hz(row: pd.Series) -> float:
    poles = [
        complex(float(row[f"pole_{index}_real"]), float(row[f"pole_{index}_imag"]))
        for index in range(3)
    ]
    magnitudes = np.asarray([abs(pole) for pole in poles], dtype=np.float64)
    if magnitudes.size == 0:
        return 0.0
    return float(np.max(magnitudes) / (2.0 * np.pi))


def _gain_magnitude(row: pd.Series) -> float:
    return float(
        np.sqrt(
            float(row["kp"]) ** 2 + float(row["ki"]) ** 2 + float(row["kd"]) ** 2
        )
    )


def _sample_metric_row(
    row: pd.Series,
    time_grid: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float | int | str]:
    truth_metrics = extract_response_metrics(time_grid, truth)
    prediction_metrics = extract_response_metrics(time_grid, prediction)
    return {
        "split": str(row["split"]),
        "sample_id": int(row["sample_id"]),
        "plant_family": str(row["plant_family"]),
        "analysis_family": _analysis_family_label(
            plant_family=str(row["plant_family"]),
            plant_has_complex_poles=bool(row["plant_has_complex_poles"]),
            plant_pole_spread_log10=float(row["plant_pole_spread_log10"]),
        ),
        "trajectory_case": _trajectory_case_label(row),
        "plant_min_damping_ratio": float(row["plant_min_damping_ratio"]),
        "natural_frequency_hz": _natural_frequency_hz(row),
        "pid_gain_magnitude": _gain_magnitude(row),
        "trajectory_rmse": float(np.sqrt(np.mean(np.square(prediction - truth)))),
        "rise_time_error": float(abs(prediction_metrics.rise_time - truth_metrics.rise_time))
        if np.isfinite(prediction_metrics.rise_time) and np.isfinite(truth_metrics.rise_time)
        else float("nan"),
        "settling_time_error": float(
            abs(prediction_metrics.settling_time - truth_metrics.settling_time)
        )
        if np.isfinite(prediction_metrics.settling_time)
        and np.isfinite(truth_metrics.settling_time)
        else float("nan"),
    }


def _reservoir_insert(
    reservoir: list[dict[str, float | int | str | None]],
    candidate: dict[str, float | int | str | None],
    seen: int,
    cap: int,
    rng: np.random.Generator,
) -> int:
    next_seen = seen + 1
    if cap <= 0:
        return next_seen
    if len(reservoir) < cap:
        reservoir.append(candidate)
        return next_seen
    replacement_index = int(rng.integers(0, next_seen))
    if replacement_index < cap:
        reservoir[replacement_index] = candidate
    return next_seen


def _pick_median_case(
    candidates: list[dict[str, float | int | str | None]],
    score_column: str,
) -> dict[str, float | int | str | None] | None:
    if not candidates:
        return None
    ordered = sorted(candidates, key=lambda row: float(row.get(score_column, 0.0)))
    return ordered[len(ordered) // 2]


def _load_samples_by_id(
    dataset_dir: Path,
    sample_ids: set[int],
    columns: list[str],
) -> dict[int, tuple[pd.Series, np.ndarray]]:
    lookup: dict[int, tuple[pd.Series, np.ndarray]] = {}
    for frame, trajectories in iter_dataset_chunks(
        dataset_dir,
        include_trajectories=True,
        columns=columns,
        progress_desc=None,
    ):
        if trajectories is None:
            raise RuntimeError("Expected trajectories while loading publication evaluation cases.")
        sample_id_array = frame["sample_id"].to_numpy(dtype=int)
        mask = np.isin(sample_id_array, np.asarray(sorted(sample_ids), dtype=int))
        if not np.any(mask):
            continue
        selected_frame = frame.loc[mask].reset_index(drop=True)
        selected_trajectories = trajectories[mask]
        for index, row in selected_frame.iterrows():
            lookup[int(row["sample_id"])] = (row.copy(), selected_trajectories[index].copy())
        if len(lookup) == len(sample_ids):
            break
    return lookup


def _plot_trajectory_quality(
    time_grid: np.ndarray,
    cases: list[dict[str, np.ndarray | str]],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    flat_axes = list(axes.flat)
    titles = [
        "Overdamped",
        "Underdamped",
        "Lightly Damped (OOD)",
        "Unstable",
    ]
    for axis, case, title in zip(flat_axes, cases, titles, strict=False):
        axis.plot(time_grid, case["truth"], label="simulation", linewidth=2.3)
        axis.plot(time_grid, case["prediction"], label="surrogate", linewidth=2.0, linestyle="--")
        axis.set_title(f"{title}\n{case['subtitle']}")
        axis.set_xlabel("Time [s]")
        axis.set_ylabel("Output y(t)")
        axis.grid(alpha=0.3)
        axis.legend()
    for axis in flat_axes[len(cases) :]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _plot_error_vs_properties(
    sampled_rows: pd.DataFrame,
    output_path: Path,
) -> None:
    metric_specs = [
        ("trajectory_rmse", "Trajectory RMSE"),
        ("rise_time_error", "Rise-Time Error [s]"),
        ("settling_time_error", "Settling-Time Error [s]"),
    ]
    property_specs = [
        ("plant_min_damping_ratio", "Damping Ratio", False),
        ("natural_frequency_hz", "Natural Frequency [Hz]", False),
        ("pid_gain_magnitude", "PID Gain Magnitude", True),
    ]

    figure, axes = plt.subplots(3, 3, figsize=(15, 12), squeeze=False)
    for row_index, (metric_column, metric_label) in enumerate(metric_specs):
        for column_index, (property_column, property_label, log_x) in enumerate(property_specs):
            axis = axes[row_index][column_index]
            frame = sampled_rows[[property_column, metric_column]].dropna()
            if frame.empty:
                axis.set_title(f"{metric_label} vs {property_label}")
                axis.text(0.5, 0.5, "no data", ha="center", va="center")
                axis.axis("off")
                continue
            axis.scatter(
                frame[property_column],
                frame[metric_column],
                s=8,
                alpha=0.18,
                color="#1f77b4",
            )
            sorted_frame = frame.sort_values(property_column).reset_index(drop=True)
            bin_count = min(24, max(8, len(sorted_frame) // 250))
            index_bins = np.array_split(np.arange(len(sorted_frame)), bin_count)
            centers = []
            medians = []
            for index_block in index_bins:
                if index_block.size == 0:
                    continue
                block = sorted_frame.iloc[index_block]
                centers.append(float(block[property_column].median()))
                medians.append(float(block[metric_column].median()))
            if centers:
                axis.plot(centers, medians, color="#111111", linewidth=2.0)
            if log_x:
                axis.set_xscale("log")
            axis.set_title(f"{metric_label} vs {property_label}")
            axis.set_xlabel(property_label)
            axis.set_ylabel(metric_label)
            axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _control_performance(
    time_grid: np.ndarray,
    trajectory: np.ndarray,
) -> tuple[float, dict[str, float]]:
    metrics = extract_response_metrics(time_grid, trajectory)
    tracking_rmse = float(np.sqrt(np.mean(np.square(trajectory - 1.0))))
    overshoot_penalty = max(metrics.overshoot_pct, 0.0) / 100.0
    settling_penalty = metrics.settling_time if np.isfinite(metrics.settling_time) else float(
        time_grid[-1] * 1.5
    )
    score = (
        tracking_rmse
        + overshoot_penalty
        + (0.05 * settling_penalty)
        + float(metrics.steady_state_error)
    )
    return score, {
        "tracking_rmse": tracking_rmse,
        "overshoot_pct": float(metrics.overshoot_pct),
        "rise_time": float(metrics.rise_time),
        "settling_time": float(metrics.settling_time),
        "steady_state_error": float(metrics.steady_state_error),
    }


def _surrogate_objective(
    models: Any,
    base_features: torch.Tensor,
    gain_logs: torch.Tensor,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    trajectory_mean: torch.Tensor,
    trajectory_std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gains = torch.exp(gain_logs)
    raw_features = torch.cat([base_features, gains], dim=0).unsqueeze(0)
    scaled = (raw_features - feature_mean) / feature_std
    stability_probability = torch.sigmoid(models.classifier(scaled)).squeeze(0)
    predicted_scaled = models.regressor(scaled).squeeze(0)
    predicted = (predicted_scaled * trajectory_std) + trajectory_mean

    tracking_mse = torch.mean((predicted - 1.0) ** 2)
    overshoot_penalty = torch.relu(torch.max(predicted) - 1.0)
    final_error = torch.abs(predicted[-1] - 1.0)
    stability_penalty = torch.relu(0.8 - stability_probability)
    smoothness_penalty = torch.mean((predicted[1:] - predicted[:-1]) ** 2)
    objective = (
        tracking_mse
        + (0.6 * overshoot_penalty)
        + (0.8 * final_error)
        + (1.5 * stability_penalty)
        + (0.05 * smoothness_penalty)
    )
    return objective, predicted, stability_probability


def _run_pid_optimization_demo(
    config: PublicationEvaluationConfig,
    dataset_dir: Path,
    models: Any,
    time_grid: np.ndarray,
    tables_dir: Path,
    plots_dir: Path,
) -> pd.DataFrame:
    required_columns = _publication_columns(models.feature_columns)
    demo_row: pd.Series | None = None
    for frame, _ in iter_dataset_chunks(
        dataset_dir,
        include_trajectories=False,
        splits={config.pid_demo_split},
        columns=required_columns,
        progress_desc=None,
    ):
        mask = (
            (frame["plant_family"] == config.pid_demo_family)
            & frame["stable"].astype(bool)
        )
        if not np.any(mask.to_numpy(dtype=bool)):
            continue
        demo_row = frame.loc[mask].iloc[0].copy()
        break
    if demo_row is None:
        raise RuntimeError("Could not locate a stable plant for the PID optimization demo.")

    plant = plant_from_sample_row(demo_row)
    tau_d = float(demo_row.get("tau_d", 0.0))
    base_kp, base_ki, base_kd = heuristic_pid_scales(plant)
    if not simulate_closed_loop(plant, base_kp, base_ki, base_kd, tau_d, time_grid).stable:
        base_kp = float(demo_row["kp"])
        base_ki = float(demo_row["ki"])
        base_kd = float(demo_row["kd"])

    kp_grid = np.geomspace(base_kp * 0.35, base_kp * 3.0, config.pid_demo_grid_kp_points)
    ki_grid = np.geomspace(base_ki * 0.35, base_ki * 3.0, config.pid_demo_grid_ki_points)
    kd_grid = np.geomspace(
        max(base_kd * 0.25, 1e-4),
        max(base_kd * 3.0, 1e-4),
        config.pid_demo_grid_kd_points,
    )

    grid_start = perf_counter()
    grid_rows: list[tuple[float, float, float, float, dict[str, float]]] = []
    for kp in kp_grid:
        for ki in ki_grid:
            for kd in kd_grid:
                result = simulate_closed_loop(
                    plant=plant,
                    kp=float(kp),
                    ki=float(ki),
                    kd=float(kd),
                    tau_d=tau_d,
                    time_grid=time_grid,
                )
                if not result.stable or result.trajectory is None:
                    grid_rows.append(
                        (
                            float(kp),
                            float(ki),
                            float(kd),
                            1e6,
                            {
                                "tracking_rmse": float("nan"),
                                "overshoot_pct": float("nan"),
                                "rise_time": float("nan"),
                                "settling_time": float("nan"),
                                "steady_state_error": float("nan"),
                            },
                        )
                    )
                    continue
                score, metrics = _control_performance(time_grid, result.trajectory)
                grid_rows.append((float(kp), float(ki), float(kd), score, metrics))
    grid_elapsed = perf_counter() - grid_start
    best_grid = min(grid_rows, key=lambda row: row[3])
    grid_simulation = simulate_closed_loop(
        plant=plant,
        kp=best_grid[0],
        ki=best_grid[1],
        kd=best_grid[2],
        tau_d=tau_d,
        time_grid=time_grid,
    )

    feature_mean = torch.tensor(
        models.feature_scaler.mean,
        dtype=torch.float32,
        device=models.device,
    )
    feature_std = torch.tensor(
        models.feature_scaler.std,
        dtype=torch.float32,
        device=models.device,
    )
    trajectory_mean = torch.tensor(
        models.trajectory_scaler.mean,
        dtype=torch.float32,
        device=models.device,
    )
    trajectory_std = torch.tensor(
        models.trajectory_scaler.std,
        dtype=torch.float32,
        device=models.device,
    )
    base_features = torch.tensor(
        [
            float(demo_row["b0"]),
            float(demo_row["a2"]),
            float(demo_row["a1"]),
            float(demo_row["a0"]),
        ],
        dtype=torch.float32,
        device=models.device,
    )
    log_lows = torch.log(
        torch.tensor([1e-2, 1e-3, 1e-4], dtype=torch.float32, device=models.device)
    )
    log_highs = torch.log(
        torch.tensor([1e2, 1e2, 10.0], dtype=torch.float32, device=models.device)
    )
    gain_logs = torch.nn.Parameter(
        torch.log(
            torch.tensor(
                [base_kp, base_ki, max(base_kd, 1e-4)],
                dtype=torch.float32,
                device=models.device,
            )
        )
    )
    surrogate_start = perf_counter()
    best_state: tuple[float, np.ndarray, np.ndarray] | None = None
    for _ in range(config.pid_demo_gradient_steps):
        if gain_logs.grad is not None:
            gain_logs.grad.zero_()
        objective, predicted, stability_probability = _surrogate_objective(
            models=models,
            base_features=base_features,
            gain_logs=gain_logs,
            feature_mean=feature_mean,
            feature_std=feature_std,
            trajectory_mean=trajectory_mean,
            trajectory_std=trajectory_std,
        )
        objective.backward()
        with torch.no_grad():
            gain_logs -= config.pid_demo_learning_rate * gain_logs.grad
            gain_logs.copy_(torch.minimum(torch.maximum(gain_logs, log_lows), log_highs))
            current_objective = float(objective.detach().cpu())
            if best_state is None or current_objective < best_state[0]:
                best_state = (
                    current_objective,
                    torch.exp(gain_logs.detach()).cpu().numpy().astype(np.float64),
                    np.asarray(
                        [
                            float(stability_probability.detach().cpu()),
                            float(predicted[-1].detach().cpu()),
                        ],
                        dtype=np.float64,
                    ),
                )
    surrogate_elapsed = perf_counter() - surrogate_start
    if best_state is None:
        raise RuntimeError("Surrogate optimization did not produce a valid result.")

    surrogate_kp, surrogate_ki, surrogate_kd = best_state[1].tolist()
    surrogate_simulation = simulate_closed_loop(
        plant=plant,
        kp=surrogate_kp,
        ki=surrogate_ki,
        kd=surrogate_kd,
        tau_d=tau_d,
        time_grid=time_grid,
    )
    if surrogate_simulation.trajectory is None:
        surrogate_score = 1e6
        surrogate_metrics = {
            "tracking_rmse": float("nan"),
            "overshoot_pct": float("nan"),
            "rise_time": float("nan"),
            "settling_time": float("nan"),
            "steady_state_error": float("nan"),
        }
    else:
        surrogate_score, surrogate_metrics = _control_performance(
            time_grid,
            surrogate_simulation.trajectory,
        )

    grid_score, grid_metrics = (
        _control_performance(time_grid, grid_simulation.trajectory)
        if grid_simulation.trajectory is not None
        else (
            1e6,
            {
                "tracking_rmse": float("nan"),
                "overshoot_pct": float("nan"),
                "rise_time": float("nan"),
                "settling_time": float("nan"),
                "steady_state_error": float("nan"),
            },
        )
    )

    results = pd.DataFrame(
        [
            {
                "method": "ode_grid_search",
                "runtime_seconds": float(grid_elapsed),
                "evaluations": int(len(grid_rows)),
                "kp": float(best_grid[0]),
                "ki": float(best_grid[1]),
                "kd": float(best_grid[2]),
                "stable": bool(grid_simulation.stable),
                "control_score": float(grid_score),
                **grid_metrics,
            },
            {
                "method": "surrogate_gradient",
                "runtime_seconds": float(surrogate_elapsed),
                "evaluations": int(config.pid_demo_gradient_steps),
                "kp": float(surrogate_kp),
                "ki": float(surrogate_ki),
                "kd": float(surrogate_kd),
                "stable": bool(surrogate_simulation.stable),
                "control_score": float(surrogate_score),
                **surrogate_metrics,
            },
        ]
    )
    results.to_csv(tables_dir / "pid_optimization_demo.csv", index=False)

    if grid_simulation.trajectory is not None and surrogate_simulation.trajectory is not None:
        figure, axis = plt.subplots(figsize=(8, 4.8))
        axis.plot(
            time_grid,
            grid_simulation.trajectory,
            label="ODE grid-search best",
            linewidth=2.0,
        )
        axis.plot(
            time_grid,
            surrogate_simulation.trajectory,
            label="Surrogate-gradient validated",
            linewidth=2.0,
            linestyle="--",
        )
        axis.axhline(1.0, color="#222222", linewidth=1.0, linestyle=":")
        axis.set_title("PID Optimization Demo")
        axis.set_xlabel("Time [s]")
        axis.set_ylabel("Output y(t)")
        axis.grid(alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(plots_dir / "pid_optimization_demo.png", dpi=200)
        plt.close(figure)

    return results


def _run_speed_benchmark(
    config: PublicationEvaluationConfig,
    dataset_dir: Path,
    models: Any,
    time_grid: np.ndarray,
    benchmark_path: Path,
    tables_dir: Path,
) -> dict[str, float | int | str]:
    rng = np.random.default_rng(config.seed)
    required_columns = _publication_columns(models.feature_columns)
    samples: list[pd.Series] = []
    seen = 0
    for frame, _ in iter_dataset_chunks(
        dataset_dir,
        include_trajectories=False,
        columns=required_columns,
        progress_desc="publication-speed-sample",
    ):
        for _, row in frame.iterrows():
            seen += 1
            if len(samples) < config.speed_benchmark_samples:
                samples.append(row.copy())
                continue
            replacement_index = int(rng.integers(0, seen))
            if replacement_index < config.speed_benchmark_samples:
                samples[replacement_index] = row.copy()
    if not samples:
        raise RuntimeError("Speed benchmark could not sample any systems.")

    frame = pd.DataFrame(samples).reset_index(drop=True)
    features = models.feature_scaler.transform(
        feature_matrix(frame, feature_columns=models.feature_columns)
    ).astype(np.float32)

    warmup_count = min(32, frame.shape[0])
    for index in range(warmup_count):
        row = frame.iloc[index]
        simulate_closed_loop(
            plant=plant_from_sample_row(row),
            kp=float(row["kp"]),
            ki=float(row["ki"]),
            kd=float(row["kd"]),
            tau_d=float(row["tau_d"]),
            time_grid=time_grid,
        )
        predict_stability_probabilities(
            models.classifier,
            features[index : index + 1],
            batch_size=1,
        )
        predict_trajectories(
            models.regressor,
            features[index : index + 1],
            models.trajectory_scaler,
            batch_size=1,
        )

    simulation_times = np.zeros(frame.shape[0], dtype=np.float64)
    surrogate_times = np.zeros(frame.shape[0], dtype=np.float64)
    for index in tqdm(range(frame.shape[0]), desc="publication-speed-benchmark", unit="system"):
        row = frame.iloc[index]
        start = perf_counter()
        simulate_closed_loop(
            plant=plant_from_sample_row(row),
            kp=float(row["kp"]),
            ki=float(row["ki"]),
            kd=float(row["kd"]),
            tau_d=float(row["tau_d"]),
            time_grid=time_grid,
        )
        simulation_times[index] = perf_counter() - start

        start = perf_counter()
        predict_stability_probabilities(
            models.classifier,
            features[index : index + 1],
            batch_size=1,
        )
        predict_trajectories(
            models.regressor,
            features[index : index + 1],
            models.trajectory_scaler,
            batch_size=1,
        )
        surrogate_times[index] = perf_counter() - start

    summary = {
        "device": str(models.device),
        "n_systems": int(frame.shape[0]),
        "simulation_runtime_mean_seconds": float(simulation_times.mean()),
        "simulation_runtime_std_seconds": float(simulation_times.std(ddof=0)),
        "surrogate_runtime_mean_seconds": float(surrogate_times.mean()),
        "surrogate_runtime_std_seconds": float(surrogate_times.std(ddof=0)),
        "speedup_factor_mean_runtime": float(
            simulation_times.mean() / max(surrogate_times.mean(), 1e-12)
        ),
    }
    dump_json(summary, benchmark_path)
    pd.DataFrame(
        [
            {
                "method": "simulation_equivalent_step_response",
                "mean_runtime_seconds": summary["simulation_runtime_mean_seconds"],
                "std_runtime_seconds": summary["simulation_runtime_std_seconds"],
            },
            {
                "method": "surrogate_forward_pass",
                "mean_runtime_seconds": summary["surrogate_runtime_mean_seconds"],
                "std_runtime_seconds": summary["surrogate_runtime_std_seconds"],
            },
        ]
    ).to_csv(tables_dir / "speed_benchmark.csv", index=False)
    return summary


def run_publication_evaluation(config: PublicationEvaluationConfig) -> Path:
    dataset_dir = resolve_path(config.dataset_dir)
    run_dir = resolve_path(config.run_dir)
    tables_dir = ensure_dir(resolve_path(config.tables_dir()))
    plots_dir = ensure_dir(resolve_path(config.plots_dir()))
    benchmark_path = resolve_path(config.speed_benchmark_file())
    ensure_dir(benchmark_path.parent)

    models = load_models(run_dir, device=config.device)
    time_grid = dataset_time_grid(dataset_dir)
    rng = np.random.default_rng(config.seed)
    _log_publication_stage(f"[publication-eval] scan splits {config.name}")

    classifier_metrics: dict[str, BinaryConfusion] = {
        name: BinaryConfusion()
        for name in ["overdamped", "critically_damped", "lightly_damped", "oscillatory"]
    }
    regressor_metrics: dict[str, RmseAccumulator] = {
        name: RmseAccumulator()
        for name in ["overdamped", "critically_damped", "lightly_damped", "oscillatory"]
    }
    property_rows: list[dict[str, float | int | str]] = []
    property_seen = 0
    case_candidates: dict[str, list[dict[str, float | int | str | None]]] = {
        "overdamped": [],
        "underdamped": [],
        "lightly_damped_ood": [],
    }
    case_seen = {name: 0 for name in case_candidates}
    unstable_candidates: list[dict[str, float | int | str | None]] = []
    unstable_seen = 0

    required_columns = _publication_columns(models.feature_columns)
    for split in ["test", "ood_test"]:
        for frame, trajectories in iter_dataset_chunks(
            dataset_dir,
            include_trajectories=True,
            splits={split},
            columns=required_columns,
            progress_desc=f"publication-eval:{split}",
        ):
            if trajectories is None:
                raise RuntimeError("Expected trajectories during publication evaluation.")
            features = models.feature_scaler.transform(
                feature_matrix(frame, feature_columns=models.feature_columns)
            ).astype(np.float32)
            truth_labels = frame["stable"].to_numpy(dtype=int)
            prediction_labels = (
                predict_stability_probabilities(
                    models.classifier,
                    features,
                    batch_size=config.inference_batch_size,
                )
                >= 0.5
            ).astype(int)
            analysis_families = np.asarray(
                [
                    _analysis_family_label(
                        plant_family=str(family),
                        plant_has_complex_poles=bool(has_complex),
                        plant_pole_spread_log10=float(spread),
                    )
                    for family, has_complex, spread in zip(
                        frame["plant_family"].to_numpy(dtype=str),
                        frame["plant_has_complex_poles"].to_numpy(dtype=bool),
                        frame["plant_pole_spread_log10"].to_numpy(dtype=float),
                        strict=False,
                    )
                ],
                dtype=object,
            )
            for category in classifier_metrics:
                mask = analysis_families == category
                if np.any(mask):
                    classifier_metrics[category].update(truth_labels[mask], prediction_labels[mask])

            unstable_mask = ~frame["stable"].to_numpy(dtype=bool)
            if np.any(unstable_mask):
                unstable_frame = frame.loc[unstable_mask].reset_index(drop=True)
                for _, unstable_row in unstable_frame.iterrows():
                    unstable_seen = _reservoir_insert(
                        reservoir=unstable_candidates,
                        candidate={
                            "sample_id": int(unstable_row["sample_id"]),
                            "score": float(unstable_row["trajectory_peak_abs"]),
                        },
                        seen=unstable_seen,
                        cap=config.case_candidate_cap,
                        rng=rng,
                    )

            stable_mask = frame["stable"].to_numpy(dtype=bool)
            if not np.any(stable_mask):
                continue
            stable_frame = frame.loc[stable_mask].reset_index(drop=True)
            truth = trajectories[stable_mask]
            stable_features = features[stable_mask]
            predictions = predict_trajectories(
                models.regressor,
                stable_features,
                models.trajectory_scaler,
                batch_size=config.inference_batch_size,
            ).astype(np.float32)
            stable_analysis_families = analysis_families[stable_mask]
            for category in regressor_metrics:
                mask = stable_analysis_families == category
                if np.any(mask):
                    regressor_metrics[category].update(truth[mask], predictions[mask])

            sample_rmses = np.sqrt(np.mean(np.square(predictions - truth), axis=1))
            for index, (_, row) in enumerate(stable_frame.iterrows()):
                property_seen += 1
                include_metric_row = len(property_rows) < config.property_plot_sample_cap
                replacement_index = -1
                if not include_metric_row:
                    replacement_index = int(rng.integers(0, property_seen))
                    include_metric_row = replacement_index < config.property_plot_sample_cap
                if include_metric_row:
                    metric_row = _sample_metric_row(
                        row=row,
                        time_grid=time_grid,
                        truth=truth[index],
                        prediction=predictions[index],
                    )
                    if replacement_index >= 0:
                        property_rows[replacement_index] = metric_row
                    else:
                        property_rows.append(metric_row)

                case_label = _trajectory_case_label(row)
                if case_label is not None:
                    case_seen[case_label] = _reservoir_insert(
                        reservoir=case_candidates[case_label],
                        candidate={
                            "sample_id": int(row["sample_id"]),
                            "score": float(sample_rmses[index]),
                        },
                        seen=case_seen[case_label],
                        cap=config.case_candidate_cap,
                        rng=rng,
                    )

    property_frame = pd.DataFrame(property_rows)
    property_frame.to_csv(tables_dir / "error_property_sample.csv", index=False)

    family_rows = []
    for category in ["overdamped", "critically_damped", "lightly_damped", "oscillatory"]:
        family_rows.append(
            {
                "family": category,
                "n_classifier_samples": classifier_metrics[category].sample_count(),
                "n_regression_samples": regressor_metrics[category].sample_count,
                "classifier_f1": classifier_metrics[category].f1(),
                "trajectory_rmse": regressor_metrics[category].rmse(),
            }
        )
    family_frame = pd.DataFrame(family_rows)
    family_frame.to_csv(tables_dir / "family_ood_analysis.csv", index=False)

    selected_case_rows: list[dict[str, float | int | str | None]] = []
    for name in ["overdamped", "underdamped", "lightly_damped_ood"]:
        selected = _pick_median_case(case_candidates[name], "score")
        if selected is not None:
            selected_case_rows.append(selected)
    unstable_case = _pick_median_case(unstable_candidates, "score")
    if unstable_case is not None:
        selected_case_rows.append(unstable_case)

    sample_lookup = _load_samples_by_id(
        dataset_dir=dataset_dir,
        sample_ids={int(row["sample_id"]) for row in selected_case_rows},
        columns=required_columns,
    )
    trajectory_cases = []
    case_names = ["overdamped", "underdamped", "lightly_damped_ood", "unstable"]
    selected_by_name = {
        "overdamped": _pick_median_case(case_candidates["overdamped"], "score"),
        "underdamped": _pick_median_case(case_candidates["underdamped"], "score"),
        "lightly_damped_ood": _pick_median_case(case_candidates["lightly_damped_ood"], "score"),
        "unstable": unstable_case,
    }
    for case_name in case_names:
        selected = selected_by_name.get(case_name)
        if selected is None:
            continue
        sample_id = int(selected["sample_id"])
        row, truth = sample_lookup[sample_id]
        feature_values = models.feature_scaler.transform(
            feature_matrix(pd.DataFrame([row]), feature_columns=models.feature_columns)
        ).astype(np.float32)
        prediction = predict_trajectories(
            models.regressor,
            feature_values,
            models.trajectory_scaler,
            batch_size=1,
        )[0].astype(np.float32)
        stability_probability = float(
            predict_stability_probabilities(models.classifier, feature_values, batch_size=1)[0]
        )
        trajectory_cases.append(
            {
                "truth": truth,
                "prediction": prediction,
                "subtitle": (
                    f"sample={sample_id} | split={row['split']} | "
                    f"p(stable)={stability_probability:.2f}"
                ),
            }
        )
    _plot_trajectory_quality(
        time_grid=time_grid,
        cases=trajectory_cases,
        output_path=plots_dir / "trajectory_quality_publication.png",
    )
    _plot_error_vs_properties(
        sampled_rows=property_frame,
        output_path=plots_dir / "error_vs_system_properties.png",
    )

    _log_publication_stage(f"[publication-eval] speed benchmark {config.name}")
    speed_summary = _run_speed_benchmark(
        config=config,
        dataset_dir=dataset_dir,
        models=models,
        time_grid=time_grid,
        benchmark_path=benchmark_path,
        tables_dir=tables_dir,
    )
    _log_publication_stage(f"[publication-eval] pid demo {config.name}")
    pid_demo_frame = _run_pid_optimization_demo(
        config=config,
        dataset_dir=dataset_dir,
        models=models,
        time_grid=time_grid,
        tables_dir=tables_dir,
        plots_dir=plots_dir,
    )

    summary = {
        "name": config.name,
        "family_table_path": str(tables_dir / "family_ood_analysis.csv"),
        "property_sample_path": str(tables_dir / "error_property_sample.csv"),
        "pid_demo_path": str(tables_dir / "pid_optimization_demo.csv"),
        "trajectory_quality_plot": str(plots_dir / "trajectory_quality_publication.png"),
        "error_vs_properties_plot": str(plots_dir / "error_vs_system_properties.png"),
        "speed_benchmark_path": str(benchmark_path),
        "critical_family_note": (
            "critically_damped is a proxy bucket defined by "
            "real-pole plants with clustered pole spread"
        ),
        "speed_benchmark": speed_summary,
        "family_rows": family_frame.to_dict(orient="records"),
        "pid_demo_rows": pid_demo_frame.to_dict(orient="records"),
    }
    dump_json(summary, tables_dir / "publication_eval_summary.json")
    (tables_dir / "publication_eval_summary.md").write_text(
        "\n".join(
            [
                "# Publication Evaluation Summary",
                "",
                "## Speed Benchmark",
                "",
                f"- systems: {int(speed_summary['n_systems'])}",
                (
                    f"- simulation mean/std [s]: "
                    f"{float(speed_summary['simulation_runtime_mean_seconds']):.6f} / "
                    f"{float(speed_summary['simulation_runtime_std_seconds']):.6f}"
                ),
                (
                    f"- surrogate mean/std [s]: "
                    f"{float(speed_summary['surrogate_runtime_mean_seconds']):.6f} / "
                    f"{float(speed_summary['surrogate_runtime_std_seconds']):.6f}"
                ),
                f"- speedup: {float(speed_summary['speedup_factor_mean_runtime']):.3f}x",
                "",
                "## Notes",
                "",
                "- lightly_damped maps to the withheld campaign OOD family",
                "- critically_damped is a proxy category built from clustered real-pole systems",
            ]
        ),
        encoding="utf-8",
    )
    _log_publication_stage(f"[publication-eval] complete {config.name}")
    return tables_dir
