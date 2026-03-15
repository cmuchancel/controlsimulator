from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
from scipy import signal
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from controlsimulator.config import PIDOptimizationComparisonConfig, save_config_snapshot
from controlsimulator.dataset import dataset_time_grid, iter_dataset_chunks
from controlsimulator.evaluate import load_models
from controlsimulator.metrics import extract_response_metrics
from controlsimulator.plants import Plant, heuristic_pid_scales, plant_from_sample_row
from controlsimulator.simulate import simulate_closed_loop
from controlsimulator.utils import dump_json, ensure_dir, resolve_path

ABSOLUTE_GAIN_LOW = np.asarray([1e-2, 1e-3, 1e-4], dtype=np.float64)
ABSOLUTE_GAIN_HIGH = np.asarray([1e2, 1e2, 10.0], dtype=np.float64)
METHOD_ORDER = [
    "ziegler_nichols",
    "grid_search",
    "bayesian_optimization",
    "surrogate_gradient",
]
METHOD_LABELS = {
    "ziegler_nichols": "Ziegler-Nichols",
    "grid_search": "Grid Search",
    "bayesian_optimization": "Bayesian Optimization",
    "surrogate_gradient": "Surrogate Gradient",
}


@dataclass(slots=True)
class ControllerEvaluation:
    final_cost: float
    stable: bool
    overshoot_pct: float
    settling_time: float
    steady_state_error: float
    rise_time: float
    trajectory: np.ndarray | None
    peak_control_effort: float | None


@dataclass(slots=True)
class SurrogateContext:
    models: Any
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    trajectory_mean: torch.Tensor
    trajectory_std: torch.Tensor
    log_gain_low: torch.Tensor
    log_gain_high: torch.Tensor
    settling_band: float
    time_horizon: float
    constant_features: dict[str, torch.Tensor]


def _log_pid_opt_stage(message: str) -> None:
    print(message, flush=True)


def _open_loop_step_response(plant: Plant, time_grid: np.ndarray) -> np.ndarray:
    system = signal.TransferFunction(plant.numerator, plant.denominator)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, response = signal.step(system, T=time_grid)
    return np.asarray(response, dtype=np.float64)


def _evaluate_controller(
    plant: Plant,
    kp: float,
    ki: float,
    kd: float,
    tau_d: float,
    time_grid: np.ndarray,
) -> ControllerEvaluation:
    result = simulate_closed_loop(
        plant=plant,
        kp=float(kp),
        ki=float(ki),
        kd=float(kd),
        tau_d=float(tau_d),
        time_grid=time_grid,
        trajectory_clip_abs=250.0,
    )
    if result.trajectory is None:
        return ControllerEvaluation(
            final_cost=float((time_grid[-1] * 2.5) + 3.0),
            stable=False,
            overshoot_pct=float("nan"),
            settling_time=float("nan"),
            steady_state_error=float("nan"),
            rise_time=float("nan"),
            trajectory=None,
            peak_control_effort=result.peak_control_effort,
        )

    metrics = extract_response_metrics(
        time_grid=time_grid,
        trajectory=result.trajectory,
        peak_control_effort=result.peak_control_effort,
    )
    settling_penalty = (
        float(metrics.settling_time)
        if np.isfinite(metrics.settling_time)
        else float(time_grid[-1] * 1.5)
    )
    overshoot_penalty = max(float(metrics.overshoot_pct), 0.0) / 100.0
    final_cost = settling_penalty + overshoot_penalty + float(metrics.steady_state_error)
    if not result.stable:
        final_cost += float(time_grid[-1])
    return ControllerEvaluation(
        final_cost=float(final_cost),
        stable=bool(result.stable),
        overshoot_pct=float(metrics.overshoot_pct),
        settling_time=float(metrics.settling_time),
        steady_state_error=float(metrics.steady_state_error),
        rise_time=float(metrics.rise_time),
        trajectory=result.trajectory.copy(),
        peak_control_effort=result.peak_control_effort,
    )


def _search_bounds(
    plant: Plant,
    config: PIDOptimizationComparisonConfig,
) -> tuple[np.ndarray, np.ndarray]:
    base_gains = np.clip(
        np.asarray(heuristic_pid_scales(plant), dtype=np.float64),
        ABSOLUTE_GAIN_LOW,
        ABSOLUTE_GAIN_HIGH,
    )
    center = np.log10(base_gains)
    spans = np.asarray(
        [
            config.search_log10_span_kp,
            config.search_log10_span_ki,
            config.search_log10_span_kd,
        ],
        dtype=np.float64,
    )
    log_low = np.maximum(np.log10(ABSOLUTE_GAIN_LOW), center - spans)
    log_high = np.minimum(np.log10(ABSOLUTE_GAIN_HIGH), center + spans)
    narrow = (log_high - log_low) < 0.35
    if np.any(narrow):
        log_low[narrow] = np.maximum(np.log10(ABSOLUTE_GAIN_LOW[narrow]), center[narrow] - 0.35)
        log_high[narrow] = np.minimum(np.log10(ABSOLUTE_GAIN_HIGH[narrow]), center[narrow] + 0.35)
    return np.power(10.0, log_low), np.power(10.0, log_high)


def _ziegler_nichols_open_loop(
    plant: Plant,
    tau_d: float,
    time_grid: np.ndarray,
) -> tuple[np.ndarray, str]:
    stable_real_parts = np.abs(np.real(plant.poles[np.real(plant.poles) < 0.0]))
    slowest_real_mag = float(np.min(stable_real_parts)) if stable_real_parts.size else 0.05
    open_loop_horizon = float(np.clip(8.0 / max(slowest_real_mag, 1e-3), time_grid[-1], 80.0))
    open_loop_grid = np.linspace(0.0, open_loop_horizon, time_grid.shape[0], dtype=np.float64)
    response = _open_loop_step_response(plant, open_loop_grid)
    derivative = np.gradient(response, open_loop_grid)
    slope_index = int(np.argmax(derivative))
    slope = float(derivative[slope_index])
    final_value = float(response[-1])
    if (
        not np.isfinite(final_value)
        or final_value <= 1e-8
        or not np.isfinite(slope)
        or slope <= 1e-8
    ):
        return (
            np.clip(
                np.asarray(heuristic_pid_scales(plant), dtype=np.float64),
                ABSOLUTE_GAIN_LOW,
                ABSOLUTE_GAIN_HIGH,
            ),
            "heuristic_fallback",
        )

    inflection_time = float(open_loop_grid[slope_index])
    inflection_value = float(response[slope_index])
    dead_time = max(
        inflection_time - (inflection_value / slope),
        float(open_loop_grid[1] - open_loop_grid[0]),
    )
    time_constant = max(final_value / slope, float(open_loop_grid[1] - open_loop_grid[0]))
    kp = 1.2 * time_constant / max(final_value * dead_time, 1e-8)
    ki = kp / max(2.0 * dead_time, 1e-8)
    kd = kp * 0.5 * dead_time
    gains = np.asarray([kp, ki, kd], dtype=np.float64)
    return np.clip(gains, ABSOLUTE_GAIN_LOW, ABSOLUTE_GAIN_HIGH), "reaction_curve"


def _grid_search_method(
    plant: Plant,
    tau_d: float,
    time_grid: np.ndarray,
    config: PIDOptimizationComparisonConfig,
) -> dict[str, float | int | str | bool]:
    lower, upper = _search_bounds(plant, config)
    kp_grid = np.geomspace(lower[0], upper[0], config.grid_kp_points)
    ki_grid = np.geomspace(lower[1], upper[1], config.grid_ki_points)
    kd_grid = np.geomspace(lower[2], upper[2], config.grid_kd_points)
    start = perf_counter()
    best: tuple[np.ndarray, ControllerEvaluation] | None = None
    simulations = 0
    for kp in kp_grid:
        for ki in ki_grid:
            for kd in kd_grid:
                evaluation = _evaluate_controller(
                    plant,
                    float(kp),
                    float(ki),
                    float(kd),
                    tau_d,
                    time_grid,
                )
                simulations += 1
                if best is None or evaluation.final_cost < best[1].final_cost:
                    best = (np.asarray([kp, ki, kd], dtype=np.float64), evaluation)
    elapsed = perf_counter() - start
    if best is None:
        raise RuntimeError("Grid search did not evaluate any controllers.")
    return {
        "method": "grid_search",
        "variant": f"{config.grid_kp_points}x{config.grid_ki_points}x{config.grid_kd_points}",
        "runtime_seconds": float(elapsed),
        "system_simulations": int(simulations),
        "kp": float(best[0][0]),
        "ki": float(best[0][1]),
        "kd": float(best[0][2]),
        "final_cost": float(best[1].final_cost),
        "stable": bool(best[1].stable),
        "overshoot_pct": float(best[1].overshoot_pct),
        "settling_time": float(best[1].settling_time),
        "steady_state_error": float(best[1].steady_state_error),
        "rise_time": float(best[1].rise_time),
        "peak_control_effort": float(best[1].peak_control_effort)
        if best[1].peak_control_effort is not None
        else float("nan"),
    }


def _bayesian_optimization_method(
    plant: Plant,
    tau_d: float,
    time_grid: np.ndarray,
    config: PIDOptimizationComparisonConfig,
    rng: np.random.Generator,
) -> dict[str, float | int | str | bool]:
    lower, upper = _search_bounds(plant, config)
    log_lower = np.log(lower)
    log_upper = np.log(upper)
    base = np.clip(np.asarray(heuristic_pid_scales(plant), dtype=np.float64), lower, upper)

    def evaluate_gains(gains: np.ndarray) -> ControllerEvaluation:
        return _evaluate_controller(
            plant=plant,
            kp=float(gains[0]),
            ki=float(gains[1]),
            kd=float(gains[2]),
            tau_d=tau_d,
            time_grid=time_grid,
        )

    start = perf_counter()
    evaluations: list[tuple[np.ndarray, ControllerEvaluation]] = []
    initial_count = max(3, config.bayes_initial_points)
    seed_points = [np.log(base)]
    while len(seed_points) < initial_count:
        seed_points.append(rng.uniform(log_lower, log_upper))
    for point in seed_points:
        gains = np.exp(point)
        evaluations.append((gains, evaluate_gains(gains)))

    total_budget = max(initial_count, config.bayes_iterations)
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(3, dtype=np.float64),
        length_scale_bounds=(1e-2, 10.0),
        nu=2.5,
    ) + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1))

    while len(evaluations) < total_budget:
        x_train = np.asarray([np.log(item[0]) for item in evaluations], dtype=np.float64)
        y_train = np.asarray([item[1].final_cost for item in evaluations], dtype=np.float64)
        candidate_points = rng.uniform(log_lower, log_upper, size=(config.bayes_candidate_pool, 3))
        candidate_points = np.vstack([candidate_points, x_train[np.argmin(y_train)].reshape(1, -1)])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=1e-6,
                    normalize_y=True,
                    random_state=config.seed,
                )
                gp.fit(x_train, y_train)
                mean, std = gp.predict(candidate_points, return_std=True)
            improvement = np.min(y_train) - mean - 1e-3
            z_score = np.divide(
                improvement,
                std,
                out=np.zeros_like(improvement),
                where=std > 1e-12,
            )
            expected_improvement = (improvement * norm.cdf(z_score)) + (std * norm.pdf(z_score))
            candidate = candidate_points[int(np.argmax(expected_improvement))]
        except Exception:
            candidate = rng.uniform(log_lower, log_upper)
        gains = np.exp(candidate)
        evaluations.append((gains, evaluate_gains(gains)))

    elapsed = perf_counter() - start
    best_gains, best_eval = min(evaluations, key=lambda item: item[1].final_cost)
    return {
        "method": "bayesian_optimization",
        "variant": f"gp_ei_{config.bayes_iterations}",
        "runtime_seconds": float(elapsed),
        "system_simulations": int(len(evaluations)),
        "kp": float(best_gains[0]),
        "ki": float(best_gains[1]),
        "kd": float(best_gains[2]),
        "final_cost": float(best_eval.final_cost),
        "stable": bool(best_eval.stable),
        "overshoot_pct": float(best_eval.overshoot_pct),
        "settling_time": float(best_eval.settling_time),
        "steady_state_error": float(best_eval.steady_state_error),
        "rise_time": float(best_eval.rise_time),
        "peak_control_effort": float(best_eval.peak_control_effort)
        if best_eval.peak_control_effort is not None
        else float("nan"),
    }


def _surrogate_feature_vector(
    context: SurrogateContext,
    gain_logs: torch.Tensor,
) -> torch.Tensor:
    gains = torch.exp(gain_logs)
    values: list[torch.Tensor] = []
    for column in context.models.feature_columns:
        if column == "kp":
            values.append(gains[0])
        elif column == "ki":
            values.append(gains[1])
        elif column == "kd":
            values.append(gains[2])
        elif column == "log10_kp":
            values.append(gain_logs[0] / np.log(10.0))
        elif column == "log10_ki":
            values.append(gain_logs[1] / np.log(10.0))
        elif column == "log10_kd":
            values.append(gain_logs[2] / np.log(10.0))
        else:
            values.append(context.constant_features[column])
    return torch.stack(values).unsqueeze(0)


def _surrogate_objective(
    context: SurrogateContext,
    gain_logs: torch.Tensor,
) -> torch.Tensor:
    raw_features = _surrogate_feature_vector(context, gain_logs)
    scaled = (raw_features - context.feature_mean) / context.feature_std
    stability_probability = torch.sigmoid(context.models.classifier(scaled)).squeeze()
    predicted_scaled = context.models.regressor(scaled).squeeze(0)
    predicted = (predicted_scaled * context.trajectory_std) + context.trajectory_mean
    overshoot = torch.relu(torch.max(predicted) - 1.0)
    steady_state_error = torch.abs(predicted[-1] - 1.0)
    band_violation = torch.sigmoid(45.0 * (torch.abs(predicted - 1.0) - context.settling_band))
    time_weights = torch.linspace(
        0.1,
        1.0,
        predicted.shape[0],
        device=predicted.device,
        dtype=predicted.dtype,
    )
    settling_proxy = (
        (band_violation * time_weights).sum() / torch.sum(time_weights)
    ) * context.time_horizon
    stability_penalty = torch.relu(0.9 - stability_probability) * context.time_horizon
    return settling_proxy + overshoot + steady_state_error + stability_penalty


def _surrogate_context(
    models: Any,
    row: pd.Series,
    time_grid: np.ndarray,
) -> SurrogateContext:
    device = models.device
    constant_features: dict[str, torch.Tensor] = {}
    for column in models.feature_columns:
        if column in {"kp", "ki", "kd", "log10_kp", "log10_ki", "log10_kd"}:
            continue
        constant_features[column] = torch.tensor(
            float(row.get(column, 0.0)),
            dtype=torch.float32,
            device=device,
        )
    return SurrogateContext(
        models=models,
        feature_mean=torch.tensor(models.feature_scaler.mean, dtype=torch.float32, device=device),
        feature_std=torch.tensor(models.feature_scaler.std, dtype=torch.float32, device=device),
        trajectory_mean=torch.tensor(
            models.trajectory_scaler.mean,
            dtype=torch.float32,
            device=device,
        ),
        trajectory_std=torch.tensor(
            models.trajectory_scaler.std,
            dtype=torch.float32,
            device=device,
        ),
        log_gain_low=torch.log(
            torch.tensor(ABSOLUTE_GAIN_LOW, dtype=torch.float32, device=device)
        ),
        log_gain_high=torch.log(
            torch.tensor(ABSOLUTE_GAIN_HIGH, dtype=torch.float32, device=device)
        ),
        settling_band=0.02,
        time_horizon=float(time_grid[-1]),
        constant_features=constant_features,
    )


def _surrogate_gradient_method(
    plant: Plant,
    row: pd.Series,
    tau_d: float,
    time_grid: np.ndarray,
    config: PIDOptimizationComparisonConfig,
    models: Any,
) -> dict[str, float | int | str | bool]:
    context = _surrogate_context(models, row, time_grid)
    initial_gains = np.clip(
        np.asarray(heuristic_pid_scales(plant), dtype=np.float64),
        ABSOLUTE_GAIN_LOW,
        ABSOLUTE_GAIN_HIGH,
    )
    gain_logs = torch.nn.Parameter(
        torch.log(torch.tensor(initial_gains, dtype=torch.float32, device=models.device))
    )
    start = perf_counter()
    best_loss = float("inf")
    best_gains = initial_gains.copy()
    for _ in range(config.surrogate_steps):
        if gain_logs.grad is not None:
            gain_logs.grad.zero_()
        objective = _surrogate_objective(context, gain_logs)
        objective.backward()
        if gain_logs.grad is None or not torch.all(torch.isfinite(gain_logs.grad)):
            break
        with torch.no_grad():
            gain_logs -= config.surrogate_learning_rate * gain_logs.grad
            gain_logs.copy_(
                torch.minimum(
                    torch.maximum(gain_logs, context.log_gain_low),
                    context.log_gain_high,
                )
            )
            current_loss = float(objective.detach().cpu())
            if current_loss < best_loss:
                best_loss = current_loss
                best_gains = torch.exp(gain_logs.detach()).cpu().numpy().astype(np.float64)
    evaluation = _evaluate_controller(
        plant=plant,
        kp=float(best_gains[0]),
        ki=float(best_gains[1]),
        kd=float(best_gains[2]),
        tau_d=tau_d,
        time_grid=time_grid,
    )
    elapsed = perf_counter() - start
    return {
        "method": "surrogate_gradient",
        "variant": f"steps_{config.surrogate_steps}",
        "runtime_seconds": float(elapsed),
        "system_simulations": 1,
        "kp": float(best_gains[0]),
        "ki": float(best_gains[1]),
        "kd": float(best_gains[2]),
        "final_cost": float(evaluation.final_cost),
        "stable": bool(evaluation.stable),
        "overshoot_pct": float(evaluation.overshoot_pct),
        "settling_time": float(evaluation.settling_time),
        "steady_state_error": float(evaluation.steady_state_error),
        "rise_time": float(evaluation.rise_time),
        "peak_control_effort": float(evaluation.peak_control_effort)
        if evaluation.peak_control_effort is not None
        else float("nan"),
    }


def _ziegler_nichols_method(
    plant: Plant,
    tau_d: float,
    time_grid: np.ndarray,
) -> dict[str, float | int | str | bool]:
    start = perf_counter()
    gains, variant = _ziegler_nichols_open_loop(plant, tau_d, time_grid)
    evaluation = _evaluate_controller(
        plant=plant,
        kp=float(gains[0]),
        ki=float(gains[1]),
        kd=float(gains[2]),
        tau_d=tau_d,
        time_grid=time_grid,
    )
    elapsed = perf_counter() - start
    return {
        "method": "ziegler_nichols",
        "variant": str(variant),
        "runtime_seconds": float(elapsed),
        "system_simulations": 2,
        "kp": float(gains[0]),
        "ki": float(gains[1]),
        "kd": float(gains[2]),
        "final_cost": float(evaluation.final_cost),
        "stable": bool(evaluation.stable),
        "overshoot_pct": float(evaluation.overshoot_pct),
        "settling_time": float(evaluation.settling_time),
        "steady_state_error": float(evaluation.steady_state_error),
        "rise_time": float(evaluation.rise_time),
        "peak_control_effort": float(evaluation.peak_control_effort)
        if evaluation.peak_control_effort is not None
        else float("nan"),
    }


def _sample_unique_plants(
    config: PIDOptimizationComparisonConfig,
    dataset_dir: Path,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    required_columns = [
        "plant_id",
        "plant_family",
        "split",
        "b0",
        "a2",
        "a1",
        "a0",
        "dc_gain",
        "dominant_pole_mag",
        "mean_pole_mag",
        "plant_order",
        "plant_min_damping_ratio",
        "plant_max_oscillation_hz",
        "plant_pole_spread_log10",
        "plant_has_complex_poles",
        "plant_max_real_part",
        "plant_min_real_part",
        "pole_0_real",
        "pole_1_real",
        "pole_2_real",
    ]
    reservoir: list[pd.Series] = []
    seen = 0
    for frame, _ in iter_dataset_chunks(
        dataset_dir,
        include_trajectories=False,
        columns=required_columns,
        progress_desc="pid-opt-sample",
    ):
        unique_rows = frame.drop_duplicates(subset="plant_id").reset_index(drop=True)
        mask = unique_rows["split"].isin(config.source_splits)
        if config.exclude_families:
            mask &= ~unique_rows["plant_family"].isin(config.exclude_families)
        mask &= (
            unique_rows[["pole_0_real", "pole_1_real", "pole_2_real"]]
            .max(axis=1)
            .to_numpy(dtype=float)
            < 0.0
        )
        eligible = unique_rows.loc[mask].reset_index(drop=True)
        for _, row in eligible.iterrows():
            seen += 1
            if len(reservoir) < config.n_plants:
                reservoir.append(row.copy())
                continue
            replacement_index = int(rng.integers(0, seen))
            if replacement_index < config.n_plants:
                reservoir[replacement_index] = row.copy()
    if len(reservoir) < config.n_plants:
        raise RuntimeError(
            f"Only found {len(reservoir)} eligible plants, need {config.n_plants}."
        )
    sampled = pd.DataFrame(reservoir).sort_values("plant_id").reset_index(drop=True)
    sampled["sample_rank"] = np.arange(1, sampled.shape[0] + 1)
    return sampled


def _summary_table(results: pd.DataFrame) -> pd.DataFrame:
    win_cost = results.groupby("plant_id")["final_cost"].transform("min")
    results = results.copy()
    results["is_best_cost"] = np.isclose(results["final_cost"], win_cost)
    summary = (
        results.groupby("method", sort=False)
        .agg(
            plants=("plant_id", "nunique"),
            stable_rate=("stable", "mean"),
            final_cost_mean=("final_cost", "mean"),
            final_cost_median=("final_cost", "median"),
            final_cost_std=("final_cost", "std"),
            runtime_mean_seconds=("runtime_seconds", "mean"),
            runtime_median_seconds=("runtime_seconds", "median"),
            system_simulations_mean=("system_simulations", "mean"),
            system_simulations_median=("system_simulations", "median"),
            win_rate=("is_best_cost", "mean"),
        )
        .reset_index()
    )
    summary["method_label"] = summary["method"].map(METHOD_LABELS)
    summary = summary[
        [
            "method",
            "method_label",
            "plants",
            "stable_rate",
            "final_cost_mean",
            "final_cost_median",
            "final_cost_std",
            "runtime_mean_seconds",
            "runtime_median_seconds",
            "system_simulations_mean",
            "system_simulations_median",
            "win_rate",
        ]
    ]
    return summary


def _plot_cost_boxplot(results: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 5.2))
    ordered = [method for method in METHOD_ORDER if method in set(results["method"])]
    data = [
        results.loc[results["method"] == method, "final_cost"].to_numpy(dtype=float)
        for method in ordered
    ]
    boxplot = axis.boxplot(
        data,
        patch_artist=True,
        tick_labels=[METHOD_LABELS[method] for method in ordered],
    )
    palette = ["#d95f02", "#1b9e77", "#7570b3", "#e7298a"]
    for patch, color in zip(boxplot["boxes"], palette, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    axis.set_ylabel("Final Cost J")
    axis.set_title("PID Optimization Cost Across 100 Plants")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def _plot_runtime_comparison(results: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 5.2))
    ordered = [method for method in METHOD_ORDER if method in set(results["method"])]
    data = [
        results.loc[results["method"] == method, "runtime_seconds"].to_numpy(dtype=float)
        for method in ordered
    ]
    boxplot = axis.boxplot(
        data,
        patch_artist=True,
        tick_labels=[METHOD_LABELS[method] for method in ordered],
    )
    palette = ["#d95f02", "#1b9e77", "#7570b3", "#e7298a"]
    for patch, color in zip(boxplot["boxes"], palette, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    axis.set_yscale("log")
    axis.set_ylabel("Runtime [s]")
    axis.set_title("PID Optimization Runtime Comparison")
    axis.grid(axis="y", alpha=0.25, which="both")
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def run_pid_optimization_comparison(config: PIDOptimizationComparisonConfig) -> Path:
    dataset_dir = resolve_path(config.dataset_dir)
    models = load_models(config.run_dir, device=config.device)
    time_grid = dataset_time_grid(dataset_dir)
    tables_dir = ensure_dir(resolve_path(config.tables_dir()))
    plots_dir = ensure_dir(resolve_path(config.plots_dir()))
    ensure_dir(resolve_path(config.summary_file()).parent)
    save_config_snapshot(config, tables_dir / f"{config.name}_config_snapshot.yaml")

    _log_pid_opt_stage(f"[pid-opt-compare] sample plants {config.name}")
    plants_frame = _sample_unique_plants(config, dataset_dir)
    plants_frame.to_csv(tables_dir / "pid_optimization_comparison_plants.csv", index=False)

    rng = np.random.default_rng(config.seed)
    rows: list[dict[str, float | int | str | bool]] = []
    _log_pid_opt_stage(f"[pid-opt-compare] optimize {config.name}")
    for _, row in tqdm(
        plants_frame.iterrows(),
        total=plants_frame.shape[0],
        desc="pid-opt-compare:plants",
        unit="plant",
    ):
        plant = plant_from_sample_row(row)
        tau_d = 0.0
        shared = {
            "name": config.name,
            "plant_id": int(row["plant_id"]),
            "plant_family": str(row["plant_family"]),
            "split": str(row["split"]),
            "sample_rank": int(row["sample_rank"]),
        }
        rows.append({**shared, **_ziegler_nichols_method(plant, tau_d, time_grid)})
        rows.append({**shared, **_grid_search_method(plant, tau_d, time_grid, config)})
        rows.append(
            {
                **shared,
                **_bayesian_optimization_method(plant, tau_d, time_grid, config, rng),
            }
        )
        rows.append(
            {
                **shared,
                **_surrogate_gradient_method(
                    plant=plant,
                    row=row,
                    tau_d=tau_d,
                    time_grid=time_grid,
                    config=config,
                    models=models,
                ),
            }
        )

    results = pd.DataFrame(rows)
    results["method_label"] = results["method"].map(METHOD_LABELS)
    results.to_csv(tables_dir / "pid_optimization_comparison_results.csv", index=False)

    summary = _summary_table(results)
    summary.to_csv(tables_dir / "pid_optimization_comparison_summary.csv", index=False)
    _plot_cost_boxplot(results, plots_dir / "pid_optimization_cost_boxplot.png")
    _plot_runtime_comparison(results, plots_dir / "pid_optimization_runtime_comparison.png")

    payload = {
        "name": config.name,
        "n_plants": int(plants_frame.shape[0]),
        "source_splits": list(config.source_splits),
        "excluded_families": list(config.exclude_families),
        "objective": "J = settling_time + overshoot_pct/100 + steady_state_error",
        "results_path": str(tables_dir / "pid_optimization_comparison_results.csv"),
        "summary_path": str(tables_dir / "pid_optimization_comparison_summary.csv"),
        "cost_plot_path": str(plots_dir / "pid_optimization_cost_boxplot.png"),
        "runtime_plot_path": str(plots_dir / "pid_optimization_runtime_comparison.png"),
        "summary_rows": summary.to_dict(orient="records"),
    }
    dump_json(payload, resolve_path(config.summary_file()))

    summary_text = summary.to_string(
        index=False,
        justify="left",
        float_format=lambda value: f"{value:.4f}",
    )
    markdown_lines = [
        f"# {config.name}",
        "",
        "Objective: `J = settling_time + overshoot_pct/100 + steady_state_error`",
        "",
        "```text",
        summary_text,
        "```",
    ]
    (tables_dir / "pid_optimization_comparison_summary.md").write_text(
        "\n".join(markdown_lines),
        encoding="utf-8",
    )

    _log_pid_opt_stage(f"[pid-opt-compare] complete {config.name}")
    return tables_dir
