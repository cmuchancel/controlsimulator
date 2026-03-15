from __future__ import annotations

import warnings
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from controlsimulator.config import SurrogateWarmStartBOConfig, save_config_snapshot
from controlsimulator.dataset import dataset_time_grid
from controlsimulator.evaluate import load_models
from controlsimulator.pid_optimization_compare import (
    ABSOLUTE_GAIN_HIGH,
    ABSOLUTE_GAIN_LOW,
    _bayesian_optimization_method,
    _evaluate_controller,
    _sample_unique_plants,
    _surrogate_context,
    _surrogate_gradient_method,
    _surrogate_objective,
)
from controlsimulator.plants import heuristic_pid_scales, plant_from_sample_row
from controlsimulator.utils import dump_json, ensure_dir, resolve_path

METHOD_ORDER = [
    "bayesian_optimization",
    "surrogate_gradient",
    "surrogate_bo",
]
METHOD_LABELS = {
    "bayesian_optimization": "Bayesian Optimization",
    "surrogate_gradient": "Surrogate Gradient",
    "surrogate_bo": "Surrogate + BO",
}


def _log_surrogate_bo_stage(message: str) -> None:
    print(message, flush=True)


def _surrogate_seed_gains(
    plant: Any,
    row: pd.Series,
    time_grid: np.ndarray,
    config: SurrogateWarmStartBOConfig,
    models: Any,
) -> tuple[np.ndarray, float]:
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
                    torch.maximum(
                        gain_logs,
                        context.log_gain_low,
                    ),
                    context.log_gain_high,
                )
            )
            current_loss = float(objective.detach().cpu())
            if current_loss < best_loss:
                best_loss = current_loss
                best_gains = torch.exp(gain_logs.detach()).cpu().numpy().astype(np.float64)
    return best_gains, perf_counter() - start


def _surrogate_centered_bounds(
    seed_gains: np.ndarray,
    config: SurrogateWarmStartBOConfig,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.clip(
        seed_gains * config.surrogate_bo_lower_factor,
        ABSOLUTE_GAIN_LOW,
        ABSOLUTE_GAIN_HIGH,
    )
    upper = np.clip(
        seed_gains * config.surrogate_bo_upper_factor,
        ABSOLUTE_GAIN_LOW,
        ABSOLUTE_GAIN_HIGH,
    )
    for index in range(3):
        if upper[index] <= lower[index]:
            lower[index] = max(ABSOLUTE_GAIN_LOW[index], seed_gains[index] * 0.9)
            upper[index] = min(ABSOLUTE_GAIN_HIGH[index], seed_gains[index] * 1.1)
        if upper[index] <= lower[index]:
            lower[index] = float(ABSOLUTE_GAIN_LOW[index])
            upper[index] = float(ABSOLUTE_GAIN_HIGH[index])
    return lower.astype(np.float64), upper.astype(np.float64)


def _surrogate_warm_start_bo_method(
    plant: Any,
    row: pd.Series,
    tau_d: float,
    time_grid: np.ndarray,
    config: SurrogateWarmStartBOConfig,
    models: Any,
    rng: np.random.Generator,
) -> dict[str, float | int | str | bool]:
    seed_gains, surrogate_runtime = _surrogate_seed_gains(
        plant=plant,
        row=row,
        time_grid=time_grid,
        config=config,
        models=models,
    )
    lower, upper = _surrogate_centered_bounds(seed_gains, config)
    log_lower = np.log(lower)
    log_upper = np.log(upper)

    def evaluate_gains(gains: np.ndarray):
        return _evaluate_controller(
            plant=plant,
            kp=float(gains[0]),
            ki=float(gains[1]),
            kd=float(gains[2]),
            tau_d=tau_d,
            time_grid=time_grid,
        )

    start = perf_counter()
    evaluations: list[tuple[np.ndarray, Any]] = [(seed_gains.copy(), evaluate_gains(seed_gains))]
    total_budget = max(1, int(config.surrogate_bo_iterations))
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(3, dtype=np.float64),
        length_scale_bounds=(1e-2, 10.0),
        nu=2.5,
    ) + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1))

    while len(evaluations) < total_budget:
        x_train = np.asarray([np.log(item[0]) for item in evaluations], dtype=np.float64)
        y_train = np.asarray([item[1].final_cost for item in evaluations], dtype=np.float64)
        candidate_points = rng.uniform(
            log_lower,
            log_upper,
            size=(config.surrogate_bo_candidate_pool, 3),
        )
        candidate_points = np.vstack([candidate_points, np.log(seed_gains).reshape(1, -1)])
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

    bo_runtime = perf_counter() - start
    best_gains, best_eval = min(evaluations, key=lambda item: item[1].final_cost)
    return {
        "method": "surrogate_bo",
        "variant": f"warm_start_gp_ei_{config.surrogate_bo_iterations}",
        "runtime_seconds": float(surrogate_runtime + bo_runtime),
        "number_of_simulations": int(len(evaluations)),
        "kp": float(best_gains[0]),
        "ki": float(best_gains[1]),
        "kd": float(best_gains[2]),
        "final_cost": float(best_eval.final_cost),
        "stable": bool(best_eval.stable),
        "overshoot_pct": float(best_eval.overshoot_pct),
        "settling_time": float(best_eval.settling_time),
        "steady_state_error": float(best_eval.steady_state_error),
        "rise_time": float(best_eval.rise_time),
        "seed_kp": float(seed_gains[0]),
        "seed_ki": float(seed_gains[1]),
        "seed_kd": float(seed_gains[2]),
    }


def _summary_table(results: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results.groupby("method", sort=False)
        .agg(
            plants=("plant_id", "nunique"),
            mean_cost=("final_cost", "mean"),
            median_cost=("final_cost", "median"),
            stability_rate=("stable", "mean"),
            mean_runtime=("runtime_seconds", "mean"),
            mean_simulations=("number_of_simulations", "mean"),
        )
        .reset_index()
    )
    summary["method_label"] = summary["method"].map(METHOD_LABELS)
    return summary[
        [
            "method",
            "method_label",
            "plants",
            "mean_cost",
            "median_cost",
            "stability_rate",
            "mean_runtime",
            "mean_simulations",
        ]
    ]


def _plot_cost_boxplot(results: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9.5, 5.0))
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
    palette = ["#1b9e77", "#7570b3", "#e7298a"]
    for patch, color in zip(boxplot["boxes"], palette, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.58)
    axis.set_ylabel("Final Cost J")
    axis.set_title("Surrogate Warm-Start BO Cost Comparison")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def _plot_simulation_counts(results: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 4.6))
    summary = (
        results.groupby("method", sort=False)["number_of_simulations"]
        .mean()
        .reindex(METHOD_ORDER)
        .dropna()
    )
    labels = [METHOD_LABELS[method] for method in summary.index]
    axis.bar(labels, summary.to_numpy(dtype=float), color=["#1b9e77", "#7570b3", "#e7298a"])
    axis.set_ylabel("Mean Number of Simulations")
    axis.set_title("Simulation Budget Comparison")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def run_surrogate_warm_start_bo_experiment(config: SurrogateWarmStartBOConfig) -> Path:
    dataset_dir = resolve_path(config.dataset_dir)
    models = load_models(config.run_dir, device=config.device)
    time_grid = dataset_time_grid(dataset_dir)
    tables_dir = ensure_dir(resolve_path(config.tables_dir()))
    plots_dir = ensure_dir(resolve_path(config.plots_dir()))
    ensure_dir(resolve_path(config.summary_file()).parent)
    save_config_snapshot(config, tables_dir / f"{config.name}_config_snapshot.yaml")

    _log_surrogate_bo_stage(f"[pid-surrogate-bo] sample plants {config.name}")
    plants_frame = _sample_unique_plants(config, dataset_dir)
    plants_frame.to_csv(tables_dir / "pid_surrogate_bo_plants.csv", index=False)

    rng = np.random.default_rng(config.seed)
    rows: list[dict[str, float | int | str | bool]] = []
    _log_surrogate_bo_stage(f"[pid-surrogate-bo] optimize {config.name}")
    for _, row in tqdm(
        plants_frame.iterrows(),
        total=plants_frame.shape[0],
        desc="pid-surrogate-bo:plants",
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
        baseline_bo = _bayesian_optimization_method(plant, tau_d, time_grid, config, rng)
        baseline_bo["number_of_simulations"] = int(baseline_bo.pop("system_simulations"))
        surrogate_gradient = _surrogate_gradient_method(
            plant=plant,
            row=row,
            tau_d=tau_d,
            time_grid=time_grid,
            config=config,
            models=models,
        )
        surrogate_gradient["number_of_simulations"] = int(
            surrogate_gradient.pop("system_simulations")
        )
        surrogate_bo = _surrogate_warm_start_bo_method(
            plant=plant,
            row=row,
            tau_d=tau_d,
            time_grid=time_grid,
            config=config,
            models=models,
            rng=rng,
        )
        rows.append({**shared, **baseline_bo})
        rows.append({**shared, **surrogate_gradient})
        rows.append({**shared, **surrogate_bo})

    results = pd.DataFrame(rows)
    results["method_label"] = results["method"].map(METHOD_LABELS)
    results.to_csv(tables_dir / "pid_surrogate_bo_results.csv", index=False)

    summary = _summary_table(results)
    summary.to_csv(tables_dir / "pid_surrogate_bo_summary.csv", index=False)
    _plot_cost_boxplot(results, plots_dir / "pid_surrogate_bo_cost_boxplot.png")
    _plot_simulation_counts(results, plots_dir / "pid_surrogate_bo_simulation_counts.png")

    payload = {
        "name": config.name,
        "n_plants": int(plants_frame.shape[0]),
        "source_splits": list(config.source_splits),
        "excluded_families": list(config.exclude_families),
        "objective": "J = settling_time + overshoot_pct/100 + steady_state_error",
        "results_path": str(tables_dir / "pid_surrogate_bo_results.csv"),
        "summary_path": str(tables_dir / "pid_surrogate_bo_summary.csv"),
        "cost_plot_path": str(plots_dir / "pid_surrogate_bo_cost_boxplot.png"),
        "simulation_plot_path": str(plots_dir / "pid_surrogate_bo_simulation_counts.png"),
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
    (tables_dir / "pid_surrogate_bo_summary.md").write_text(
        "\n".join(markdown_lines),
        encoding="utf-8",
    )

    _log_surrogate_bo_stage(f"[pid-surrogate-bo] complete {config.name}")
    return tables_dir
