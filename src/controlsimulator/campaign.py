from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import pandas as pd

from controlsimulator.benchmark import benchmark_models
from controlsimulator.config import DatasetConfig, EvaluationConfig, TrainingConfig, load_config
from controlsimulator.dataset import export_dataset_layout, generate_dataset
from controlsimulator.evaluate import evaluate_models
from controlsimulator.train import train_models
from controlsimulator.utils import format_seconds, load_json, resolve_path


def _log_campaign_stage(message: str) -> None:
    print(message, flush=True)


def run_campaign(
    dataset_config: DatasetConfig,
    training_config: TrainingConfig,
    evaluation_config: EvaluationConfig,
) -> Path:
    _log_campaign_stage(f"[campaign] dataset {dataset_config.name}")
    dataset_dir = generate_dataset(dataset_config)
    if dataset_config.export_dataset_layout:
        _log_campaign_stage(f"[campaign] export {dataset_config.name}")
        export_dataset_layout(dataset_dir)

    training_config = replace(training_config, dataset_dir=str(dataset_dir))
    _log_campaign_stage(f"[campaign] train {training_config.name}")
    run_dir = train_models(training_config)

    evaluation_config = replace(
        evaluation_config,
        dataset_dir=str(dataset_dir),
        run_dir=str(run_dir),
    )
    _log_campaign_stage(f"[campaign] evaluate {evaluation_config.name}")
    report_dir = evaluate_models(evaluation_config)
    _log_campaign_stage(f"[campaign] benchmark {evaluation_config.name}")
    benchmark_models(evaluation_config)

    training_curves = run_dir / "training_curves.png"
    if training_curves.exists():
        shutil.copy2(training_curves, report_dir / "training_curves.png")

    _write_campaign_report(
        dataset_dir=dataset_dir,
        run_dir=run_dir,
        report_dir=report_dir,
    )
    _log_campaign_stage(f"[campaign] complete {evaluation_config.name}")
    return report_dir


def run_campaign_from_paths(
    dataset_config_path: str | Path,
    training_config_path: str | Path,
    evaluation_config_path: str | Path,
) -> Path:
    dataset_config = load_config(resolve_path(dataset_config_path), DatasetConfig)
    training_config = load_config(resolve_path(training_config_path), TrainingConfig)
    evaluation_config = load_config(resolve_path(evaluation_config_path), EvaluationConfig)
    return run_campaign(dataset_config, training_config, evaluation_config)


def _write_campaign_report(
    dataset_dir: Path,
    run_dir: Path,
    report_dir: Path,
) -> None:
    dataset_metadata = load_json(dataset_dir / "metadata.json")
    train_summary = load_json(run_dir / "train_summary.json")
    evaluation_summary = load_json(report_dir / "evaluation_summary.json")
    benchmark_summary = load_json(report_dir / "benchmark_summary.json")
    sample_errors = pd.read_csv(report_dir / "sample_errors.csv")
    family_metrics = pd.read_csv(report_dir / "family_metrics.csv")

    worst_failures = sample_errors.sort_values("trajectory_rmse", ascending=False).head(5)
    hardest_family = (
        family_metrics.loc[family_metrics["task"] == "regressor"]
        .sort_values("trajectory_rmse", ascending=False)
        .head(5)
    )
    export_dir = dataset_dir / "dataset"

    classifier_test = evaluation_summary["classifier"].get("test", {})
    classifier_ood = evaluation_summary["classifier"].get("ood_test", {})
    regressor_test = evaluation_summary["regressor"].get("test", {})
    regressor_ood = evaluation_summary["regressor"].get("ood_test", {})
    gain_stats = dataset_metadata.get("gain_stats", {})
    pole_stats = dataset_metadata.get("pole_stats", {})

    lines = [
        "# Campaign Report",
        "",
        "## Dataset",
        "",
        f"- dataset size: {int(dataset_metadata['n_samples']):,} simulations",
        f"- plants: {int(dataset_metadata['n_plants']):,}",
        f"- time steps: {int(dataset_metadata['n_time_steps'])}",
        f"- stable fraction: {float(dataset_metadata['stable_fraction_pct']):.2f}%",
        f"- unstable responses: {int(dataset_metadata.get('unstable_count', 0)):,}",
        f"- oscillatory responses: {int(dataset_metadata.get('oscillatory_count', 0)):,}",
        f"- near-instability responses: {int(dataset_metadata.get('near_instability_count', 0)):,}",
        (
            f"- generation time: "
            f"{format_seconds(float(dataset_metadata.get('generation_seconds', 0.0)))}"
        ),
        f"- train samples: {int(train_summary.get('train_samples', 0)):,}",
        f"- val samples: {int(train_summary.get('val_samples', 0)):,}",
        "",
        "## Pole Distribution",
        "",
        _format_triplet(
            label="sampled real-part min / median / max",
            low=float(pole_stats.get("real_part_min", float("nan"))),
            mid=float(pole_stats.get("real_part_p50", float("nan"))),
            high=float(pole_stats.get("real_part_max", float("nan"))),
            fmt=".4f",
        ),
        f"- |imag| p95: {pole_stats.get('imag_part_abs_p95', float('nan')):.4f}",
    ]
    for family, count in sorted(dataset_metadata.get("family_counts", {}).items()):
        lines.append(f"- {family}: {int(count):,}")

    lines.extend(
        [
            "",
            "## PID Gains",
            "",
            _format_triplet(
                label="Kp p05 / median / p95",
                low=float(gain_stats.get("kp_p05", float("nan"))),
                mid=float(gain_stats.get("kp_median", float("nan"))),
                high=float(gain_stats.get("kp_p95", float("nan"))),
                fmt=".4g",
            ),
            _format_triplet(
                label="Ki p05 / median / p95",
                low=float(gain_stats.get("ki_p05", float("nan"))),
                mid=float(gain_stats.get("ki_median", float("nan"))),
                high=float(gain_stats.get("ki_p95", float("nan"))),
                fmt=".4g",
            ),
            _format_triplet(
                label="Kd p05 / median / p95",
                low=float(gain_stats.get("kd_p05", float("nan"))),
                mid=float(gain_stats.get("kd_median", float("nan"))),
                high=float(gain_stats.get("kd_p95", float("nan"))),
                fmt=".4g",
            ),
            "",
            "## Classifier",
            "",
            _format_classifier_metrics("test", classifier_test),
            _format_classifier_metrics("ood", classifier_ood),
            "",
            "## Trajectory Regressor",
            "",
            _format_regressor_metrics("test", regressor_test),
            _format_regressor_error_metrics("test", regressor_test),
            _format_regressor_metrics("ood", regressor_ood),
            _format_regressor_error_metrics("ood", regressor_ood),
            "",
            "## Baselines",
            "",
            _format_baseline_metric(
                "mean-baseline test RMSE",
                regressor_test,
                "mean_baseline_trajectory_rmse",
            ),
            _format_baseline_metric(
                "k-NN-baseline test RMSE",
                regressor_test,
                "knn_baseline_trajectory_rmse",
            ),
            _format_baseline_metric(
                "mean-baseline ood RMSE",
                regressor_ood,
                "mean_baseline_trajectory_rmse",
            ),
            _format_baseline_metric(
                "k-NN-baseline ood RMSE",
                regressor_ood,
                "knn_baseline_trajectory_rmse",
            ),
            "",
            "## Benchmark",
            "",
            (
                f"- single-sample speedup: "
                f"{benchmark_summary.get('single_speedup_x', float('nan')):.2f}x"
            ),
            f"- batch speedup: {benchmark_summary.get('batch_speedup_x', float('nan')):.2f}x",
            "",
            "## Failure Cases",
            "",
            "- hardest regressor families:",
        ]
    )

    for row in hardest_family.itertuples(index=False):
        lines.append(f"- {row.plant_family} ({row.split}): RMSE={float(row.trajectory_rmse):.4f}")

    lines.append("- largest sample errors:")
    for row in worst_failures.itertuples(index=False):
        lines.append(
            f"- sample {int(row.sample_id)} | split={row.split} | "
            f"family={row.plant_family} | RMSE={float(row.trajectory_rmse):.4f}"
        )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- dataset export: {export_dir}",
            f"- training curves: {report_dir / 'training_curves.png'}",
            f"- evaluation metrics: {report_dir / 'metrics.json'}",
            f"- family metrics: {report_dir / 'family_metrics.csv'}",
            f"- sample errors: {report_dir / 'sample_errors.csv'}",
            f"- plots: {evaluation_summary.get('artifacts_plot_dir', '')}",
            "",
            "## Observations",
            "",
            (
                "- the surrogate succeeds when plant poles and controller gains stay "
                "in the dense regions of the training distribution"
            ),
            (
                "- the surrogate fails most often on the highest-RMSE families listed "
                "above and on the worst per-sample errors in the failure table"
            ),
        ]
    )

    (report_dir / "campaign_report.md").write_text("\n".join(lines), encoding="utf-8")


def _format_triplet(label: str, low: float, mid: float, high: float, fmt: str) -> str:
    return (
        f"- {label}: "
        f"{low:{fmt}} / {mid:{fmt}} / {high:{fmt}}"
    )


def _format_classifier_metrics(name: str, metrics: dict[str, float]) -> str:
    return (
        f"- {name} accuracy / precision / recall / f1: "
        f"{metrics.get('accuracy', float('nan')):.4f} / "
        f"{metrics.get('precision', float('nan')):.4f} / "
        f"{metrics.get('recall', float('nan')):.4f} / "
        f"{metrics.get('f1', float('nan')):.4f}"
    )


def _format_regressor_metrics(name: str, metrics: dict[str, float]) -> str:
    return (
        f"- {name} RMSE / MAE: "
        f"{metrics.get('trajectory_rmse', float('nan')):.4f} / "
        f"{metrics.get('trajectory_mae', float('nan')):.4f}"
    )


def _format_regressor_error_metrics(name: str, metrics: dict[str, float]) -> str:
    return (
        f"- {name} overshoot / rise / settling / SSE MAE: "
        f"{metrics.get('overshoot_pct_mae', float('nan')):.4f} / "
        f"{metrics.get('rise_time_mae', float('nan')):.4f} / "
        f"{metrics.get('settling_time_mae', float('nan')):.4f} / "
        f"{metrics.get('steady_state_error_mae', float('nan')):.4f}"
    )


def _format_baseline_metric(name: str, metrics: dict[str, float], key: str) -> str:
    return f"- {name}: {metrics.get(key, float('nan')):.4f}"
