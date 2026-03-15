from __future__ import annotations

import argparse

from controlsimulator.benchmark import benchmark_models
from controlsimulator.campaign import run_campaign_from_paths
from controlsimulator.config import (
    DatasetConfig,
    EvaluationConfig,
    PIDOptimizationComparisonConfig,
    PublicationEvaluationConfig,
    SurrogateWarmStartBOConfig,
    TrainingConfig,
    load_config,
)
from controlsimulator.dataset import generate_dataset
from controlsimulator.evaluate import evaluate_models
from controlsimulator.pid_optimization_compare import run_pid_optimization_comparison
from controlsimulator.pid_surrogate_bo import run_surrogate_warm_start_bo_experiment
from controlsimulator.publication_eval import run_publication_evaluation
from controlsimulator.train import train_models
from controlsimulator.utils import resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(prog="controlsimulator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in [
        "generate-dataset",
        "train",
        "evaluate",
        "benchmark",
        "demo",
        "publication-eval",
        "pid-opt-compare",
        "pid-surrogate-bo",
    ]:
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True)

    subparsers.add_parser("overnight")
    campaign = subparsers.add_parser("campaign")
    campaign.add_argument("--dataset-config", required=True)
    campaign.add_argument("--training-config", required=True)
    campaign.add_argument("--evaluation-config", required=True)
    subparsers.add_parser("overnight-campaign")
    args = parser.parse_args()

    if args.command == "generate-dataset":
        config = load_config(resolve_path(args.config), DatasetConfig)
        dataset_dir = generate_dataset(config)
        print(f"dataset ready: {dataset_dir}")
        return
    if args.command == "train":
        config = load_config(resolve_path(args.config), TrainingConfig)
        run_dir = train_models(config)
        print(f"training run ready: {run_dir}")
        return
    if args.command == "evaluate":
        config = load_config(resolve_path(args.config), EvaluationConfig)
        report_dir = evaluate_models(config)
        print(f"evaluation ready: {report_dir}")
        return
    if args.command == "benchmark":
        config = load_config(resolve_path(args.config), EvaluationConfig)
        report_dir = benchmark_models(config)
        print(f"benchmark ready: {report_dir}")
        return
    if args.command == "demo":
        config = load_config(resolve_path(args.config), EvaluationConfig)
        report_dir = evaluate_models(config)
        print(f"demo plots ready: {report_dir / 'plots'}")
        return
    if args.command == "publication-eval":
        config = load_config(resolve_path(args.config), PublicationEvaluationConfig)
        output_dir = run_publication_evaluation(config)
        print(f"publication evaluation ready: {output_dir}")
        return
    if args.command == "pid-opt-compare":
        config = load_config(resolve_path(args.config), PIDOptimizationComparisonConfig)
        output_dir = run_pid_optimization_comparison(config)
        print(f"pid optimization comparison ready: {output_dir}")
        return
    if args.command == "pid-surrogate-bo":
        config = load_config(resolve_path(args.config), SurrogateWarmStartBOConfig)
        output_dir = run_surrogate_warm_start_bo_experiment(config)
        print(f"pid surrogate bo ready: {output_dir}")
        return
    if args.command == "overnight":
        _run_overnight()
        return
    if args.command == "campaign":
        report_dir = run_campaign_from_paths(
            dataset_config_path=args.dataset_config,
            training_config_path=args.training_config,
            evaluation_config_path=args.evaluation_config,
        )
        print(f"campaign ready: {report_dir}")
        return
    if args.command == "overnight-campaign":
        report_dir = run_campaign_from_paths(
            dataset_config_path="configs/datasets/third_order_campaign.yaml",
            training_config_path="configs/training/third_order_campaign.yaml",
            evaluation_config_path="configs/evaluation/third_order_campaign.yaml",
        )
        print(f"campaign ready: {report_dir}")
        return
    raise ValueError(f"Unknown command: {args.command}")


def _run_overnight() -> None:
    dataset_config = load_config(resolve_path("configs/datasets/full_v4.yaml"), DatasetConfig)
    training_config = load_config(resolve_path("configs/training/training_v4.yaml"), TrainingConfig)
    evaluation_config = load_config(
        resolve_path("configs/evaluation/evaluation_v4.yaml"),
        EvaluationConfig,
    )

    generate_dataset(dataset_config)
    train_models(training_config)
    evaluate_models(evaluation_config)
    benchmark_models(evaluation_config)
    print("overnight pipeline completed")


if __name__ == "__main__":
    main()
