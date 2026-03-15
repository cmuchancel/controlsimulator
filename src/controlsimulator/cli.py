from __future__ import annotations

import argparse

from controlsimulator.benchmark import benchmark_models
from controlsimulator.config import DatasetConfig, EvaluationConfig, TrainingConfig, load_config
from controlsimulator.dataset import generate_dataset
from controlsimulator.evaluate import evaluate_models
from controlsimulator.train import train_models
from controlsimulator.utils import resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(prog="controlsimulator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ["generate-dataset", "train", "evaluate", "benchmark", "demo"]:
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True)

    subparsers.add_parser("overnight")
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
    if args.command == "overnight":
        _run_overnight()
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
