from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def assign_dataset_splits(
    samples: pd.DataFrame,
    val_fraction: float,
    test_fraction: float,
    ood_families: list[str],
    seed: int,
) -> pd.Series:
    plant_frame = samples[["plant_id", "plant_family"]].drop_duplicates().sort_values("plant_id")
    ood_mask = plant_frame["plant_family"].isin(ood_families)
    split_map: dict[int, str] = {
        int(plant_id): "ood_test" for plant_id in plant_frame.loc[ood_mask, "plant_id"].tolist()
    }

    in_distribution = plant_frame.loc[~ood_mask]
    if in_distribution.empty:
        raise ValueError("No in-distribution plants remain after applying the OOD family split.")

    train_val_ids, test_ids = _safe_split(
        in_distribution["plant_id"],
        in_distribution["plant_family"],
        test_fraction,
        seed,
    )
    train_val_frame = in_distribution[in_distribution["plant_id"].isin(train_val_ids)]
    train_ids, val_ids = _safe_split(
        train_val_frame["plant_id"],
        train_val_frame["plant_family"],
        val_fraction / (1.0 - test_fraction),
        seed + 1,
    )

    for plant_id in train_ids:
        split_map[int(plant_id)] = "train"
    for plant_id in val_ids:
        split_map[int(plant_id)] = "val"
    for plant_id in test_ids:
        split_map[int(plant_id)] = "test"

    return samples["plant_id"].map(split_map)


def _safe_split(
    plant_ids: pd.Series,
    families: pd.Series,
    test_size: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    ids = plant_ids.astype(int).tolist()
    labels = families.astype(str).tolist()
    try:
        left, right = train_test_split(
            ids,
            test_size=test_size,
            random_state=seed,
            stratify=labels,
        )
    except ValueError:
        left, right = train_test_split(ids, test_size=test_size, random_state=seed)
    return list(left), list(right)


def assert_no_plant_leakage(samples: pd.DataFrame) -> None:
    for split_a in samples["split"].unique():
        ids_a = set(samples.loc[samples["split"] == split_a, "plant_id"])
        for split_b in samples["split"].unique():
            if split_a == split_b:
                continue
            ids_b = set(samples.loc[samples["split"] == split_b, "plant_id"])
            overlap = ids_a & ids_b
            if overlap:
                message = f"Plant leakage between {split_a} and {split_b}: {sorted(overlap)}"
                raise AssertionError(message)
