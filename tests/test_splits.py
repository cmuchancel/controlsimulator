from __future__ import annotations

import pandas as pd

from controlsimulator.splits import assert_no_plant_leakage, assign_dataset_splits


def test_split_assignment_prevents_plant_leakage() -> None:
    samples = pd.DataFrame(
        {
            "plant_id": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "plant_family": [
                "first_order",
                "first_order",
                "first_order",
                "first_order",
                "second_order",
                "second_order",
                "second_order",
                "second_order",
                "third_order",
                "third_order",
                "lightly_damped_second_order",
                "lightly_damped_second_order",
            ],
        }
    )
    samples["split"] = assign_dataset_splits(
        samples,
        val_fraction=0.2,
        test_fraction=0.2,
        ood_families=["lightly_damped_second_order"],
        seed=11,
    )
    assert_no_plant_leakage(samples)
    grouped = samples.groupby("plant_id")["split"].nunique()
    assert grouped.max() == 1
