from __future__ import annotations

import numpy as np

from controlsimulator.plants import FAMILY_CHOICES, sample_plant


def test_sampled_plants_are_stable() -> None:
    rng = np.random.default_rng(7)
    for plant_id in range(40):
        plant = sample_plant(rng, plant_id=plant_id, families=FAMILY_CHOICES)
        assert np.all(np.real(plant.poles) < 0.0)
        assert plant.dominant_pole_mag > 0.0
        assert plant.denominator[0] == 1.0


def test_campaign_unstable_family_can_sample_positive_poles() -> None:
    rng = np.random.default_rng(11)
    plant = sample_plant(
        rng,
        plant_id=0,
        families=["campaign_third_order_unstable"],
    )

    assert plant.denominator.shape == (4,)
    assert np.max(np.real(plant.poles)) > 0.0
    assert 0.5 <= plant.numerator[-1] <= 3.0
