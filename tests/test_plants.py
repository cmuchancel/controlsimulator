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
