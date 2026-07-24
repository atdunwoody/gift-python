from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from gift_habitat import (
    avg_hydraulics,
    habitat,
    load_example_curve,
    substrate_suitability,
)


class HabitatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.depth_curve = load_example_curve(
            "depth",
            species="rainbow",
            life_stage="parr",
        )
        cls.velocity_curve = load_example_curve(
            "velocity",
            species="rainbow",
            life_stage="parr",
        )

    def test_matches_documented_r_output(self) -> None:
        hydraulics = avg_hydraulics(
            slope=0.01,
            bankfull_width=10,
            bankfull_depth=0.5,
            max_bankfull_depth=0.75,
            d84_mm=100,
        )
        result = habitat(
            hydraulics,
            self.depth_curve,
            self.velocity_curve,
        )
        expected = np.array(
            [
                [0.001, 0.07580115, 0.1293719, 1, 1.330329, 0.01304593],
                [0.002, 0.10185502, 0.1765005, 1, 1.626481, 0.02923999],
                [0.003, 0.11985860, 0.2084220, 1, 1.827366, 0.04564974],
                [0.004, 0.13483312, 0.2351875, 1, 1.985960, 0.06297691],
                [0.005, 0.14586234, 0.2566764, 1, 2.114073, 0.07914966],
                [0.006, 0.15651724, 0.2779703, 1, 2.230608, 0.09704739],
            ]
        )
        np.testing.assert_allclose(
            result.iloc[:6].to_numpy(),
            expected,
            rtol=0,
            atol=5e-7,
        )

    def test_substrate_suitability_uses_lower_inclusive_classes(self) -> None:
        curve = pd.DataFrame(
            {
                "lower": [0, 10],
                "upper": [10, 20],
                "suit": [0.25, 1.0],
            }
        )
        gsd = np.array([1, 2, 3, 4, 10, 11, 12, 13, 14, 15])
        self.assertAlmostEqual(substrate_suitability(curve, gsd), 0.7)

    def test_one_substrate_input_defaults_to_one_with_warning(self) -> None:
        hydraulics = avg_hydraulics(
            0.01,
            10,
            0.5,
            100,
            discharges=[0.01, 0.02, 0.03],
        )
        curve = pd.DataFrame(
            {"lower": [0], "upper": [1000], "suit": [0.5]}
        )
        with self.assertWarnsRegex(UserWarning, "Both substrate_curve"):
            result = habitat(
                hydraulics,
                self.depth_curve,
                self.velocity_curve,
                substrate_curve=curve,
            )
        np.testing.assert_array_equal(result["s.suit"], 1.0)


if __name__ == "__main__":
    unittest.main()
