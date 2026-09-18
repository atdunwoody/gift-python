from __future__ import annotations

import unittest

import numpy as np

from gift_habitat import avg_hydraulics


class AvgHydraulicsTests(unittest.TestCase):
    def test_matches_documented_r_output_with_derived_shape(self) -> None:
        result = avg_hydraulics(
            slope=0.01,
            bankfull_width=10,
            bankfull_depth=0.5,
            d84_mm=100,
        )
        expected = np.array(
            [
                [0.001, 0.04809630, 2.469268, 0.01947684, 0.02078792],
                [0.002, 0.07151588, 3.010765, 0.02375331, 0.02796553],
                [0.003, 0.09024328, 3.381748, 0.02668452, 0.03324022],
                [0.004, 0.10648644, 3.673706, 0.02898549, 0.03756106],
                [0.005, 0.12094234, 3.914475, 0.03089580, 0.04134046],
                [0.006, 0.13439002, 4.127162, 0.03256184, 0.04464409],
            ]
        )
        np.testing.assert_allclose(
            result.iloc[:6].to_numpy(),
            expected,
            rtol=0,
            atol=5e-7,
        )
        self.assertAlmostEqual(result.attrs["shape_factor"], 0.2)

    def test_matches_documented_r_output_with_max_depth(self) -> None:
        result = avg_hydraulics(
            slope=0.01,
            bankfull_width=10,
            bankfull_depth=0.5,
            max_bankfull_depth=0.75,
            d84_mm=100,
        )
        expected = np.array(
            [
                [0.001, 0.03354854, 1.330329, 0.02521728, 0.02980415],
                [0.002, 0.05007830, 1.626481, 0.03078886, 0.03993549],
                [0.003, 0.06321439, 1.827366, 0.03459238, 0.04745422],
                [0.004, 0.07463971, 1.985960, 0.03758297, 0.05358751],
                [0.005, 0.08468275, 2.114073, 0.04005611, 0.05904123],
                [0.006, 0.09419307, 2.230608, 0.04222676, 0.06369544],
            ]
        )
        np.testing.assert_allclose(
            result.iloc[:6].to_numpy(),
            expected,
            rtol=0,
            atol=5e-7,
        )

    def test_custom_discharge_values_are_preserved(self) -> None:
        result = avg_hydraulics(
            0.01,
            10,
            0.5,
            100,
            discharges=[0.0025, 0.0045],
        )
        np.testing.assert_array_equal(result["Q"], [0.0025, 0.0045])

    def test_rejects_unrealistic_shape_factor(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds 0.7"):
            avg_hydraulics(0.01, 10, 0.5, 100, shape_factor=0.8)


if __name__ == "__main__":
    unittest.main()

