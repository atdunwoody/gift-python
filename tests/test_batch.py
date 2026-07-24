from __future__ import annotations

import unittest

import pandas as pd

from gift_habitat import load_example_curve, model_reaches


class BatchTests(unittest.TestCase):
    def test_models_multiple_reaches_at_reach_specific_discharge(self) -> None:
        reaches = pd.DataFrame(
            {
                "segment_uid": [101, 102],
                "slope": [0.01, 0.015],
                "width_m": [10.0, 8.0],
                "depth_m": [0.5, 0.45],
                "d84_mm": [100.0, 80.0],
                "flow_m3s": [0.01, 0.02],
            }
        )
        result = model_reaches(
            reaches,
            load_example_curve(
                "depth",
                species="rainbow",
                life_stage="parr",
            ),
            load_example_curve(
                "velocity",
                species="rainbow",
                life_stage="parr",
            ),
            slope_col="slope",
            width_col="width_m",
            depth_col="depth_m",
            d84_col="d84_mm",
            id_col="segment_uid",
            discharge_col="flow_m3s",
        )
        self.assertEqual(result["reach_id"].tolist(), [101, 102])
        self.assertEqual(result["Q"].tolist(), [0.01, 0.02])
        self.assertTrue((result["WUA"] > 0).all())


if __name__ == "__main__":
    unittest.main()

