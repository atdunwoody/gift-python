"""Check group selection and statistical calculations for the DA comparison."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "examples/hsi_vs_drainage_area.py"
spec = importlib.util.spec_from_file_location("da_analysis", SCRIPT)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class DrainageAreaAnalysisTests(unittest.TestCase):
    def test_discovers_complete_groups_and_skips_generic(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for name in (
                "steelhead_spawning_depth.csv", "steelhead_spawning_velocity.csv",
                "steelhead_spawning_substrate.csv", "generic_salmon_spawning_depth.csv",
            ):
                (folder / name).touch()
            self.assertEqual(analysis.curve_groups(folder), ["steelhead_spawning"])
            (folder / "chinook_rearing_depth.csv").touch()
            with self.assertRaisesRegex(FileNotFoundError, "Missing velocity curve"):
                analysis.curve_groups(folder)

    def test_association_uses_all_values_including_zeros_and_adjusts_p(self):
        da = np.geomspace(1, 100, 12)
        frame = pd.DataFrame({
            "group": ["spawning"] * len(da), "DA_km2": da,
            "HSI_mean": np.arange(len(da), dtype=float) / 20,
            "HSI_max": np.arange(len(da) - 1, -1, -1, dtype=float) / 20,
            "WUA_mean": np.arange(len(da), dtype=float),
        })
        frame.loc[0, "HSI_mean"] = 0.0
        result = analysis.test_da_association(frame).set_index("metric")
        self.assertEqual(set(result["n"]), {len(da)})
        self.assertAlmostEqual(result.loc["HSI_mean", "spearman_rho"], 1.0)
        self.assertAlmostEqual(result.loc["HSI_max", "spearman_rho"], -1.0)
        self.assertAlmostEqual(result.loc["HSI_mean", "slope_per_10x_DA"], 0.275)
        self.assertTrue((result["spearman_p_BH"] <= 1).all())
        bins = analysis.da_bin_medians(frame)
        self.assertEqual(bins["n"].sum(), len(da))

    def test_component_trend_decomposition_sums_to_hsi_mean_slope(self):
        x = np.linspace(0, 2, 12)
        frame = pd.DataFrame({
            "group": "spawning", "DA_km2": 10 ** x,
            "HSI_depth_contribution_mean": -0.5 + 0.10 * x,
            "HSI_velocity_contribution_mean": -0.3 + 0.04 * x,
            "HSI_substrate_contribution_mean": -0.1 - 0.01 * x,
        })
        frame["HSI_mean"] = 1 + frame[list(analysis.COMPONENTS.values())].sum(axis=1)
        result = analysis.component_attribution(frame)
        self.assertAlmostEqual(result["slope_per_10x_DA"].sum(), 0.13)
        self.assertAlmostEqual(result["HSI_mean_slope_per_10x_DA"].iloc[0], 0.13)
        self.assertEqual(result.loc[result["largest_aligned_contributor"], "component"].iloc[0], "depth")
        self.assertAlmostEqual(result["share_of_HSI_mean_slope_pct"].sum(), 100)


if __name__ == "__main__":
    unittest.main()
