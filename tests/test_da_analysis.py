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

    def test_width_adjustment_aligns_duplicate_ids_by_position(self):
        reaches = pd.DataFrame({
            analysis.ID_FIELD: [101, 101],
            analysis.WIDTH_FIELD: [2.0, 4.0],
            analysis.DEPTH_FIELD: [0.2, 0.4],
            analysis.SLOPE_FIELD: [0.02, 0.01],
            analysis.SUMMER_WIDTH_FIELD: [1.0, 0.0],
        })
        modeled = pd.DataFrame({
            "group": ["juvenile", "juvenile", "spawning", "spawning"],
            analysis.ID_FIELD: [101] * 4,
            "WUA_mean": [2.0, 8.0, 4.0, 16.0],
        })
        result = analysis.add_driver_columns(modeled, reaches,
                                             ["juvenile", "spawning"])
        np.testing.assert_allclose(result["WUA_per_bankfull_width"],
                                   [1.0, 2.0, 2.0, 4.0])
        np.testing.assert_allclose(result["WUA_per_summer_width"].iloc[[0, 2]],
                                   [2.0, 4.0])
        self.assertTrue(result["WUA_per_summer_width"].iloc[[1, 3]].isna().all())
        with self.assertRaisesRegex(ValueError, "do not align"):
            analysis.add_driver_columns(modeled.iloc[:-1], reaches,
                                        ["juvenile", "spawning"])

    def test_general_salmonid_curves_have_expected_sources_and_boundaries(self):
        curves = analysis.general_salmonid_curves()
        self.assertIn(analysis.GENERAL_GROUP, analysis.curve_groups(analysis.CURVES_DIR))
        self.assertEqual(len(curves["depth"]), 41)
        self.assertEqual(len(curves["velocity"]), 41)
        for kind in ("depth", "velocity"):
            source = pd.read_csv(
                analysis.ROOT / f"src/gift_habitat/data/{kind}_suit_ptolemy.csv"
            )
            expected = source.loc[
                source["species"].eq("salmonid")
                & source["life_stage"].eq("general"), [kind, "suit"],
            ].reset_index(drop=True)
            pd.testing.assert_frame_equal(curves[kind], expected)
        self.assertAlmostEqual(
            curves["depth"].loc[curves["depth"]["suit"].idxmax(), "depth"],
            0.65,
        )
        self.assertAlmostEqual(
            curves["velocity"].loc[curves["velocity"]["suit"].idxmax(), "velocity"],
            0.30,
        )
        substrate = curves["substrate"]
        np.testing.assert_allclose(substrate["lower"].iloc[1:],
                                   substrate["upper"].iloc[:-1])
        for grain_size, score in ((5.0, 0.1), (30.0, 0.3),
                                  (100.0, 0.5), (200.0, 0.7)):
            matched = substrate.loc[
                (substrate["lower"] <= grain_size)
                & (grain_size < substrate["upper"])
            ]
            self.assertEqual(len(matched), 1)
            self.assertAlmostEqual(float(matched["suit"].iloc[0]), score)


if __name__ == "__main__":
    unittest.main()
