from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd

from gift_habitat import (
    avg_hydraulics, habitat, integrate_wua_curve, load_example_curve, model_reaches,
)
from gift_habitat.cli import main as cli_main

try:
    import geopandas as gpd
    from shapely.geometry import LineString
except ImportError:
    gpd = None


class NetworkMetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.depth_curve = load_example_curve("depth", species="rainbow", life_stage="parr")
        cls.velocity_curve = load_example_curve("velocity", species="rainbow", life_stage="parr")
        cls.options = dict(
            slope_col="slope", width_col="width_m", depth_col="depth_m",
            d84_col="d84_mm", id_col="segment_uid",
        )

    def setUp(self) -> None:
        self.reaches = pd.DataFrame({
            "segment_uid": [101, 101],  # Repeated IDs must not duplicate output features.
            "slope": [0.01, 0.012],
            "width_m": [10.0, 10.0],
            "depth_m": [0.5, 0.5],
            "d84_mm": [100.0, 100.0],
            "late_summer_wetted_width_pred_m": [5.0, 20.0],
            "label": ["upstream", "downstream"],
            "Q_low": [0.1, 0.2],
            "Q_peak": [1.0, 1.0],
            "Q_high": [6.0, 6.0],
        }, index=[7, 7])

    def model(self, reaches=None, **kwargs):
        return model_reaches(
            self.reaches if reaches is None else reaches,
            self.depth_curve, self.velocity_curve, **self.options, **kwargs,
        )

    def expected_wua(self, discharges):
        hydraulics = avg_hydraulics(0.01, 10.0, 0.5, 100.0, discharges=discharges)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return habitat(hydraulics, self.depth_curve, self.velocity_curve)["WUA"].to_numpy()

    def test_exact_area_of_unsorted_curve_with_unequal_flow_intervals(self):
        # Three trapezoids have areas 2, 4, and 4, respectively.
        curve = pd.DataFrame({"Q": [4.0, 0.0, 1.0, 2.0], "WUA": [0.0, 0.0, 4.0, 4.0]})
        self.assertAlmostEqual(integrate_wua_curve(curve), 10.0)

    def test_integration_rejects_incomplete_or_ambiguous_curves(self):
        for q, wua in [([1.0], [3.0]), ([1.0, 1.0], [2.0, 3.0]),
                       ([1.0, 2.0], [2.0, np.nan])]:
            with self.subTest(q=q, wua=wua), self.assertRaises(ValueError):
                integrate_wua_curve(pd.DataFrame({"Q": q, "WUA": wua}))

    def test_summary_reports_mean_max_and_full_discharge_range(self):
        result = self.model()
        pd.testing.assert_frame_equal(result[self.reaches.columns], self.reaches)
        self.assertNotIn("WUA by flowrate", result.columns)
        full = avg_hydraulics(0.01, 10.0, 0.5, 100.0, full_curve=True)
        legacy = avg_hydraulics(0.01, 10.0, 0.5, 100.0)
        self.assertEqual(result.iloc[0]["WUA_Q_min_m3s"], full["Q"].min())
        self.assertEqual(result.iloc[0]["WUA_Q_max_m3s"], full.attrs["simulated_bankfull_discharge_m3s"])
        self.assertLess(legacy["Q"].max(), result.iloc[0]["WUA_Q_max_m3s"])
        self.assertEqual(result.iloc[0]["WUA_Q_count"], 981)
        curve = habitat(full, self.depth_curve, self.velocity_curve)
        hsi = curve["d.suit"] * curve["v.suit"] * curve["s.suit"]
        self.assertAlmostEqual(result.iloc[0]["WUA_mean"], curve["WUA"].mean())
        self.assertAlmostEqual(result.iloc[0]["WUA_max"], curve["WUA"].max())
        self.assertAlmostEqual(
            result.iloc[0]["HSI_mean"], hsi.mean()
        )
        self.assertAlmostEqual(
            result.iloc[0]["HSI_max"], hsi.max()
        )
        self.assertAlmostEqual(result.iloc[0]["HSI_cubic_root_mean"], np.cbrt(hsi).mean())
        self.assertAlmostEqual(result.iloc[0]["HSI_cubic_root_max"], np.cbrt(hsi).max())
        self.assertAlmostEqual(
            result.iloc[0][[
                "HSI_depth_contribution_mean", "HSI_velocity_contribution_mean",
                "HSI_substrate_contribution_mean",
            ]].sum(), result.iloc[0]["HSI_mean"] - 1,
        )
        self.assertNotIn("WUA_dimensionless_mean", result.columns)
        self.assertAlmostEqual(result.iloc[0]["d.suit_mean"], curve["d.suit"].mean())
        self.assertAlmostEqual(result.iloc[0]["v.suit_mean"], curve["v.suit"].mean())
        max_rows = curve.loc[curve["WUA"] == curve["WUA"].max()]
        max_index = int(max_rows.index[0])
        self.assertAlmostEqual(result.iloc[0]["GIFT_depth_mean_m"], full["di"].mean())
        self.assertAlmostEqual(result.iloc[0]["GIFT_velocity_mean_mps"], full["Ui"].mean())
        self.assertAlmostEqual(result.iloc[0]["GIFT_depth_at_WUA_max_m"], full.iloc[max_index]["di"])
        self.assertAlmostEqual(result.iloc[0]["GIFT_velocity_at_WUA_max_mps"], full.iloc[max_index]["Ui"])
        self.assertAlmostEqual(
            result.iloc[0]["WUA_Q_at_max_m3s"], max_rows["Q"].iloc[0]
        )
        self.assertFalse(np.isclose(result.iloc[0]["WUA_mean"], result.iloc[1]["WUA_mean"]))
        self.assertNotIn("WUA_auc", result.columns)

    def test_normalized_wua_uses_requested_reach_width(self):
        result = self.model(normalize_width_col="late_summer_wetted_width_pred_m")
        np.testing.assert_allclose(
            result["WUA_mean_normalized"],
            result["WUA_mean"] / self.reaches["late_summer_wetted_width_pred_m"],
        )
        np.testing.assert_allclose(
            result["WUA_max_normalized"],
            result["WUA_max"] / self.reaches["late_summer_wetted_width_pred_m"],
        )
        self.assertNotIn("WUA_normalized", result.columns)

    def test_normalized_biological_flow_metric_uses_same_width(self):
        result = self.model(
            normalize_width_col="late_summer_wetted_width_pred_m",
            flow_cols=["Q_low", "Q_peak"],
            flow_units="m3/s",
        )
        np.testing.assert_allclose(
            result["WUA by flowrate normalized"],
            result["WUA by flowrate"] / self.reaches["late_summer_wetted_width_pred_m"],
        )

    def test_invalid_normalization_width_only_nulls_normalized_metrics(self):
        reaches = pd.concat([self.reaches.iloc[:1]] * 3, ignore_index=True)
        reaches["late_summer_wetted_width_pred_m"] = [np.nan, 0.0, -1.0]
        with self.assertWarnsRegex(UserWarning, "normalization widths"):
            result = self.model(
                reaches, normalize_width_col="late_summer_wetted_width_pred_m"
            )
        self.assertTrue(result["WUA_mean"].notna().all())
        self.assertTrue(result["WUA_max"].notna().all())
        self.assertTrue(result["HSI_mean"].notna().all())
        self.assertTrue(result["HSI_max"].notna().all())
        self.assertTrue(result["HSI_cubic_root_mean"].notna().all())
        self.assertTrue(result["HSI_cubic_root_max"].notna().all())
        self.assertTrue(result["WUA_mean_normalized"].isna().all())
        self.assertTrue(result["WUA_max_normalized"].isna().all())

    def test_curve_output_can_include_normalized_wua(self):
        reaches = self.reaches.iloc[:1].copy()
        curves = self.model(
            reaches,
            output="curves",
            discharges=[0.1, 1.0],
            normalize_width_col="late_summer_wetted_width_pred_m",
        )
        np.testing.assert_allclose(
            curves["WUA_normalized"],
            curves["WUA"] / reaches.iloc[0]["late_summer_wetted_width_pred_m"],
        )
        hydraulics = avg_hydraulics(0.01, 10.0, 0.5, 100.0, discharges=[0.1, 1.0])
        np.testing.assert_allclose(curves["depth_m"], hydraulics["di"])
        np.testing.assert_allclose(curves["velocity_mps"], hydraulics["Ui"])
        np.testing.assert_allclose(
            curves["HSI"], curves["d.suit"] * curves["v.suit"] * curves["s.suit"],
        )
        np.testing.assert_allclose(curves["HSI_cubic_root"], np.cbrt(curves["HSI"]))

    def test_normalization_field_must_exist(self):
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            self.model(normalize_width_col="missing_width")

    def test_small_reach_uses_full_range_even_below_default_grid(self):
        tiny = self.reaches.iloc[:1].copy()
        tiny["width_m"] = 0.1
        tiny["depth_m"] = 0.01
        tiny["slope"] = 0.001
        result = self.model(tiny)
        self.assertLess(result.iloc[0]["WUA_Q_max_m3s"], 0.001)
        self.assertTrue(np.isfinite(result.iloc[0]["WUA_mean"]))
        self.assertTrue(np.isfinite(result.iloc[0]["WUA_max"]))
        self.assertEqual(result.iloc[0]["WUA_Q_count"], 981)

    def test_median_is_of_wua_values_not_wua_at_median_flow(self):
        fields = ["Q_low", "Q_peak", "Q_high"]
        result = self.model(flow_cols=fields, flow_units="m3/s")
        wua = self.expected_wua([0.1, 1.0, 6.0])
        self.assertAlmostEqual(result.iloc[0]["WUA by flowrate"], np.median(wua))
        self.assertFalse(np.isclose(result.iloc[0]["WUA by flowrate"], wua[1]))
        self.assertEqual(json.loads(result.iloc[0]["WUA flow fields"]), fields)
        np.testing.assert_allclose(result["WUA_mean"], self.model()["WUA_mean"])
        np.testing.assert_allclose(result["WUA_max"], self.model()["WUA_max"])

    def test_cfs_and_cubic_meters_per_second_produce_equivalent_metrics(self):
        fields = ["Q_low", "Q_peak", "Q_high"]
        cfs = self.reaches.copy()
        # International foot = exactly 0.3048 meters.
        cfs[fields] = cfs[fields] / (0.3048 ** 3)
        si_result = self.model(flow_cols=fields, flow_units="m3/s")
        cfs_result = self.model(cfs, flow_cols=fields, flow_units="cfs")
        np.testing.assert_allclose(si_result["WUA by flowrate"], cfs_result["WUA by flowrate"], rtol=1e-12)
        np.testing.assert_array_equal(si_result["WUA_mean"], cfs_result["WUA_mean"])
        np.testing.assert_array_equal(si_result["WUA_max"], cfs_result["WUA_max"])
        self.assertEqual(cfs_result.iloc[0]["WUA flow units"], "cfs")

    def test_one_even_and_repeated_flow_values(self):
        reaches = self.reaches.iloc[:1].copy()
        reaches["Q_repeat"] = reaches["Q_low"]
        for fields, q in [(["Q_low"], [0.1]),
                          (["Q_low", "Q_peak"], [0.1, 1.0]),
                          (["Q_low", "Q_repeat", "Q_peak"], [0.1, 0.1, 1.0])]:
            with self.subTest(fields=fields):
                result = self.model(reaches, flow_cols=fields, flow_units="m3/s")
                self.assertAlmostEqual(result.iloc[0]["WUA by flowrate"], np.median(self.expected_wua(q)))
                self.assertEqual(json.loads(result.iloc[0]["WUA flow fields"]), fields)

    def test_zero_missing_invalid_and_out_of_range_flows_are_accounted_for(self):
        reaches = self.reaches.iloc[:1].copy()
        reaches["dry"] = 0.0
        reaches["missing"] = np.nan
        reaches["negative"] = -1.0
        reaches["text"] = "bad"
        reaches["infinite"] = np.inf
        reaches["too_low"] = 1e-12
        reaches["too_high"] = 1e10
        fields = ["Q_low", "dry", "missing", "negative", "text", "infinite", "too_low", "too_high"]
        with self.assertWarnsRegex(UserWarning, "excluded one or more selected flows"):
            result = self.model(reaches, flow_cols=fields, flow_units="m3/s")
        self.assertAlmostEqual(result.iloc[0]["WUA by flowrate"], self.expected_wua([0.1])[0] / 2.0)
        self.assertEqual(json.loads(result.iloc[0]["WUA flow fields"]), ["Q_low", "dry"])
        self.assertEqual(json.loads(result.iloc[0]["WUA flow fields excluded"]), {
            "missing": "missing", "negative": "invalid flow", "text": "not numeric",
            "infinite": "invalid flow", "too_low": "below simulated range",
            "too_high": "above simulated range",
        })
        with self.assertWarns(UserWarning):
            empty = self.model(reaches, flow_cols=fields[2:], flow_units="m3/s")
        self.assertTrue(np.isnan(empty.iloc[0]["WUA by flowrate"]))
        self.assertEqual(empty.iloc[0]["WUA flow fields"], "[]")
        dry = self.model(reaches, flow_cols=["dry"], flow_units="m3/s")
        self.assertEqual(dry.iloc[0]["WUA by flowrate"], 0.0)

    def test_flow_options_validate_units_fields_and_modes(self):
        for kwargs in [dict(flow_cols=["Q_low"]),
                       dict(flow_cols=["Q_low"], flow_units="gpm"),
                       dict(flow_cols=["absent"], flow_units="m3/s"),
                       dict(flow_cols=["Q_low", "Q_low"], flow_units="m3/s"),
                       dict(flow_units="cfs"),
                       dict(discharges=[0.1]),
                       dict(flow_cols=["Q_low"], flow_units="m3/s", output="curves")]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.model(**kwargs)
        with self.assertRaises(TypeError):
            self.model(flow_cols="Q_low", flow_units="m3/s")

    def test_empty_network_has_summary_schema(self):
        result = self.model(self.reaches.iloc[:0], flow_cols=["Q_low"], flow_units="cfs")
        self.assertEqual(len(result), 0)
        self.assertIn("WUA_mean", result.columns)
        self.assertIn("WUA_max", result.columns)
        self.assertIn("HSI_mean", result.columns)
        self.assertIn("HSI_cubic_root_mean", result.columns)
        self.assertIn("WUA_Q_at_max_m3s", result.columns)
        self.assertIn("GIFT_depth_mean_m", result.columns)
        self.assertNotIn("WUA_auc", result.columns)
        self.assertIn("WUA by flowrate", result.columns)

    def test_missing_hydraulic_inputs_keep_features_and_null_all_model_metrics(self):
        reaches = pd.concat([self.reaches.iloc[:1]] * 4)
        reaches["width_m"] = pd.array([10.0, np.nan, pd.NA, np.inf], dtype="Float64")
        progress = []
        with self.assertWarnsRegex(UserWarning, "3 segment\\(s\\).*results are null"):
            result = self.model(
                reaches, flow_cols=["Q_low"], flow_units="m3/s",
                progress_callback=lambda done, total: progress.append((done, total)),
            )
        self.assertEqual(progress, [(1, 4), (2, 4), (3, 4), (4, 4)])
        self.assertEqual(len(result), len(reaches))
        self.assertEqual(result.index.tolist(), reaches.index.tolist())
        self.assertTrue(pd.notna(result.iloc[0]["WUA_mean"]))
        self.assertTrue(pd.notna(result.iloc[0]["WUA_max"]))
        for column in (
            "WUA_mean", "WUA_max", "HSI_mean", "HSI_max",
            "HSI_cubic_root_mean", "HSI_cubic_root_max",
            "HSI_depth_contribution_mean", "HSI_velocity_contribution_mean",
            "HSI_substrate_contribution_mean", "WUA_Q_min_m3s", "WUA_Q_max_m3s",
            "WUA_Q_at_max_m3s", "WUA_Q_count",
            "GIFT_depth_mean_m", "GIFT_velocity_mean_mps",
            "GIFT_depth_at_WUA_max_m", "GIFT_velocity_at_WUA_max_mps",
            "d.suit_mean", "v.suit_mean", "s.suit", "WUA by flowrate",
            "WUA flow fields", "WUA flow units",
            "WUA flow fields excluded",
        ):
            self.assertTrue(result.iloc[1:][column].isna().all(), column)

    def test_missing_selected_discharge_keeps_null_curve_row(self):
        reaches = self.reaches.copy()
        reaches.iloc[1, reaches.columns.get_loc("Q_low")] = np.nan
        progress = []
        with self.assertWarnsRegex(UserWarning, "1 segment\\(s\\).*results are null"):
            result = self.model(
                reaches, output="curves", discharge_col="Q_low",
                progress_callback=lambda done, total: progress.append((done, total)),
            )
        self.assertEqual(progress, [(1, 2), (2, 2)])
        self.assertEqual(result["reach_id"].tolist(), [101, 101])
        self.assertTrue(result.iloc[1].drop("reach_id").isna().all())
        self.assertTrue(pd.notna(result.iloc[0]["WUA"]))

    def test_invalid_finite_hydraulic_values_still_raise(self):
        reaches = self.reaches.iloc[:1].copy()
        reaches["width_m"] = -1.0
        with self.assertRaisesRegex(ValueError, "bankfull_width must be greater than zero"):
            self.model(reaches)

    @unittest.skipIf(gpd is None, "Install the network extra for GeoPackage checks")
    def test_cli_writes_summary_and_geopackage_with_exact_field_names(self):
        streams = gpd.GeoDataFrame(
            self.reaches,
            geometry=[LineString([(0, 0), (10, 10)]), LineString([(10, 10), (20, 20)])],
            crs="EPSG:32610",
        )
        in_memory = self.model(streams, flow_cols=["Q_low", "Q_peak", "Q_high"], flow_units="m3/s")
        self.assertIsInstance(in_memory, gpd.GeoDataFrame)
        self.assertEqual(in_memory.crs, streams.crs)
        self.assertTrue(in_memory.geometry.equals(streams.geometry))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "input.gpkg"
            summary = root / "summary.csv"
            curves = root / "curves.csv"
            streams.to_file(source, layer="streams", driver="GPKG", index=False)
            self.assertEqual(cli_main([
                str(source), str(summary), "--layer", "streams",
                "--width-col", "width_m", "--depth-col", "depth_m", "--d84-col", "d84_mm",
                "--flow-cols", "Q_low", "Q_peak", "Q_high", "--flow-units", "m3/s",
                "--curves-csv", str(curves),
            ]), 0)
            saved = gpd.read_file(summary.with_suffix(".gpkg"), layer="gift_wua")
            csv = pd.read_csv(summary)
            self.assertEqual(len(saved), 2)
            self.assertEqual(saved["segment_uid"].tolist(), [101, 101])
            self.assertEqual(saved.crs, streams.crs)
            self.assertEqual(saved.geometry.to_wkt().tolist(), streams.geometry.to_wkt().tolist())
            self.assertEqual(saved["label"].tolist(), streams["label"].tolist())
            for column in (
                "WUA by flowrate", "WUA_mean", "WUA_max", "WUA_Q_at_max_m3s",
                "d.suit_mean", "v.suit_mean", "GIFT_depth_mean_m",
                "GIFT_velocity_mean_mps", "GIFT_depth_at_WUA_max_m",
                "GIFT_velocity_at_WUA_max_mps",
            ):
                np.testing.assert_allclose(saved[column], in_memory[column])
                np.testing.assert_allclose(csv[column], in_memory[column])
            self.assertEqual(json.loads(saved.iloc[0]["WUA flow fields"]), ["Q_low", "Q_peak", "Q_high"])
            self.assertNotIn("geometry", csv.columns)
            self.assertEqual(len(pd.read_csv(curves)), 2 * 981)
            # Default mode still writes one full-curve metric per segment.
            summary_only = root / "summary_only.csv"
            self.assertEqual(cli_main([
                str(source), str(summary_only), "--layer", "streams",
                "--width-col", "width_m", "--depth-col", "depth_m", "--d84-col", "d84_mm",
            ]), 0)
            self.assertNotIn("WUA by flowrate", pd.read_csv(summary_only).columns)
            np.testing.assert_allclose(pd.read_csv(summary_only)["WUA_mean"], csv["WUA_mean"])
            np.testing.assert_allclose(pd.read_csv(summary_only)["WUA_max"], csv["WUA_max"])
            np.testing.assert_allclose(
                pd.read_csv(summary_only)["WUA_Q_at_max_m3s"], csv["WUA_Q_at_max_m3s"]
            )

    def test_cli_rejects_missing_units_and_input_overwrite_before_reading(self):
        with self.assertRaisesRegex(SystemExit, "requires --flow-units"):
            cli_main(["input.gpkg", "output.csv", "--flow-cols", "Q_low"])
        with self.assertRaisesRegex(SystemExit, "--grain-sizes requires --substrate-curve"):
            cli_main(["input.gpkg", "output.csv", "--grain-sizes", "samples.csv"])
        with self.assertRaisesRegex(SystemExit, "distinct output paths"):
            cli_main(["input.gpkg", "input.csv"])


if __name__ == "__main__":
    unittest.main()
