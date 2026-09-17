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

    def test_summary_preserves_rows_and_integrates_through_bankfull(self):
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
        integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        self.assertAlmostEqual(result.iloc[0]["WUA_auc"], integrate(curve["WUA"], curve["Q"]))
        self.assertFalse(np.isclose(result.iloc[0]["WUA_auc"], result.iloc[1]["WUA_auc"]))

    def test_small_reach_uses_full_range_even_below_default_grid(self):
        tiny = self.reaches.iloc[:1].copy()
        tiny["width_m"] = 0.1
        tiny["depth_m"] = 0.01
        tiny["slope"] = 0.001
        result = self.model(tiny)
        self.assertLess(result.iloc[0]["WUA_Q_max_m3s"], 0.001)
        self.assertTrue(np.isfinite(result.iloc[0]["WUA_auc"]))
        self.assertEqual(result.iloc[0]["WUA_Q_count"], 981)

    def test_median_is_of_wua_values_not_wua_at_median_flow(self):
        fields = ["Q_low", "Q_peak", "Q_high"]
        result = self.model(flow_cols=fields, flow_units="m3/s")
        wua = self.expected_wua([0.1, 1.0, 6.0])
        self.assertAlmostEqual(result.iloc[0]["WUA by flowrate"], np.median(wua))
        self.assertFalse(np.isclose(result.iloc[0]["WUA by flowrate"], wua[1]))
        self.assertEqual(json.loads(result.iloc[0]["WUA flow fields"]), fields)
        np.testing.assert_allclose(result["WUA_auc"], self.model()["WUA_auc"])

    def test_cfs_and_cubic_meters_per_second_produce_equivalent_metrics(self):
        fields = ["Q_low", "Q_peak", "Q_high"]
        cfs = self.reaches.copy()
        # International foot = exactly 0.3048 meters.
        cfs[fields] = cfs[fields] / (0.3048 ** 3)
        si_result = self.model(flow_cols=fields, flow_units="m3/s")
        cfs_result = self.model(cfs, flow_cols=fields, flow_units="cfs")
        np.testing.assert_allclose(si_result["WUA by flowrate"], cfs_result["WUA by flowrate"], rtol=1e-12)
        np.testing.assert_array_equal(si_result["WUA_auc"], cfs_result["WUA_auc"])
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
        self.assertIn("WUA_auc", result.columns)
        self.assertIn("WUA by flowrate", result.columns)

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
            for column in ("WUA by flowrate", "WUA_auc"):
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
            np.testing.assert_allclose(pd.read_csv(summary_only)["WUA_auc"], csv["WUA_auc"])

    def test_cli_rejects_missing_units_and_input_overwrite_before_reading(self):
        with self.assertRaisesRegex(SystemExit, "requires --flow-units"):
            cli_main(["input.gpkg", "output.csv", "--flow-cols", "Q_low"])
        with self.assertRaisesRegex(SystemExit, "--grain-sizes requires --substrate-curve"):
            cli_main(["input.gpkg", "output.csv", "--grain-sizes", "samples.csv"])
        with self.assertRaisesRegex(SystemExit, "distinct output paths"):
            cli_main(["input.gpkg", "input.csv"])


if __name__ == "__main__":
    unittest.main()
