from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from gift_habitat import group_grain_sizes, load_example_curve, model_reaches


class BatchTests(unittest.TestCase):
    def test_reports_capped_shape_factors_by_comid_and_preserves_rows(self) -> None:
        reaches = pd.DataFrame({
            "COMID": [23428532, 23428532, 42, 43],
            "slope": [0.01] * 4,
            "width_m": [10.0] * 4,
            "depth_m": [0.5, 0.5, 0.5, np.nan],
            "d84_mm": [100.0] * 4,
            "factor": [0.8, 0.75, 0.7, 0.9],
        })
        depth = load_example_curve("depth", species="rainbow", life_stage="parr")
        velocity = load_example_curve("velocity", species="rainbow", life_stage="parr")
        options = dict(
            slope_col="slope", width_col="width_m", depth_col="depth_m",
            d84_col="d84_mm", id_col="COMID", shape_factor_col="factor",
        )
        result = model_reaches(reaches, depth, velocity, **options)
        self.assertEqual(result["COMID"].tolist(), reaches["COMID"].tolist())
        self.assertEqual(result.attrs["shape_factor_exceedance_count"], 2)
        self.assertEqual(result.attrs["shape_factor_exceedances"], [
            (23428532, 0.8), (23428532, 0.75),
        ])
        np.testing.assert_allclose(
            result["GIFT_shape_factor_raw"].iloc[:3], [0.8, 0.75, 0.7],
        )
        np.testing.assert_allclose(
            result["GIFT_shape_factor_used"].iloc[:3], [0.7, 0.7, 0.7],
        )
        self.assertTrue(np.isnan(result["GIFT_shape_factor_used"].iloc[3]))
        self.assertTrue(np.isfinite(result["WUA_mean"].iloc[:3]).all())

        curves = model_reaches(
            reaches.iloc[:2], depth, velocity,
            output="curves", discharges=[0.001], **options,
        )
        self.assertEqual(curves.attrs["shape_factor_exceedance_count"], 2)

    def test_reach_specific_substrate_scales_full_and_selected_flow_wua(self) -> None:
        reaches = pd.DataFrame({
            "COMID": [101, 102], "slope": [0.01, 0.01],
            "width_m": [10.0, 10.0], "depth_m": [0.5, 0.5],
            "d84_mm": [100.0, 100.0], "flow_m3s": [0.05, 0.05],
        })
        depth = load_example_curve("depth", species="rainbow", life_stage="parr")
        velocity = load_example_curve("velocity", species="rainbow", life_stage="parr")
        samples = pd.DataFrame({
            "COMID": [101] * 10 + [102] * 10,
            "grain_size_mm": [5.0] * 10 + [50.0] * 10,
        })
        curve = pd.DataFrame({
            "lower": [0.0, 10.0], "upper": [10.0, 100.0],
            "suit": [0.25, 0.75],
        })
        options = dict(
            slope_col="slope", width_col="width_m", depth_col="depth_m",
            d84_col="d84_mm", id_col="COMID", flow_cols=["flow_m3s"],
            flow_units="m3/s",
        )
        baseline = model_reaches(reaches, depth, velocity, **options)
        with_substrate = model_reaches(
            reaches, depth, velocity,
            substrate_curve=curve,
            gsd_by_reach=group_grain_sizes(samples, id_col="COMID"),
            **options,
        )
        np.testing.assert_allclose(with_substrate["s.suit"], [0.25, 0.75])
        for field in ("WUA_mean", "WUA_max", "WUA by flowrate"):
            np.testing.assert_allclose(
                with_substrate[field],
                baseline[field] * with_substrate["s.suit"],
                rtol=1e-12,
            )

        native = model_reaches(
            reaches, depth, velocity, slope_col="slope", width_col="width_m",
            depth_col="depth_m", d84_col="d84_mm", id_col="COMID",
            output="curves", discharges=[0.05], substrate_curve=curve,
            gsd_by_reach=group_grain_sizes(samples, id_col="COMID"),
        )
        np.testing.assert_allclose(native["s.suit"], [0.25, 0.75])

    def test_missing_grain_sizes_and_unpaired_inputs_fail(self) -> None:
        reaches = pd.DataFrame({
            "COMID": [101, 102], "slope": [0.01, 0.01],
            "width_m": [10.0, 10.0], "depth_m": [0.5, 0.5],
            "d84_mm": [100.0, 100.0],
        })
        depth = load_example_curve("depth", species="rainbow", life_stage="parr")
        velocity = load_example_curve("velocity", species="rainbow", life_stage="parr")
        curve = pd.DataFrame({"lower": [0], "upper": [1000], "suit": [0.5]})
        options = dict(
            slope_col="slope", width_col="width_m", depth_col="depth_m",
            d84_col="d84_mm", id_col="COMID",
        )
        with self.assertRaisesRegex(ValueError, "requires substrate_curve"):
            model_reaches(reaches, depth, velocity, gsd=[100] * 10, **options)
        with self.assertRaisesRegex(ValueError, "Reach 102.*no grain-size"):
            model_reaches(
                reaches, depth, velocity, substrate_curve=curve,
                gsd_by_reach={101: [100] * 10}, **options,
            )

    def test_substrate_curve_uses_each_reachs_d84_without_a_gsd(self) -> None:
        reaches = pd.DataFrame({
            "COMID": [101, 102], "slope": [0.01, 0.01],
            "width_m": [10.0, 10.0], "depth_m": [0.5, 0.5],
            "d84_mm": [10.0, 100.0], "flow_m3s": [0.05, 0.05],
        })
        depth = load_example_curve("depth", species="rainbow", life_stage="parr")
        velocity = load_example_curve("velocity", species="rainbow", life_stage="parr")
        curve = pd.DataFrame({
            "lower": [0, 50], "upper": [50, 256], "suit": [0.25, 0.75],
        })
        options = dict(
            slope_col="slope", width_col="width_m", depth_col="depth_m",
            d84_col="d84_mm", id_col="COMID", flow_cols=["flow_m3s"],
            flow_units="m3/s",
        )
        baseline = model_reaches(reaches, depth, velocity, **options)
        result = model_reaches(
            reaches, depth, velocity, substrate_curve=curve, **options,
        )
        np.testing.assert_allclose(result["s.suit"], [0.25, 0.75])
        for field in ("WUA_mean", "WUA_max", "WUA by flowrate"):
            np.testing.assert_allclose(
                result[field], baseline[field] * result["s.suit"], rtol=1e-12,
            )
        with self.assertRaisesRegex(ValueError, "finite, nonnegative"):
            group_grain_sizes(
                pd.DataFrame({"COMID": [101], "grain_size_mm": [-1]}),
                id_col="COMID",
            )

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
            output="curves",
            discharge_col="flow_m3s",
        )
        self.assertEqual(result["reach_id"].tolist(), [101, 102])
        self.assertEqual(result["Q"].tolist(), [0.01, 0.02])
        self.assertTrue((result["WUA"] > 0).all())


if __name__ == "__main__":
    unittest.main()
