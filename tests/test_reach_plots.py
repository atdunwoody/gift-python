"""Check reach labels and shared scales for selected discharge curves."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "examples/stream_network.py"
spec = importlib.util.spec_from_file_location("stream_network", SCRIPT)
example = importlib.util.module_from_spec(spec)
geopandas_stub = ModuleType("geopandas")
geopandas_stub.GeoDataFrame = pd.DataFrame
with patch.dict(sys.modules, {"geopandas": geopandas_stub}):
    spec.loader.exec_module(example)


class ReachPlotTests(unittest.TestCase):
    def test_reach_title_and_substrate_note(self):
        reach = pd.Series({
            "COMID": 101,
            "GNIS_NAME": "Sample Creek",
        })
        self.assertEqual(example._reach_plot_title(reach),
                         "Sample Creek (COMID: 101)")
        self.assertEqual(example._reach_plot_title(pd.Series({
            "COMID": 102, "GNIS_NAME": None,
        })), "Unnamed stream (COMID: 102)")
        curve = pd.DataFrame({"s.suit": [0.5, 0.7]})
        self.assertEqual(example._substrate_note(curve),
                         "Substrate suitability: 0.600")

    def test_primary_axes_share_limits_by_metric_across_reaches(self):
        curves = [
            pd.DataFrame({"WUA": [0, 2], "HSI": [0, 0.25]}),
            pd.DataFrame({"WUA": [0, 5], "HSI": [0, 0.10]}),
        ]
        limits = example._shared_primary_limits(curves)
        self.assertAlmostEqual(limits["WUA"], 5.4)
        self.assertAlmostEqual(limits["HSI"], 0.27)


if __name__ == "__main__":
    unittest.main()
