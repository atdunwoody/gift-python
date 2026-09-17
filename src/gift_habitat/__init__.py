"""Python implementation of the Geomorphic Instream Flow Tool."""

from .batch import group_grain_sizes, integrate_wua_curve, model_reaches
from .compat import AvgHydraulics, Habitat
from .curves import available_example_curves, load_example_curve
from .habitat import habitat, substrate_suitability, substrate_suitability_at_size
from .hydraulics import avg_hydraulics

__all__ = [
    "AvgHydraulics",
    "Habitat",
    "available_example_curves",
    "avg_hydraulics",
    "habitat",
    "group_grain_sizes",
    "integrate_wua_curve",
    "load_example_curve",
    "model_reaches",
    "substrate_suitability",
    "substrate_suitability_at_size",
]

__version__ = "0.2.0"

