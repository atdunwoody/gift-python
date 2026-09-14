"""Python implementation of the Geomorphic Instream Flow Tool."""

from .batch import integrate_wua_curve, model_reaches
from .compat import AvgHydraulics, Habitat
from .curves import available_example_curves, load_example_curve
from .habitat import habitat, substrate_suitability
from .hydraulics import avg_hydraulics

__all__ = [
    "AvgHydraulics",
    "Habitat",
    "available_example_curves",
    "avg_hydraulics",
    "habitat",
    "integrate_wua_curve",
    "load_example_curve",
    "model_reaches",
    "substrate_suitability",
]

__version__ = "0.2.0"

