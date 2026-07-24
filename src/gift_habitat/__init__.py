"""Python implementation of the Geomorphic Instream Flow Tool."""

from .batch import model_reaches
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
    "load_example_curve",
    "model_reaches",
    "substrate_suitability",
]

__version__ = "0.1.0"

