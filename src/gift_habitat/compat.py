"""Thin wrappers using the original R function and parameter names."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .habitat import habitat
from .hydraulics import avg_hydraulics


def AvgHydraulics(
    S: float,
    wb: float,
    db: float,
    D84: float,
    db_max: float | None = None,
    b_value: float | None = None,
    xs_output: bool = True,
    output_dir: str | Path = ".",
) -> pd.DataFrame:
    """Compatibility wrapper for the R ``AvgHydraulics`` function."""
    return avg_hydraulics(
        slope=S,
        bankfull_width=wb,
        bankfull_depth=db,
        d84_mm=D84,
        max_bankfull_depth=db_max,
        shape_factor=b_value,
        output_dir=output_dir if xs_output else None,
    )


def Habitat(
    hydraulics: pd.DataFrame,
    d_curve: pd.DataFrame,
    v_curve: pd.DataFrame,
    s_curve: pd.DataFrame | None = None,
    gsd=None,
    wua_output: bool = True,
    output_dir: str | Path = ".",
) -> pd.DataFrame:
    """Compatibility wrapper for the R ``Habitat`` function."""
    return habitat(
        hydraulics,
        depth_curve=d_curve,
        velocity_curve=v_curve,
        substrate_curve=s_curve,
        gsd=gsd,
        output_dir=output_dir if wua_output else None,
    )

