"""Convenience functions for applying GIFT to multiple stream reaches."""

from __future__ import annotations

from collections.abc import Iterable
import warnings

import numpy as np
import pandas as pd

from .habitat import habitat
from .hydraulics import avg_hydraulics


def model_reaches(
    reaches: pd.DataFrame,
    depth_curve: pd.DataFrame,
    velocity_curve: pd.DataFrame,
    *,
    slope_col: str,
    width_col: str,
    depth_col: str,
    d84_col: str,
    id_col: str | None = None,
    max_depth_col: str | None = None,
    shape_factor_col: str | None = None,
    discharges: float | Iterable[float] | None = None,
    discharge_col: str | None = None,
    substrate_curve: pd.DataFrame | None = None,
    gsd: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Run GIFT for every row in a DataFrame or GeoDataFrame.

    The result is a long table with one or more discharge rows per reach. If
    ``discharge_col`` is provided, each reach is evaluated only at the
    discharge stored in that row. Otherwise every reach uses ``discharges``
    or the original GIFT discharge grid when ``discharges`` is omitted.
    """
    if not isinstance(reaches, pd.DataFrame):
        raise TypeError("reaches must be a pandas DataFrame or GeoDataFrame")
    if discharges is not None and discharge_col is not None:
        raise ValueError("Use either discharges or discharge_col, not both")

    requested_columns = [
        slope_col,
        width_col,
        depth_col,
        d84_col,
    ]
    for optional_column in (
        id_col,
        max_depth_col,
        shape_factor_col,
        discharge_col,
    ):
        if optional_column is not None:
            requested_columns.append(optional_column)

    missing = sorted(set(requested_columns) - set(reaches.columns))
    if missing:
        raise ValueError(f"reaches is missing required columns: {missing}")

    shared_discharges: float | np.ndarray | None
    if discharges is None or np.isscalar(discharges):
        shared_discharges = discharges
    else:
        shared_discharges = np.asarray(list(discharges), dtype=float)

    outputs: list[pd.DataFrame] = []
    for index, reach in reaches.iterrows():
        reach_id = reach[id_col] if id_col is not None else index
        reach_discharges = (
            float(reach[discharge_col])
            if discharge_col is not None
            else shared_discharges
        )
        hydraulics = avg_hydraulics(
            slope=reach[slope_col],
            bankfull_width=reach[width_col],
            bankfull_depth=reach[depth_col],
            d84_mm=reach[d84_col],
            max_bankfull_depth=(
                reach[max_depth_col] if max_depth_col is not None else None
            ),
            shape_factor=(
                reach[shape_factor_col]
                if shape_factor_col is not None
                else None
            ),
            discharges=reach_discharges,
        )
        if hydraulics.empty:
            raise ValueError(
                f"No requested discharge falls within the simulated range "
                f"for reach {reach_id!r}"
            )

        # A one-flow-per-reach network is intentional when discharge_col is
        # used, so the original R warning about fewer than three hydraulic
        # rows would otherwise be emitted once for every feature.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="hydraulics contains fewer than three rows",
                category=UserWarning,
            )
            reach_result = habitat(
                hydraulics,
                depth_curve,
                velocity_curve,
                substrate_curve=substrate_curve,
                gsd=gsd,
            )
        reach_result.insert(0, "reach_id", reach_id)
        outputs.append(reach_result)

    if not outputs:
        return pd.DataFrame(
            columns=[
                "reach_id",
                "Q",
                "d.suit",
                "v.suit",
                "s.suit",
                "w",
                "WUA",
            ]
        )
    return pd.concat(outputs, ignore_index=True)
