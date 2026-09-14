"""Apply GIFT and attribute WUA metrics to individual stream segments."""

from __future__ import annotations

from collections.abc import Iterable
import json
import warnings

import numpy as np
import pandas as pd

from ._validation import numeric_frame
from .habitat import habitat
from .hydraulics import _approx_like_r, avg_hydraulics

CFS_TO_M3S = 0.028316846592
_SUMMARY_COLUMNS = ("WUA_auc", "WUA_Q_min_m3s", "WUA_Q_max_m3s", "WUA_Q_count")
_FLOW_COLUMNS = (
    "WUA by flowrate",
    "WUA flow fields",
    "WUA flow units",
    "WUA flow fields excluded",
)


def integrate_wua_curve(curve: pd.DataFrame) -> float:
    """Integrate WUA against Q (m3/s) using the trapezoidal rule.

    Uses the actual, potentially unequal discharge intervals. Returns
    (m2/m)*(m3/s), without dividing by the flow range or reach length.
    At least two distinct, finite discharge coordinates are required.
    """
    values = numeric_frame(curve, "curve", ("Q", "WUA"))
    if (values[["Q", "WUA"]] < 0).any(axis=None):
        raise ValueError("curve Q and WUA must be nonnegative")
    values = values.sort_values("Q", kind="stable")
    q = values["Q"].to_numpy(dtype=float)
    wua = values["WUA"].to_numpy(dtype=float)
    if q.size < 2 or np.any(np.diff(q) <= 0):
        raise ValueError("curve needs at least two distinct Q values and no duplicates")
    # Explicit trapezoids support both NumPy 1.x and 2.x.
    return float(np.sum(np.diff(q) * (wua[:-1] + wua[1:]) / 2.0))


def _flow_options(
    reaches: pd.DataFrame,
    flow_cols: Iterable[str] | None,
    flow_units: str | None,
) -> list[str]:
    if isinstance(flow_cols, (str, bytes)):
        raise TypeError("flow_cols must be a list of field names, not a string")
    fields = list(flow_cols) if flow_cols is not None else []
    if any(not isinstance(field, str) or not field for field in fields):
        raise ValueError("flow_cols must contain non-empty field names")
    if len(fields) != len(set(fields)):
        raise ValueError("flow_cols contains duplicate field names")
    if fields and flow_units not in ("cfs", "m3/s"):
        raise ValueError("Specify flow_units='cfs' or flow_units='m3/s' with flow_cols")
    if not fields and flow_units is not None:
        raise ValueError("flow_units requires at least one field in flow_cols")
    missing = sorted(set(fields) - set(reaches.columns))
    if missing:
        raise ValueError(f"reaches is missing flow columns: {missing}")
    return fields


def _habitat_at_flows(
    hydraulics: pd.DataFrame,
    depth_curve: pd.DataFrame,
    velocity_curve: pd.DataFrame,
    substrate_curve: pd.DataFrame | None,
    gsd: np.ndarray | None,
) -> pd.DataFrame:
    # Selected flows can intentionally contain fewer than three rows.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="hydraulics contains fewer than three rows",
            category=UserWarning,
        )
        return habitat(
            hydraulics, depth_curve, velocity_curve,
            substrate_curve=substrate_curve, gsd=gsd,
        )


def _flow_summary(
    reach: pd.Series,
    fields: list[str],
    units: str,
    hydraulics: pd.DataFrame,
    depth_curve: pd.DataFrame,
    velocity_curve: pd.DataFrame,
    substrate_curve: pd.DataFrame | None,
    gsd: np.ndarray | None,
) -> dict[str, object]:
    """Evaluate each usable field before taking an equally weighted median."""
    used: list[str] = []
    excluded: dict[str, str] = {}
    flows: list[float] = []
    q_min, q_max = hydraulics["Q"].min(), hydraulics["Q"].max()
    factor = CFS_TO_M3S if units == "cfs" else 1.0
    for field in fields:
        value = reach[field]
        if pd.isna(value):
            excluded[field] = "missing"
            continue
        try:
            q = float(value) * factor
        except (TypeError, ValueError, OverflowError):
            excluded[field] = "not numeric"
            continue
        if isinstance(value, (bool, np.bool_)) or not np.isfinite(q) or q < 0:
            excluded[field] = "invalid flow"
            continue
        # Zero flow is explicitly treated as a dry channel with zero WUA.
        # Small tolerances allow cfs round-trips at the simulation endpoints.
        if q > 0 and q < q_min:
            if np.isclose(q, q_min, rtol=1e-12, atol=0.0):
                q = float(q_min)
            else:
                excluded[field] = "below simulated range"
                continue
        if q > q_max:
            if np.isclose(q, q_max, rtol=1e-12, atol=0.0):
                q = float(q_max)
            else:
                excluded[field] = "above simulated range"
                continue
        used.append(field)
        flows.append(q)

    q_values = np.asarray(flows, dtype=float)
    wua_values = np.zeros(q_values.size, dtype=float)
    positive = q_values > 0
    if positive.any():
        selected = pd.DataFrame({"Q": q_values[positive]})
        for column in ("Ai", "Wi", "di", "Ui"):
            selected[column] = _approx_like_r(
                hydraulics["Q"].to_numpy(),
                hydraulics[column].to_numpy(),
                q_values[positive],
            )
        # Calculate habitat at each exact requested discharge, rather than
        # interpolating WUA or evaluating WUA at the median discharge.
        wua_values[positive] = _habitat_at_flows(
            selected, depth_curve, velocity_curve, substrate_curve, gsd,
        )["WUA"].to_numpy(dtype=float)

    return {
        "WUA by flowrate": float(np.median(wua_values)) if used else np.nan,
        "WUA flow fields": json.dumps(used, ensure_ascii=False),
        "WUA flow units": units,
        "WUA flow fields excluded": json.dumps(excluded, ensure_ascii=False),
    }


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
    flow_cols: Iterable[str] | None = None,
    flow_units: str | None = None,
    output: str = "summary",
    discharges: float | Iterable[float] | None = None,
    discharge_col: str | None = None,
    full_curve: bool = False,
    substrate_curve: pd.DataFrame | None = None,
    gsd: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Run GIFT for each input segment and return attributed segment metrics.

    By default, returns a copy of ``reaches`` with one row per input row,
    preserving attributes, row order, index, geometry and CRS. ``WUA_auc`` is
    the area under the full native WUA-discharge curve, from the lowest
    simulated positive flow through simulated bankfull. Bounds are saved in
    ``WUA_Q_min_m3s`` and ``WUA_Q_max_m3s``. No extrapolation or normalization
    is applied. The integral has units (m2/m)*(m3/s).

    Add ``flow_cols=[...]`` and explicitly set ``flow_units='cfs'`` or
    ``'m3/s'`` to also calculate ``WUA by flowrate``: the median of WUA values
    at the selected per-segment flows, in m2/m. Zero flow contributes zero
    WUA. Missing, invalid and unsupported positive flows are excluded and
    recorded with reasons; no usable flows gives NaN. ``WUA flow fields``
    records the fields actually included as a JSON list. Different fields
    with identical flows each contribute once. Field names must be unique.

    ``output='curves'`` retains the earlier long table behavior, with one row
    per reach and discharge. Only that mode accepts ``discharges`` or
    ``discharge_col`` (both in m3/s). Use ``full_curve=True`` in curves mode
    to export the native curve used for integration, or omit it to use the
    original GIFT discharge grid. Summary mode always uses the full curve.
    """
    if not isinstance(reaches, pd.DataFrame):
        raise TypeError("reaches must be a pandas DataFrame or GeoDataFrame")
    if not reaches.columns.is_unique:
        raise ValueError("reaches must have unique column names")
    if output not in ("summary", "curves"):
        raise ValueError("output must be 'summary' or 'curves'")
    if discharges is not None and discharge_col is not None:
        raise ValueError("Use either discharges or discharge_col, not both")
    selected_discharges = discharges is not None or discharge_col is not None
    if output == "summary" and selected_discharges:
        raise ValueError(
            "Summary mode always integrates the full curve. Use flow_cols and "
            "flow_units for biological flows, or output='curves' for "
            "discharges/discharge_col."
        )
    if full_curve and selected_discharges:
        raise ValueError("full_curve cannot be combined with selected discharges")
    fields = _flow_options(reaches, flow_cols, flow_units)
    if output == "curves" and fields:
        raise ValueError("flow_cols requires output='summary'")

    requested_columns = [slope_col, width_col, depth_col, d84_col]
    requested_columns.extend(
        name for name in (id_col, max_depth_col, shape_factor_col, discharge_col)
        if name is not None
    )
    missing = sorted(set(requested_columns) - set(reaches.columns))
    if missing:
        raise ValueError(f"reaches is missing required columns: {missing}")

    shared_discharges = discharges
    if discharges is not None and not np.isscalar(discharges):
        shared_discharges = np.asarray(list(discharges), dtype=float)
    # Materialize once so generator inputs work consistently across segments.
    shared_gsd = np.asarray(list(gsd), dtype=float) if gsd is not None else None
    outputs: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    excluded_reaches = 0
    for position, (index, reach) in enumerate(reaches.iterrows()):
        # Access the original column to preserve integer IDs without iterrows
        # coercion. Summary attribution is positional, never a join on IDs.
        reach_id = reaches[id_col].iloc[position] if id_col is not None else index
        reach_discharges = (
            float(reach[discharge_col]) if discharge_col is not None
            else shared_discharges
        )
        try:
            hydraulics = avg_hydraulics(
                slope=reach[slope_col],
                bankfull_width=reach[width_col],
                bankfull_depth=reach[depth_col],
                d84_mm=reach[d84_col],
                max_bankfull_depth=(
                    reach[max_depth_col] if max_depth_col is not None else None
                ),
                shape_factor=(
                    reach[shape_factor_col] if shape_factor_col is not None else None
                ),
                discharges=reach_discharges,
                full_curve=(output == "summary" or full_curve),
            )
        except (ValueError, TypeError, RuntimeError) as exc:
            raise ValueError(f"Reach {reach_id!r} (row {position}): {exc}") from exc
        if hydraulics.empty:
            raise ValueError(
                f"No requested discharge falls within the simulated range "
                f"for reach {reach_id!r}"
            )
        curve = _habitat_at_flows(
            hydraulics, depth_curve, velocity_curve, substrate_curve, shared_gsd,
        )
        if output == "curves":
            curve.insert(0, "reach_id", reach_id)
            outputs.append(curve)
            continue

        summary = {
            "WUA_auc": integrate_wua_curve(curve),
            "WUA_Q_min_m3s": float(curve["Q"].min()),
            "WUA_Q_max_m3s": float(curve["Q"].max()),
            "WUA_Q_count": len(curve),
        }
        if fields:
            summary.update(_flow_summary(
                reach, fields, flow_units, hydraulics,
                depth_curve, velocity_curve, substrate_curve, shared_gsd,
            ))
            excluded_reaches += summary["WUA flow fields excluded"] != "{}"
        summaries.append(summary)

    if output == "curves":
        if not outputs:
            return pd.DataFrame(columns=[
                "reach_id", "Q", "d.suit", "v.suit", "s.suit", "w", "WUA",
            ])
        return pd.concat(outputs, ignore_index=True)

    # Preserve repeated IDs and duplicate index labels without fan-out.
    result = reaches.copy()
    # Remove optional metrics from a prior run when the option is disabled.
    if not fields:
        result = result.drop(columns=list(_FLOW_COLUMNS), errors="ignore")
    for column in (*_SUMMARY_COLUMNS, *(_FLOW_COLUMNS if fields else ())):
        values = [summary[column] for summary in summaries]
        dtype = (
            str if column in _FLOW_COLUMNS[1:]
            else int if column == "WUA_Q_count" else float
        )
        result[column] = np.asarray(values, dtype=dtype)
    if excluded_reaches:
        warnings.warn(
            f"{excluded_reaches} segment(s) excluded one or more selected flows. "
            "WUA by flowrate uses the remaining fields; no usable fields gives "
            "NaN. Inspect 'WUA flow fields' and 'WUA flow fields excluded'.",
            UserWarning,
            stacklevel=2,
        )
    return result
