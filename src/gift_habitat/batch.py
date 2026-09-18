"""Apply GIFT and attribute WUA metrics to individual stream segments."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
import json
from numbers import Real
import warnings

import numpy as np
import pandas as pd

from ._validation import numeric_frame
from .habitat import habitat
from .hydraulics import _approx_like_r, avg_hydraulics

CFS_TO_M3S = 0.028316846592
_SUMMARY_COLUMNS = (
    "WUA_mean",
    "WUA_max",
    "WUA_dimensionless_mean",
    "WUA_dimensionless_max",
    "WUA_Q_min_m3s",
    "WUA_Q_max_m3s",
    "WUA_Q_at_max_m3s",
    "WUA_Q_count",
    "d.suit_mean",
    "v.suit_mean",
    "s.suit",
)
_NORMALIZED_SUMMARY_COLUMNS = (
    "WUA_mean_normalized",
    "WUA_max_normalized",
)
_LEGACY_SUMMARY_COLUMNS = ("WUA_auc",)
_FLOW_COLUMNS = (
    "WUA by flowrate",
    "WUA flow fields",
    "WUA flow units",
    "WUA flow fields excluded",
)
_NORMALIZED_FLOW_COLUMN = "WUA by flowrate normalized"


def _missing_model_input(value: object) -> bool:
    """Identify absent hydraulic inputs without hiding invalid finite values."""
    return bool(pd.isna(value)) or (
        isinstance(value, Real) and not np.isfinite(value)
    )
def group_grain_sizes(
    samples: pd.DataFrame,
    *,
    id_col: str,
    size_col: str = "grain_size_mm",
) -> dict[object, np.ndarray]:
    """Group individual grain-size observations (mm) by reach ID.

    Each row is one observation, as in the GIFT ``gsd`` argument. A D84
    percentile used for hydraulics does not replace these observations.
    """
    if not isinstance(samples, pd.DataFrame):
        raise TypeError("samples must be a pandas DataFrame")
    if not samples.columns.is_unique:
        raise ValueError("samples must have unique column names")
    missing = {id_col, size_col} - set(samples.columns)
    if missing:
        raise ValueError(f"samples is missing required columns: {sorted(missing)}")
    if samples.empty or samples[id_col].isna().any():
        raise ValueError(f"samples must contain non-missing {id_col} values")
    try:
        sizes = pd.to_numeric(samples[size_col], errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{size_col} must contain numeric grain sizes in mm") from exc
    if not np.isfinite(sizes).all() or (sizes < 0).any():
        raise ValueError(f"{size_col} must contain finite, nonnegative grain sizes in mm")
    values = samples[[id_col]].copy()
    values[size_col] = sizes
    return {
        reach_id: group[size_col].to_numpy(dtype=float)
        for reach_id, group in values.groupby(id_col, sort=False)
    }


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
    substrate_size_mm: float | None = None,
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
            substrate_size_mm=substrate_size_mm,
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
    substrate_size_mm: float | None,
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
            substrate_size_mm,
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
    normalize_width_col: str | None = None,
    flow_cols: Iterable[str] | None = None,
    flow_units: str | None = None,
    output: str = "summary",
    discharges: float | Iterable[float] | None = None,
    discharge_col: str | None = None,
    full_curve: bool = False,
    substrate_curve: pd.DataFrame | None = None,
    gsd: Iterable[float] | None = None,
    gsd_by_reach: Mapping[object, Iterable[float]] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Run GIFT for each input segment and return attributed segment metrics.

    By default, returns a copy of ``reaches`` with one row per input row,
    preserving attributes, row order, index, geometry and CRS. ``WUA_mean``
    and ``WUA_max`` summarize the full native WUA-discharge curve, from the
    lowest simulated positive flow through simulated bankfull. The modeled
    discharge range is saved in ``WUA_Q_min_m3s`` and ``WUA_Q_max_m3s``, and
    ``WUA_Q_at_max_m3s`` records the lowest modeled discharge at which the
    maximum WUA occurs. ``WUA_mean`` is the arithmetic mean of the native
    modeled WUA values and is not discharge-weighted.
    ``WUA_dimensionless_mean`` and ``WUA_dimensionless_max`` summarize
    ``d.suit * v.suit * s.suit`` over the same native discharge curve, without
    multiplying by wetted width. ``d.suit_mean`` and
    ``v.suit_mean`` are the arithmetic means of depth and velocity suitability
    across those same native discharge simulations. If
    ``normalize_width_col`` is supplied, ``WUA_mean_normalized`` and
    ``WUA_max_normalized`` divide those metrics by that reach-level wetted
    width in meters. These normalized values are dimensionless.

    Add ``flow_cols=[...]`` and explicitly set ``flow_units='cfs'`` or
    ``'m3/s'`` to also calculate ``WUA by flowrate``: the median of WUA values
    at the selected per-segment flows, in m2/m. Zero flow contributes zero
    WUA. Missing, invalid and unsupported positive flows are excluded and
    recorded with reasons; no usable flows gives NaN. ``WUA flow fields``
    records the fields actually included as a JSON list. When
    ``normalize_width_col`` is also supplied, ``WUA by flowrate normalized``
    divides the biological-flow WUA metric by the same reach-level width.
    Different fields with identical flows each contribute once. Field names
    must be unique.

    ``output='curves'`` retains the earlier long table behavior, with one row
    per reach and discharge. Only that mode accepts ``discharges`` or
    ``discharge_col`` (both in m3/s). Use ``full_curve=True`` in curves mode
    to export the native curve used for the summary statistics, or omit it
    to use the original GIFT discharge grid. Summary mode always uses the full
    native curve.

    With ``substrate_curve`` alone, use each reach's ``d84_col`` value as a
    representative substrate size and look up its class suitability. To use
    the original GIFT grain-size-distribution calculation instead, also supply
    one shared ``gsd`` or ``gsd_by_reach`` keyed by ``id_col``. Every reach
    must have observations in the latter case. The score is returned as
    ``s.suit`` in summary mode.
    Reaches with missing or non-finite values in mapped hydraulic inputs (or
    the selected discharge column) retain their input row and have null model
    metrics. In curves mode, each such reach gets one row with null curve
    values. ``progress_callback`` receives (processed_reaches, total_reaches)
    after each reach, including skipped reaches.
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
            "Summary mode always summarizes the full native curve. Use "
            "flow_cols and flow_units for biological flows, or output='curves' for "
            "discharges/discharge_col."
        )
    if full_curve and selected_discharges:
        raise ValueError("full_curve cannot be combined with selected discharges")
    fields = _flow_options(reaches, flow_cols, flow_units)
    if output == "curves" and fields:
        raise ValueError("flow_cols requires output='summary'")
    if gsd is not None and gsd_by_reach is not None:
        raise ValueError("Use either gsd or gsd_by_reach, not both")
    if substrate_curve is None and (gsd is not None or gsd_by_reach is not None):
        raise ValueError("gsd or gsd_by_reach requires substrate_curve")
    if gsd_by_reach is not None:
        if id_col is None:
            raise ValueError("gsd_by_reach requires id_col")
        if not isinstance(gsd_by_reach, Mapping):
            raise TypeError("gsd_by_reach must map reach IDs to grain-size observations")
        per_reach_gsd = {
            reach_id: np.asarray(list(sizes), dtype=float)
            for reach_id, sizes in gsd_by_reach.items()
        }
    else:
        per_reach_gsd = None

    requested_columns = [slope_col, width_col, depth_col, d84_col]
    requested_columns.extend(
        name for name in (
            id_col, max_depth_col, shape_factor_col, discharge_col,
            normalize_width_col,
        )
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
    skipped_reaches = 0
    invalid_normalization_widths = 0
    total_reaches = len(reaches)
    for position, (index, reach) in enumerate(reaches.iterrows()):
        # Access the original column to preserve integer IDs without iterrows
        # coercion. Summary attribution is positional, never a join on IDs.
        reach_id = reaches[id_col].iloc[position] if id_col is not None else index
        model_columns = (slope_col, width_col, depth_col, d84_col)
        optional_columns = (max_depth_col, shape_factor_col, discharge_col)
        if any(_missing_model_input(reach[column]) for column in (
            *model_columns, *(name for name in optional_columns if name is not None),
        )):
            skipped_reaches += 1
            if output == "curves":
                null_curve = {
                    "reach_id": reach_id, "Q": np.nan, "d.suit": np.nan,
                    "v.suit": np.nan, "s.suit": np.nan, "w": np.nan,
                    "WUA": np.nan,
                }
                if normalize_width_col is not None:
                    null_curve["WUA_normalized"] = np.nan
                outputs.append(pd.DataFrame([null_curve]))
            else:
                null_columns = [
                    *_SUMMARY_COLUMNS,
                    *(_NORMALIZED_SUMMARY_COLUMNS if normalize_width_col is not None else ()),
                    *(_FLOW_COLUMNS if fields else ()),
                ]
                if fields and normalize_width_col is not None:
                    null_columns.append(_NORMALIZED_FLOW_COLUMN)
                summaries.append({column: None for column in null_columns})
            if progress_callback is not None:
                progress_callback(position + 1, total_reaches)
            continue
        if per_reach_gsd is not None:
            if pd.isna(reach_id) or reach_id not in per_reach_gsd:
                raise ValueError(
                    f"Reach {reach_id} (row {position}) has no grain-size observations"
                )
            reach_gsd = per_reach_gsd[reach_id]
        else:
            reach_gsd = shared_gsd
        substrate_size_mm = (
            reach[d84_col]
            if substrate_curve is not None and reach_gsd is None
            else None
        )
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
            hydraulics, depth_curve, velocity_curve, substrate_curve, reach_gsd,
            substrate_size_mm,
        )
        if output == "curves":
            if normalize_width_col is not None:
                try:
                    normalization_width = float(reach[normalize_width_col])
                except (TypeError, ValueError):
                    normalization_width = np.nan
                if np.isfinite(normalization_width) and normalization_width > 0:
                    curve["WUA_normalized"] = curve["WUA"] / normalization_width
                else:
                    curve["WUA_normalized"] = np.nan
                    invalid_normalization_widths += 1
            curve.insert(0, "reach_id", reach_id)
            outputs.append(curve)
            if progress_callback is not None:
                progress_callback(position + 1, total_reaches)
            continue

        wua = curve["WUA"].to_numpy(dtype=float)
        dimensionless_wua = (
            curve["d.suit"].to_numpy(dtype=float)
            * curve["v.suit"].to_numpy(dtype=float)
            * curve["s.suit"].to_numpy(dtype=float)
        )
        q = curve["Q"].to_numpy(dtype=float)
        max_wua = float(np.max(wua))
        # Curves are ordered by increasing discharge. In the event of an exact
        # WUA tie, report the lowest modeled discharge attaining the maximum.
        q_at_max = float(q[np.flatnonzero(wua == max_wua)[0]])
        summary = {
            "WUA_mean": float(np.mean(wua)),
            "WUA_max": max_wua,
            "WUA_dimensionless_mean": float(np.mean(dimensionless_wua)),
            "WUA_dimensionless_max": float(np.max(dimensionless_wua)),
            "WUA_Q_min_m3s": float(np.min(q)),
            "WUA_Q_max_m3s": float(np.max(q)),
            "WUA_Q_at_max_m3s": q_at_max,
            "WUA_Q_count": len(curve),
            "d.suit_mean": float(curve["d.suit"].mean()),
            "v.suit_mean": float(curve["v.suit"].mean()),
            "s.suit": float(curve["s.suit"].iloc[0]),
        }
        normalization_width = None
        if normalize_width_col is not None:
            try:
                candidate_width = float(reach[normalize_width_col])
            except (TypeError, ValueError):
                candidate_width = np.nan
            if np.isfinite(candidate_width) and candidate_width > 0:
                normalization_width = candidate_width
                summary["WUA_mean_normalized"] = summary["WUA_mean"] / candidate_width
                summary["WUA_max_normalized"] = summary["WUA_max"] / candidate_width
            else:
                summary["WUA_mean_normalized"] = np.nan
                summary["WUA_max_normalized"] = np.nan
                invalid_normalization_widths += 1
        if fields:
            summary.update(_flow_summary(
                reach, fields, flow_units, hydraulics,
                depth_curve, velocity_curve, substrate_curve, reach_gsd,
                substrate_size_mm,
            ))
            if normalize_width_col is not None:
                summary[_NORMALIZED_FLOW_COLUMN] = (
                    summary["WUA by flowrate"] / normalization_width
                    if normalization_width is not None
                    else np.nan
                )
            excluded_reaches += summary["WUA flow fields excluded"] != "{}"
        summaries.append(summary)
        if progress_callback is not None:
            progress_callback(position + 1, total_reaches)

    if skipped_reaches:
        warnings.warn(
            f"{skipped_reaches} segment(s) have missing or non-finite model "
            "inputs; their model results are null.",
            UserWarning,
            stacklevel=2,
        )

    if output == "curves":
        if not outputs:
            columns = [
                "reach_id", "Q", "d.suit", "v.suit", "s.suit", "w", "WUA",
            ]
            if normalize_width_col is not None:
                columns.append("WUA_normalized")
            return pd.DataFrame(columns=columns)
        if invalid_normalization_widths:
            warnings.warn(
                f"{invalid_normalization_widths} segment(s) have missing, non-finite, "
                "or nonpositive normalization widths; normalized WUA is null for "
                "those segments.",
                UserWarning,
                stacklevel=2,
            )
        return pd.concat(outputs, ignore_index=True)

    # Preserve repeated IDs and duplicate index labels without fan-out.
    result = reaches.copy()
    # Remove deprecated summary metrics from prior runs so stale integrated
    # values cannot be mistaken for results from the current calculation.
    result = result.drop(columns=list(_LEGACY_SUMMARY_COLUMNS), errors="ignore")
    # Remove optional metrics from a prior run when the option is disabled.
    if not fields:
        result = result.drop(columns=[*_FLOW_COLUMNS, _NORMALIZED_FLOW_COLUMN], errors="ignore")
    if normalize_width_col is None:
        result = result.drop(
            columns=[*_NORMALIZED_SUMMARY_COLUMNS, _NORMALIZED_FLOW_COLUMN],
            errors="ignore",
        )
    output_columns = [
        *_SUMMARY_COLUMNS,
        *(_NORMALIZED_SUMMARY_COLUMNS if normalize_width_col is not None else ()),
        *(_FLOW_COLUMNS if fields else ()),
    ]
    if fields and normalize_width_col is not None:
        output_columns.append(_NORMALIZED_FLOW_COLUMN)
    for column in output_columns:
        values = [summary[column] for summary in summaries]
        if column == "WUA_Q_count":
            result[column] = pd.array(values, dtype="Int64")
        elif column in _FLOW_COLUMNS[1:]:
            result[column] = np.asarray(values, dtype=object)
        else:
            result[column] = np.asarray(values, dtype=float)
    if invalid_normalization_widths:
        warnings.warn(
            f"{invalid_normalization_widths} segment(s) have missing, non-finite, "
            "or nonpositive normalization widths; normalized WUA is null for "
            "those segments.",
            UserWarning,
            stacklevel=2,
        )
    if excluded_reaches:
        warnings.warn(
            f"{excluded_reaches} segment(s) excluded one or more selected flows. "
            "WUA by flowrate uses the remaining fields; no usable fields gives "
            "NaN. Inspect 'WUA flow fields' and 'WUA flow fields excluded'.",
            UserWarning,
            stacklevel=2,
        )
    return result
