"""Reach-averaged hydraulic simulation used by GIFT."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from ._validation import finite_scalar

_DELTA_X = 0.0001
_DELTA_Y = 0.001
_GRAVITY = 9.81


def _default_discharge_grid() -> np.ndarray:
    """Reproduce the fixed discharge grid in the original R function."""
    return np.concatenate(
        (
            np.arange(1, 101, dtype=float) * 0.001,
            np.arange(11, 101, dtype=float) * 0.01,
            np.arange(11, 101, dtype=float) * 0.1,
            np.arange(11, 101, dtype=float),
            np.arange(11, 101, dtype=float) * 10.0,
            np.arange(11, 101, dtype=float) * 100.0,
        )
    )


def _as_discharge_grid(
    discharges: float | Iterable[float] | None,
) -> np.ndarray:
    if discharges is None:
        return _default_discharge_grid()

    if np.isscalar(discharges):
        values = np.asarray([discharges], dtype=float)
    else:
        values = np.asarray(list(discharges), dtype=float)

    if values.ndim != 1 or values.size == 0:
        raise ValueError("discharges must contain at least one value")
    if not np.all(np.isfinite(values)):
        raise ValueError("discharges contains missing or non-finite values")
    if np.any(values <= 0):
        raise ValueError("discharges must be greater than zero")
    return values


def _mean_duplicate_x(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply R approx()'s default mean treatment for tied x values."""
    unique_x, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    if unique_x.size == x.size:
        return x, y

    sums = np.zeros(unique_x.size, dtype=float)
    np.add.at(sums, inverse, y)
    return unique_x, sums / counts


def _approx_like_r(
    x: np.ndarray,
    y: np.ndarray,
    xout: np.ndarray,
) -> np.ndarray:
    """Linear interpolation with R approx() defaults relevant to GIFT."""
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2:
        raise ValueError("At least two finite hydraulic simulations are required")

    order = np.argsort(x, kind="mergesort")
    x, y = _mean_duplicate_x(x[order], y[order])

    result = np.interp(xout, x, y)
    outside = (xout < x[0]) | (xout > x[-1])
    result[outside] = np.nan
    return result


def _shape_factor(
    bankfull_width: float,
    bankfull_depth: float,
    max_bankfull_depth: float | None,
    shape_factor: float | None,
) -> float:
    if shape_factor is not None:
        result = finite_scalar(shape_factor, "shape_factor")
    elif max_bankfull_depth is not None:
        result = 1.0 - (bankfull_depth / max_bankfull_depth)
    else:
        # This is the formula in the executable R source. One equation in the
        # original user guide omits the divisor of 100.
        result = (bankfull_width / bankfull_depth) / 100.0

    if result < 0:
        raise ValueError("shape_factor must be greater than or equal to zero")
    if result > 0.7:
        raise ValueError(
            "shape_factor exceeds 0.7; the original model identifies this "
            "range as unrealistic"
        )
    return result


def _cross_section_grid(
    bankfull_width: float,
    bankfull_depth: float,
    shape_factor: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    max_depth = bankfull_depth / (1.0 - shape_factor)
    x_coordinates = np.array(
        [0.0, shape_factor * bankfull_width, 0.99 * bankfull_width, bankfull_width],
        dtype=float,
    )
    y_coordinates = 5.0 * bankfull_depth - np.array(
        [0.0, bankfull_depth, max_depth, 0.0],
        dtype=float,
    )

    # np.interp matches R approx() for the usual strictly increasing
    # coordinates. Average tied coordinates to also cover shape_factor == 0.
    x_coordinates, y_coordinates = _mean_duplicate_x(
        x_coordinates,
        y_coordinates,
    )
    x_grid = np.linspace(0.0, bankfull_width, 10_001)
    y_grid = np.interp(x_grid, x_coordinates, y_coordinates)
    return x_grid, y_grid, max_depth


def _simulate_water_levels(
    slope: float,
    bankfull_width: float,
    bankfull_depth: float,
    d84_mm: float,
    shape_factor: float,
) -> pd.DataFrame:
    """Simulate the original discrete cross section without a Python loop."""
    x_grid, y_grid, max_depth = _cross_section_grid(
        bankfull_width,
        bankfull_depth,
        shape_factor,
    )
    bed_minimum_index = int(np.argmin(y_grid))

    relative_depths = np.arange(20, 1001, dtype=float) * _DELTA_Y * max_depth
    water_levels = 5.0 * bankfull_depth - max_depth + relative_depths

    left_bed = y_grid[: bed_minimum_index + 1]
    right_bed = y_grid[bed_minimum_index:]

    # The cross section decreases monotonically to its minimum and then
    # increases. These searches reproduce depths[depths >= 0] in the R code.
    left_indices = np.searchsorted(-left_bed, -water_levels, side="left")
    right_offsets = np.searchsorted(
        right_bed,
        water_levels,
        side="right",
    ) - 1
    right_indices = bed_minimum_index + right_offsets

    point_counts = right_indices - left_indices + 1
    if np.any(point_counts < 2):
        raise RuntimeError("The simulated wetted cross section is too narrow")

    cumulative_y = np.concatenate(([0.0], np.cumsum(y_grid)))
    wetted_y_sum = (
        cumulative_y[right_indices + 1] - cumulative_y[left_indices]
    )

    grid_spacing = float(np.max(x_grid)) * _DELTA_X
    wetted_width = point_counts * grid_spacing
    area = (point_counts * water_levels - wetted_y_sum) * grid_spacing
    mean_depth = area / wetted_width

    segment_lengths = np.hypot(np.diff(y_grid), grid_spacing)
    cumulative_length = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    wetted_perimeter = (
        cumulative_length[right_indices] - cumulative_length[left_indices]
    )
    hydraulic_radius = area / wetted_perimeter

    d84_m = d84_mm / 1000.0
    a1 = 6.5
    a2 = 2.5
    relative_submergence = (
        a1
        * a2
        * (hydraulic_radius / d84_m)
        / np.sqrt(
            a1**2
            + a2**2 * (hydraulic_radius / d84_m) ** (5.0 / 3.0)
        )
    )
    velocity = relative_submergence * np.sqrt(
        _GRAVITY * hydraulic_radius * slope
    )
    discharge = velocity * area

    return pd.DataFrame(
        {
            "Q": discharge,
            "Ai": area,
            "Wi": wetted_width,
            "di": mean_depth,
            "Ui": velocity,
        }
    )


def _write_cross_section(
    output_dir: Path,
    bankfull_width: float,
    bankfull_depth: float,
    shape_factor: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    max_depth = bankfull_depth / (1.0 - shape_factor)
    cross_section = pd.DataFrame(
        {
            "x": [
                0.0,
                shape_factor * bankfull_width,
                0.99 * bankfull_width,
                bankfull_width,
            ],
            "y": [0.0, -bankfull_depth, -max_depth, 0.0],
        }
    )
    cross_section.to_csv(output_dir / "channel_xs.csv", index=False)

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError(
            "Install gift-habitat[plot] to write cross-section figures"
        ) from exc

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(cross_section["x"], cross_section["y"], color="black")
    axis.set(
        xlabel="Width (m)",
        ylabel="Depth (m)",
        ylim=(cross_section["y"].min() * 1.2, 0.0),
    )
    figure.tight_layout()
    figure.savefig(output_dir / "channel_xs.jpeg", dpi=300)
    plt.close(figure)


def avg_hydraulics(
    slope: float,
    bankfull_width: float,
    bankfull_depth: float,
    d84_mm: float,
    *,
    max_bankfull_depth: float | None = None,
    shape_factor: float | None = None,
    discharges: float | Iterable[float] | None = None,
    full_curve: bool = False,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Simulate reach-averaged hydraulics below bankfull flow.

    Parameters
    ----------
    slope
        Channel gradient in m/m.
    bankfull_width
        Reach-averaged bankfull width in m.
    bankfull_depth
        Reach-averaged mean bankfull depth in m.
    d84_mm
        Bed-material D84 in mm.
    max_bankfull_depth
        Optional reach-averaged maximum bankfull depth in m.
    shape_factor
        Optional user-specified channel shape factor. This overrides
        ``max_bankfull_depth`` when both are supplied.
    discharges
        Optional positive discharge value or values in m3/s. The original
        GIFT discharge grid is used when omitted. Values outside the simulated
        subbankfull range are omitted.
    full_curve
        Return all native water-level simulations, including both discharge
        endpoints, for summary statistics over the entire modeled range. Cannot be
        combined with ``discharges``. The default single-reach grid is
        unchanged when this is False.
    output_dir
        Optional directory for ``channel_xs.csv`` and ``channel_xs.jpeg``.

    Returns
    -------
    pandas.DataFrame
        Columns match the R package: discharge ``Q`` (m3/s), area ``Ai``
        (m2), wetted width ``Wi`` (m), mean depth ``di`` (m), and mean
        velocity ``Ui`` (m/s).
    """
    if full_curve and discharges is not None:
        raise ValueError("full_curve cannot be combined with discharges")
    slope = finite_scalar(slope, "slope", positive=True)
    bankfull_width = finite_scalar(
        bankfull_width,
        "bankfull_width",
        positive=True,
    )
    bankfull_depth = finite_scalar(
        bankfull_depth,
        "bankfull_depth",
        positive=True,
    )
    d84_mm = finite_scalar(d84_mm, "d84_mm", positive=True)

    if bankfull_width > 100:
        warnings.warn(
            "bankfull_width is outside the original recommended range",
            UserWarning,
            stacklevel=2,
        )
    if bankfull_depth > 5:
        warnings.warn(
            "bankfull_depth is outside the original recommended range",
            UserWarning,
            stacklevel=2,
        )
    if d84_mm <= 1:
        warnings.warn(
            "Confirm that d84_mm is expressed in millimeters",
            UserWarning,
            stacklevel=2,
        )
    if d84_mm > 400:
        warnings.warn(
            "d84_mm may be outside the original recommended range",
            UserWarning,
            stacklevel=2,
        )

    if max_bankfull_depth is not None:
        max_bankfull_depth = finite_scalar(
            max_bankfull_depth,
            "max_bankfull_depth",
            positive=True,
        )
        if bankfull_depth > max_bankfull_depth:
            raise ValueError(
                "bankfull_depth must not exceed max_bankfull_depth"
            )

    resolved_shape_factor = _shape_factor(
        bankfull_width,
        bankfull_depth,
        max_bankfull_depth,
        shape_factor,
    )
    simulated = _simulate_water_levels(
        slope,
        bankfull_width,
        bankfull_depth,
        d84_mm,
        resolved_shape_factor,
    )
    if full_curve:
        result = simulated.sort_values("Q", kind="stable").reset_index(drop=True)
    else:
        target_discharges = _as_discharge_grid(discharges)
        result = pd.DataFrame({"Q": target_discharges})
        for column in ("Ai", "Wi", "di", "Ui"):
            result[column] = _approx_like_r(
                simulated["Q"].to_numpy(),
                simulated[column].to_numpy(),
                target_discharges,
            )
        result = result.dropna(subset=["Ai"]).reset_index(drop=True)

    max_depth = bankfull_depth / (1.0 - resolved_shape_factor)
    result.attrs.update(
        {
            "shape_factor": resolved_shape_factor,
            "simulated_max_depth_m": max_depth,
            "simulated_min_discharge_m3s": float(simulated["Q"].min()),
            "simulated_bankfull_discharge_m3s": float(simulated["Q"].iloc[-1]),
        }
    )

    if output_dir is not None:
        _write_cross_section(
            Path(output_dir),
            bankfull_width,
            bankfull_depth,
            resolved_shape_factor,
        )

    return result

