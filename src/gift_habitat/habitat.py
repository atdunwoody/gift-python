"""Habitat suitability and weighted usable area calculations."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from ._validation import numeric_frame

_GRAVITY = 9.81


def _validate_suitability_curve(
    curve: object,
    name: str,
    value_columns: tuple[str, ...],
) -> pd.DataFrame:
    result = numeric_frame(curve, name, (*value_columns, "suit"))
    for column in value_columns:
        if (result[column] < 0).any():
            raise ValueError(f"{name}.{column} contains values below zero")
    if ((result["suit"] < 0) | (result["suit"] > 1)).any():
        raise ValueError(f"{name}.suit must be between zero and one")
    return result


def _as_gsd(gsd: Sequence[float] | np.ndarray | pd.Series) -> np.ndarray:
    if isinstance(gsd, (str, bytes)) or not isinstance(
        gsd,
        (Sequence, np.ndarray, pd.Series),
    ):
        raise TypeError("gsd must be a one-dimensional numeric sequence")
    try:
        result = np.asarray(gsd, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("gsd must be numeric") from exc
    if result.ndim != 1 or result.size == 0:
        raise ValueError("gsd must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(result)):
        raise ValueError("gsd contains missing or non-finite values")
    if np.any(result < 0):
        raise ValueError("gsd contains grain sizes below zero")
    if np.max(result) < 2:
        warnings.warn(
            "Confirm that gsd is expressed in millimeters",
            UserWarning,
            stacklevel=3,
        )
    if result.size < 10:
        warnings.warn(
            "gsd contains fewer than 10 observations",
            UserWarning,
            stacklevel=3,
        )
    return result


def substrate_suitability(
    substrate_curve: pd.DataFrame,
    gsd: Sequence[float] | np.ndarray | pd.Series,
) -> float:
    """Calculate the reach-averaged substrate suitability used by GIFT."""
    curve = _validate_suitability_curve(
        substrate_curve,
        "substrate_curve",
        ("lower", "upper"),
    )
    if (curve["upper"] <= curve["lower"]).any():
        raise ValueError(
            "Each substrate_curve upper bound must exceed its lower bound"
        )

    grain_sizes = _as_gsd(gsd)
    counts = np.array(
        [
            np.count_nonzero(
                (grain_sizes >= lower) & (grain_sizes < upper)
            )
            for lower, upper in zip(curve["lower"], curve["upper"], strict=True)
        ],
        dtype=float,
    )
    classified_count = counts.sum()
    if classified_count == 0:
        raise ValueError(
            "No gsd observations fall within the substrate_curve classes"
        )
    return float(np.dot(counts, curve["suit"]) / classified_count)


def substrate_suitability_at_size(
    substrate_curve: pd.DataFrame,
    grain_size_mm: float,
) -> float:
    """Look up suitability for one representative grain size in millimeters.

    This class lookup is an alternative to the original GIFT calculation,
    which averages suitability across a grain-size distribution.
    """
    curve = _validate_suitability_curve(
        substrate_curve, "substrate_curve", ("lower", "upper"),
    )
    if (curve["upper"] <= curve["lower"]).any():
        raise ValueError("Each substrate_curve upper bound must exceed its lower bound")
    try:
        size = float(grain_size_mm)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("grain_size_mm must be a finite, nonnegative number") from exc
    if not np.isfinite(size) or size < 0:
        raise ValueError("grain_size_mm must be a finite, nonnegative number")
    matching = curve.loc[(curve["lower"] <= size) & (size < curve["upper"])]
    if len(matching) != 1:
        raise ValueError(
            f"grain_size_mm={size:g} must fall in exactly one substrate class"
        )
    return float(matching["suit"].iloc[0])


def _normal_pdf(values: np.ndarray, mean: float, sd: float) -> np.ndarray:
    standardized = (values - mean) / sd
    return np.exp(-0.5 * standardized**2) / (sd * np.sqrt(2.0 * np.pi))


def _lognormal_pdf(values: np.ndarray, meanlog: float, sdlog: float) -> np.ndarray:
    result = np.zeros_like(values, dtype=float)
    positive = values > 0
    log_values = np.log(values[positive])
    standardized = (log_values - meanlog) / sdlog
    result[positive] = (
        np.exp(-0.5 * standardized**2)
        / (values[positive] * sdlog * np.sqrt(2.0 * np.pi))
    )
    return result


def _logistic(value: float) -> float:
    if value >= 0:
        return float(1.0 / (1.0 + np.exp(-value)))
    exponential = np.exp(value)
    return float(exponential / (1.0 + exponential))


def _nearest_suitability(
    curve_values: np.ndarray,
    curve_suitability: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    if np.all(curve_values[1:] >= curve_values[:-1]):
        # Search sorted curves without allocating a values-by-targets array.
        # Use the first duplicate and favor the lower value on an exact tie,
        # preserving argmin's first-row behavior (R's which.min()).
        right = np.searchsorted(curve_values, targets, side="left").clip(
            0, len(curve_values) - 1
        )
        left = np.searchsorted(
            curve_values,
            curve_values[np.maximum(right - 1, 0)],
            side="left",
        )
        choose_left = (
            np.abs(curve_values[left] - targets)
            <= np.abs(curve_values[right] - targets)
        )
        indices = np.where(choose_left, left, right)
    else:
        # Unsorted curves retain the original first-row tie behavior.
        indices = np.abs(curve_values[:, np.newaxis] - targets).argmin(axis=0)
    return curve_suitability[indices]


def _write_wua_plot(output_dir: Path, result: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError(
            "Install gift-habitat[plot] to write weighted-usable-area figures"
        ) from exc

    smoothed = result["WUA"].rolling(window=5, center=True).mean()
    if smoothed.notna().sum() == 0:
        smoothed = result["WUA"]
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(result["Q"], smoothed, linewidth=2)
    axis.set(
        xlabel=r"Discharge ($m^3\,s^{-1}$)",
        ylabel=r"WUA ($m^2\,m^{-1}$)",
        xlim=(0.0, float(result["Q"].max())),
        ylim=(0.0, float(smoothed.max(skipna=True)) * 1.1),
    )
    axis.grid()
    figure.tight_layout()
    figure.savefig(output_dir / "WUA_Q.jpeg", dpi=300)
    plt.close(figure)


def habitat(
    hydraulics: pd.DataFrame,
    depth_curve: pd.DataFrame,
    velocity_curve: pd.DataFrame,
    *,
    substrate_curve: pd.DataFrame | None = None,
    gsd: Sequence[float] | np.ndarray | pd.Series | None = None,
    substrate_size_mm: float | None = None,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Calculate reach-averaged suitability and weighted usable area.

    Suitability values are assigned using the nearest curve coordinate, as in
    the original R implementation. They are not linearly interpolated.

    Parameters
    ----------
    hydraulics
        Output from :func:`gift_habitat.avg_hydraulics`.
    depth_curve
        DataFrame with numeric ``depth`` (m) and ``suit`` columns.
    velocity_curve
        DataFrame with numeric ``velocity`` (m/s) and ``suit`` columns.
    substrate_curve
        Optional DataFrame with ``lower`` and ``upper`` grain-size bounds
        (mm) and a ``suit`` column.
    gsd
        Optional grain-size observations in mm. Supply with
        ``substrate_curve`` for the original GIFT distribution-weighted score.
    substrate_size_mm
        Optional representative grain size in mm (for example D84). Uses the
        containing substrate class directly, instead of averaging over a GSD.
        Supply this or ``gsd``, not both.
    output_dir
        Optional directory for ``WUA_Q.jpeg``.

    Returns
    -------
    pandas.DataFrame
        Columns match the R package: ``Q``, ``d.suit``, ``v.suit``,
        ``s.suit``, ``w``, and ``WUA``.
    """
    hydraulics = numeric_frame(
        hydraulics,
        "hydraulics",
        ("Q", "Ai", "Wi", "di", "Ui"),
    )
    if len(hydraulics) < 3:
        warnings.warn(
            "hydraulics contains fewer than three rows",
            UserWarning,
            stacklevel=2,
        )
    if (
        (hydraulics[["Q", "Ai", "Wi", "di", "Ui"]] <= 0)
        .any(axis=None)
    ):
        raise ValueError("hydraulics values must be greater than zero")

    depth_curve = _validate_suitability_curve(
        depth_curve,
        "depth_curve",
        ("depth",),
    )
    velocity_curve = _validate_suitability_curve(
        velocity_curve,
        "velocity_curve",
        ("velocity",),
    )

    if gsd is not None and substrate_size_mm is not None:
        raise ValueError("Use either gsd or substrate_size_mm, not both")
    if substrate_size_mm is not None:
        if substrate_curve is None:
            raise ValueError("substrate_size_mm requires substrate_curve")
        substrate_score = substrate_suitability_at_size(
            substrate_curve, substrate_size_mm,
        )
    elif (substrate_curve is None) != (gsd is None):
        warnings.warn(
            "Both substrate_curve and gsd are required; substrate "
            "suitability defaults to 1.0",
            UserWarning,
            stacklevel=2,
        )
        substrate_score = 1.0
    elif substrate_curve is None:
        substrate_score = 1.0
    else:
        substrate_score = substrate_suitability(substrate_curve, gsd)

    bins = np.arange(0, 121, dtype=float) * 0.05
    normal_depth = _normal_pdf(bins, mean=1.0, sd=0.52)
    lognormal_depth = _lognormal_pdf(bins, meanlog=0.0, sdlog=1.09)

    depth_values = depth_curve["depth"].to_numpy(dtype=float)
    depth_suitability = depth_curve["suit"].to_numpy(dtype=float)
    velocity_values = velocity_curve["velocity"].to_numpy(dtype=float)
    velocity_suitability = velocity_curve["suit"].to_numpy(dtype=float)

    rows: list[tuple[float, float, float, float, float, float]] = []
    for hydraulic in hydraulics.itertuples(index=False):
        froude = hydraulic.Ui / np.sqrt(_GRAVITY * hydraulic.di)

        mixing_logit = -4.72 - 2.84 * np.log(froude)
        mixing = _logistic(float(mixing_logit))
        depth_distribution = (
            (1.0 - mixing) * normal_depth + mixing * lognormal_depth
        )
        depth_distribution = np.maximum(depth_distribution, 0.0)
        absolute_depths = hydraulic.di * bins

        velocity_scale = -0.150 - 0.252 * np.log(froude)
        velocity_distribution = velocity_scale * (
            3.33 * np.exp(-bins / 0.693)
            + 0.117 * np.exp(-((bins - 8.0) / 1.73) ** 2)
        ) + (1.0 - velocity_scale) * (
            0.653 * np.exp(-((bins - 1.0) / 0.664) ** 2)
        )
        velocity_distribution = np.maximum(velocity_distribution, 0.0)
        absolute_velocities = hydraulic.Ui * bins

        selected_depth_suitability = _nearest_suitability(
            depth_values,
            depth_suitability,
            absolute_depths,
        )
        selected_velocity_suitability = _nearest_suitability(
            velocity_values,
            velocity_suitability,
            absolute_velocities,
        )
        depth_score = float(
            np.dot(depth_distribution, selected_depth_suitability)
            / depth_distribution.sum()
        )
        velocity_score = float(
            np.dot(velocity_distribution, selected_velocity_suitability)
            / velocity_distribution.sum()
        )
        wua = (
            depth_score
            * velocity_score
            * substrate_score
            * hydraulic.Wi
        )
        rows.append(
            (
                hydraulic.Q,
                depth_score,
                velocity_score,
                substrate_score,
                hydraulic.Wi,
                wua,
            )
        )

    result = pd.DataFrame(
        rows,
        columns=("Q", "d.suit", "v.suit", "s.suit", "w", "WUA"),
    )
    if output_dir is not None:
        _write_wua_plot(Path(output_dir), result)
    return result
