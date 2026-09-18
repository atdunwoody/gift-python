"""Compare HSI, WUA, and their component trends with drainage area.

Edit the paths and column names below and run this file directly in VSCode.
Requires the installed gift-habitat package, SciPy, and Matplotlib. GeoPackage
attributes are read with SQLite, so GeoPandas is not needed for this analysis.
"""

from pathlib import Path
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, spearmanr

from gift_habitat import model_reaches


ROOT = Path(__file__).resolve().parents[1]
NETWORK_PATH = ROOT / "examples/Inputs/UGR_channel_geometry_predictions.gpkg"
CURVES_DIR = ROOT / "examples/Inputs"
OUTPUT_DIR = ROOT / "examples/outputs/hsi_vs_drainage_area"
ID_FIELD = "COMID"
DA_FIELD = "TotDASqKm"  # km2; use div_da_km2 if that is the desired drainage area.
SLOPE_FIELD = "SLOPE"  # m/m or ft/ft
WIDTH_FIELD = "bf_width_pred_m"  # m
DEPTH_FIELD = "bf_depth_pred_m"  # m
D84_FIELD = "D84_pred"  # mm
N_DA_BINS = 5
METRICS = ("HSI_mean", "HSI_max", "WUA_mean")
COMPONENTS = {
    "depth": "HSI_depth_contribution_mean",
    "velocity": "HSI_velocity_contribution_mean",
    "substrate": "HSI_substrate_contribution_mean",
}


def curve_groups(curves_dir: Path) -> list[str]:
    """Identify only groups with matching depth, velocity and substrate curves."""
    groups = []
    for depth_path in sorted(curves_dir.glob("*_depth.csv")):
        group = depth_path.name.removesuffix("_depth.csv")
        if group.startswith("generic_salmon"):
            continue
        for kind in ("velocity", "substrate"):
            other = curves_dir / f"{group}_{kind}.csv"
            if not other.is_file():
                raise FileNotFoundError(f"Missing {kind} curve for {group}: {other}")
        groups.append(group)
    if not groups:
        raise ValueError(f"No complete non-generic curve groups in {curves_dir}")
    return groups


def read_network_attributes(path: Path, fields: list[str]) -> pd.DataFrame:
    """Read only the required attribute columns; geometries are not modeled."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, usecols=fields)
    if path.suffix.lower() != ".gpkg":
        raise ValueError("NETWORK_PATH must be a GeoPackage or CSV")
    with sqlite3.connect(path) as connection:
        layers = connection.execute(
            "SELECT table_name FROM gpkg_contents WHERE data_type = 'features'"
        ).fetchall()
        if len(layers) != 1:
            raise ValueError(f"Expected one feature layer in {path}; found {len(layers)}")
        table = layers[0][0].replace('"', '""')
        columns = connection.execute(f'PRAGMA table_info("{table}")').fetchall()
        available = {column[1] for column in columns}
        missing = set(fields) - available
        if missing:
            raise ValueError(f"Missing network fields: {sorted(missing)}")
        selected = ", ".join('"' + field.replace('"', '""') + '"' for field in fields)
        return pd.read_sql_query(f'SELECT {selected} FROM "{table}"', connection)


def adjust_bh(p_values: pd.Series) -> np.ndarray:
    """Benjamini-Hochberg adjusted p values across all groups and metrics."""
    p = p_values.to_numpy(dtype=float)
    adjusted = np.full(len(p), np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    if finite.size:
        ordered = finite[np.argsort(p[finite])]
        ranked = p[ordered] * len(ordered) / np.arange(1, len(ordered) + 1)
        adjusted[ordered] = np.minimum.accumulate(ranked[::-1])[::-1].clip(0, 1)
    return adjusted


def test_da_association(by_reach: pd.DataFrame) -> pd.DataFrame:
    """Test monotonic association and report change per tenfold increase in DA."""
    rows = []
    for (group, metric), values in (
        by_reach.melt(
            id_vars=["group", "DA_km2"], value_vars=list(METRICS),
            var_name="metric", value_name="score",
        ).groupby(["group", "metric"], sort=True)
    ):
        valid = values.loc[
            np.isfinite(values["DA_km2"]) & (values["DA_km2"] > 0)
            & np.isfinite(values["score"])
        ]
        x = np.log10(valid["DA_km2"].to_numpy(dtype=float))
        y = valid["score"].to_numpy(dtype=float)
        if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
            rho = p = slope = intercept = r2 = np.nan
        else:
            rho, p = spearmanr(x, y)
            fit = linregress(x, y)
            slope, intercept, r2 = fit.slope, fit.intercept, fit.rvalue**2
        rows.append({
            "group": group, "metric": metric, "n": len(valid),
            "spearman_rho": rho, "spearman_p": p,
            "slope_per_10x_DA": slope, "intercept_at_1_km2": intercept,
            "linear_r2": r2,
        })
    result = pd.DataFrame(rows)
    result["spearman_p_BH"] = adjust_bh(result["spearman_p"])
    return result


def da_bin_medians(by_reach: pd.DataFrame) -> pd.DataFrame:
    """Summarize the same reach sample in pooled log-DA quintiles."""
    unique_reaches = by_reach.loc[by_reach["group"] == by_reach["group"].iloc[0]]
    breaks = np.unique(np.quantile(
        np.log10(unique_reaches["DA_km2"]), np.linspace(0, 1, N_DA_BINS + 1),
    ))
    if len(breaks) < 2:
        raise ValueError("Drainage area has no usable spread for binning")
    data = by_reach.copy()
    data["DA_bin"] = pd.cut(np.log10(data["DA_km2"]), breaks, include_lowest=True, labels=False)
    aggregations = {
        "n": ("DA_km2", "size"),
        "DA_median_km2": ("DA_km2", "median"),
    }
    aggregations.update({f"{metric}_median": (metric, "median") for metric in METRICS})
    medians = data.groupby(["group", "DA_bin"], observed=True).agg(
        **aggregations,
    ).reset_index()
    medians["DA_bin"] += 1
    return medians


def component_attribution(by_reach: pd.DataFrame) -> pd.DataFrame:
    """Decompose the observed HSI_mean trend into three additive trends.

    Each reach's Shapley terms sum to HSI_mean - 1, so their least-squares
    slopes versus log10(DA) sum exactly to the HSI_mean slope. These describe
    the observed association and do not isolate causal effects.
    """
    rows = []
    for group, frame in by_reach.groupby("group", sort=True):
        x = np.log10(frame["DA_km2"].to_numpy(dtype=float))
        overall_slope = linregress(x, frame["HSI_mean"]).slope
        component_slopes = {
            name: linregress(x, frame[field]).slope
            for name, field in COMPONENTS.items()
        }
        # The term aligned with the overall slope has the largest positive
        # contribution to that trend; opposing terms get negative shares.
        if np.isclose(overall_slope, 0, atol=1e-12):
            dominant = None
        else:
            direction = np.sign(overall_slope)
            dominant = max(component_slopes, key=lambda name: direction * component_slopes[name])
        for name, slope in component_slopes.items():
            field = COMPONENTS[name]
            rho, p = spearmanr(x, frame[field])
            rows.append({
                "group": group, "component": name, "n": len(frame),
                "spearman_rho": rho, "spearman_p": p,
                "slope_per_10x_DA": slope,
                "HSI_mean_slope_per_10x_DA": overall_slope,
                "share_of_HSI_mean_slope_pct": (
                    slope / overall_slope * 100 if dominant is not None else np.nan
                ),
                "largest_aligned_contributor": name == dominant,
            })
    return pd.DataFrame(rows)


def save_plots(by_reach: pd.DataFrame, bins: pd.DataFrame, groups: list[str]) -> None:
    for metric in METRICS:
        figure, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True,
                                    sharey=metric != "WUA_mean")
        for axis, group in zip(axes.flat, groups):
            subset = by_reach.loc[by_reach["group"] == group]
            median = bins.loc[bins["group"] == group]
            points = axis.scatter(
                subset["DA_km2"], subset[metric],
                c=subset["d.suit_mean"], cmap="viridis", vmin=0, vmax=1,
                s=10, alpha=0.6, linewidths=0, rasterized=True,
            )
            axis.plot(
                median["DA_median_km2"], median[f"{metric}_median"],
                color="black", marker="o", linewidth=1.5, label="DA-bin median",
            )
            axis.set_title(group.replace("_", " "), fontsize=10)
            axis.set_xscale("log")
            axis.grid(alpha=0.2)
        for axis in list(axes.flat)[len(groups):]:
            axis.set_visible(False)
        figure.supxlabel("Drainage area (km²; log scale)")
        figure.supylabel(f"{metric} (m²/m)" if metric == "WUA_mean" else metric)
        axes.flat[0].legend(loc="best", fontsize=8)
        colorbar = figure.colorbar(points, ax=[a for a in axes.flat if a.get_visible()],
                                   shrink=0.84, pad=0.02)
        colorbar.set_label("Mean depth suitability (0–1)")
        figure.savefig(OUTPUT_DIR / f"{metric}_vs_DA.png", dpi=200)
        plt.close(figure)


def save_component_plot(attribution: pd.DataFrame, groups: list[str]) -> None:
    """Plot the exact additive decomposition of each group's HSI_mean slope."""
    figure, axes = plt.subplots(len(groups), 1, figsize=(10, 2.0 * len(groups)),
                                sharex=True, layout="constrained")
    colors = {"depth": "#2b83ba", "velocity": "#008c7d", "substrate": "#d77b21"}
    for axis, group in zip(np.atleast_1d(axes), groups):
        frame = attribution.loc[attribution["group"] == group].set_index("component")
        axis.barh(list(COMPONENTS), frame.loc[list(COMPONENTS), "slope_per_10x_DA"],
                  color=[colors[name] for name in COMPONENTS])
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_ylabel(group.replace("_", " "), rotation=0, ha="right", va="center")
        axis.grid(axis="x", alpha=0.2)
    axes[-1].set_xlabel("Contribution to HSI_mean slope per tenfold increase in DA")
    figure.savefig(OUTPUT_DIR / "HSI_mean_component_contributions.png", dpi=200)
    plt.close(figure)


def main() -> None:
    groups = curve_groups(CURVES_DIR)
    # All groups use exactly the same reaches and hydraulic inputs. Excluding
    # incomplete rows up front makes group-to-group comparisons consistent.
    fields = list(dict.fromkeys((
        ID_FIELD, DA_FIELD, SLOPE_FIELD, WIDTH_FIELD, DEPTH_FIELD, D84_FIELD,
    )))
    streams = read_network_attributes(NETWORK_PATH, fields)
    for field in (DA_FIELD, SLOPE_FIELD, WIDTH_FIELD, DEPTH_FIELD, D84_FIELD):
        streams[field] = pd.to_numeric(streams[field], errors="coerce")
    valid = np.isfinite(streams[[DA_FIELD, SLOPE_FIELD, WIDTH_FIELD, DEPTH_FIELD, D84_FIELD]]).all(axis=1)
    valid &= (streams[[DA_FIELD, SLOPE_FIELD, WIDTH_FIELD, DEPTH_FIELD, D84_FIELD]] > 0).all(axis=1)
    valid &= streams[ID_FIELD].notna()
    reaches = streams.loc[valid].copy().reset_index(drop=True)
    if reaches.empty:
        raise ValueError("No valid reaches with positive DA and complete hydraulic inputs")
    print(f"Modeling {len(reaches)} of {len(streams)} reaches in {len(groups)} groups", flush=True)
    results = []
    for group in groups:
        print(f"Modeling {group}", flush=True)
        curves = {
            kind: pd.read_csv(CURVES_DIR / f"{group}_{kind}.csv")
            for kind in ("depth", "velocity", "substrate")
        }
        modeled = model_reaches(
            reaches, curves["depth"], curves["velocity"],
            slope_col=SLOPE_FIELD, width_col=WIDTH_FIELD, depth_col=DEPTH_FIELD,
            d84_col=D84_FIELD, id_col=ID_FIELD, substrate_curve=curves["substrate"],
        )
        subset = modeled[[
            ID_FIELD, DA_FIELD, D84_FIELD, *METRICS,
            "HSI_cubic_root_mean", "HSI_cubic_root_max",
            *COMPONENTS.values(), "d.suit_mean", "v.suit_mean", "s.suit",
            "GIFT_depth_mean_m", "GIFT_velocity_mean_mps",
        ]].copy()
        subset.insert(0, "group", group)
        subset.rename(columns={DA_FIELD: "DA_km2"}, inplace=True)
        results.append(subset)

    by_reach = pd.concat(results, ignore_index=True)
    summary = test_da_association(by_reach)
    bins = da_bin_medians(by_reach)
    attribution = component_attribution(by_reach)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    by_reach.to_csv(OUTPUT_DIR / "hsi_wua_by_reach.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "hsi_wua_da_tests.csv", index=False)
    bins.to_csv(OUTPUT_DIR / "hsi_wua_da_bins.csv", index=False)
    attribution.to_csv(OUTPUT_DIR / "hsi_component_da_attribution.csv", index=False)
    save_plots(by_reach, bins, groups)
    save_component_plot(attribution, groups)
    print(summary.to_string(index=False))
    print("\nHSI_mean component slopes per tenfold increase in DA:")
    print(attribution.to_string(index=False))
    print(f"Saved tables and plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
