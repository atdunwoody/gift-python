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

from gift_habitat import avg_hydraulics, habitat, model_reaches


ROOT = Path(__file__).resolve().parents[1]
NETWORK_PATH = ROOT / "examples/Inputs/UGR_channel_geometry_predictions.gpkg"
CURVES_DIR = ROOT / "examples/Inputs"
OUTPUT_DIR = ROOT / "examples/outputs/hsi_vs_drainage_area"
GENERAL_GROUP = "salmonid_general_sensitivity"
ID_FIELD = "COMID"
DA_FIELD = "TotDASqKm"  # km2; use div_da_km2 if that is the desired drainage area.
SLOPE_FIELD = "SLOPE"  # m/m or ft/ft
WIDTH_FIELD = "bf_width_pred_m"  # m
DEPTH_FIELD = "bf_depth_pred_m"  # m
D84_FIELD = "D84_pred"  # mm
SUMMER_WIDTH_FIELD = "late_summer_wetted_width_pred_m"  # m; use None to omit
N_DA_BINS = 5
METRICS = ("HSI_mean", "HSI_max", "WUA_mean")
DRIVER_LABELS = {
    "bankfull_width_m": "Predicted bankfull width (m)",
    "bankfull_depth_m": "Predicted bankfull depth (m)",
    "slope_m_per_m": "Channel slope (m/m)",
    "D84_pred": "Predicted D84 (mm)",
    "GIFT_depth_mean_m": "Mean modeled depth (m)",
    "GIFT_velocity_mean_mps": "Mean modeled velocity (m/s)",
}
WIDTH_ADJUSTED_METRICS = {
    "WUA_per_bankfull_width": "WUA / predicted bankfull width",
    "WUA_per_summer_width": "WUA / predicted late-summer width",
}
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


def general_salmonid_curves() -> dict[str, pd.DataFrame]:
    """Read the mixed-source general salmonid scenario from Inputs.

    The depth and velocity CSVs retain Ptolemy's `salmonid/general` values.
    The substrate CSV uses WDFW/Ecology (2022), Table 1, generic juvenile and
    resident adult salmon/trout rearing values. Applying these classes to D84
    approximates the source's field-based substrate/cover observations;
    bedrock and cover cannot be inferred from D84.

    Sources: https://sgronsdahl.github.io/GIFT/guidance.html
    https://apps.ecology.wa.gov/publications/documents/0411007.pdf
    """
    return {
        kind: pd.read_csv(CURVES_DIR / f"{GENERAL_GROUP}_{kind}.csv")
        for kind in ("depth", "velocity", "substrate")
    }


def read_group_curves(group: str) -> dict[str, pd.DataFrame]:
    if group == GENERAL_GROUP:
        return general_salmonid_curves()
    return {
        kind: pd.read_csv(CURVES_DIR / f"{group}_{kind}.csv")
        for kind in ("depth", "velocity", "substrate")
    }


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
    for metric in (*DRIVER_LABELS, "d.suit_mean", "v.suit_mean", "s.suit",
                   *WIDTH_ADJUSTED_METRICS):
        if metric in data:
            aggregations[f"{metric}_median"] = (metric, "median")
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
                color="#999999", alpha=0.75, marker="o", linewidth=2,
                markersize=5, label="DA-bin median",
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


def add_driver_columns(by_reach: pd.DataFrame, reaches: pd.DataFrame,
                       groups: list[str]) -> pd.DataFrame:
    """Align shared hydraulic inputs by row, including when IDs repeat."""
    if len(by_reach) != len(reaches) * len(groups):
        raise ValueError("Reach results and hydraulic input rows do not align")
    inputs = reaches.reset_index(drop=True)
    for index, group in enumerate(groups):
        part = by_reach.iloc[index * len(inputs):(index + 1) * len(inputs)]
        if not part["group"].eq(group).all() or not np.array_equal(
            part[ID_FIELD].to_numpy(), inputs[ID_FIELD].to_numpy()
        ):
            raise ValueError("Reach results must follow the input row order for every group")
    result = by_reach.copy().reset_index(drop=True)
    names = {
        WIDTH_FIELD: "bankfull_width_m",
        DEPTH_FIELD: "bankfull_depth_m",
        SLOPE_FIELD: "slope_m_per_m",
    }
    if SUMMER_WIDTH_FIELD is not None:
        names[SUMMER_WIDTH_FIELD] = "summer_width_m"
    for source, target in names.items():
        result[target] = np.tile(inputs[source].to_numpy(dtype=float), len(groups))
    result["WUA_per_bankfull_width"] = (
        result["WUA_mean"] / result["bankfull_width_m"]
    )
    if SUMMER_WIDTH_FIELD is not None:
        result["WUA_per_summer_width"] = np.where(
            result["summer_width_m"] > 0,
            result["WUA_mean"] / result["summer_width_m"], np.nan,
        )
    return result


def _rho(da: pd.Series, values: pd.Series) -> float:
    usable = np.isfinite(da) & (da > 0) & np.isfinite(values)
    if usable.sum() < 3 or values.loc[usable].nunique() < 2:
        return np.nan
    return float(spearmanr(da.loc[usable], values.loc[usable]).statistic)


def save_driver_plot(by_reach: pd.DataFrame, bins: pd.DataFrame,
                     first_group: str) -> None:
    """Show how DA tracks the shared inputs and resulting hydraulics."""
    data = by_reach.loc[by_reach["group"] == first_group]
    medians = bins.loc[bins["group"] == first_group]
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for axis, (field, label) in zip(axes.flat, DRIVER_LABELS.items()):
        axis.scatter(data["DA_km2"], data[field], s=9, color="#5685a0",
                     alpha=0.22, linewidths=0, rasterized=True)
        axis.plot(medians["DA_median_km2"], medians[f"{field}_median"],
                  "o-", color="#999999", alpha=0.75, linewidth=2,
                  markersize=5)
        axis.set(xscale="log", yscale="log", title=label,
                 xlabel="Drainage area (km²)", ylabel=label)
        axis.text(0.04, 0.96, f"Spearman ρ = {_rho(data['DA_km2'], data[field]):.2f}",
                  transform=axis.transAxes, va="top", fontsize=9,
                  bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
        axis.grid(alpha=0.2)
    figure.suptitle("Hydraulic inputs and outputs along the drainage-area gradient\n"
                    "One point per reach; grey lines show drainage-area quintile medians")
    figure.savefig(OUTPUT_DIR / "DA_hydraulic_drivers.png", dpi=200)
    plt.close(figure)


def save_suitability_plot(by_reach: pd.DataFrame, bins: pd.DataFrame,
                          groups: list[str]) -> None:
    """Separate depth, velocity and substrate associations for each group."""
    colors = {"d.suit_mean": "#2166ac", "v.suit_mean": "#1b9e77",
              "s.suit": "#cf6914"}
    labels = {"d.suit_mean": "Depth", "v.suit_mean": "Velocity",
              "s.suit": "Substrate"}
    nrows = max(3, (len(groups) + 1) // 2)
    figure, axes = plt.subplots(nrows, 2, figsize=(12, 3.8 * nrows), sharex=True,
                                sharey=True, layout="constrained")
    for axis, group in zip(axes.flat, groups):
        data = by_reach.loc[by_reach["group"] == group]
        medians = bins.loc[bins["group"] == group]
        for field, color in colors.items():
            axis.scatter(data["DA_km2"], data[field], s=5, alpha=0.07,
                         color=color, linewidths=0, rasterized=True)
            axis.plot(medians["DA_median_km2"], medians[f"{field}_median"],
                      "o-", color=color, linewidth=1.8, markersize=4,
                      label=labels[field])
        axis.set(xscale="log", ylim=(-0.02, 1.02),
                 title=group.replace("_", " "))
        axis.grid(alpha=0.2)
    for axis in axes[-1]:
        axis.set_xlabel("Drainage area (km²)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Mean component suitability (0–1)")
    for axis in list(axes.flat)[len(groups):]:
        axis.axis("off")
    axes.flat[0].legend(loc="upper right", frameon=False, fontsize=9)
    figure.suptitle("Species and life-stage suitability along the drainage-area gradient\n"
                    "Faint points are reaches; lines show drainage-area quintile medians. "
                    "Component means do not multiply to mean HSI.")
    figure.savefig(OUTPUT_DIR / "DA_suitability_components.png", dpi=200)
    plt.close(figure)


def save_width_adjusted_plot(by_reach: pd.DataFrame, bins: pd.DataFrame,
                             groups: list[str]) -> None:
    """Show whether the WUA trend persists after dividing by width."""
    metrics = ["WUA_per_bankfull_width"]
    if SUMMER_WIDTH_FIELD is not None:
        metrics.append("WUA_per_summer_width")
    colors = ["#2166ac", "#cf6914"]
    ncols = 4 if len(groups) > 5 else 3
    figure, axes = plt.subplots(2, ncols, figsize=(4.5 * ncols, 8), sharex=True,
                                sharey=True, layout="constrained")
    for axis, group in zip(axes.flat, groups):
        data = by_reach.loc[by_reach["group"] == group]
        medians = bins.loc[bins["group"] == group]
        for metric, color in zip(metrics, colors):
            valid = np.isfinite(data[metric]) & (data[metric] > 0)
            axis.scatter(data.loc[valid, "DA_km2"], data.loc[valid, metric],
                         s=7, alpha=0.12, color=color, linewidths=0,
                         rasterized=True)
            axis.plot(medians["DA_median_km2"], medians[f"{metric}_median"],
                      "o-", color=color, linewidth=1.8, markersize=4,
                      label=WIDTH_ADJUSTED_METRICS[metric])
        axis.set(xscale="log", yscale="log", title=group.replace("_", " "))
        names = {"WUA_per_bankfull_width": "BF width",
                 "WUA_per_summer_width": "Summer width"}
        axis.text(0.04, 0.96,
                  "\n".join(f"{names[m]}: ρ = {_rho(data['DA_km2'], data[m]):.2f}"
                            for m in metrics),
                  color="#253b4a", transform=axis.transAxes, va="top", fontsize=9,
                  bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
        axis.grid(alpha=0.2)
    for axis in axes[-1]:
        axis.set_xlabel("Drainage area (km²)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Mean WUA / predicted width (unitless)")
    for axis in list(axes.flat)[len(groups):]:
        axis.axis("off")
    legend_axis = axes.flat[len(groups)]
    legend_axis.legend(*axes.flat[0].get_legend_handles_labels(),
                       loc="upper left", frameon=False, fontsize=9)
    legend_axis.text(0, 0.5, "Width ratios are scaling diagnostics,\n"
                       "not habitat indices at a common flow.\n"
                       "Widths and flow ranges also vary with DA.",
                       transform=legend_axis.transAxes, va="top", fontsize=10)
    figure.suptitle("WUA after adjusting for predicted channel width\n"
                    "Faint points are reaches; lines show drainage-area quintile medians")
    figure.savefig(OUTPUT_DIR / "DA_WUA_width_adjustment.png", dpi=200)
    plt.close(figure)


def save_general_curves_plot(curves: dict[str, pd.DataFrame]) -> None:
    """Plot source criteria for transparent review of the mixed-source scenario."""
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), layout="constrained")
    for axis, kind, units in zip(
        axes[:2], ("depth", "velocity"), ("Water depth (m)", "Velocity (m/s)"),
    ):
        frame = curves[kind]
        axis.plot(frame[kind], frame["suit"], "o-", color="#2166ac", markersize=3)
        axis.set(xlabel=units, ylabel="Suitability (0–1)", xlim=(0, 2),
                 ylim=(-0.02, 1.05), title=f"General salmonid {kind} (Ptolemy)")
        axis.grid(alpha=0.2)
    substrate = curves["substrate"]
    axes[2].stairs(substrate["suit"],
                   np.r_[substrate["lower"].to_numpy(), substrate["upper"].iloc[-1]],
                   color="#cf6914", linewidth=2)
    axes[2].set(xscale="log", xlim=(1, 1_000), ylim=(-0.02, 1.05),
                xlabel="Representative D84 (mm)", ylabel="Suitability (0–1)",
                title="Generic rearing substrate (WDFW/Ecology)")
    axes[2].grid(alpha=0.2)
    figure.suptitle("General salmonid sensitivity criteria (two source sets)")
    figure.savefig(OUTPUT_DIR / "general_salmonid_input_curves.png", dpi=200)
    plt.close(figure)


def geometry_sensitivity(reaches: pd.DataFrame,
                         curves: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Vary width or depth alone; keep other inputs at network medians.

    These full-curve means compare native water-level samples. They do not
    represent hydraulics at an identical absolute discharge across channels.
    """
    reference = {
        name: float(reaches[field].median()) for name, field in (
            ("width_m", WIDTH_FIELD), ("depth_m", DEPTH_FIELD),
            ("slope_m_per_m", SLOPE_FIELD), ("D84_mm", D84_FIELD),
        )
    }
    rows = []
    for field, name in ((WIDTH_FIELD, "bankfull_width_m"),
                        (DEPTH_FIELD, "bankfull_depth_m")):
        values = np.quantile(reaches[field], np.linspace(0.1, 0.9, 17))
        for value in values:
            width = float(value) if field == WIDTH_FIELD else reference["width_m"]
            depth = float(value) if field == DEPTH_FIELD else reference["depth_m"]
            hydraulics = avg_hydraulics(
                slope=reference["slope_m_per_m"], bankfull_width=width,
                bankfull_depth=depth, d84_mm=reference["D84_mm"], full_curve=True,
            )
            suited = habitat(hydraulics, curves["depth"], curves["velocity"])
            rows.append({
                "varied_input": name, "varied_value_m": float(value),
                "mean_modeled_depth_m": float(hydraulics["di"].mean()),
                "depth_suitability_mean": float(suited["d.suit"].mean()),
                "velocity_suitability_mean": float(suited["v.suit"].mean()),
                "shape_factor_used": float(hydraulics.attrs["shape_factor"]),
                **{f"fixed_median_{key}": val for key, val in reference.items()},
            })
    return pd.DataFrame(rows)


def save_geometry_sensitivity_plot(sensitivity: pd.DataFrame) -> None:
    """Compare depth and velocity responses with other inputs held fixed."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True,
                                layout="constrained")
    for axis, (name, label) in zip(axes, (
        ("bankfull_width_m", "Predicted bankfull width (m)"),
        ("bankfull_depth_m", "Predicted bankfull depth (m)"),
    )):
        subset = sensitivity.loc[sensitivity["varied_input"] == name]
        axis.plot(subset["varied_value_m"], subset["depth_suitability_mean"],
                  "o-", color="#2166ac", markersize=3, linewidth=1.8,
                  label="Depth suitability")
        axis.plot(subset["varied_value_m"], subset["velocity_suitability_mean"],
                  "s--", color="#cf6914", markersize=3, linewidth=1.6,
                  label="Velocity suitability")
        field = "width_m" if name == "bankfull_width_m" else "depth_m"
        axis.axvline(subset[f"fixed_median_{field}"].iloc[0],
                     color="#777777", linestyle="--", linewidth=1)
        axis.set(xlabel=label, ylabel="Mean suitability (0–1)",
                 ylim=(0, 1))
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, loc="upper left")
    figure.suptitle("GIFT geometry sensitivity with general salmonid curves\n"
                    "One input varied from its 10th to 90th percentile; other inputs fixed at network medians")
    figure.savefig(OUTPUT_DIR / "general_salmonid_geometry_sensitivity.png", dpi=200)
    plt.close(figure)


def main() -> None:
    groups = curve_groups(CURVES_DIR)
    groups.sort(key=lambda group: group == GENERAL_GROUP)
    # All groups use exactly the same reaches and hydraulic inputs. Excluding
    # incomplete rows up front makes group-to-group comparisons consistent.
    fields = list(dict.fromkeys((
        ID_FIELD, DA_FIELD, SLOPE_FIELD, WIDTH_FIELD, DEPTH_FIELD, D84_FIELD,
        *(field for field in (SUMMER_WIDTH_FIELD,) if field is not None),
    )))
    streams = read_network_attributes(NETWORK_PATH, fields)
    for field in (DA_FIELD, SLOPE_FIELD, WIDTH_FIELD, DEPTH_FIELD, D84_FIELD):
        streams[field] = pd.to_numeric(streams[field], errors="coerce")
    if SUMMER_WIDTH_FIELD is not None:
        streams[SUMMER_WIDTH_FIELD] = pd.to_numeric(
            streams[SUMMER_WIDTH_FIELD], errors="coerce",
        )
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
        curves = read_group_curves(group)
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

    by_reach = add_driver_columns(pd.concat(results, ignore_index=True), reaches, groups)
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
    save_driver_plot(by_reach, bins, groups[0])
    save_suitability_plot(by_reach, bins, groups)
    save_width_adjusted_plot(by_reach, bins, groups)
    general_curves = read_group_curves(GENERAL_GROUP)
    save_general_curves_plot(general_curves)
    sensitivity = geometry_sensitivity(reaches, general_curves)
    sensitivity.to_csv(OUTPUT_DIR / "general_salmonid_geometry_sensitivity.csv", index=False)
    save_geometry_sensitivity_plot(sensitivity)
    print(summary.to_string(index=False))
    print("\nHSI_mean component slopes per tenfold increase in DA:")
    print(attribution.to_string(index=False))
    print(f"Saved tables and plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
