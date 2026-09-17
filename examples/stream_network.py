"""Edit the settings below, then run this script to attribute a stream network."""

from pathlib import Path

import geopandas as gpd
import pandas as pd

from gift_habitat import group_grain_sizes, model_reaches


# Input and output paths. Use a separate output GeoPackage.
NETWORK_PATH = Path(r"examples\inputs\UGR_channel_geometry_predictions.gpkg")
INPUT_LAYER = None  # Set a layer name if the GeoPackage contains several layers.
OUTPUT_PATH = Path(r"examples\outputs\gift_network_results.gpkg")
OUTPUT_LAYER = "gift_wua"
DEPTH_CURVE_PATH = Path(r"examples\inputs\steelhead_spawning_depth.csv")
VELOCITY_CURVE_PATH = Path(r"examples\inputs\steelhead_spawning_velocity.csv")
SUBSTRATE_CURVE_PATH = Path(r"examples\inputs\steelhead_spawning_substrate.csv")  # Path(r"C:\path\to\substrate_suitability.csv")

# Hydraulic fields: width/depth in meters, slope in m/m or ft/ft, D84 in mm.
ID_FIELD = "COMID"
SLOPE_FIELD = "SLOPE"
WIDTH_FIELD = "bf_width_pred_m"
DEPTH_FIELD = "bf_depth_pred_m"
D84_FIELD = "D84_pred"
MAX_DEPTH_FIELD = None
SHAPE_FACTOR_FIELD = None

# WUA_auc is always calculated over the full simulated curve.
# Populate this list to ALSO calculate median WUA at biological flow fields.
# Example: BIOLOGICAL_FLOW_FIELDS = ["Q_August", "Q_September", "Q_October"]
BIOLOGICAL_FLOW_FIELDS = []
BIOLOGICAL_FLOW_UNITS = "cfs"  # Choose "cfs" or "m3/s" for all selected fields.
PROGRESS_STEP_PERCENT = 5  # Report completed reaches at roughly 5% intervals.

###############################################################################
####################### Optional Inputs Below This Line ########################
################################################################################
# Used only if the suitability CSVs contain species/life_stage columns.
SPECIES = "rainbow"
LIFE_STAGE = "parr"
# Optionally set GRAIN_SIZES_PATH to average suitability over the observed
# grain-size distribution as in the original GIFT R implementation. This CSV
# needs one observation per row, with COMID and grain_size_mm (mm) by default.
GRAIN_SIZES_PATH = None  # Path(r"C:\path\to\grain_sizes_by_reach.csv")
GRAIN_ID_FIELD = "COMID"
GRAIN_SIZE_FIELD = "grain_size_mm"

def read_curve(path: Path) -> pd.DataFrame:
    curve = pd.read_csv(path)
    if "species" in curve.columns:
        curve = curve.loc[curve["species"] == SPECIES]
    if "life_stage" in curve.columns:
        curve = curve.loc[curve["life_stage"] == LIFE_STAGE]
    if curve.empty:
        raise ValueError(f"No suitability rows match {SPECIES!r}/{LIFE_STAGE!r}: {path}")
    return curve.reset_index(drop=True)


def main() -> None:
    csv_path = OUTPUT_PATH.with_suffix(".csv")
    if OUTPUT_PATH.suffix.lower() != ".gpkg":
        raise ValueError("OUTPUT_PATH must use the .gpkg extension")
    if GRAIN_SIZES_PATH is not None and SUBSTRATE_CURVE_PATH is None:
        raise ValueError("GRAIN_SIZES_PATH requires SUBSTRATE_CURVE_PATH")
    inputs = {
        p.resolve() for p in (
            NETWORK_PATH, DEPTH_CURVE_PATH, VELOCITY_CURVE_PATH,
            SUBSTRATE_CURVE_PATH, GRAIN_SIZES_PATH,
        ) if p is not None
    }
    if inputs.intersection({OUTPUT_PATH.resolve(), csv_path.resolve()}):
        raise ValueError("Use output paths that do not overwrite any input file")
    options = {"layer": INPUT_LAYER} if INPUT_LAYER else {}
    streams = gpd.read_file(NETWORK_PATH, **options)
    substrate_curve = (
        read_curve(SUBSTRATE_CURVE_PATH) if SUBSTRATE_CURVE_PATH is not None
        else None
    )
    gsd_by_reach = (
        group_grain_sizes(
            pd.read_csv(GRAIN_SIZES_PATH),
            id_col=GRAIN_ID_FIELD,
            size_col=GRAIN_SIZE_FIELD,
        ) if GRAIN_SIZES_PATH is not None else None
    )
    if not 1 <= PROGRESS_STEP_PERCENT <= 100:
        raise ValueError("PROGRESS_STEP_PERCENT must be between 1 and 100")
    last_reported_bucket = 0

    def report_progress(completed: int, total: int) -> None:
        nonlocal last_reported_bucket
        percent = completed * 100 // total
        bucket = percent // PROGRESS_STEP_PERCENT
        if bucket > last_reported_bucket or completed == total:
            print(f"Processed {completed:,}/{total:,} reaches ({percent}%)", flush=True)
            last_reported_bucket = bucket

    result = model_reaches(
        streams,
        read_curve(DEPTH_CURVE_PATH),
        read_curve(VELOCITY_CURVE_PATH),
        id_col=ID_FIELD,
        slope_col=SLOPE_FIELD,
        width_col=WIDTH_FIELD,
        depth_col=DEPTH_FIELD,
        d84_col=D84_FIELD,
        substrate_curve=substrate_curve,
        gsd_by_reach=gsd_by_reach,
        max_depth_col=MAX_DEPTH_FIELD,
        shape_factor_col=SHAPE_FACTOR_FIELD,
        flow_cols=BIOLOGICAL_FLOW_FIELDS,
        flow_units=BIOLOGICAL_FLOW_UNITS if BIOLOGICAL_FLOW_FIELDS else None,
        progress_callback=report_progress,
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_file(OUTPUT_PATH, layer=OUTPUT_LAYER, driver="GPKG", index=False)
    result.drop(columns=[result.geometry.name]).to_csv(csv_path, index=False)
    print(f"Wrote {len(result):,} segments to {OUTPUT_PATH}")
    print(f"Wrote segment summary to {csv_path}")


if __name__ == "__main__":
    main()
