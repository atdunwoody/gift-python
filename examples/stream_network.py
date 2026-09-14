"""Edit the settings below, then run this script to attribute a stream network."""

from pathlib import Path

import geopandas as gpd
import pandas as pd

from gift_habitat import model_reaches


# Input and output paths. Use a separate output GeoPackage.
NETWORK_PATH = Path(r"C:\path\to\stream_network.gpkg")
INPUT_LAYER = None  # Set a layer name if the GeoPackage contains several layers.
OUTPUT_PATH = Path(r"C:\path\to\gift_network_results.gpkg")
OUTPUT_LAYER = "gift_wua"
DEPTH_CURVE_PATH = Path(r"C:\path\to\depth_suitability.csv")
VELOCITY_CURVE_PATH = Path(r"C:\path\to\velocity_suitability.csv")

# Hydraulic fields: width/depth in meters, slope in m/m or ft/ft, D84 in mm.
ID_FIELD = "COMID"
SLOPE_FIELD = "slope_ft_ft"
WIDTH_FIELD = "BF_width_m"
DEPTH_FIELD = "BF_depth_m"
D84_FIELD = "D84_mm"
MAX_DEPTH_FIELD = None
SHAPE_FACTOR_FIELD = None

# WUA_auc is always calculated over the full simulated curve.
# Populate this list to ALSO calculate median WUA at biological flow fields.
# Example: BIOLOGICAL_FLOW_FIELDS = ["Q_August", "Q_September", "Q_October"]
BIOLOGICAL_FLOW_FIELDS = []
BIOLOGICAL_FLOW_UNITS = "cfs"  # Choose "cfs" or "m3/s" for all selected fields.

# Used only if the suitability CSVs contain species/life_stage columns.
SPECIES = "rainbow"
LIFE_STAGE = "parr"


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
    inputs = {p.resolve() for p in (NETWORK_PATH, DEPTH_CURVE_PATH, VELOCITY_CURVE_PATH)}
    if inputs.intersection({OUTPUT_PATH.resolve(), csv_path.resolve()}):
        raise ValueError("Use output paths that do not overwrite any input file")
    options = {"layer": INPUT_LAYER} if INPUT_LAYER else {}
    streams = gpd.read_file(NETWORK_PATH, **options)
    result = model_reaches(
        streams,
        read_curve(DEPTH_CURVE_PATH),
        read_curve(VELOCITY_CURVE_PATH),
        id_col=ID_FIELD,
        slope_col=SLOPE_FIELD,
        width_col=WIDTH_FIELD,
        depth_col=DEPTH_FIELD,
        d84_col=D84_FIELD,
        max_depth_col=MAX_DEPTH_FIELD,
        shape_factor_col=SHAPE_FACTOR_FIELD,
        flow_cols=BIOLOGICAL_FLOW_FIELDS,
        flow_units=BIOLOGICAL_FLOW_UNITS if BIOLOGICAL_FLOW_FIELDS else None,
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_file(OUTPUT_PATH, layer=OUTPUT_LAYER, driver="GPKG", index=False)
    result.drop(columns=[result.geometry.name]).to_csv(csv_path, index=False)
    print(f"Wrote {len(result):,} segments to {OUTPUT_PATH}")
    print(f"Wrote segment summary to {csv_path}")


if __name__ == "__main__":
    main()
