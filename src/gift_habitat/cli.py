"""Command-line interface for applying GIFT across a stream network."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from .batch import group_grain_sizes, model_reaches
from .curves import load_example_curve


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the GIFT hydraulic-habitat model to each feature in a "
            "stream-network file. Write one WUA summary row per segment "
            "to CSV and an attributed GeoPackage."
        )
    )
    parser.add_argument("network", type=Path, help="Input spatial file")
    parser.add_argument("output_csv", type=Path, help="Output per-segment summary CSV")
    parser.add_argument("--layer", help="Input layer name when needed")
    parser.add_argument(
        "--output-network", type=Path,
        help="Output GeoPackage; defaults to output_csv with a .gpkg extension",
    )
    parser.add_argument("--output-layer", default="gift_wua")
    parser.add_argument(
        "--flow-cols", nargs="+",
        help="Biologically relevant flow fields to evaluate and summarize by median WUA",
    )
    parser.add_argument(
        "--flow-units", choices=("cfs", "m3/s"),
        help="Units for all --flow-cols; required when those fields are selected",
    )
    parser.add_argument(
        "--curves-csv", type=Path,
        help="Optional long CSV; defaults to the full native curves used for WUA_auc",
    )
    parser.add_argument("--id-col", default="segment_uid")
    parser.add_argument("--slope-col", default="slope")
    parser.add_argument("--width-col", default="bankfull_width_m")
    parser.add_argument("--depth-col", default="bankfull_depth_m")
    parser.add_argument("--d84-col", default="D84_mm")
    parser.add_argument("--max-depth-col")
    parser.add_argument("--shape-factor-col")
    parser.add_argument(
        "--discharge-col",
        help="For --curves-csv only: evaluate each reach at this field's flow (m3/s)",
    )
    parser.add_argument(
        "--q-values",
        nargs="+",
        type=float,
        help="For --curves-csv only: common discharge values in m3/s",
    )
    parser.add_argument("--depth-curve", type=Path)
    parser.add_argument("--velocity-curve", type=Path)
    parser.add_argument(
        "--substrate-curve", type=Path,
        help="CSV with lower, upper (mm), and suit columns; uses reach D84 by default",
    )
    parser.add_argument(
        "--grain-sizes", type=Path,
        help="CSV with one grain-size observation per row and a matching reach ID",
    )
    parser.add_argument("--grain-id-col", help="Reach ID field in --grain-sizes; defaults to --id-col")
    parser.add_argument("--grain-size-col", default="grain_size_mm")
    parser.add_argument("--species", default="rainbow")
    parser.add_argument("--life-stage", default="parr")
    return parser


def _custom_curve(
    path: Path,
    *,
    species: str,
    life_stage: str,
) -> pd.DataFrame:
    curve = pd.read_csv(path)
    if "species" in curve.columns:
        curve = curve.loc[curve["species"] == species]
    if "life_stage" in curve.columns:
        curve = curve.loc[curve["life_stage"] == life_stage]
    if curve.empty:
        raise ValueError(
            f"No rows in {path} match species={species!r} and "
            f"life_stage={life_stage!r}"
        )
    return curve.reset_index(drop=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the stream-network command-line interface."""
    args = _parser().parse_args(argv)
    if (args.depth_curve is None) != (args.velocity_curve is None):
        raise SystemExit(
            "--depth-curve and --velocity-curve must be supplied together"
        )
    if args.grain_sizes is not None and args.substrate_curve is None:
        raise SystemExit("--grain-sizes requires --substrate-curve")
    if args.discharge_col is not None and args.q_values is not None:
        raise SystemExit("Use either --discharge-col or --q-values, not both")
    if args.flow_cols and args.flow_units is None:
        raise SystemExit("--flow-cols requires --flow-units cfs or --flow-units m3/s")
    if args.flow_units is not None and not args.flow_cols:
        raise SystemExit("--flow-units requires --flow-cols")
    has_selected_q = args.discharge_col is not None or args.q_values is not None
    if has_selected_q and args.curves_csv is None:
        raise SystemExit(
            "--discharge-col and --q-values require --curves-csv. "
            "For median WUA on the network use --flow-cols and --flow-units."
        )
    output_network = args.output_network or args.output_csv.with_suffix(".gpkg")
    if output_network.suffix.lower() != ".gpkg":
        raise SystemExit("--output-network must be a .gpkg file to preserve field names")
    input_paths = {path.resolve() for path in (
        args.network, args.depth_curve, args.velocity_curve,
        args.substrate_curve, args.grain_sizes,
    ) if path is not None}
    output_paths = [path.resolve() for path in (
        args.output_csv, output_network, args.curves_csv,
    ) if path is not None]
    if len(set(output_paths)) != len(output_paths) or input_paths.intersection(output_paths):
        raise SystemExit("Use distinct output paths that do not overwrite any input file")

    try:
        import geopandas as gpd
    except ImportError as exc:
        raise SystemExit(
            "Install gift-habitat[network] to read a stream-network file"
        ) from exc

    read_options = {"layer": args.layer} if args.layer else {}
    reaches = gpd.read_file(args.network, **read_options)

    if args.depth_curve is None:
        depth_curve = load_example_curve(
            "depth",
            species=args.species,
            life_stage=args.life_stage,
        )
        velocity_curve = load_example_curve(
            "velocity",
            species=args.species,
            life_stage=args.life_stage,
        )
    else:
        depth_curve = _custom_curve(
            args.depth_curve,
            species=args.species,
            life_stage=args.life_stage,
        )
        velocity_curve = _custom_curve(
            args.velocity_curve,
            species=args.species,
            life_stage=args.life_stage,
        )

    substrate_curve = None
    gsd_by_reach = None
    if args.substrate_curve is not None:
        substrate_curve = _custom_curve(
            args.substrate_curve,
            species=args.species,
            life_stage=args.life_stage,
        )
        if args.grain_sizes is not None:
            gsd_by_reach = group_grain_sizes(
                pd.read_csv(args.grain_sizes),
                id_col=args.grain_id_col or args.id_col,
                size_col=args.grain_size_col,
            )

    model_options = dict(
        slope_col=args.slope_col,
        width_col=args.width_col,
        depth_col=args.depth_col,
        d84_col=args.d84_col,
        id_col=args.id_col,
        max_depth_col=args.max_depth_col,
        shape_factor_col=args.shape_factor_col,
        substrate_curve=substrate_curve,
        gsd_by_reach=gsd_by_reach,
    )
    result = model_reaches(
        reaches, depth_curve, velocity_curve,
        flow_cols=args.flow_cols, flow_units=args.flow_units,
        **model_options,
    )
    curves = None
    if args.curves_csv is not None:
        curves = model_reaches(
            reaches, depth_curve, velocity_curve,
            output="curves", full_curve=not has_selected_q,
            discharges=args.q_values, discharge_col=args.discharge_col,
            **model_options,
        )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_network.parent.mkdir(parents=True, exist_ok=True)
    result.to_file(output_network, layer=args.output_layer, driver="GPKG", index=False)
    result.drop(columns=[result.geometry.name]).to_csv(args.output_csv, index=False)
    if curves is not None:
        args.curves_csv.parent.mkdir(parents=True, exist_ok=True)
        curves.to_csv(args.curves_csv, index=False)
        print(f"Wrote {len(curves):,} reach-discharge rows to {args.curves_csv}")
    print(f"Wrote {len(result):,} segment summaries to {args.output_csv}")
    print(f"Wrote attributed network to {output_network} (layer: {args.output_layer})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

