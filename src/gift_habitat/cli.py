"""Command-line interface for applying GIFT across a stream network."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from .batch import model_reaches
from .curves import load_example_curve


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the GIFT hydraulic-habitat model to each feature in a "
            "stream-network file and write a long CSV of WUA results."
        )
    )
    parser.add_argument("network", type=Path, help="Input spatial file")
    parser.add_argument("output_csv", type=Path, help="Output long-format CSV")
    parser.add_argument("--layer", help="Input layer name when needed")
    parser.add_argument("--id-col", default="segment_uid")
    parser.add_argument("--slope-col", default="slope")
    parser.add_argument("--width-col", default="bankfull_width_m")
    parser.add_argument("--depth-col", default="bankfull_depth_m")
    parser.add_argument("--d84-col", default="D84_mm")
    parser.add_argument("--max-depth-col")
    parser.add_argument("--shape-factor-col")
    parser.add_argument(
        "--discharge-col",
        help="Evaluate each reach at the discharge stored in this field",
    )
    parser.add_argument(
        "--q-values",
        nargs="+",
        type=float,
        help="Common discharge values in m3/s; omit for the GIFT grid",
    )
    parser.add_argument("--depth-curve", type=Path)
    parser.add_argument("--velocity-curve", type=Path)
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
    if args.discharge_col is not None and args.q_values is not None:
        raise SystemExit("Use either --discharge-col or --q-values, not both")

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

    result = model_reaches(
        reaches,
        depth_curve,
        velocity_curve,
        slope_col=args.slope_col,
        width_col=args.width_col,
        depth_col=args.depth_col,
        d84_col=args.d84_col,
        id_col=args.id_col,
        max_depth_col=args.max_depth_col,
        shape_factor_col=args.shape_factor_col,
        discharges=args.q_values,
        discharge_col=args.discharge_col,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(result):,} modeled reach-discharge rows to {args.output_csv}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

