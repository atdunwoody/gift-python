"""Access to the suitability curves bundled with the original GIFT guide."""

from __future__ import annotations

from importlib import resources

import pandas as pd

_FILES = {
    "depth": "depth_suit_ptolemy.csv",
    "velocity": "velocity_suit_ptolemy.csv",
    "substrate": "substrate_suit_ptolemy.csv",
}


def _read_curve_file(kind: str) -> pd.DataFrame:
    try:
        filename = _FILES[kind]
    except KeyError as exc:
        raise ValueError(
            f"kind must be one of {sorted(_FILES)}, not {kind!r}"
        ) from exc
    data_path = resources.files("gift_habitat").joinpath("data", filename)
    with data_path.open("rb") as source:
        return pd.read_csv(source)


def available_example_curves() -> pd.DataFrame:
    """List available species and life-stage combinations by curve type."""
    records: list[pd.DataFrame] = []
    for kind in _FILES:
        curve = _read_curve_file(kind)
        combinations = curve[["species", "life_stage"]].drop_duplicates()
        combinations.insert(0, "kind", kind)
        records.append(combinations)
    return (
        pd.concat(records, ignore_index=True)
        .sort_values(["kind", "species", "life_stage"])
        .reset_index(drop=True)
    )


def load_example_curve(
    kind: str,
    *,
    species: str,
    life_stage: str,
) -> pd.DataFrame:
    """Load one curve from the original GIFT Ptolemy example data."""
    curve = _read_curve_file(kind)
    selected = curve.loc[
        (curve["species"] == species) & (curve["life_stage"] == life_stage)
    ].copy()
    if selected.empty:
        raise ValueError(
            f"No {kind!r} curve is available for species={species!r}, "
            f"life_stage={life_stage!r}"
        )
    return selected.reset_index(drop=True)

