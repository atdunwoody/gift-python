"""Run the two original GIFT model stages for one stream reach."""

from pathlib import Path

from gift_habitat import avg_hydraulics, habitat, load_example_curve


def main() -> None:
    output_dir = Path("gift_outputs")

    hydraulics = avg_hydraulics(
        slope=0.01,
        bankfull_width=10.0,
        bankfull_depth=0.5,
        max_bankfull_depth=0.75,
        d84_mm=100.0,
        output_dir=output_dir,
    )

    depth_curve = load_example_curve(
        "depth",
        species="rainbow",
        life_stage="parr",
    )
    velocity_curve = load_example_curve(
        "velocity",
        species="rainbow",
        life_stage="parr",
    )
    wua = habitat(
        hydraulics,
        depth_curve,
        velocity_curve,
        output_dir=output_dir,
    )

    hydraulics.to_csv(output_dir / "hydraulics.csv", index=False)
    wua.to_csv(output_dir / "habitat.csv", index=False)
    print(wua.head())


if __name__ == "__main__":
    main()

