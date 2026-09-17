# GIFT Habitat for Python

Python implementation of the [Geomorphic Instream Flow Tool (GIFT)](https://github.com/SGronsdahl/GIFT) for simulating reach-averaged hydraulics and weighted usable habitat area.

The package reproduces the two principal functions in the original R package:

| R function        | Python function    | Purpose                                                                   |
| ----------------- | ------------------ | ------------------------------------------------------------------------- |
| `AvgHydraulics()` | `avg_hydraulics()` | Simulate mean hydraulic conditions over a range of subbankfull discharges |
| `Habitat()`       | `habitat()`        | Apply habitat suitability curves and calculate weighted usable area       |
| Not included      | `model_reaches()`  | Attribute full-curve and optional flow-specific WUA metrics to each segment |

## Requirements

* Python 3.10 or later
* NumPy
* pandas
* Matplotlib, optional for plots
* GeoPandas and Pyogrio, optional for stream-network files

## Installation

Activate the conda environment in which the package will be used:

```powershell
conda activate gift
```

From the extracted `gift-python` directory, install the package with plotting and spatial-network support:

```powershell
python -m pip install -e ".[plot,network]"
```

Alternatively, install the included wheel:

```powershell
python -m pip install "dist\gift_habitat-0.2.0-py3-none-any.whl[plot,network]"
```

Verify the installation:

```powershell
python -c "import gift_habitat; print(gift_habitat.__version__)"
```

## Input units

GIFT expects the following units:

| Input                  | Description                      | Required unit |
| ---------------------- | -------------------------------- | ------------- |
| `slope`                | Channel gradient                 | m/m or ft/ft  |
| `bankfull_width`       | Mean bankfull width              | m             |
| `bankfull_depth`       | Mean bankfull depth              | m             |
| `max_bankfull_depth`   | Maximum bankfull depth, optional | m             |
| `d84_mm`               | Bed-material D84                 | mm            |
| `discharge`            | Single-reach or curve-table discharge | m³/s       |
| Network `flow_cols`     | Biological flow fields           | cfs or m³/s, explicitly selected |
| Depth suitability      | Suitable water depth             | m             |
| Velocity suitability   | Suitable water velocity          | m/s           |
| Substrate class bounds | Grain-size class limits          | mm            |

Slope is dimensionless, so a field expressed in ft/ft can be used without conversion.

## Habitat suitability curves

Depth and velocity suitability curves are supplied as pandas DataFrames, usually read from CSV files.

### Depth suitability curve

Required columns:

```csv
depth,suit
0.00,0.00
0.10,0.25
0.25,0.75
0.50,1.00
0.75,0.50
1.00,0.00
```

* `depth` is water depth in meters.
* `suit` is habitat suitability from 0 to 1.

### Velocity suitability curve

Required columns:

```csv
velocity,suit
0.00,0.00
0.10,0.40
0.30,1.00
0.60,0.75
1.00,0.20
1.50,0.00
```

* `velocity` is water velocity in m/s.
* `suit` is habitat suitability from 0 to 1.

Suitability values are assigned using the nearest curve coordinate, consistent with the original R implementation. Values are not linearly interpolated.

If a CSV contains several species or life stages, filter it before running the model:

```python
import pandas as pd

depth_curve = pd.read_csv("depth_suitability.csv")

depth_curve = depth_curve.loc[
    (depth_curve["species"] == "chinook")
    & (depth_curve["life_stage"] == "juvenile")
].copy()
```

## Model a single reach

```python
from pathlib import Path

import pandas as pd

from gift_habitat import avg_hydraulics, habitat


OUTPUT_DIR = Path("gift_outputs")

depth_curve = pd.read_csv("depth_suitability.csv")
velocity_curve = pd.read_csv("velocity_suitability.csv")

hydraulics = avg_hydraulics(
    slope=0.01,
    bankfull_width=10.0,
    bankfull_depth=0.5,
    max_bankfull_depth=0.75,
    d84_mm=100.0,
    output_dir=OUTPUT_DIR,
)

wua = habitat(
    hydraulics,
    depth_curve,
    velocity_curve,
    output_dir=OUTPUT_DIR,
)

hydraulics.to_csv(OUTPUT_DIR / "hydraulics.csv", index=False)
wua.to_csv(OUTPUT_DIR / "weighted_usable_area.csv", index=False)

print(wua.head())
```

If `max_bankfull_depth` is unavailable, omit it:

```python
hydraulics = avg_hydraulics(
    slope=0.01,
    bankfull_width=10.0,
    bankfull_depth=0.5,
    d84_mm=100.0,
)
```

The model will derive the channel-shape factor from the bankfull width-to-depth ratio.

### Model selected discharges

Use `discharges` to evaluate only specified flows:

```python
hydraulics = avg_hydraulics(
    slope=0.01,
    bankfull_width=10.0,
    bankfull_depth=0.5,
    d84_mm=100.0,
    discharges=[0.1, 0.25, 0.5],
)
```

If `discharges` is omitted, the model uses the original GIFT discharge grid and returns all modeled flows below bankfull discharge.

## Use the bundled example curves

The package includes the Ptolemy suitability curves distributed with the original GIFT documentation.

```python
from gift_habitat import load_example_curve

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
```

List the available curve combinations:

```python
from gift_habitat import available_example_curves

print(available_example_curves())
```

## Include substrate suitability

D84 affects the hydraulic calculation through Ferguson's flow-resistance
relation. It can also serve as a representative grain size for a direct lookup
in a substrate suitability curve. The direct lookup assigns the suitability of
the class containing D84 to the entire reach. The [original R `Habitat()`](https://github.com/SGronsdahl/GIFT/blob/main/R/Habitat.R)
instead averages class suitabilities over a supplied grain-size distribution
(`gsd`). Both methods are supported, and they can give different results.

The substrate curve must contain `lower`, `upper`, and `suit` columns:

```csv
lower,upper,suit
0.0,2.0,0.0
2.0,16.0,0.4
16.0,64.0,1.0
64.0,256.0,0.7
256.0,1000.0,0.2
```

For a single reach, use D84 (mm) as the representative substrate size:

```python
import pandas as pd
from gift_habitat import habitat

substrate_curve = pd.read_csv("substrate_suitability.csv")
wua = habitat(
    hydraulics, depth_curve, velocity_curve,
    substrate_curve=substrate_curve,
    substrate_size_mm=100.0,
)
```

The class includes its lower bound and excludes its upper bound. A D84 of
64 mm, for example, uses the suitability of the 64–256 mm class. If D84 falls
in no class or more than one class, the lookup raises an error.

To use the original R distribution-weighted method, supply grain-size
observations instead of `substrate_size_mm`:

```python
import pandas as pd

from gift_habitat import habitat

substrate_curve = pd.read_csv("substrate_suitability.csv")
grain_sizes = pd.read_csv("pebble_count.csv")["grain_size_mm"].to_numpy()

wua = habitat(
    hydraulics,
    depth_curve,
    velocity_curve,
    substrate_curve=substrate_curve,
    gsd=grain_sizes,
)
```

For a single reach, `habitat()` uses substrate suitability of 1.0 when all
substrate inputs are omitted. For network runs, supplying only a substrate
curve applies the D84 lookup to each reach. Supply grain-size observations as
well to use the distribution-weighted method instead.

For a stream network with grain-size distributions, supply observations for
**each reach**. The CSV has one measured or simulated grain-size observation
(mm) per row (abbreviated example):

```csv
COMID,grain_size_mm
1001,8
1001,12
1001,30
1002,62
1002,105
```

Both CSVs can include additional columns; suitability rows can be filtered by
`species` and `life_stage`. For a GSD, each observation receives the suitability
of its class; the classified observations' mean is the reach's `s.suit`. For a
D84 lookup, `s.suit` is the value of the one class containing D84. Both methods
multiply depth suitability, velocity suitability, and wetted width by
`s.suit` at every modeled flow.

```python
from gift_habitat import group_grain_sizes, model_reaches

substrate_curve = pd.read_csv("substrate_suitability.csv")
gsd_by_reach = group_grain_sizes(
    pd.read_csv("grain_sizes_by_reach.csv"),
    id_col="COMID",
    size_col="grain_size_mm",
)
results = model_reaches(
    streams, depth_curve, velocity_curve,
    slope_col="slope_ft_ft", width_col="BF_width_m",
    depth_col="BF_depth_m", d84_col="D84_mm", id_col="COMID",
    substrate_curve=substrate_curve, gsd_by_reach=gsd_by_reach,
)
print(results[["COMID", "s.suit", "WUA_mean", "WUA_max", "WUA_Q_at_max_m3s"]])
```

Omit `gsd_by_reach` in this example to use `D84_mm` for the direct class lookup.
For a GSD, repeated reach IDs share the same observations. Every modeled reach
must have samples, or the run stops with its ID and row number. `gsd=` can be
used when one grain-size distribution should apply to all reaches.

## Model a stream network

`model_reaches()` returns one row per input segment by default. It preserves the input attributes, geometry, CRS, row order, and index, and adds the arithmetic mean and maximum WUA over the full native simulated WUA-discharge curve. Repeated segment IDs remain separate features; no join is needed.

For a script with editable settings, use `examples/stream_network.py`. Set the input paths and hydraulic field names. Leave `BIOLOGICAL_FLOW_FIELDS = []` to report only the full-curve mean and maximum WUA statistics. To add the biological-flow metric, populate that list and set `BIOLOGICAL_FLOW_UNITS` to `"cfs"` or `"m3/s"`.
If a mapped hydraulic value is missing or non-finite, the feature remains in the output with null model metrics. A summary warning gives the number skipped. The example reports percentage progress every 5% (configurable with `PROGRESS_STEP_PERCENT`). API callers can pass `progress_callback`, which receives `(processed_reaches, total_reaches)` after each feature.

```python
from pathlib import Path

import geopandas as gpd
import pandas as pd

from gift_habitat import model_reaches

NETWORK_PATH = Path(r"C:\path\to\stream_network.gpkg")
OUTPUT_PATH = Path(r"C:\path\to\gift_results.gpkg")

streams = gpd.read_file(NETWORK_PATH)
depth_curve = pd.read_csv(r"C:\path\to\depth_suitability.csv")
velocity_curve = pd.read_csv(r"C:\path\to\velocity_suitability.csv")

results = model_reaches(
    streams,
    depth_curve,
    velocity_curve,
    slope_col="slope_ft_ft",
    width_col="BF_width_m",
    depth_col="BF_depth_m",
    d84_col="D84_mm",
    id_col="COMID",
)

# One feature per original stream segment, with full-curve WUA statistics attributed directly.
results.to_file(OUTPUT_PATH, layer="gift_wua", driver="GPKG", index=False)
results.drop(columns=[results.geometry.name]).to_csv(
    OUTPUT_PATH.with_suffix(".csv"), index=False,
)
```

Replace the example field names with the corresponding fields in the network. Each run uses one selected species/life-stage combination from the supplied suitability curves. Filter multi-species suitability tables before calling the API, as shown above; the CLI and example script can filter them for you.

### Full-curve WUA metrics

For each segment, the summary uses all 981 native water-level simulations across the full modeled discharge domain. It reports:

- `WUA_mean`: arithmetic mean of the modeled WUA values. This is a simple mean across native simulation points and is not discharge-weighted.
- `WUA_max`: maximum modeled WUA.
- `WUA_Q_min_m3s` and `WUA_Q_max_m3s`: minimum and maximum modeled discharge, which define the discharge range summarized.
- `WUA_Q_at_max_m3s`: lowest modeled discharge at which `WUA_max` occurs.
- `WUA_Q_count`: number of native curve points included in the summary.

The modeled domain begins at the flow corresponding to 2% of the modeled maximum bankfull depth and ends at the simulated bankfull discharge. No curve is extrapolated below the first simulation or above bankfull. WUA is usable area per unit reach length in m²/m.

### Add WUA at biologically relevant flows

Select any number of flow fields and explicitly specify their common units. This option adds a separate biological-flow metric while retaining the full-curve mean and maximum WUA statistics:

```python
results = model_reaches(
    streams,
    depth_curve,
    velocity_curve,
    slope_col="slope_ft_ft",
    width_col="BF_width_m",
    depth_col="BF_depth_m",
    d84_col="D84_mm",
    id_col="COMID",
    flow_cols=["Q_August", "Q_September", "Q_October"],
    flow_units="cfs",  # Use "m3/s" for cubic meters per second.
)

results.to_file(OUTPUT_PATH, layer="gift_wua", driver="GPKG", index=False)
```

For each segment, the model converts the selected discharges to m³/s when necessary, interpolates the hydraulic variables to those exact discharges, calculates WUA separately at each discharge, and then takes the median of those WUA values. It does **not** calculate WUA at the median discharge or interpolate values from the WUA curve. The conversion is `1 cfs = 0.028316846592 m³/s`.

Each selected field has equal weight. With one valid field, the metric is its WUA. With an even number of fields, the median is the mean of the two middle WUA values. Different fields containing the same flow each contribute once. Duplicate field names are rejected.

The output field is named exactly **`WUA by flowrate`**. The second field, **`WUA flow fields`**, records the contributing input field names as a JSON list, for example `["Q_August", "Q_September"]`.

Missing, nonnumeric, negative, non-finite, or out-of-range positive flows are excluded. `WUA flow fields excluded` records each omitted field and its reason, and a summary warning identifies runs with omissions. The median uses the remaining fields; if none can be evaluated, `WUA by flowrate` is null. A specified zero flow is treated as a dry channel and contributes WUA = 0. Positive flows outside the simulated range are not clamped or extrapolated. This zero-flow convention affects only the biological-flow metric, not the full-curve summary statistics.

All selected fields must use the same declared units. Units must be supplied whenever `flow_cols` is nonempty. When flow evaluation is disabled, any prior optional WUA flow metrics are removed from the returned result so they are not mistaken for results of the current run.

### Retain discharge-by-discharge results

For the earlier long-format API behavior, explicitly set `output="curves"`:

```python
curves = model_reaches(
    streams,
    depth_curve,
    velocity_curve,
    slope_col="slope_ft_ft",
    width_col="BF_width_m",
    depth_col="BF_depth_m",
    d84_col="D84_mm",
    id_col="COMID",
    output="curves",
    full_curve=True,  # Export the native curve used for the summary statistics.
)
curves.to_csv("network_wua_curves.csv", index=False)
```

In curves mode, omit `full_curve=True` to use the original GIFT discharge grid, or pass `discharges=[0.1, 0.25, 0.5]` or `discharge_col="Q_m3s"` to evaluate specified flows in m³/s. These discharge selectors are available only in curves mode and cannot be combined with `full_curve=True`. Use `flow_cols` and `flow_units` for the optional median metric in summary mode. Curve output remains a long table, so repeated reach IDs are not unique join keys.

The single-reach functions retain their previous default numerical behavior. `avg_hydraulics(..., full_curve=True)` is also available for native hydraulic output, and `integrate_wua_curve(curve)` can integrate a supplied WUA table.

## Command-line network workflow

The package installs the `gift-network` command. The positional output path is now a **per-segment summary CSV**. The command also writes a GeoPackage with the same stem, unless `--output-network` supplies another `.gpkg` path. The default output layer is `gift_wua`; use `--output-layer` to change it.

PowerShell example for the full-curve summary metrics:

```powershell
gift-network "C:\path\to\stream_network.gpkg" "C:\path\to\network_wua.csv" `
  --id-col COMID `
  --slope-col slope_ft_ft `
  --width-col BF_width_m `
  --depth-col BF_depth_m `
  --d84-col D84_mm `
  --depth-curve "C:\path\to\depth_suitability.csv" `
  --velocity-curve "C:\path\to\velocity_suitability.csv" `
  --species chinook `
  --life-stage juvenile
```

To also calculate the biological-flow metric, append these arguments to the command:

```powershell
--flow-cols Q_August Q_September Q_October --flow-units cfs
```

To use each reach's D84 as its substrate class in both WUA metrics, also pass:

```powershell
--substrate-curve "C:\path\to\substrate_suitability.csv"
```

To calculate the original GSD-weighted score instead, additionally pass
`--grain-sizes "C:\path\to\grain_sizes_by_reach.csv"`. The grain-size CSV uses
`--id-col` as its reach ID field by default. Set
`--grain-id-col` or `--grain-size-col` if its headers differ. The script in
`examples/stream_network.py` provides the same inputs as editable settings
for direct use in VSCode.

Use `--flow-units "m3/s"` for fields in cubic meters per second. Quote field names containing spaces.

To additionally save the native WUA curves, append `--curves-csv "C:\path\to\network_wua_curves.csv"`. The legacy `--q-values` and `--discharge-col` arguments now apply only to this optional curve CSV and require `--curves-csv`. They do not change the full-curve summary statistics. The prior one-flow-per-reach use case can instead be summarized on the network with `--flow-cols Q_m3s --flow-units "m3/s"`.

If the `gift-network` command is not recognized, use:

```powershell
python -m gift_habitat.cli --help
```

## Output fields

### Stream-network summary

All input segment attributes and geometries are retained in the GeoPackage; the CSV omits geometry. Use GeoPackage output to retain the exact field names, including spaces. Existing values in the output metric fields are replaced on rerun.

| Field | Description | Unit |
| ----- | ----------- | ---- |
| `WUA_mean` | Arithmetic mean WUA across the full native curve | m²/m |
| `WUA_max` | Maximum WUA across the full native curve | m²/m |
| `s.suit` | Substrate suitability applied to the reach (1.0 without substrate inputs) | Dimensionless |
| `WUA_Q_min_m3s` | Minimum modeled discharge | m³/s |
| `WUA_Q_max_m3s` | Maximum modeled discharge, simulated bankfull flow | m³/s |
| `WUA_Q_at_max_m3s` | Lowest modeled discharge at which maximum WUA occurs | m³/s |
| `WUA_Q_count` | Number of native curve points summarized | Count |
| `WUA by flowrate` | Optional median WUA at the selected flow fields | m²/m |
| `WUA flow fields` | Optional JSON list of contributing flow field names, in selection order | Text |
| `WUA flow units` | Optional declared units of the input flow fields | `cfs` or `m3/s` |
| `WUA flow fields excluded` | Optional JSON mapping of omitted fields to reasons; `{}` if none | Text |

### Hydraulic results

| Field | Description                 | Unit |
| ----- | --------------------------- | ---- |
| `Q`   | Discharge                   | m³/s |
| `Ai`  | Wetted cross-sectional area | m²   |
| `Wi`  | Wetted width                | m    |
| `di`  | Mean water depth            | m    |
| `Ui`  | Mean water velocity         | m/s  |

### Habitat results

| Field    | Description                                | Unit          |
| -------- | ------------------------------------------ | ------------- |
| `Q`      | Discharge                                  | m³/s          |
| `d.suit` | Depth suitability                          | Dimensionless |
| `v.suit` | Velocity suitability                       | Dimensionless |
| `s.suit` | Substrate suitability                      | Dimensionless |
| `w`      | Wetted width                               | m             |
| `WUA`    | Weighted usable area per unit reach length | m²/m          |

When `output_dir` is supplied, the single-reach functions also create:

* `channel_xs.csv`
* `channel_xs.jpeg`
* `WUA_Q.jpeg`

## R-compatible function names

The package includes wrappers using the original R function names and parameter names:

```python
from gift_habitat import AvgHydraulics, Habitat

hydraulics = AvgHydraulics(
    S=0.01,
    wb=10.0,
    db=0.5,
    db_max=0.75,
    D84=100.0,
    xs_output=False,
)

wua = Habitat(
    hydraulics=hydraulics,
    d_curve=depth_curve,
    v_curve=velocity_curve,
    wua_output=False,
)
```

The lower-case Python functions are recommended for new analyses.

## Validation

The Python implementation is tested against the numerical results published in the original GIFT documentation. Additional tests check the full-range mean and maximum WUA summaries, discharge-at-maximum reporting, full-range endpoints, per-field median WUA, cfs conversion, missing and unsupported flows, repeated segment IDs, and GeoPackage/CSV output with exact field names. The standalone integration utility retains its analytical integration tests. Install the network extra to run the GeoPackage checks.

Run the test suite from the source directory:

```powershell
python -m unittest discover -s tests -v
```

## Limitations

* GIFT produces reach-averaged hydraulic and habitat estimates.
* It does not represent individual pools, riffles, or other channel units.
* Results depend on the accuracy of bankfull geometry, slope, D84, discharge, and suitability curves.
* Full-curve summary statistics are limited to the simulated positive-flow range through bankfull. Unsupported positive biological flows are excluded and reported.
* WUA results should be interpreted as model-based habitat indices rather than direct measurements of habitat use.

## Source and license

This package is a Python port of GIFT developed by Stefan Gronsdahl:

https://github.com/SGronsdahl/GIFT

The port follows the GNU General Public License version 3 or later. See `LICENSE` and `NOTICE.md` for details.
