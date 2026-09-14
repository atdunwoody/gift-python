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

D84 affects the hydraulic calculation through Ferguson's flow-resistance relation. It does not directly provide substrate suitability.

Substrate suitability requires:

1. A substrate suitability curve.
2. An observed or simulated grain-size distribution.

The substrate curve must contain `lower`, `upper`, and `suit` columns:

```csv
lower,upper,suit
0.0,2.0,0.0
2.0,16.0,0.4
16.0,64.0,1.0
64.0,256.0,0.7
256.0,1000.0,0.2
```

Example:

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

If either the substrate curve or grain-size distribution is omitted, substrate suitability defaults to 1.0.

## Model a stream network

`model_reaches()` now returns one row per input segment by default. It preserves the input attributes, geometry, CRS, row order, and index, and adds `WUA_auc`, the area under the entire simulated WUA-discharge curve. Repeated segment IDs remain separate features; no join is needed.

For a script with editable settings, use `examples/stream_network.py`. Set the input paths and hydraulic field names. Leave `BIOLOGICAL_FLOW_FIELDS = []` for full-curve integration only. To add the biological-flow metric, populate that list and set `BIOLOGICAL_FLOW_UNITS` to `"cfs"` or `"m3/s"`.

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

# One feature per original stream segment, with WUA_auc attributed directly.
results.to_file(OUTPUT_PATH, layer="gift_wua", driver="GPKG", index=False)
results.drop(columns=[results.geometry.name]).to_csv(
    OUTPUT_PATH.with_suffix(".csv"), index=False,
)
```

Replace the example field names with the corresponding fields in the network. Each run uses one selected species/life-stage combination from the supplied suitability curves. Filter multi-species suitability tables before calling the API, as shown above; the CLI and example script can filter them for you.

### Full-curve WUA metric

For each segment, `WUA_auc` is calculated by trapezoidal integration of WUA against discharge in m³/s:

```text
WUA_auc = sum((Q[i+1] - Q[i]) * (WUA[i] + WUA[i+1]) / 2)
```

The calculation uses all 981 native water-level simulations and their actual discharge spacing, sorted by discharge. It includes both endpoints, avoiding truncation by the original fixed discharge grid. The modeled domain begins at the flow corresponding to 2% of the modeled maximum bankfull depth and ends at the simulated bankfull discharge. “Entire curve” means this full modeled domain, not all possible discharges. No curve is extrapolated below the first simulation or above bankfull, and no assumed origin point is added to the integral.

WUA in this implementation is usable area per unit reach length, in m²/m. Therefore, `WUA_auc` has units **(m²/m) × (m³/s)**. It is not a mean WUA, is not normalized by the discharge range, is not multiplied by segment length, and is not weighted by flow duration. The integration bounds are recorded because the modeled flow range varies among segments.

### Add WUA at biologically relevant flows

Select any number of flow fields and explicitly specify their common units. This option adds a second metric while retaining `WUA_auc`:

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

Missing, nonnumeric, negative, non-finite, or out-of-range positive flows are excluded. `WUA flow fields excluded` records each omitted field and its reason, and a summary warning identifies runs with omissions. The median uses the remaining fields; if none can be evaluated, `WUA by flowrate` is null. A specified zero flow is treated as a dry channel and contributes WUA = 0. Positive flows outside the simulated range are not clamped or extrapolated. This zero-flow convention affects only the biological-flow metric, not the full-curve integral.

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
    full_curve=True,  # Export the native curve used for WUA_auc.
)
curves.to_csv("network_wua_curves.csv", index=False)
```

In curves mode, omit `full_curve=True` to use the original GIFT discharge grid, or pass `discharges=[0.1, 0.25, 0.5]` or `discharge_col="Q_m3s"` to evaluate specified flows in m³/s. These discharge selectors are available only in curves mode and cannot be combined with `full_curve=True`. Use `flow_cols` and `flow_units` for the optional median metric in summary mode. Curve output remains a long table, so repeated reach IDs are not unique join keys.

The single-reach functions retain their previous default numerical behavior. `avg_hydraulics(..., full_curve=True)` is also available for native hydraulic output, and `integrate_wua_curve(curve)` can integrate a supplied WUA table.

## Command-line network workflow

The package installs the `gift-network` command. The positional output path is now a **per-segment summary CSV**. The command also writes a GeoPackage with the same stem, unless `--output-network` supplies another `.gpkg` path. The default output layer is `gift_wua`; use `--output-layer` to change it.

PowerShell example for the full-curve metric:

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

Use `--flow-units "m3/s"` for fields in cubic meters per second. Quote field names containing spaces.

To additionally save the native WUA curves, append `--curves-csv "C:\path\to\network_wua_curves.csv"`. The legacy `--q-values` and `--discharge-col` arguments now apply only to this optional curve CSV and require `--curves-csv`. They do not change the full-curve integral. The prior one-flow-per-reach use case can instead be summarized on the network with `--flow-cols Q_m3s --flow-units "m3/s"`.

If the `gift-network` command is not recognized, use:

```powershell
python -m gift_habitat.cli --help
```

## Output fields

### Stream-network summary

All input segment attributes and geometries are retained in the GeoPackage; the CSV omits geometry. Use GeoPackage output to retain the exact field names, including spaces. Existing values in the output metric fields are replaced on rerun.

| Field | Description | Unit |
| ----- | ----------- | ---- |
| `WUA_auc` | Area under the entire native WUA-discharge curve | (m²/m) × (m³/s) |
| `WUA_Q_min_m3s` | Lower integration bound | m³/s |
| `WUA_Q_max_m3s` | Upper integration bound, simulated bankfull flow | m³/s |
| `WUA_Q_count` | Number of native curve points integrated | Count |
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

The Python implementation is tested against the numerical results published in the original GIFT documentation. Additional tests check analytical trapezoidal integration, full-range endpoints, per-field median WUA, cfs conversion, missing and unsupported flows, repeated segment IDs, and GeoPackage/CSV output with exact field names. Install the network extra to run the GeoPackage checks.

Run the test suite from the source directory:

```powershell
python -m unittest discover -s tests -v
```

## Limitations

* GIFT produces reach-averaged hydraulic and habitat estimates.
* It does not represent individual pools, riffles, or other channel units.
* Results depend on the accuracy of bankfull geometry, slope, D84, discharge, and suitability curves.
* The full-curve integral is limited to the simulated positive-flow range through bankfull. Unsupported positive biological flows are excluded and reported.
* WUA results should be interpreted as model-based habitat indices rather than direct measurements of habitat use.

## Source and license

This package is a Python port of GIFT developed by Stefan Gronsdahl:

https://github.com/SGronsdahl/GIFT

The port follows the GNU General Public License version 3 or later. See `LICENSE` and `NOTICE.md` for details.
