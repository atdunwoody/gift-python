# GIFT Habitat for Python

Python implementation of the [Geomorphic Instream Flow Tool (GIFT)](https://github.com/SGronsdahl/GIFT) for simulating reach-averaged hydraulics and weighted usable habitat area.

The package reproduces the two principal functions in the original R package:

| R function        | Python function    | Purpose                                                                   |
| ----------------- | ------------------ | ------------------------------------------------------------------------- |
| `AvgHydraulics()` | `avg_hydraulics()` | Simulate mean hydraulic conditions over a range of subbankfull discharges |
| `Habitat()`       | `habitat()`        | Apply habitat suitability curves and calculate weighted usable area       |
| Not included      | `model_reaches()`  | Apply GIFT to every feature in a stream network                           |

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
python -m pip install dist\gift_habitat-0.1.0-py3-none-any.whl
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
| `discharge`            | Stream discharge                 | m³/s          |
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

`model_reaches()` accepts a pandas DataFrame or GeoDataFrame. It returns a long table containing one row for each reach-discharge combination.

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

# Use this conversion - HIQ1_5_HIST is in cfs.
streams["Q_m3s"] = streams["HIQ1_5_HIST"] * 0.0283168466

results = model_reaches(
    streams,
    depth_curve,
    velocity_curve,
    slope_col="slope_ft_ft",
    width_col="BF_width_m",
    depth_col="BF_depth_m",
    d84_col="D84_mm",
    id_col="COMID",
    discharge_col="Q_m3s",
)

output = streams.merge(
    results,
    left_on="COMID",
    right_on="reach_id",
    how="left",
)

output.to_file(
    OUTPUT_PATH,
    layer="gift_wua",
    driver="GPKG",
)
```

Replace the example field names with the corresponding fields in the input network.

### Model common discharges across the network

To evaluate the same flows for every reach:

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
    discharges=[0.1, 0.25, 0.5],
)
```

This produces multiple rows per reach. Save the results as a long-format CSV or select a single discharge before joining them back to the spatial network:

```python
results.to_csv("network_wua_curves.csv", index=False)
```

## Command-line network workflow

The package installs the `gift-network` command.

Example PowerShell command:

```powershell
gift-network "C:\path\to\stream_network.gpkg" "C:\path\to\network_wua.csv" `
  --id-col COMID `
  --slope-col slope_ft_ft `
  --width-col BF_width_m `
  --depth-col BF_depth_m `
  --d84-col D84_mm `
  --discharge-col Q_m3s `
  --depth-curve "C:\path\to\depth_suitability.csv" `
  --velocity-curve "C:\path\to\velocity_suitability.csv" `
  --species chinook `
  --life-stage juvenile
```

If the `gift-network` command is not recognized, use:

```powershell
python -m gift_habitat.cli --help
```

## Output fields

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

The Python implementation is tested against the numerical results published in the original GIFT documentation.

Run the test suite from the source directory:

```powershell
python -m unittest discover -s tests -v
```

## Limitations

* GIFT produces reach-averaged hydraulic and habitat estimates.
* It does not represent individual pools, riffles, or other channel units.
* Results depend on the accuracy of bankfull geometry, slope, D84, discharge, and suitability curves.
* Discharges above the simulated bankfull range are omitted.
* WUA results should be interpreted as model-based habitat indices rather than direct measurements of habitat use.

## Source and license

This package is a Python port of GIFT developed by Stefan Gronsdahl:

https://github.com/SGronsdahl/GIFT

The port follows the GNU General Public License version 3 or later. See `LICENSE` and `NOTICE.md` for details.
