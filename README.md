# HEIMDALL / BIFROST-HEIMDALL

**Spectral Solar and Lunar Irradiance Model**

*Dr Stacey Connan-McGinty*

This repository contains two related models. The irradiance and surface transmission components are globally applicable; however the bio-optical model used for water column attenuation (Kostakis et al. 2020) is derived from Arctic conditions and is most appropriate for Arctic applications.


- **HEIMDALL** — computes spectrally resolved solar and lunar irradiance above the surface, and irradiance entering the ocean through open water. HEIMDALL is a published model.
- **BIFROST-HEIMDALL** — extends HEIMDALL by adding transmission through snow and sea ice, enabling irradiance estimates below a sea ice cover. BIFROST-HEIMDALL is currently under review.

Which model is active is controlled by settings in `config.yaml` — see [Model output level](#model-output-level) below.

Both models are designed for ecological and oceanographic applications where spectral light availability — particularly photosynthetically active radiation (PAR) — is needed across the Arctic domain.

**Spectral range:** 405–695 nm (30 wavelengths, 10 nm steps)  
**Grid:** User-defined locations or a regular spatial grid at any latitude; TOPAZ4 sea ice data available for 50–90°N  
**Time axis:** Regular time grid at any interval, or explicit timestamp list

---

## What the model computes

- Solar and lunar direct and diffuse spectral irradiance [W m⁻² nm⁻¹]
- Spectral PAR [µmol photons m⁻² s⁻¹ nm⁻¹] and broadband PAR [µmol photons m⁻² s⁻¹]
- Broadband irradiance [W m⁻²]
- Solar and lunar zenith angles [degrees]
- Cloud cover fraction
- Full water column depth profile using a bio-optical attenuation model
- Optionally: irradiance transmitted through snow, sea ice, and open water (BIFROST-HEIMDALL)

---

## File structure

```
HEIMDALL/
├── main.py                        — Master orchestration script
├── config.yaml                    — User configuration (edit this to run the model)
├── config_loader.py               — YAML config parsing and validation
├── solar_lunar_positions.py       — Solar/lunar zenith angles via Skyfield (DE421)
├── load_lut_and_interpolate.py    — Irradiance LUT loading and interpolation
├── lunar_irradiance.py            — Lunar spectral irradiance scaling
├── irradiance_conversions.py      — PAR and broadband irradiance conversions
├── fetch_cloud_cover.py           — ERA5 cloud cover download from CDS (used when cloud_mode: 2 in config.yaml)
├── fetch_ice_data.py              — TOPAZ4 sea ice data from Copernicus Marine (used when above_surface_only: false and open_water_only: false in config.yaml)
├── compute_transmission.py        — BIFROST surface transmission (snow/ice/open water)
├── compute_depth_attenuation.py   — Beer-Lambert water column attenuation
├── save_to_netcdf.py              — NetCDF output via xarray
├── requirements.txt               — Python dependencies
├── CDS_SETUP.md                   — CDS account and credentials setup guide (MUST be completed before running cloud_mode: 2)
├── COPERNICUS_MARINE_SETUP.md     — Copernicus Marine credentials setup guide (MUST be completed before computing transmission through snow and sea ice)
├── LUT/                           — Spectral lookup tables (.mat files)
└── model_outputs/                 — Model run output files (.nc) saved here (created automatically if it does not exist)
```

---

## Installation

**1. Ensure Python 3.13 or later is installed.**

**2. Create and activate a virtual environment:**

```bash
python -m venv .HEIMDALL_venv
source .HEIMDALL_venv/bin/activate      # macOS / Linux
.HEIMDALL_venv\Scripts\activate         # Windows
```

**3. Install dependencies:**

```bash
pip install -r requirements.txt
```

**4. Set up external data credentials** (only needed for certain model levels — see below):

- Cloud cover (`cloud_mode: 2`): see [CDS_SETUP.md](CDS_SETUP.md)
- Sea ice and depth profile (Levels 2 and 3): see [COPERNICUS_MARINE_SETUP.md](COPERNICUS_MARINE_SETUP.md)

---

## Configuration

All model settings are controlled through `config.yaml`. The key options are described below.

### Time

```yaml
time:
  use_custom: false
  start: "2021-07-15 00:00:00"
  end:   "2021-07-15 23:00:00"
  resolution_minutes: 60
```

Set `start` and `end` to the same value for a single-timestamp run. For an explicit list of timestamps (e.g. from instrument data), set `use_custom: true` and provide a `timestamps` list.

### Location

```yaml
# Option A — explicit list
locations:
  - latitude: 75.0
    longitude: 0.0

# Option B — regular spatial grid
spatial_grid:
  use_grid: true

  lat_start: 66.0
  lat_end:   90.0
  lat_step:  2.0

  lon_start: -180.0
  lon_end:    180.0
  lon_step:   2.0
```

### Cloud mode

```yaml
cloud_mode: 0   # 0 = clear sky, 1 = overcast, 2 = ERA5 (requires CDS credentials)
```

### Model output level

The table below shows which model is active for each configuration:

| Level | `above_surface_only` | `open_water_only` | `compute_depth_profile` | Model            | Description                             |
|-------|----------------------|-------------------|-------------------------|------------------|-----------------------------------------|
| 1     | `true`               | —                 | —                       | HEIMDALL         | Above-surface irradiance only           |
| 2     | `false`              | `true`            | `false`                 | HEIMDALL         | + Fresnel open water transmission       |
| 3     | `false`              | `true`            | `true`                  | HEIMDALL         | + Open water depth profile              |
| 2     | `false`              | `false`           | `false`                 | BIFROST-HEIMDALL | + Snow/ice/open water transmission      |
| 3     | `false`              | `false`           | `true`                  | BIFROST-HEIMDALL | + Snow/ice transmission + depth profile |

Levels 2 and 3 with `open_water_only: false` require Copernicus Marine credentials to download TOPAZ4 sea ice data.

### Depth profile (Level 3 only)

```yaml
depth:
  compute_depth_profile: true
  max_depth: 50               # metres
  depth_resolution_m: 1       # metres per layer
  chlorophyll: 0.5            # mg m⁻³ — single value (homogeneous column)
  salinity: 34.0              # PSU    — single value (homogeneous column)
```

To specify depth-varying chlorophyll or salinity, provide a list with one value per depth layer. The list length must equal `max_depth / depth_resolution_m`:

```yaml
depth:
  compute_depth_profile: true
  max_depth: 50
  depth_resolution_m: 1
  chlorophyll:        # mg m⁻³ — one value per layer (50 values for max_depth=50, depth_resolution_m=1)
    - 0.8             # 0–1 m
    - 0.8             # 1–2 m
    - 0.6             # 2–3 m
    - 0.5             # 3–4 m
    # ... continue for all 50 layers
  salinity: 34.0      # PSU — salinity can remain a single value if homogeneous
```

---

## Running the model

```bash
python main.py
```

Output is saved to `model_outputs/` as a timestamped NetCDF file (e.g. `model_output_20210715T120000Z.nc`).

To run the model from another script:

```python
from main import run_model
results = run_model("config.yaml")
```

`results` is a Python dictionary containing all output arrays — see `main.py` for the full key listing and array shapes.

---

## Output variables

### Always present (Level 1+)

| Variable               | Shape     | Units               | Description                        |
|------------------------|-----------|---------------------|------------------------------------|
| `solar_direct`         | (L, T, W) | W m⁻² nm⁻¹          | Solar direct spectral irradiance   |
| `solar_diffuse`        | (L, T, W) | W m⁻² nm⁻¹          | Solar diffuse spectral irradiance  |
| `lunar_direct`         | (L, T, W) | W m⁻² nm⁻¹          | Lunar direct spectral irradiance   |
| `lunar_diffuse`        | (L, T, W) | W m⁻² nm⁻¹          | Lunar diffuse spectral irradiance  |
| `spectral_par`         | (L, T, W) | µmol m⁻² s⁻¹ nm⁻¹   | Spectral PAR per 10 nm band        |
| `broadband_par`        | (L, T)    | µmol m⁻² s⁻¹        | Broadband PAR                      |
| `broadband_irradiance` | (L, T)    | W m⁻²               | Broadband irradiance               |
| `solar_zenith`         | (L, T)    | degrees             | Solar zenith angle                 |
| `lunar_zenith`         | (L, T)    | degrees             | Lunar zenith angle                 |
| `cloud_cover`          | (L, T)    | 0–1                 | Cloud fraction                     |
| `timestamps`           | (T,)      | Unix seconds        | Timestamps used in the simulation  |
| `locations`            | (L,)      | —                   | Location objects (lat, lon)        |
| `wavelengths_nm`       | (W,)      | nm                  | Wavelength array (405–695 nm)      |

### Below-surface (Level 2+)

Same variables as above prefixed with `below_` — irradiance immediately below the snow/ice/ocean surface after transmission.

### Underwater (Level 3)

Same variables prefixed with `underwater_` with shape (L, T, Z, W) for spectral quantities and (L, T, Z) for broadband quantities, where Z is the depth axis. Also includes `depth_steps` (Z,) as a coordinate.

*L = locations, T = timestamps, W = 30 wavelengths, Z = depth levels*

---

## Physics

### HEIMDALL
- **Solar/lunar geometry:** Skyfield (DE421 ephemeris), topocentric zenith angles with optional atmospheric refraction
- **Irradiance LUTs:** Spectrally resolved direct and diffuse irradiance as a function of zenith angle and cloud cover
- **Lunar scaling:** Solar irradiance scaled by lunar albedo, phase function and solid angle
- **Cloud cover data:** ERA5 total cloud cover (reanalysis-era5-single-levels) from Copernicus Climate Data Store
- **Open water transmission:** Fresnel reflection for direct radiation (zenith-dependent); fixed 2.88% diffuse reflectance
- **Water column attenuation:** Beer-Lambert with a Barents Sea bio-optical model; absorption from water, phytoplankton, CDOM, and NAP; direct and diffuse kept separate using Snell's law refraction (n = 1.34)

### BIFROST (extension)
- **Sea ice transmission:** Beer-Lambert through snow (if present) and ice; FYI/MYI distinguished at 2.33 m thickness threshold
- **Surface weighting:** T = (1 − SIC) × T_open_water + SIC × T_ice
- **Sea ice data:** TOPAZ4 daily reanalysis (ARCTIC_MULTIYEAR_PHY_002_003) from Copernicus Marine

---

## References

- **HEIMDALL:** Midnight Sun to Polar Night: A Model of Seasonal Light in the Barents Sea. https://doi.org/10.1029/2022MS003198
