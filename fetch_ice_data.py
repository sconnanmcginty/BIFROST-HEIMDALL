"""
fetch_ice_data.py

Downloads sea ice concentration (SIC), sea ice thickness (SIT), and snow
depth (SD) from the Copernicus Marine Service (TOPAZ4 reanalysis,
ARCTIC_MULTIYEAR_PHY_002_003) for the model's spatial/temporal domain.

Returns (L, T) numpy arrays for use in the BIFROST transmission module.
Locations outside the TOPAZ4 spatial domain (lat < 50°N) are treated as
ice-free (zeros) without downloading data.

Requires Copernicus Marine credentials. See COPERNICUS_MARINE_SETUP.md.
"""

import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import copernicusmarine

from typing import List, Optional
from config_loader import TimeConfig, Location

# Suppress verbose warnings from the copernicusmarine library
logging.getLogger("copernicusmarine").setLevel(logging.ERROR)

# TOPAZ4 daily reanalysis dataset
DATASET_ID   = "cmems_mod_arc_phy_my_topaz4_P1D-m"
VARIABLES    = ["siconc", "sithick", "sisnthick"]
PRODUCT_PAGE = (
    "https://data.marine.copernicus.eu/product/ARCTIC_MULTIYEAR_PHY_002_003/description"
)


def fetch_ice_data(
    time_config: TimeConfig,
    locations: List[Location],
) -> dict:
    """
    Download TOPAZ4 sea ice data for the model domain and return (L, T) arrays.

    Locations outside the TOPAZ4 spatial domain are treated as ice-free (zeros).
    A temporal coverage check is only performed for runs where at least one
    location falls within the TOPAZ4 spatial domain.

    Ice data is daily. Each model timestep is matched to the nearest available
    daily value. The downloaded file is deleted after extraction.

    Parameters
    ----------
    time_config : TimeConfig
        Model time axis (from config_loader).
    locations : list of Location
        Model spatial grid.

    Returns
    -------
    dict with keys:
        'sic' : np.ndarray (L, T)  sea ice concentration [0-1]
        'sit' : np.ndarray (L, T)  sea ice thickness [m]
        'sd'  : np.ndarray (L, T)  snow depth on ice [m]
    """
    L = len(locations)
    T = len(time_config.output_datetimes)

    sic = np.zeros((L, T))
    sit = np.zeros((L, T))
    sd  = np.zeros((L, T))

    # --- Step 1: query dataset bounds (single describe() call) ---
    bounds = _get_dataset_bounds()

    # --- Step 2: identify locations within TOPAZ4 spatial domain ---
    if bounds is not None:
        lat_min, lat_max = bounds["lat_min"], bounds["lat_max"]
        lon_min, lon_max = bounds["lon_min"], bounds["lon_max"]
    else:
        # Metadata unavailable — fall back to known TOPAZ4 domain
        lat_min, lat_max = 50.0, 90.0
        lon_min, lon_max = -180.0, 180.0

    in_coverage_idx = [
        i for i, loc in enumerate(locations)
        if lat_min <= loc.latitude <= lat_max
        and lon_min <= loc.longitude <= lon_max
    ]
    out_of_coverage = L - len(in_coverage_idx)

    if out_of_coverage > 0:
        print(
            f"  {out_of_coverage} of {L} location(s) are outside the TOPAZ4 spatial "
            f"domain (lat {lat_min:.0f}°–{lat_max:.0f}°N) — treated as ice-free."
        )

    # --- Step 3: if no locations need ice data, return zeros ---
    if not in_coverage_idx:
        print("  No locations within TOPAZ4 spatial coverage — skipping ice data download.")
        return {"sic": sic, "sit": sit, "sd": sd}

    # --- Step 4: temporal check (only now that we know ice data is needed) ---
    if bounds is not None:
        _check_temporal_coverage(time_config, bounds)

    # --- Step 5: download for in-coverage locations only ---
    in_coverage_locs = [locations[i] for i in in_coverage_idx]
    tmp_path = _tmp_filename(time_config, in_coverage_locs)

    try:
        _download_topaz4(time_config, in_coverage_locs, tmp_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            print(f"  Partial download removed: {tmp_path}")
        raise

    # --- Step 6: extract and map back into full (L, T) arrays ---
    ice_subset = _extract_ice_arrays(tmp_path, time_config, in_coverage_locs)

    for result_idx, orig_idx in enumerate(in_coverage_idx):
        sic[orig_idx] = ice_subset["sic"][result_idx]
        sit[orig_idx] = ice_subset["sit"][result_idx]
        sd[orig_idx]  = ice_subset["sd"][result_idx]

    return {"sic": sic, "sit": sit, "sd": sd}


# ---------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------

def _get_dataset_bounds() -> Optional[dict]:
    """
    Query TOPAZ4 metadata and return temporal and spatial bounds.

    Returns a dict with keys: time_start, time_end, lat_min, lat_max,
    lon_min, lon_max — or None if the metadata cannot be retrieved.
    """
    print("  Checking TOPAZ4 dataset coverage...")

    try:
        catalogue = copernicusmarine.describe(dataset_id=DATASET_ID)
        part      = catalogue.products[0].datasets[0].versions[0].parts[0]

        # Coordinate bounds are on variable coordinates within services.
        # Find siconc from any service that has coordinates populated.
        time_coord = lat_coord = lon_coord = None

        for service in part.services:
            for var in service.variables:
                if var.short_name == "siconc":
                    for coord in var.coordinates:
                        if coord.minimum_value is not None:
                            if coord.coordinate_id == "time":
                                time_coord = coord
                            elif coord.coordinate_id == "latitude":
                                lat_coord = coord
                            elif coord.coordinate_id == "longitude":
                                lon_coord = coord
                if time_coord and lat_coord and lon_coord:
                    break
            if time_coord and lat_coord and lon_coord:
                break

        if not all([time_coord, lat_coord, lon_coord]):
            raise RuntimeError("Could not locate all coordinates in dataset metadata.")

        bounds = {
            "time_start": datetime.utcfromtimestamp(time_coord.minimum_value / 1000),
            "time_end":   datetime.utcfromtimestamp(time_coord.maximum_value / 1000),
            "lat_min":    lat_coord.minimum_value,
            "lat_max":    lat_coord.maximum_value,
            "lon_min":    lon_coord.minimum_value,
            "lon_max":    lon_coord.maximum_value,
        }

        print(
            f"  TOPAZ4 dataset coverage: "
            f"{bounds['time_start'].strftime('%d %b %Y')} – "
            f"{bounds['time_end'].strftime('%d %b %Y')} | "
            f"lat: {bounds['lat_min']:.0f}°–{bounds['lat_max']:.0f}°N"
        )

        return bounds

    except Exception as e:
        print(
            f"  WARNING: Could not retrieve dataset metadata "
            f"({type(e).__name__}: {e}). Proceeding without coverage check."
        )
        return None


def _check_temporal_coverage(time_config: TimeConfig, bounds: dict) -> None:
    """
    Raise a ValueError if the requested dates fall outside dataset coverage.
    Only called when at least one location is within the TOPAZ4 spatial domain.
    """
    requested_start = time_config.output_datetimes[0]
    requested_end   = time_config.output_datetimes[-1]

    if requested_start.tzinfo is not None:
        requested_start = requested_start.replace(tzinfo=None)
    if requested_end.tzinfo is not None:
        requested_end = requested_end.replace(tzinfo=None)

    print(
        f"  Requested period:        "
        f"{requested_start.strftime('%d %b %Y')} – {requested_end.strftime('%d %b %Y')}"
    )

    ds_start = bounds["time_start"]
    ds_end   = bounds["time_end"]

    outside_start = requested_start < ds_start
    outside_end   = requested_end   > ds_end

    if outside_start or outside_end:
        print()
        print("  WARNING: Requested dates are outside the available sea ice dataset coverage.")
        if outside_start:
            print(f"    Requested start ({requested_start.date()}) is before "
                  f"dataset start ({ds_start.date()}).")
        if outside_end:
            print(f"    Requested end ({requested_end.date()}) is after "
                  f"dataset end ({ds_end.date()}).")
        print(
            f"\n  Note: ARCTIC_MULTIYEAR_PHY_002_003 is a reanalysis product "
            f"updated approximately bi-annually. Dates beyond the coverage end "
            f"will not be available until the next product update.\n"
            f"  Product page: {PRODUCT_PAGE}\n"
        )
        raise ValueError(
            f"Sea ice data: requested dates [{requested_start.date()} – {requested_end.date()}] "
            f"are outside the available coverage of the TOPAZ4 reanalysis dataset "
            f"[{ds_start.date()} – {ds_end.date()}]."
        )


def _tmp_filename(time_config: TimeConfig, locations: List[Location]) -> str:
    """Build a descriptive temporary filename from the model domain."""
    start = time_config.output_datetimes[0].strftime("%Y%m%d")
    end   = time_config.output_datetimes[-1].strftime("%Y%m%d")
    lats  = [loc.latitude  for loc in locations]
    lons  = [loc.longitude for loc in locations]
    return (
        f"topaz4_ice_{start}_{end}"
        f"_lat{min(lats):.0f}to{max(lats):.0f}"
        f"_lon{min(lons):.0f}to{max(lons):.0f}.nc"
    )


def _download_topaz4(
    time_config: TimeConfig,
    locations: List[Location],
    output_path: str,
) -> None:
    """Download TOPAZ4 sea ice variables for the model domain."""
    datetimes = time_config.output_datetimes
    lats = [loc.latitude  for loc in locations]
    lons = [loc.longitude for loc in locations]

    # Bounding box with 1° padding, clamped to valid ranges
    min_lat = max(-90.0, min(lats) - 1.0)
    max_lat = min( 90.0, max(lats) + 1.0)
    min_lon = max(-180.0, min(lons) - 1.0)
    max_lon = min( 180.0, max(lons) + 1.0)

    start_str = datetimes[0].strftime("%Y-%m-%dT00:00:00")
    end_str   = datetimes[-1].strftime("%Y-%m-%dT23:59:59")

    print(
        f"  Downloading TOPAZ4 ice data | "
        f"{start_str[:10]} – {end_str[:10]} | "
        f"lat: {min_lat:.1f}–{max_lat:.1f} | "
        f"lon: {min_lon:.1f}–{max_lon:.1f}"
    )

    copernicusmarine.subset(
        dataset_id=DATASET_ID,
        variables=VARIABLES,
        minimum_latitude=min_lat,
        maximum_latitude=max_lat,
        minimum_longitude=min_lon,
        maximum_longitude=max_lon,
        start_datetime=start_str,
        end_datetime=end_str,
        output_filename=output_path,
        overwrite=True,
    )

    print(f"  Ice data saved to: {output_path}")


def _extract_ice_arrays(
    nc_path: str,
    time_config: TimeConfig,
    locations: List[Location],
) -> dict:
    """Load TOPAZ4 netCDF and interpolate to the model (L, T) grid."""
    ds = xr.open_dataset(nc_path)

    # Build timezone-naive DatetimeIndex for xarray selection
    times_pd = pd.DatetimeIndex(
        [dt.replace(tzinfo=None) for dt in time_config.output_datetimes]
    )

    L = len(locations)
    T = len(time_config.output_datetimes)

    sic = np.zeros((L, T))
    sit = np.zeros((L, T))
    sd  = np.zeros((L, T))

    for li, loc in enumerate(locations):
        # Nearest TOPAZ4 grid point in space
        kwargs = dict(latitude=loc.latitude, longitude=loc.longitude, method="nearest")

        sic_ts = ds["siconc"].sel(**kwargs).sel(time=times_pd, method="nearest")
        sit_ts = ds["sithick"].sel(**kwargs).sel(time=times_pd, method="nearest")
        sd_ts  = ds["sisnthick"].sel(**kwargs).sel(time=times_pd, method="nearest")

        # Clip to valid ranges and fill NaN — NaN treated as ice-free
        sic[li] = np.clip(np.nan_to_num(sic_ts.values, nan=0.0), 0.0, 1.0)
        sit[li] = np.maximum(np.nan_to_num(sit_ts.values, nan=0.0), 0.0)
        sd[li]  = np.maximum(np.nan_to_num(sd_ts.values,  nan=0.0), 0.0)

    ds.close()
    os.remove(nc_path)

    return {"sic": sic, "sit": sit, "sd": sd}
