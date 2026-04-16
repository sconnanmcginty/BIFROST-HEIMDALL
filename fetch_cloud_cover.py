"""
fetch_cloud_cover.py

Downloads ERA5 total cloud cover from the Copernicus Climate Data Store (CDS)
for the model's spatial/temporal domain and returns a (L, T) numpy array.

Requires a .cdsapirc credentials file in your home directory.
See CDS_SETUP.md for account registration and setup instructions.
"""

import os
from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr
import cdsapi

from typing import List
from config_loader import TimeConfig, Location


def fetch_cloud_cover(time_config: TimeConfig, locations: List[Location]) -> np.ndarray:
    """
    Return ERA5 total cloud cover interpolated to the model's (L, T) grid.

    Downloads ERA5 data from the Copernicus CDS, extracts cloud fractions
    for the model domain, and returns the result. The downloaded file is
    deleted after extraction so it does not persist on disk.

    Requests are split by calendar month to avoid over-downloading when
    timestamps are scattered across non-contiguous periods. If a download
    fails partway through, any partial file is deleted so the next run
    starts clean.

    Parameters
    ----------
    time_config : TimeConfig
        Model time axis (from config_loader).
    locations : list of Location
        Model spatial grid.

    Returns
    -------
    cloud_array : np.ndarray, shape (L, T), values in [0, 1]
        Cloud fraction at each model grid point and timestep.
    """
    tmp_path = _tmp_filename(time_config, locations)

    try:
        _download_era5(time_config, locations, tmp_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            print(f"  Partial download removed: {tmp_path}")
        raise

    return _extract_cloud_array(tmp_path, time_config, locations)


# ---------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------

def _tmp_filename(time_config: TimeConfig, locations: List[Location]) -> str:
    """Build a descriptive temporary filename from the model domain."""
    start = time_config.output_datetimes[0].strftime("%Y%m%d")
    end   = time_config.output_datetimes[-1].strftime("%Y%m%d")
    lats  = [loc.latitude  for loc in locations]
    lons  = [loc.longitude for loc in locations]
    return (
        f"era5_cloud_{start}_{end}"
        f"_lat{min(lats):.0f}to{max(lats):.0f}"
        f"_lon{min(lons):.0f}to{max(lons):.0f}.nc"
    )


def _era5_time_coord(ds: xr.Dataset) -> str:
    """Return the name of the time coordinate in an ERA5 dataset."""
    return "valid_time" if "valid_time" in ds.coords else "time"


def _download_era5(
    time_config: TimeConfig,
    locations: List[Location],
    output_path: str,
) -> None:
    """
    Download ERA5 total cloud cover for the model domain to output_path.

    Splits the request into one CDS call per calendar month so that
    scattered timestamps never trigger a large Cartesian over-request.
    Monthly downloads are written to temporary files, merged in memory,
    and saved as a single file. Temporary files are always cleaned
    up, even if an error occurs.
    """
    datetimes = time_config.output_datetimes

    lats = [loc.latitude  for loc in locations]
    lons = [loc.longitude for loc in locations]

    # Bounding box [N, W, S, E] with 1° padding, clamped to valid ranges
    area = [
        min( 90.0, max(lats) + 1.0),
        max(-180.0, min(lons) - 1.0),
        max( -90.0, min(lats) - 1.0),
        min( 180.0, max(lons) + 1.0),
    ]

    # Group timestamps by (year, month) to avoid Cartesian over-requests
    groups = defaultdict(list)
    for dt in datetimes:
        groups[(dt.year, dt.month)].append(dt)

    client = cdsapi.Client()
    temp_files = []
    datasets = []

    try:
        for (year, month), group_dts in sorted(groups.items()):
            days  = sorted({f"{dt.day:02d}"     for dt in group_dts})
            hours = sorted({f"{dt.hour:02d}:00" for dt in group_dts})

            print(
                f"  Downloading ERA5 cloud cover | "
                f"{year}-{month:02d} | "
                f"days: {days[0]}–{days[-1]} | "
                f"lat: {area[2]:.1f}–{area[0]:.1f} | "
                f"lon: {area[1]:.1f}–{area[3]:.1f}"
            )

            tmp_path = f"era5_cloud_{year}{month:02d}_tmp.nc"
            temp_files.append(tmp_path)

            client.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": ["reanalysis"],
                    "variable": ["total_cloud_cover"],
                    "year": [str(year)],
                    "month": [f"{month:02d}"],
                    "day": days,
                    "time": hours,
                    "area": area,
                    "data_format": "netcdf",
                    "download_format": "unarchived",
                },
            ).download(tmp_path)

        # Merge monthly files along the time dimension
        datasets = [xr.open_dataset(f) for f in temp_files]
        time_coord = _era5_time_coord(datasets[0])

        merged = xr.concat(datasets, dim=time_coord) if len(datasets) > 1 else datasets[0]
        merged.to_netcdf(output_path)

        print(f"  Cloud cover data saved to: {output_path}")

    finally:
        for ds in datasets:
            try:
                ds.close()
            except Exception:
                pass
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)


def _extract_cloud_array(
    nc_path: str,
    time_config: TimeConfig,
    locations: List[Location],
) -> np.ndarray:
    """Load the ERA5 netCDF and interpolate to the model (L, T) grid."""
    ds = xr.open_dataset(nc_path)

    # ERA5 variable name for total cloud cover is 'tcc'
    tcc_var = next(
        (v for v in ds.data_vars if v.lower() in ("tcc", "total_cloud_cover")),
        list(ds.data_vars)[0],
    )

    time_coord = _era5_time_coord(ds)

    # Build a pandas DatetimeIndex (timezone-naive) for xarray selection
    times_pd = pd.DatetimeIndex(
        [dt.replace(tzinfo=None) for dt in time_config.output_datetimes]
    )

    L = len(locations)
    T = len(time_config.output_datetimes)
    cloud_array = np.zeros((L, T))

    for li, loc in enumerate(locations):
        # Nearest ERA5 grid point in space
        point = ds[tcc_var].sel(
            latitude=loc.latitude,
            longitude=loc.longitude,
            method="nearest",
        )
        # Nearest ERA5 hour for each model timestep
        point_at_times = point.sel(
            {time_coord: times_pd},
            method="nearest",
        )
        cloud_array[li] = np.clip(point_at_times.values, 0.0, 1.0)

    ds.close()
    os.remove(nc_path)
    return cloud_array
