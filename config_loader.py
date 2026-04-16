"""
config_loader.py

Reads and validates the model configuration from config.yaml.
Generates structured dataclasses for:

    - Time axis (TimeConfig)
    - Locations or spatial grid (Location)
    - Atmospheric settings (AtmosphereConfig)
    - Cloud mode and (optional) satellite config

This module isolates configuration handling from model logic,
ensuring reproducibility and easy modification of model inputs.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence, List, Optional

import numpy as np
import yaml 

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


# ------------------------------------------------------------
# Dataclasses
# ------------------------------------------------------------

@dataclass
class TimeConfig:
    timestamps_unix: np.ndarray
    output_datetimes: List[datetime]
    resolution_seconds: int


@dataclass
class Location:
    latitude: float
    longitude: float


@dataclass
class AtmosphereConfig:
    temperature_C: float
    pressure_mbar: float


@dataclass
class DepthConfig:
    depth_steps: np.ndarray   # (num_steps,) depth levels in metres, including 0
    chl: np.ndarray           # (num_layers,) chlorophyll per layer [mg m⁻³]
    sal: np.ndarray           # (num_layers,) salinity per layer [PSU]


@dataclass
class ModelConfig:
    time: TimeConfig
    locations: List[Location]
    cloud_mode: int
    atmosphere: AtmosphereConfig
    above_surface_only: bool = True
    open_water_only: bool = False
    depth: Optional[DepthConfig] = None


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def _parse_datetime(dt_str: str) -> datetime:
    """
    Convert a datetime string into a timezone-aware UTC datetime.

    Internal use only: external modules should not call this directly.
    """
    dt = datetime.strptime(dt_str, DATETIME_FORMAT)
    return dt.replace(tzinfo=timezone.utc)


def _build_time_axis(start_str: str, end_str: str, resolution_minutes: float) -> TimeConfig:
    """
    Build the model time axis from user-provided start/end times and resolution.

    Produces:
        - Unix timestamps array
        - Corresponding UTC datetime objects
        - Step size in seconds

    Internal use only.
    """
    start_dt = _parse_datetime(start_str)
    end_dt = _parse_datetime(end_str)

    if end_dt < start_dt:
        raise ValueError("End date/time must not be earlier than start date/time.")

    resolution_seconds = int(resolution_minutes * 60)

    start_unix = start_dt.timestamp()
    end_unix = end_dt.timestamp()

    if start_unix == end_unix:
        timestamps_unix = np.array([start_unix])
    else:
        timestamps_unix = np.arange(
            start_unix,
            end_unix + resolution_seconds,
            resolution_seconds,
        )

    output_datetimes = [
        datetime.fromtimestamp(ts, tz=timezone.utc) for ts in timestamps_unix
    ]

    return TimeConfig(
        timestamps_unix=timestamps_unix,
        output_datetimes=output_datetimes,
        resolution_seconds=resolution_seconds,
    )


def _generate_grid(
    lat_start: float,
    lat_end: float,
    lat_step: float,
    lon_start: float,
    lon_end: float,
    lon_step: float,
) -> List[Location]:
    """
    Generate a rectangular grid of Location objects.

    Example:
        lat_start=55 → lat_end=57 → lat_step=1
        lon_start=-5 → lon_end=-3 → lon_step=1

        Produces 3 × 3 = 9 Location points.

    Internal use only.
    """
    latitudes = np.arange(lat_start, lat_end + lat_step, lat_step)
    longitudes = np.arange(lon_start, lon_end + lon_step, lon_step)

    return [
        Location(lat, lon)
        for lat in latitudes
        for lon in longitudes
    ]


# ------------------------------------------------------------
# Main loader
# ------------------------------------------------------------

def load_model_config(path: str = "config.yaml") -> ModelConfig:
    """Load and validate configuration file into Python dataclasses."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    # -------------------------------
    # TIME
    # -------------------------------
    tcfg = cfg["time"]
    use_custom = tcfg.get("use_custom", False)

    if use_custom:
        timestamp_strings = tcfg.get("timestamps", [])
        if not timestamp_strings:
            raise ValueError("time.use_custom=True but no timestamps provided.")

        datetimes = [_parse_datetime(s) for s in timestamp_strings]

        timestamps_unix = np.array([dt.timestamp() for dt in datetimes], dtype=float)

        # infer resolution if possible
        if len(timestamps_unix) > 1:
            resolution_seconds = int(np.median(np.diff(timestamps_unix)))
        else:
            resolution_seconds = 0

        time_config = TimeConfig(
            timestamps_unix=timestamps_unix,
            output_datetimes=datetimes,
            resolution_seconds=resolution_seconds,
        )

    else:
        time_config = _build_time_axis(
            start_str=tcfg["start"],
            end_str=tcfg["end"],
            resolution_minutes=tcfg["resolution_minutes"],
        )

    # -------------------------------
    # LOCATIONS or SPATIAL GRID
    # -------------------------------
    grid_cfg = cfg.get("spatial_grid", {})
    use_grid = grid_cfg.get("use_grid", False)

    if use_grid:
        locations = _generate_grid(
            lat_start=grid_cfg["lat_start"],
            lat_end=grid_cfg["lat_end"],
            lat_step=grid_cfg["lat_step"],
            lon_start=grid_cfg["lon_start"],
            lon_end=grid_cfg["lon_end"],
            lon_step=grid_cfg["lon_step"],
        )
    else:
        loc_list = cfg.get("locations", [])
        if not loc_list:
            raise ValueError("No locations provided and grid mode is disabled.")
        locations = [
            Location(latitude=float(loc["latitude"]), longitude=float(loc["longitude"]))
            for loc in loc_list
        ]

    # -------------------------------
    # CLOUD MODE
    # -------------------------------
    cloud_mode = int(cfg.get("cloud_mode", 0))
    if cloud_mode not in (0, 1, 2):
        raise ValueError("cloud_mode must be 0, 1, or 2.")

    # -------------------------------
    # ATMOSPHERIC SETTINGS
    # -------------------------------
    atm_cfg = cfg.get("atmosphere", {})
    atmosphere = AtmosphereConfig(
        temperature_C=float(atm_cfg.get("temperature_C", 0.0)),
        pressure_mbar=float(atm_cfg.get("pressure_mbar", 0.0)),
    )

    # -------------------------------
    # TRANSMISSION MODE
    # -------------------------------
    above_surface_only = bool(cfg.get("above_surface_only", True))
    open_water_only    = bool(cfg.get("open_water_only", False))

    # -------------------------------
    # DEPTH ATTENUATION (optional)
    # -------------------------------
    depth_config = None
    depth_cfg = cfg.get("depth", None)

    if depth_cfg is not None:
        if above_surface_only:
            raise ValueError(
                "depth section is present in config but above_surface_only=true. "
                "Set above_surface_only=false to enable depth attenuation."
            )

        if not depth_cfg.get("compute_depth_profile", True):
            depth_cfg = None

    if depth_cfg is not None:
        max_depth = float(depth_cfg["max_depth"])
        res       = float(depth_cfg["depth_resolution_m"])

        depths      = np.arange(res, max_depth + res * 0.5, res)
        depth_steps = np.concatenate([[0.0], depths])
        num_layers  = len(depths)

        def _parse_constituent(value, name):
            if isinstance(value, list):
                arr = np.array(value, dtype=float)
                if len(arr) != num_layers:
                    raise ValueError(
                        f"depth.{name} has {len(arr)} values but depth grid has "
                        f"{num_layers} layers (max_depth={max_depth}, "
                        f"depth_resolution_m={res})."
                    )
                return arr
            return np.full(num_layers, float(value))

        chl = _parse_constituent(depth_cfg["chlorophyll"], "chlorophyll")
        sal = _parse_constituent(depth_cfg["salinity"],    "salinity")

        depth_config = DepthConfig(depth_steps=depth_steps, chl=chl, sal=sal)

    # -------------------------------
    # Return unified configuration
    # -------------------------------
    return ModelConfig(
        time=time_config,
        locations=locations,
        cloud_mode=cloud_mode,
        atmosphere=atmosphere,
        above_surface_only=above_surface_only,
        open_water_only=open_water_only,
        depth=depth_config,
    )