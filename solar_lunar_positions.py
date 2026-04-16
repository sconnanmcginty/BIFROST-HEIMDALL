"""
solar_lunar_positions.py

Utilities for computing solar and lunar geometric properties using the
Skyfield astronomy library. This module provides functions for:

    • Solar zenith angle (topocentric)
    • Lunar zenith angle (topocentric)
    • Geocentric Earth-Moon distance
    • Lunar phase angle (Miller & Turner 2009 convention)

These quantities are required for models of above-surface irradiance,
lunar radiance contributions, and any application requiring accurate
solar-lunar geometry.

-----------------------------------------------------------------------
Zenith Angle Calculations
-----------------------------------------------------------------------
The function 'calculate_zenith()' computes solar and lunar zenith angles
at a given latitude, longitude, and time. Positions are calculated
topocentrically using:

    observer = earth + wgs84.latlon(lat, lon, elevation_m)

Solar and lunar altitudes are obtained from their apparent topocentric
positions, and zenith angle is computed as:

    zenith = 90° - altitude

Temperature and pressure parameters allow atmospheric
refraction to be included or disabled. Note these are disabled by
default but can be included by end user.

-----------------------------------------------------------------------
Lunar Phase and Distance
-----------------------------------------------------------------------
The function 'calculate_lunar_phase_and_distance()' computes:

    1. Earth-Moon distance (AU)
    2. Lunar phase angle (degrees)

Phase angle is derived from the difference in apparent ecliptic
longitudes of the Sun and Moon, computed in the geocentric frame.
The classical elongation angle (0° = New Moon, 180° = Full Moon) is
converted to the convention used by Miller & Turner (2009), in which:

    0°   = Full Moon  
    180° = New Moon

This convention aligns with lunar phase-function lookup tables used in
illumination modelling.

-----------------------------------------------------------------------
Caching
-----------------------------------------------------------------------
Ephemeris files (DE421) and Skyfield timescales are loaded once and
cached using @lru_cache for efficiency.

-----------------------------------------------------------------------
Example Usage
-----------------------------------------------------------------------
    lunar_zen, solar_zen = calculate_zenith(lat=56.0, lon=-3.5,
                                            timestamp=unix_time)

    distance_au, phase_deg = calculate_lunar_phase_and_distance(unix_time)
"""



from datetime import datetime, timezone
from functools import lru_cache
from typing import Tuple

from skyfield.api import load, wgs84
from skyfield.framelib import ecliptic_frame


@lru_cache()
def _get_ephemeris():
    """Load JPL DE421 ephemeris once (cached)."""
    return load("de421.bsp")


@lru_cache()
def _get_timescale():
    """Load Skyfield timescale once (cached)."""
    return load.timescale()


def calculate_zenith(
        lat: float,
        lon: float,
        timestamp: float,
        elevation_m: float = 0,
        temperature_C: float = 0,
        pressure_mbar: float = 0
    ) -> Tuple[float, float]:
    """
    Compute solar and lunar zenith angles (degrees) at a given time/location.

    Parameters
    ----------
    lat, lon : float
        Geographic coordinates in decimal degrees.
    timestamp : float
        Unix epoch timestamp (seconds since 1970-01-01 UTC).
    elevation_m : float
        Observer elevation above mean sea level (m).
    temperature_C : float
        Ambient temperature for refraction correction.
    pressure_mbar : float
        Ambient pressure for refraction correction.

    Returns
    -------
    lunar_zenith, solar_zenith : float
        Zenith angles in degrees.
    """

    eph = _get_ephemeris()
    ts = _get_timescale()

    earth = eph["earth"]
    sun = eph["sun"]
    moon = eph["moon"]

    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    t = ts.from_datetime(dt)

    location = earth + wgs84.latlon(lat, lon, elevation_m=elevation_m)

    # Solar altitude
    sun_alt, _, _ = (
        location.at(t)
        .observe(sun)
        .apparent()
        .altaz(temperature_C=temperature_C, pressure_mbar=pressure_mbar)
    )

    # Lunar altitude 
    moon_alt, _, _ = (
        location.at(t)
        .observe(moon)
        .apparent()
        .altaz(temperature_C=temperature_C, pressure_mbar=pressure_mbar)
    )

    solar_zenith = 90.0 - sun_alt.degrees
    lunar_zenith = 90.0 - moon_alt.degrees

    return lunar_zenith, solar_zenith


def calculate_zenith_batch(
        lat: float,
        lon: float,
        timestamps,
        elevation_m: float = 0,
        temperature_C: float = 0,
        pressure_mbar: float = 0,
    ):
    """
    Compute solar and lunar zenith angles for multiple timestamps at one location.

    Parameters
    ----------
    lat, lon : float
        Geographic coordinates in decimal degrees.
    timestamps : sequence of float
        Unix epoch timestamps (seconds since 1970-01-01 UTC).
    elevation_m : float
        Observer elevation above mean sea level (m).
    temperature_C, pressure_mbar : float
        Atmospheric refraction parameters.

    Returns
    -------
    lunar_zenith, solar_zenith : np.ndarray, shape (T,)
        Zenith angles in degrees for each timestamp.
    """
    eph = _get_ephemeris()
    ts  = _get_timescale()

    earth = eph["earth"]
    sun   = eph["sun"]
    moon  = eph["moon"]

    dts = [datetime.fromtimestamp(t, tz=timezone.utc) for t in timestamps]
    t   = ts.from_datetimes(dts)

    location = earth + wgs84.latlon(lat, lon, elevation_m=elevation_m)

    sun_alts, _, _  = (
        location.at(t)
        .observe(sun)
        .apparent()
        .altaz(temperature_C=temperature_C, pressure_mbar=pressure_mbar)
    )
    moon_alts, _, _ = (
        location.at(t)
        .observe(moon)
        .apparent()
        .altaz(temperature_C=temperature_C, pressure_mbar=pressure_mbar)
    )

    return 90.0 - moon_alts.degrees, 90.0 - sun_alts.degrees


def calculate_lunar_phase_and_distance(timestamp: float):
    """
    Compute lunar distance (AU) and lunar phase angle (degrees)
    using the Miller & Turner (2009) convention:

        - 0°  = Full Moon
        - 180° = New Moon

    Parameters
    ----------
    timestamp : float
        Unix timestamp (UTC).

    Returns
    -------
    lunar_distance_au : float
        Earth-Moon distance in astronomical units.
    lunar_phase_angle : float
        Lunar phase in degrees in the Miller & Turner convention.
    """

    eph = _get_ephemeris()
    ts = _get_timescale()

    earth = eph["earth"]
    sun = eph["sun"]
    moon = eph["moon"]

    # Convert timestamp to Skyfield time
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    t = ts.from_datetime(dt)

    # Earth-centered position 
    e = earth.at(t)

    # Compute apparent ecliptic longitudes of Sun and Moon
    _, sun_lon, _  = e.observe(sun ).apparent().frame_latlon(ecliptic_frame)
    _, moon_lon, m_distance = e.observe(moon).apparent().frame_latlon(ecliptic_frame)

    # Classical elongation: 0=new, 180=full, back to 360
    classical_phase = (moon_lon.degrees - sun_lon.degrees) % 360.0

    # Miller & Turner convention: 0=full, 180=new
    lunar_phase_angle = abs(180.0 - classical_phase)

    lunar_distance_au = m_distance.au

    return lunar_distance_au, lunar_phase_angle