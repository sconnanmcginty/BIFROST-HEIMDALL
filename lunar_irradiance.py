"""
lunar_irradiance.py

Compute lunar spectral irradiance (direct and diffuse) by scaling solar
LUT-based irradiance using:

    E_moon(λ) =
        E_sun(λ) *
        A_moon(λ) *
        P_moon(phase_angle, λ) *
        (Ω_moon / π)

where:
    - E_sun(λ) is the solar spectral irradiance (from solar LUTs)
    - A_moon(λ) is the wavelength-dependent lunar spectral albedo (LUT)
    - P_moon is the lunar phase function (LUT)
    - Ω_moon is the lunar solid angle = π (R_moon / distance)^2

This module **does not compute** the lunar phase or distance itself —
that is provided by the solar_lunar_positions.py module:

    calculate_lunar_phase_and_distance(timestamp)

and lunar zenith angles (LZA) are obtained using:

    calculate_zenith(lat, lon, timestamp)

"""

import numpy as np

# Import solar LUT-based irradiance functions
from load_lut_and_interpolate import (
    get_direct_irradiance,
    get_diffuse_irradiance,
    get_lut_value,
    get_direct_irradiance_spectrum,
    get_diffuse_irradiance_spectrum,
    get_lunar_albedo_spectrum,
    get_lunar_phase_fn_spectrum,
)

# Geometry (distance + phase) comes from solar_lunar_positions
from solar_lunar_positions import (
    calculate_zenith,
    calculate_lunar_phase_and_distance)

# ------------------------------------------------------------
# Physical constants
# ------------------------------------------------------------

MOON_RADIUS_M = 1737400          # Moon radius in meters
AU_TO_M = 1.495978707e11         # Astronomical Unit in meters


# ------------------------------------------------------------
# Solid angle calculation
# ------------------------------------------------------------

def lunar_solid_angle(distance_au: float) -> float:
    """
    Compute the solid angle of the Moon as seen from Earth.

    Parameters
    ----------
    distance_au : float
        Earth-Moon distance in astronomical units.

    Returns
    -------
    float
        Solid angle in steradians.
    """
    distance_m = distance_au * AU_TO_M
    return np.pi * (MOON_RADIUS_M / distance_m) ** 2


# ------------------------------------------------------------
# Lunar Direct Irradiance
# ------------------------------------------------------------

def lunar_direct_irradiance(
    wavelength_nm: float,
    lza_deg: float,
    cloud: float,
    lunar_distance_au: float,
    lunar_phase_angle_deg: float,
) -> float:
    """
    Compute direct-beam lunar spectral irradiance at a given wavelength.

    Parameters
    ----------
    wavelength_nm : float
        Wavelength [nm].
    lza_deg : float
        Lunar zenith angle, degrees.
    cloud : float
        Cloud fraction (0-1).
    lunar_distance_au : float
        Earth-Moon distance [AU].
    lunar_phase_angle_deg : float
        Lunar phase angle (Miller & Turner convention):
            0°   = full moon
            180° = new moon

    Returns
    -------
    float
        Lunar direct irradiance at the given wavelength [W m⁻² nm⁻¹].
    """

    # Solar direct spectral irradiance from LUT
    E_sun = get_direct_irradiance(
        wavelength=wavelength_nm,
        sza=lza_deg,
        cloud=cloud,
    )

    # Spectral lunar albedo from LUT
    A = get_lut_value(
        "lunar_albedo",
        wavelength=wavelength_nm,
    )

    # Lunar phase function from LUT (depends on phase angle and wavelength)
    P = get_lut_value(
        "lunar_phase_fn",
        phase_angle=lunar_phase_angle_deg,
        wavelength=wavelength_nm,
    )

    # Solid angle of the Moon
    Omega = lunar_solid_angle(lunar_distance_au)

    # Scale solar irradiance by lunar properties
    return E_sun * A * P * (Omega / np.pi)


# ------------------------------------------------------------
# Lunar Diffuse Irradiance
# ------------------------------------------------------------

def lunar_diffuse_irradiance(
    wavelength_nm: float,
    lza_deg: float,
    cloud: float,
    lunar_distance_au: float,
    lunar_phase_angle_deg: float,
) -> float:
    """
    Compute diffuse lunar spectral irradiance.

    Parameters
    ----------
    wavelength_nm : float
        Wavelength [nm].
    lza_deg : float
        Lunar zenith angle [degrees].
    cloud : float
        Cloud fraction (0-1).
    lunar_distance_au : float
        Earth-Moon distance [AU].
    lunar_phase_angle_deg : float
        Lunar phase angle (Miller & Turner convention).

    Returns
    -------
    float
        Lunar diffuse irradiance at the given wavelength [W m⁻² nm⁻¹].
    """

    # Solar diffuse spectral irradiance from LUT
    E_sun = get_diffuse_irradiance(
        wavelength=wavelength_nm,
        sza=lza_deg,
        cloud=cloud,
    )

    # Spectral lunar albedo
    A = get_lut_value(
        "lunar_albedo",
        wavelength=wavelength_nm,
    )

    # Lunar phase function
    P = get_lut_value(
        "lunar_phase_fn",
        phase_angle=lunar_phase_angle_deg,
        wavelength=wavelength_nm,
    )

    Omega = lunar_solid_angle(lunar_distance_au)

    return E_sun * A * P * (Omega / np.pi)


# ------------------------------------------------------------
# Batch Lunar Irradiance (all wavelengths at once)
# ------------------------------------------------------------

def lunar_direct_irradiance_spectrum(
    wavelengths_nm: np.ndarray,
    lza_deg: float,
    cloud: float,
    lunar_distance_au: float,
    lunar_phase_angle_deg: float,
) -> np.ndarray:
    """
    Direct lunar spectral irradiance for all wavelengths at once.

    Parameters
    ----------
    wavelengths_nm : np.ndarray, shape (W,)
    lza_deg, cloud, lunar_distance_au, lunar_phase_angle_deg : float

    Returns
    -------
    np.ndarray, shape (W,)
    """
    E_sun = get_direct_irradiance_spectrum(wavelengths_nm, lza_deg, cloud)
    A     = get_lunar_albedo_spectrum(wavelengths_nm)
    P     = get_lunar_phase_fn_spectrum(lunar_phase_angle_deg, wavelengths_nm)
    Omega = lunar_solid_angle(lunar_distance_au)
    return E_sun * A * P * (Omega / np.pi)


def lunar_diffuse_irradiance_spectrum(
    wavelengths_nm: np.ndarray,
    lza_deg: float,
    cloud: float,
    lunar_distance_au: float,
    lunar_phase_angle_deg: float,
) -> np.ndarray:
    """
    Diffuse lunar spectral irradiance for all wavelengths at once.

    Parameters
    ----------
    wavelengths_nm : np.ndarray, shape (W,)
    lza_deg, cloud, lunar_distance_au, lunar_phase_angle_deg : float

    Returns
    -------
    np.ndarray, shape (W,)
    """
    E_sun = get_diffuse_irradiance_spectrum(wavelengths_nm, lza_deg, cloud)
    A     = get_lunar_albedo_spectrum(wavelengths_nm)
    P     = get_lunar_phase_fn_spectrum(lunar_phase_angle_deg, wavelengths_nm)
    Omega = lunar_solid_angle(lunar_distance_au)
    return E_sun * A * P * (Omega / np.pi)