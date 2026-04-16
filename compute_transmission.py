"""
compute_transmission.py

BIFROST surface transmission module for HEIMDALL.

Computes the fraction of above-surface spectral irradiance that passes through
the ocean surface, accounting for:

    - Open water: Fresnel reflection (direct) and fixed diffuse reflectance
    - Sea ice:    Beer-Lambert attenuation through snow (if present) + ice
                  FYI (SIT < 2.33 m) and MYI (SIT >= 2.33 m) distinguished
                  following Melling (2022)

Surface-type weighting (no melt ponds):
    T = (1 - SIC) * T_OW  +  SIC * T_ICE

Transmission is applied separately to each irradiance component:
    - Solar/lunar direct:  T_OW uses Fresnel at the respective zenith angle
    - Solar/lunar diffuse: T_OW uses fixed diffuse reflectance (2.88%)

All operations are fully vectorised over the (L, T, W) model domain.

References
----------
Melling (2022) — FYI/MYI thickness threshold of 2.33 m
Fresnel equations — refractive indices n_air=1.00, n_water=1.34
"""

import os

import numpy as np
import scipy.io as sio

# Physical constants (same values as irradiance_conversions.py)
H   = 6.62607015e-34   # Planck constant [J s]
C   = 2.99792458e8     # Speed of light [m s⁻¹]
N_A = 6.02214076e23    # Avogadro's number [mol⁻¹]

# Fixed diffuse transmittance through the ocean surface (1 - 2.88% reflectance)
T_OW_DIFFUSE = 1 - 0.0288

# Refractive indices
N_AIR   = 1.00
N_WATER = 1.34

# LUT file paths (relative to this file)
_LUT_DIR  = os.path.join(os.path.dirname(__file__), "LUT")
_KD_PATH  = os.path.join(_LUT_DIR, "ice_snow_attenuation_lut.mat")
_ALB_PATH = os.path.join(_LUT_DIR, "ice_snow_albedo_lut.mat")


def apply_transmission(
    results: dict,
    sic: np.ndarray,
    sit: np.ndarray,
    sd: np.ndarray,
) -> None:
    """
    Compute below-surface irradiance and add it to the results dictionary.

    Applies BIFROST transmission to all four above-surface irradiance
    components (solar direct/diffuse, lunar direct/diffuse) and derives
    below-surface spectral PAR, broadband PAR, and broadband irradiance.
    Results are added in-place under 'below_*' keys.

    Parameters
    ----------
    results : dict
        Model results dictionary from run_model(). Modified in place.
    sic : np.ndarray, shape (L, T)
        Sea ice concentration [0-1].
    sit : np.ndarray, shape (L, T)
        Sea ice thickness [m].
    sd : np.ndarray, shape (L, T)
        Snow depth on ice [m].
    """
    luts = _load_luts()

    # --- Transmission factors ---

    # Ice: Beer-Lambert through snow + ice, wavelength-dependent (L, T, W)
    t_ice = _t_ice(sit, sd, luts)

    # Open water direct: Fresnel at solar/lunar zenith angles (L, T, 1)
    t_ow_solar = _t_ow_direct(results["solar_zenith"])[:, :, np.newaxis]
    t_ow_lunar = _t_ow_direct(results["lunar_zenith"])[:, :, np.newaxis]

    # Surface-type weights (L, T, 1)
    ow_frac  = (1.0 - sic)[:, :, np.newaxis]
    ice_frac = sic[:, :, np.newaxis]

    # --- Surface-weighted transmission per component (L, T, W) ---
    T_solar_direct  = ow_frac * t_ow_solar    + ice_frac * t_ice
    T_solar_diffuse = ow_frac * T_OW_DIFFUSE  + ice_frac * t_ice
    T_lunar_direct  = ow_frac * t_ow_lunar    + ice_frac * t_ice
    T_lunar_diffuse = ow_frac * T_OW_DIFFUSE  + ice_frac * t_ice

    # --- Apply to above-surface irradiance (L, T, W) ---
    below_solar_direct  = results["solar_direct"]  * T_solar_direct
    below_solar_diffuse = results["solar_diffuse"] * T_solar_diffuse
    below_lunar_direct  = results["lunar_direct"]  * T_lunar_direct
    below_lunar_diffuse = results["lunar_diffuse"] * T_lunar_diffuse

    total_below = (below_solar_direct + below_solar_diffuse +
                   below_lunar_direct + below_lunar_diffuse)

    # --- Derived below-surface quantities ---
    wavelengths_m = results["wavelengths_nm"] * 1e-9  # (W,)

    below_spectral_par = (
        (total_below * 10.0 * wavelengths_m) / (H * C * N_A) * 1e6
    )  # (L, T, W)

    # --- Store in results ---
    results["below_solar_direct"]        = below_solar_direct
    results["below_solar_diffuse"]       = below_solar_diffuse
    results["below_lunar_direct"]        = below_lunar_direct
    results["below_lunar_diffuse"]       = below_lunar_diffuse
    results["below_spectral_par"]        = below_spectral_par
    results["below_broadband_par"]       = np.sum(below_spectral_par, axis=-1)
    results["below_broadband_irradiance"] = np.sum(total_below * 10.0, axis=-1)


# ---------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------

def _load_luts() -> dict:
    """Load Kd and albedo lookup tables from LUT .mat files."""
    kds  = sio.loadmat(_KD_PATH)
    albs = sio.loadmat(_ALB_PATH)
    return {
        "snow_kd":  kds["Snow_Kd"].flatten(),   # (W,)
        "fyi_kd":   kds["FYI_Kd"].flatten(),    # (W,)
        "myi_kd":   kds["MYI_Kd"].flatten(),    # (W,)
        "snow_alb": albs["Snow_a"].flatten(),    # (W,)
        "fyi_alb":  albs["FYI_a"].flatten(),     # (W,)
        "myi_alb":  albs["MYI_a"].flatten(),     # (W,)
    }


def _t_ice(sit: np.ndarray, sd: np.ndarray, luts: dict) -> np.ndarray:
    """
    Beer-Lambert transmission through snow (if present) and ice.

    Parameters
    ----------
    sit : (L, T)  sea ice thickness [m]
    sd  : (L, T)  snow depth [m]
    luts : dict   loaded LUT arrays

    Returns
    -------
    (L, T, W) transmission array
    """
    is_myi = (sit >= 2.33)[:, :, np.newaxis]   # (L, T, 1)

    kd_ice  = np.where(is_myi, luts["myi_kd"],  luts["fyi_kd"])   # (L, T, W)
    alb_ice = np.where(is_myi, luts["myi_alb"], luts["fyi_alb"])   # (L, T, W)

    sit_3d = sit[:, :, np.newaxis]  # (L, T, 1)
    sd_3d  = sd[:, :, np.newaxis]   # (L, T, 1)

    t_with_snow = (
        (1 - luts["snow_alb"]) *
        np.exp(-luts["snow_kd"] * sd_3d) *
        np.exp(-kd_ice * sit_3d)
    )
    t_without_snow = (1 - alb_ice) * np.exp(-kd_ice * sit_3d)

    has_snow = (sd > 0)[:, :, np.newaxis]  # (L, T, 1)
    return np.where(has_snow, t_with_snow, t_without_snow)  # (L, T, W)


def _t_ow_direct(zenith_deg: np.ndarray) -> np.ndarray:
    """
    Fresnel transmission for direct radiation.

    Parameters
    ----------
    zenith_deg : (L, T)  solar or lunar zenith angle [degrees]

    Returns
    -------
    (L, T) transmission array; zero where zenith >= 90°
    """
    safe = np.clip(zenith_deg, 0.0, 89.999)
    theta  = np.radians(safe)
    cos_t  = np.cos(theta)
    sin_t  = np.sin(theta)
    cos_r  = np.sqrt(np.maximum(0.0, 1.0 - (N_AIR / N_WATER * sin_t) ** 2))

    Rs = ((N_AIR * cos_t - N_WATER * cos_r) /
          (N_AIR * cos_t + N_WATER * cos_r)) ** 2
    Rp = ((N_AIR * cos_r - N_WATER * cos_t) /
          (N_AIR * cos_r + N_WATER * cos_t)) ** 2

    t = 1.0 - (Rs + Rp) / 2.0
    return np.where(zenith_deg >= 90.0, 0.0, t)  # (L, T)
