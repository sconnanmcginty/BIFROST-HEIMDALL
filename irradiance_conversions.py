# irradiance_conversions.py

from dataclasses import dataclass
from typing import Sequence

import numpy as np

# Physical constants
H = 6.62607015e-34        # Planck constant [J s]
C = 2.99792458e8          # Speed of light [m s⁻¹]
N_A = 6.02214076e23       # Avogadro's number [mol⁻¹]


def spectral_par_bands(
    wavelengths_nm: Sequence[float],
    E_lambda_W_m2_nm: Sequence[float],
    band_width_nm: float = 10.0,
) -> np.ndarray:
    """
    Compute spectral PAR (µmol photons m⁻² s⁻¹) for multiple wavelength
    bands simultaneously.

    Parameters
    ----------
    wavelengths_nm : array-like
        Central wavelengths of each spectral band [nm].
    E_lambda_W_m2_nm : array-like
        Spectral irradiance values at those wavelengths [W m⁻² nm⁻¹].
    band_width_nm : float, optional
        Width of each wavelength band [nm]. Default is 10 nm.

    Returns
    -------
    par_umol_m2_s : np.ndarray
        Array of spectral PAR values for each band, expressed in
        µmol photons m⁻² s⁻¹. The array has the same shape as the input
        wavelength array.

    """
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)
    E_lambda_W_m2_nm = np.asarray(E_lambda_W_m2_nm, dtype=float)

    # Convert to meters
    wavelengths_m = wavelengths_nm * 1e-9

    # Convert to band irradiance (W m⁻² over Δλ)
    E_band = E_lambda_W_m2_nm * band_width_nm

    # Convert irradiance to photon flux (photons m⁻² s⁻¹)
    photons = E_band * wavelengths_m / (H * C)

    # Convert photons → µmol photons
    par_umol_m2_s = photons / N_A * 1e6

    return par_umol_m2_s



def broadband_irradiance(
    E_lambda_W_m2_nm: Sequence[float],
    band_width_nm: float = 10.0,
) -> float:
    """
    Sum spectral irradiance over wavelength to get broadband
    irradiance [W m⁻²].

    """
    E_lambda_W_m2_nm = np.asarray(E_lambda_W_m2_nm, dtype=float)

    return np.sum(E_lambda_W_m2_nm * band_width_nm) 



def broadband_par(
    wavelengths_nm: Sequence[float],
    E_lambda_W_m2_nm: Sequence[float],
    band_width_nm: float = 10.0,
) -> float:
    """
    Compute broadband PAR (µmol photons m⁻² s⁻¹)
    by summing PAR contributions from each wavelength band.

    Parameters
    ----------
    wavelengths_nm : array-like
        Central wavelengths [nm].
    E_lambda_W_m2_nm : array-like
        Spectral irradiance [W m⁻² nm⁻¹] at each wavelength.
    band_width_nm : float
        Bandwidth represented by each wavelength (defaults to 10 nm).

    Returns
    -------
    float
        Broadband PAR in µmol photons m⁻² s⁻¹.
    """
    par_values = spectral_par_bands(
        wavelengths_nm,
        E_lambda_W_m2_nm,
        band_width_nm=band_width_nm,
    )

    return float(np.sum(par_values))