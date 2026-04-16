"""
compute_depth_attenuation.py

Propagates below-surface irradiance through the water column using
Beer-Lambert attenuation with the Kostakis et al. (2020) Barents Sea
bio-optical model.

Attenuation coefficients
------------------------
    K_dir(λ) = [a(λ) + b_b(λ)] / μ_dir
    K_dif(λ) = [a(λ) + b_b(λ)] / μ_dif

    μ_dir = cos(θ_w),  θ_w from Snell's law at air-water interface (n = 1.34)
    μ_dif = 0.83 (fixed)

Total absorption a(λ)
---------------------
    a(λ) = a_w(λ)
          + a_CDOM(λ, Sal)     — a_CDOM_440 = -0.012·Sal + 0.464
                                  a_CDOM(λ)  = a_CDOM_440 · exp(-0.0168·(λ-440))
          + a_NAP(λ, Chl)      — a_NAP_440  = 0.0124·Chl^0.724
                                  a_NAP(λ)   = a_NAP_440 · exp(-0.011·(λ-440))
          + a_ph(λ, Chl)       — A_ph · Chl^(1-B_ph)

Total backscattering b_b(λ)
---------------------------
    b_b(λ) = b_bw(λ) + A_bb · Chl^B_bb

Solar direct, solar diffuse, lunar direct, and lunar diffuse are
propagated separately. The surface boundary condition (z=0) is taken
directly from the BIFROST below-surface output.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from irradiance_conversions import spectral_par_bands

# ---------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------
N_WATER = 1.34      # refractive index of seawater (PAR range)
MU_DIF  = 0.83      # fixed mean cosine for diffuse light

WAVELENGTHS_NM = np.arange(405, 696, 10)   # 30 wavelengths

LUT_DIR      = Path(__file__).parent / "LUT"
BIO_OPT_FILE = "bio_optical_parameters.mat"


# ---------------------------------------------------------------
# LUT loading
# ---------------------------------------------------------------

@lru_cache()
def _load_bio_optical_lut() -> dict:
    """Load and cache the bio-optical parameter LUT."""
    path = LUT_DIR / BIO_OPT_FILE
    if not path.exists():
        raise FileNotFoundError(f"Bio-optical LUT not found: {path}")
    mat = loadmat(path)
    return {
        "H20_a_b_bb": np.squeeze(mat["H20_a_b_bb"]),    # (30, 4)
        "Aph_Bph":    np.squeeze(mat["Aph_Bph"]),       # (30, 3)
        "A_bb_B_bb":  np.squeeze(mat["A_bb_B_bb"]),     # (30, 3)
    }


# ---------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------

def _compute_mu_dir(zenith_deg: np.ndarray) -> np.ndarray:
    """
    Mean cosine for direct light after refraction at the air-water interface.

    Applies Snell's law with n_water = 1.34. Valid for all zenith angles —
    sin(zenith)/1.34 is always ≤ 1/1.34 < 1, so arcsin is always defined.
    When direct irradiance is zero (zenith ≥ 90°) the attenuation result
    is zero regardless of μ_dir.
    """
    sin_theta_w = np.sin(np.deg2rad(zenith_deg)) / N_WATER
    theta_w     = np.arcsin(np.clip(sin_theta_w, -1.0, 1.0))
    return np.cos(theta_w)


def _bio_optical_coefficients(
    chl: np.ndarray,
    sal: np.ndarray,
) -> tuple:
    """
    Compute total absorption a(λ) and backscattering b_b(λ) per depth layer.

    Parameters
    ----------
    chl : np.ndarray, shape (num_layers,)
        Chlorophyll concentration [mg m⁻³] for each depth layer.
    sal : np.ndarray, shape (num_layers,)
        Salinity [PSU] for each depth layer.

    Returns
    -------
    a_total  : np.ndarray, shape (num_layers, W)
    bb_total : np.ndarray, shape (num_layers, W)
    """
    lut = _load_bio_optical_lut()
    wl  = WAVELENGTHS_NM                                       # (W,)

    # --- Pure water ---
    a_w  = lut["H20_a_b_bb"][:, 1]                            # (W,)
    bb_w = lut["H20_a_b_bb"][:, 3]                            # (W,)

    # --- Phytoplankton absorption: A_ph × Chl^(1-B_ph) ---
    A_ph = lut["Aph_Bph"][:, 1]                               # (W,)
    B_ph = lut["Aph_Bph"][:, 2]                               # (W,)
    a_ph = A_ph * (chl[:, np.newaxis] ** (1.0 - B_ph))        # (num_layers, W)

    # --- CDOM absorption: from salinity ---
    a_cdom_440 = -0.012 * sal + 0.464                          # (num_layers,)
    a_cdom     = (a_cdom_440[:, np.newaxis]
                  * np.exp(-0.0168 * (wl - 440)))              # (num_layers, W)

    # --- NAP absorption: from chlorophyll ---
    a_nap_440 = 0.0124 * (chl ** 0.724)                        # (num_layers,)
    a_nap     = (a_nap_440[:, np.newaxis]
                 * np.exp(-0.011 * (wl - 440)))                # (num_layers, W)

    # --- Particulate backscattering: A_bb × Chl^B_bb ---
    A_bb = lut["A_bb_B_bb"][:, 1]                             # (W,)
    B_bb = lut["A_bb_B_bb"][:, 2]                             # (W,)
    bb_p = A_bb * (chl[:, np.newaxis] ** B_bb)                # (num_layers, W)

    a_total  = a_w + a_ph + a_cdom + a_nap                    # (num_layers, W)
    bb_total = bb_w + bb_p                                     # (num_layers, W)

    return a_total, bb_total


# ---------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------

def apply_depth_attenuation(results: dict, depth_config) -> None:
    """
    Propagate below-surface irradiance through the water column.

    Reads below_solar_direct, below_solar_diffuse, below_lunar_direct,
    below_lunar_diffuse from the results dict (set by BIFROST) and adds
    underwater_* arrays in-place.

    The surface level (z=0) of each underwater array is set to the
    BIFROST below-surface value. Subsequent levels are computed by
    Beer-Lambert attenuation through each layer.

    Parameters
    ----------
    results : dict
        Model results dict containing below_* and zenith angle arrays.
    depth_config : DepthConfig
        Depth levels, chlorophyll, and salinity from config.
    """
    depth_steps = depth_config.depth_steps      # (num_steps,) including 0
    chl         = depth_config.chl              # (num_layers,)
    sal         = depth_config.sal              # (num_layers,)

    num_steps  = len(depth_steps)               # depth levels including surface
    num_layers = num_steps - 1                  # number of layer transitions

    L, T, W = results["below_solar_direct"].shape

    print(f"  Propagating irradiance through {num_layers} depth layers "
          f"(0–{depth_steps[-1]:.0f} m, Δz={depth_steps[1]-depth_steps[0]:.0f} m)...")

    # --- Allocate output arrays (L, T, num_steps, W) ---
    uw_solar_direct  = np.zeros((L, T, num_steps, W))
    uw_solar_diffuse = np.zeros((L, T, num_steps, W))
    uw_lunar_direct  = np.zeros((L, T, num_steps, W))
    uw_lunar_diffuse = np.zeros((L, T, num_steps, W))

    # --- Surface boundary condition (z=0) from BIFROST ---
    uw_solar_direct [:, :, 0, :] = results["below_solar_direct"]
    uw_solar_diffuse[:, :, 0, :] = results["below_solar_diffuse"]
    uw_lunar_direct [:, :, 0, :] = results["below_lunar_direct"]
    uw_lunar_diffuse[:, :, 0, :] = results["below_lunar_diffuse"]

    # --- Mean cosines for direct light (L, T) ---
    mu_dir_solar = _compute_mu_dir(results["solar_zenith"])    # (L, T)
    mu_dir_lunar = _compute_mu_dir(results["lunar_zenith"])    # (L, T)

    # --- Bio-optical coefficients (num_layers, W) ---
    a_total, bb_total = _bio_optical_coefficients(chl, sal)
    K_sum = a_total + bb_total                                 # (num_layers, W)

    # --- Layer-by-layer Beer-Lambert propagation ---
    for zi in range(num_layers):
        dz = float(depth_steps[zi + 1] - depth_steps[zi])

        # Diffuse: K_dif is (W,) — broadcasts with (L, T, W) from the right
        K_dif = K_sum[zi] / MU_DIF                            # (W,)

        # Direct: K_dir is (L, T, W)
        K_dir_solar = (K_sum[zi][np.newaxis, np.newaxis, :]
                       / mu_dir_solar[:, :, np.newaxis])
        K_dir_lunar = (K_sum[zi][np.newaxis, np.newaxis, :]
                       / mu_dir_lunar[:, :, np.newaxis])

        uw_solar_direct [:, :, zi+1, :] = (uw_solar_direct [:, :, zi, :]
                                            * np.exp(-K_dir_solar * dz))
        uw_solar_diffuse[:, :, zi+1, :] = (uw_solar_diffuse[:, :, zi, :]
                                            * np.exp(-K_dif * dz))
        uw_lunar_direct [:, :, zi+1, :] = (uw_lunar_direct [:, :, zi, :]
                                            * np.exp(-K_dir_lunar * dz))
        uw_lunar_diffuse[:, :, zi+1, :] = (uw_lunar_diffuse[:, :, zi, :]
                                            * np.exp(-K_dif * dz))

    # --- Derived quantities ---
    total_uw = (uw_solar_direct + uw_solar_diffuse
                + uw_lunar_direct + uw_lunar_diffuse)          # (L, T, num_steps, W)

    # spectral_par_bands broadcasts over last axis (W) → (L, T, num_steps, W)
    uw_spectral_par  = spectral_par_bands(WAVELENGTHS_NM, total_uw)
    uw_broadband_par = np.sum(uw_spectral_par, axis=-1)        # (L, T, num_steps)
    uw_broadband_irr = np.sum(total_uw * 10.0, axis=-1)        # (L, T, num_steps)

    # --- Store in results dict ---
    results["underwater_solar_direct"]        = uw_solar_direct
    results["underwater_solar_diffuse"]       = uw_solar_diffuse
    results["underwater_lunar_direct"]        = uw_lunar_direct
    results["underwater_lunar_diffuse"]       = uw_lunar_diffuse
    results["underwater_spectral_par"]        = uw_spectral_par
    results["underwater_broadband_par"]       = uw_broadband_par
    results["underwater_broadband_irradiance"] = uw_broadband_irr
    results["depth_steps"]                    = depth_steps
