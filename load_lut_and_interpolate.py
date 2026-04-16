"""
load_lut_and_interpolate.py

Universal loader and interpolator for lunar and solar irradiance lookup
tables (LUTs).

This module provides a consistent interface for loading LUT .mat files,
building cached interpolators, and retrieving values using clean keyword-
based queries. LUT axes are automatically constructed from metadata or
loaded directly from .mat files.

The following LUT types are supported:

-----------------------------------------------------------------------
LUNAR LOOKUP TABLES
-----------------------------------------------------------------------

1. Lunar Phase Function
    - File: "lunar_phase_lut.mat"
    - LUT: Phase_Functions (shape: 14 x 30)
    - Axes:
        * phase_angle_axis: Angles (shape: 1 x 14)
        * wavelength_axis: 405-695 nm (10 nm steps)

    Notes:
        * Phase angle values range from 0° (full moon) to 180° (new moon).
        * A lunar phase function value of 1 corresponds to full moon and
          0 corresponds to new moon, representing the illuminated fraction
          of the lunar disk.
        * Reference: Miller & Turner (2009),
          DOI: 10.1109/TGRS.2009.2012696

2. Lunar Spectral Albedo
    - File: "lunar_phase_lut.mat"
    - LUT: Spectral_Albedos (shape: 1 x 4)
    - Axis:
        * wavelength_axis: [400, 500, 600, 700]


-----------------------------------------------------------------------
SOLAR LOOKUP TABLES
-----------------------------------------------------------------------

3. Direct Irradiance (Ed_Direct)
    - File: "solar_irradiance_lut.mat"
    - LUT: Ed_Direct (shape: 30 x 10 x 11)
    - Axes:
        * wavelength_axis: 405-695 nm (10 nm steps)
        * sza_axis: 0-90° (10° steps)
        * cloud_axis: 0.0-1.0 (0.1 steps)

    Notes:
        * Direct irradiance is physically zero beyond 90° solar zenith.
        * For SZA > 90°, values are clamped to 90° before interpolation.


4. Diffuse Irradiance (Ed_Diffuse)
    - File: "solar_irradiance_lut.mat"
    - LUT: Ed_Diffuse (shape: 30 x 19 x 11)
    - Axes:
        * wavelength_axis: 405-695 nm (10 nm steps)
        * sza_axis: diffuse_zeniths (19 values), consisting of:
              - 0-80° in 10° steps, followed by
              - extended empirical angles
        * cloud_axis: 0.0-1.0 (0.1 steps)

    Notes:
        * Diffuse irradiance extends beyond the solar horizon.
        * For SZA > 108°, values are clamped to the LUT maximum (108°). 
          This reflects baseline irradiance value where SZA no longer
          a dependent variable.


-----------------------------------------------------------------------
GENERAL NOTES
-----------------------------------------------------------------------

- All interpolators are built using scipy.interpolate.RegularGridInterpolator.
- All heavy operations (MAT loading and interpolator setup) are cached
  using @lru_cache for high efficiency.
- LUTs are queried using:

      get_lut_value(lut_name, axis1=value1, axis2=value2, ...)

- Wrapper functions are provided for convenience:

      get_direct_irradiance(...)
      get_diffuse_irradiance(...)
      get_lunar_phase_function(...)
      get_lunar_spectral_albedo(...)

These wrappers apply physical constraints such as SZA clamping and
automatically route user input to the correct LUT.
"""

from pathlib import Path
from functools import lru_cache

import numpy as np
from scipy.io import loadmat
from scipy.interpolate import RegularGridInterpolator


# LUT Registry — describes where each LUT lives and what axes it uses
LUT_DIR = Path(__file__).parent / "LUT"

if not LUT_DIR.is_dir():
    raise FileNotFoundError(
        f"Required LUT directory not found:\n{LUT_DIR}\n"
        "Ensure that the LUT folder is located next to this module."
    )


LUT_REGISTRY = {

    # -----------------------------------------------------------
    # Lunar Phase Function LUT
    # -----------------------------------------------------------
    "lunar_phase_fn": {
        "file": "lunar_phase_lut.mat",
        "axes": ["phase_angle", "wavelength"],
        "mat_vars": {
            "lut": "Phase_Functions",      # (14 × 30)
            "phase_angle_axis": "Phase_Angles",  # (1 × 14)
        },
        "wavelength_axis": list(range(405, 696, 10)),  # 30 wavelengths
    },

    # -----------------------------------------------------------
    # Lunar Spectral Albedo LUT
    # -----------------------------------------------------------
    "lunar_albedo": {
        "file": "lunar_phase_lut.mat",
        "axes": ["wavelength"],
        "mat_vars": {
            "lut": "Spectral_Albedos",      # (1 × 4)
        },
        "wavelength_axis": [400, 500, 600, 700],
    },

    # -----------------------------------------------------------
    # SOLAR DIRECT IRRADIANCE LUT
    # -----------------------------------------------------------
    "direct_irradiance": {
        "file": "solar_irradiance_lut.mat",

        "axes": ["wavelength", "sza", "cloud"],

        "mat_vars": {
            "lut": "Ed_Direct",             # (30 x 10 x 11)
        },

        "wavelength_axis": list(range(405, 696, 10)),     # 405–695 nm (30 values)
        "cloud_axis": [i / 10 for i in range(11)],        # 0.0–1.0 in 0.1 steps
        "sza_axis": list(range(0, 91, 10)),               # 0–90° in 10° steps (generated)
    },


    # -----------------------------------------------------------
    # SOLAR DIFFUSE IRRADIANCE LUT
    # -----------------------------------------------------------
    "diffuse_irradiance": {
        "file": "solar_irradiance_lut.mat",

        "axes": ["wavelength", "sza", "cloud"],

        "mat_vars": {
            "lut": "Ed_Diffuse",           
        },

        "sza_axis_mat": "diffuse_zeniths",

        "wavelength_axis": list(range(405, 696, 10)),
        "cloud_axis": [i / 10 for i in range(11)],
    },
}

@lru_cache()
def _load_mat(file_name: str):
    """Load a MAT file and return the dictionary."""
    path = LUT_DIR / file_name

    if not path.exists():
        raise FileNotFoundError(f"LUT file not found: {path}")

    return loadmat(path)

@lru_cache()
def _build_interpolator(lut_name: str):
    """
    Build and cache a RegularGridInterpolator for the given LUT.
    """
    meta = LUT_REGISTRY[lut_name]
    mat = _load_mat(meta["file"])

    # Get LUT array and squeeze
    lut_key = meta["mat_vars"]["lut"]
    lut = np.squeeze(mat[lut_key])

    axes = []

    # Build axes in order defined by meta["axes"]
    for ax in meta["axes"]:

        # Wavelength axis
        if ax == "wavelength":
            axes.append(np.asarray(meta["wavelength_axis"]))

        # Cloud axis
        elif ax == "cloud":
            axes.append(np.asarray(meta["cloud_axis"]))

        # Solar zenith axis
        elif ax == "sza":

            # DIRECT irradiance uses generated SZA axis
            if "sza_axis" in meta:
                axes.append(np.asarray(meta["sza_axis"]))

            # DIFFUSE irradiance uses MAT-file axis
            elif "sza_axis_mat" in meta:
                axis = mat[meta["sza_axis_mat"]].squeeze()
                axes.append(axis)

            else:
                raise ValueError(
                    f"No solar zenith axis available for LUT '{lut_name}'."
                )

        # Lunar phase angle axis (for lunar_phase_fn LUT)
        elif ax == "phase_angle":
            phase_axis_key = meta["mat_vars"]["phase_angle_axis"]
            axis = mat[phase_axis_key].squeeze()
            axes.append(axis)

        # Unknown axis
        else:
            raise ValueError(f"Unknown axis '{ax}' for LUT '{lut_name}'.")

    # Build and return the interpolator
    return RegularGridInterpolator(tuple(axes), lut)


# Public function: generic LUT value retriever
def get_lut_value(lut_name: str, **kwargs):
    """
    Retrieve a value from a LUT by providing the necessary axes.

    Examples:
        get_lut_value("lunar_phase_fn", phase_angle=20, wavelength=550)
        get_lut_value("lunar_albedo", wavelength=600)
    """

    if lut_name not in LUT_REGISTRY:
        raise ValueError(f"Unknown LUT name '{lut_name}'. "
                         f"Available: {list(LUT_REGISTRY.keys())}")

    meta = LUT_REGISTRY[lut_name]
    axes_order = meta["axes"]

    # Ensure user provided exactly the expected axes
    if set(kwargs.keys()) != set(axes_order):
        raise ValueError(
            f"LUT '{lut_name}' expects axes {axes_order}, but received {list(kwargs.keys())}"
        )

    # Order inputs for interpolator
    query_point = [kwargs[ax] for ax in axes_order]

    interp = _build_interpolator(lut_name)

    return float(interp([query_point])[0])


def get_direct_irradiance_spectrum(wavelengths: np.ndarray, sza: float, cloud: float) -> np.ndarray:
    """
    Direct irradiance for all wavelengths at once.

    Parameters
    ----------
    wavelengths : np.ndarray, shape (W,)
    sza, cloud  : float

    Returns
    -------
    np.ndarray, shape (W,)
    """
    sza = min(float(sza), 90.0)
    interp = _build_interpolator("direct_irradiance")
    n   = len(wavelengths)
    pts = np.column_stack([wavelengths, np.full(n, sza), np.full(n, cloud)])
    return interp(pts)


def get_diffuse_irradiance_spectrum(wavelengths: np.ndarray, sza: float, cloud: float) -> np.ndarray:
    """
    Diffuse irradiance for all wavelengths at once.

    Parameters
    ----------
    wavelengths : np.ndarray, shape (W,)
    sza, cloud  : float

    Returns
    -------
    np.ndarray, shape (W,)
    """
    sza = min(float(sza), 108.0)
    interp = _build_interpolator("diffuse_irradiance")
    n   = len(wavelengths)
    pts = np.column_stack([wavelengths, np.full(n, sza), np.full(n, cloud)])
    return interp(pts)


def get_lunar_albedo_spectrum(wavelengths: np.ndarray) -> np.ndarray:
    """
    Lunar spectral albedo for all wavelengths at once.

    Returns
    -------
    np.ndarray, shape (W,)
    """
    interp = _build_interpolator("lunar_albedo")
    pts = wavelengths.reshape(-1, 1)
    return interp(pts)


def get_lunar_phase_fn_spectrum(phase_angle: float, wavelengths: np.ndarray) -> np.ndarray:
    """
    Lunar phase function for all wavelengths at once.

    Returns
    -------
    np.ndarray, shape (W,)
    """
    interp = _build_interpolator("lunar_phase_fn")
    n   = len(wavelengths)
    pts = np.column_stack([np.full(n, phase_angle), wavelengths])
    return interp(pts)


def get_direct_irradiance(wavelength: float, sza: float, cloud: float) -> float:
    """
    Direct irradiance lookup.

    Notes
    -----
    Direct solar irradiance is only defined for SZA ≤ 90°.
    Values above this limit are clamped to 90° before interpolation.
    """
    if sza > 90:
        sza = 90

    return get_lut_value(
        "direct_irradiance",
        wavelength=wavelength,
        sza=sza,
        cloud=cloud
    )

def get_diffuse_irradiance(wavelength: float, sza: float, cloud: float) -> float:
    """
    Diffuse irradiance lookup.

    Notes
    -----
    Diffuse skylight LUT supports solar zenith angles up to 108°.
    Values above this limit are clamped to 108° before interpolation.
    """
    if sza > 108:
        sza = 108

    return get_lut_value(
        "diffuse_irradiance",
        wavelength=wavelength,
        sza=sza,
        cloud=cloud
    )
