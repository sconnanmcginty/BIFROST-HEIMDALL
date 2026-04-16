import numpy as np
import xarray as xr
from datetime import datetime, timezone


def _underwater_vars(results: dict) -> dict:
    """Build underwater data_vars entries when depth attenuation was applied."""
    return {
        "underwater_solar_direct": (
            ("location", "time", "depth", "wavelength"),
            results["underwater_solar_direct"],
            {"units": "W m-2 nm-1"},
        ),
        "underwater_solar_diffuse": (
            ("location", "time", "depth", "wavelength"),
            results["underwater_solar_diffuse"],
            {"units": "W m-2 nm-1"},
        ),
        "underwater_lunar_direct": (
            ("location", "time", "depth", "wavelength"),
            results["underwater_lunar_direct"],
            {"units": "W m-2 nm-1"},
        ),
        "underwater_lunar_diffuse": (
            ("location", "time", "depth", "wavelength"),
            results["underwater_lunar_diffuse"],
            {"units": "W m-2 nm-1"},
        ),
        "underwater_spectral_par": (
            ("location", "time", "depth", "wavelength"),
            results["underwater_spectral_par"],
            {"units": "µmol photons m-2 s-1"},
        ),
        "underwater_broadband_par": (
            ("location", "time", "depth"),
            results["underwater_broadband_par"],
            {"units": "µmol photons m-2 s-1"},
        ),
        "underwater_broadband_irradiance": (
            ("location", "time", "depth"),
            results["underwater_broadband_irradiance"],
            {"units": "W m-2"},
        ),
    }


def _below_surface_vars(results: dict) -> dict:
    """Build the below-surface data_vars entries when transmission was applied."""
    return {
        "below_solar_direct": (
            ("location", "time", "wavelength"),
            results["below_solar_direct"],
            {"units": "W m-2 nm-1"},
        ),
        "below_solar_diffuse": (
            ("location", "time", "wavelength"),
            results["below_solar_diffuse"],
            {"units": "W m-2 nm-1"},
        ),
        "below_lunar_direct": (
            ("location", "time", "wavelength"),
            results["below_lunar_direct"],
            {"units": "W m-2 nm-1"},
        ),
        "below_lunar_diffuse": (
            ("location", "time", "wavelength"),
            results["below_lunar_diffuse"],
            {"units": "W m-2 nm-1"},
        ),
        "below_spectral_par": (
            ("location", "time", "wavelength"),
            results["below_spectral_par"],
            {"units": "µmol photons m-2 s-1"},
        ),
        "below_broadband_par": (
            ("location", "time"),
            results["below_broadband_par"],
            {"units": "µmol photons m-2 s-1"},
        ),
        "below_broadband_irradiance": (
            ("location", "time"),
            results["below_broadband_irradiance"],
            {"units": "W m-2"},
        ),
    }


def save_results_to_netcdf(results: dict, filename: str):
    """
    Save model output dictionary to a NetCDF file.

    Dimensions
    ----------
    location : spatial points (lat/lon pairs)
    time     : model timestamps
    wavelength : spectral bands (nm)

    Parameters
    ----------
    results : dict
        Output dictionary returned by run_model().
    filename : str
        Path to output NetCDF file.
    """

    # ----------------------------
    # Extract coordinates
    # ----------------------------
    wavelengths = results["wavelengths_nm"]

    times_unix = results["timestamps"]
    times_dt = np.array([
        datetime.fromtimestamp(t, tz=timezone.utc)
        for t in times_unix
    ], dtype="datetime64[ns]")

    latitudes = np.array([loc.latitude for loc in results["locations"]])
    longitudes = np.array([loc.longitude for loc in results["locations"]])

    # ----------------------------
    # Build xarray Dataset
    # ----------------------------
    ds = xr.Dataset(

        data_vars=dict(

            # --- Solar ---
            solar_direct=(
                ("location", "time", "wavelength"),
                results["solar_direct"],
                {"units": "W m-2 nm-1"}
            ),
            solar_diffuse=(
                ("location", "time", "wavelength"),
                results["solar_diffuse"],
                {"units": "W m-2 nm-1"}
            ),

            # --- Lunar ---
            lunar_direct=(
                ("location", "time", "wavelength"),
                results["lunar_direct"],
                {"units": "W m-2 nm-1"}
            ),
            lunar_diffuse=(
                ("location", "time", "wavelength"),
                results["lunar_diffuse"],
                {"units": "W m-2 nm-1"}
            ),

            # --- PAR ---
            spectral_par=(
                ("location", "time", "wavelength"),
                results["spectral_par"],
                {"units": "µmol photons m-2 s-1"}
            ),
            broadband_par=(
                ("location", "time"),
                results["broadband_par"],
                {"units": "µmol photons m-2 s-1"}
            ),

            # --- Broadband irradiance ---
            broadband_irradiance=(
                ("location", "time"),
                results["broadband_irradiance"],
                {"units": "W m-2"}
            ),

            # --- Geometry ---
            solar_zenith=(
                ("location", "time"),
                results["solar_zenith"],
                {"units": "degrees"}
            ),
            lunar_zenith=(
                ("location", "time"),
                results["lunar_zenith"],
                {"units": "degrees"}
            ),

            # --- Cloud cover ---
            cloud_cover=(
                ("location", "time"),
                results["cloud_cover"],
                {"units": "fraction", "valid_range": [0, 1]}
            ),

            **(_below_surface_vars(results) if "below_solar_direct" in results else {}),
        **(_underwater_vars(results)    if "underwater_solar_direct" in results else {}),
        ),

        coords=dict(
            wavelength=("wavelength", wavelengths, {"units": "nm"}),
            time=("time", times_dt),
            latitude=("location", latitudes, {"units": "degrees_north"}),
            longitude=("location", longitudes, {"units": "degrees_east"}),
            **({"depth": ("depth", results["depth_steps"], {"units": "m",
                          "positive": "down"})}
               if "depth_steps" in results else {}),
        ),

        attrs=dict(
            title="Solar–Lunar Spectral Irradiance Model Output",
            created=datetime.utcnow().isoformat() + "Z",
            time_reference="UTC",
            conventions="CF-1.8",
        )
    )

    # ----------------------------
    # Save to NetCDF
    # ----------------------------
    ds.to_netcdf(filename)

    print(f"Saved NetCDF file: {filename}")