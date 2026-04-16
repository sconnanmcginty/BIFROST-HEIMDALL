"""
main.py

Master script for computing:
    - Solar direct and diffuse spectral irradiance
    - Lunar direct and diffuse spectral irradiance
    - Spectral PAR (µmol photons m⁻² s⁻¹ nm⁻¹)
    - Broadband PAR (µmol photons m⁻² s⁻¹)
    - Broadband irradiance (W m⁻²)
    - Solar and lunar zenith angles (degrees)

This script loops over:
    L = number of model locations (from config.yaml)
    T = number of timestamps (from config.yaml)
    W = number of wavelengths (30 fixed values: 405-695 nm in 10 nm steps)

-----------------------------------------------------------------------
OUTPUT STRUCTURE (returned dictionary)
-----------------------------------------------------------------------

results = {

    "wavelengths_nm": (W,)
        1-D wavelength array.

    "solar_direct":      (L, T, W)
    "solar_diffuse":     (L, T, W)
        Solar spectral irradiance [W m⁻² nm⁻¹].

    "lunar_direct":      (L, T, W)
    "lunar_diffuse":     (L, T, W)
        Lunar spectral irradiance [W m⁻² nm⁻¹].

    "spectral_par":      (L, T, W)
        Spectral PAR per 10 nm band [µmol photons m⁻² s⁻¹].

    "broadband_par":     (L, T)
        Broadband PAR integrated across all wavelengths.

    "broadband_irradiance": (L, T)
        Broadband irradiance [W m⁻²].

    "solar_zenith":      (L, T)
        Solar zenith angle [degrees].

    "lunar_zenith":      (L, T)
        Lunar zenith angle [degrees].

    "cloud_cover":       (L, T)
        Cloud fraction used in irradiance calculations [0–1].
        Constant 0.0 for cloud_mode=0, 1.0 for cloud_mode=1,
        or spatially/temporally varying ERA5 values for cloud_mode=2.

    -----------------------------------------------------------------------
    Below-surface keys (present only when above_surface_only = false)
    -----------------------------------------------------------------------

    "below_solar_direct":       (L, T, W)
    "below_solar_diffuse":      (L, T, W)
    "below_lunar_direct":       (L, T, W)
    "below_lunar_diffuse":      (L, T, W)
        Below-surface spectral irradiance [W m⁻² nm⁻¹], after BIFROST
        transmission through snow, ice, and open water.

    "below_spectral_par":       (L, T, W)
        Below-surface spectral PAR [µmol photons m⁻² s⁻¹].

    "below_broadband_par":      (L, T)
        Below-surface broadband PAR [µmol photons m⁻² s⁻¹].

    "below_broadband_irradiance": (L, T)
        Below-surface broadband irradiance [W m⁻²].

    "locations": list[Location]
        Latitude/longitude locations used by the model.

    "timestamps": (T,)
        Unix timestamps used in the simulation.

}

-----------------------------------------------------------------------
NOTES
-----------------------------------------------------------------------
• Solar and lunar zenith angles come from Skyfield (topocentric geometry).
• Lunar irradiance is computed by scaling solar irradiance using:
      lunar_albedo(λ) · lunar_phase_fn(phase, λ) · (Ω_moon / π)

-----------------------------------------------------------------------
USAGE
-----------------------------------------------------------------------
    from main import run_model
    results = run_model("config.yaml")

"""

import os
import numpy as np
from datetime import datetime, timezone

# --- Load internal modules ---
from config_loader import load_model_config
from solar_lunar_positions import (
    calculate_zenith_batch,
    calculate_lunar_phase_and_distance,
)
from load_lut_and_interpolate import (
    get_direct_irradiance_spectrum,
    get_diffuse_irradiance_spectrum,
)
from lunar_irradiance import (
    lunar_direct_irradiance_spectrum,
    lunar_diffuse_irradiance_spectrum,
)
from irradiance_conversions import (
    spectral_par_bands,
    broadband_par,
    broadband_irradiance,
)

from save_to_netcdf import save_results_to_netcdf
from fetch_cloud_cover import fetch_cloud_cover
from fetch_ice_data import fetch_ice_data
from compute_transmission import apply_transmission
from compute_depth_attenuation import apply_depth_attenuation

# ---------------------------------------------------------------
# CONSTANT MODEL PARAMETERS
# ---------------------------------------------------------------
WAVELENGTHS_NM = np.arange(405, 696, 10)  # 405–695 nm, 10 nm gap
NUM_W = len(WAVELENGTHS_NM)


# ---------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------
def run_model(config_path="config.yaml"):
    """Run irradiance model and return results dictionary."""

    # -----------------------------------------------------------
    # Load configuration
    # -----------------------------------------------------------
    cfg = load_model_config(config_path)

    time_stamps = cfg.time.timestamps_unix
    num_T = len(time_stamps)

    locations = cfg.locations
    num_L = len(locations)

    cloud_mode = cfg.cloud_mode
    temperature = cfg.atmosphere.temperature_C
    pressure = cfg.atmosphere.pressure_mbar

    # -----------------------------------------------------------
    # Cloud cover: fetch from CDS ERA5 if cloud_mode = 2,
    # otherwise use the scalar value (0 = clear, 1 = overcast).
    # -----------------------------------------------------------
    if cloud_mode == 2:
        print("Fetching cloud cover data from Copernicus CDS ERA5...")
        cloud_array = fetch_cloud_cover(cfg.time, locations)
    else:
        cloud_array = None

    # -----------------------------------------------------------
    # Allocate output arrays
    # All spectral arrays → shape (L, T, W)
    # All broadband arrays → shape (L, T)
    # -----------------------------------------------------------
    solar_direct   = np.zeros((num_L, num_T, NUM_W))
    solar_diffuse  = np.zeros((num_L, num_T, NUM_W))
    lunar_direct   = np.zeros((num_L, num_T, NUM_W))
    lunar_diffuse  = np.zeros((num_L, num_T, NUM_W))

    spectral_par   = np.zeros((num_L, num_T, NUM_W))
    broadband_par_total  = np.zeros((num_L, num_T))
    broadband_irr_total  = np.zeros((num_L, num_T))

    solar_zenith   = np.zeros((num_L, num_T))
    lunar_zenith   = np.zeros((num_L, num_T))
    cloud_cover    = np.zeros((num_L, num_T))

    lunar_distances = np.zeros(num_T)
    lunar_phases    = np.zeros(num_T)
    for ti, ts in enumerate(time_stamps):
        lunar_distances[ti], lunar_phases[ti] = calculate_lunar_phase_and_distance(ts)

    # -----------------------------------------------------------
    # MAIN LOOPS
    # -----------------------------------------------------------
    for li, loc in enumerate(locations):

        print(f"Processing location {li+1}/{num_L}: {loc}")

        lz_arr, sz_arr = calculate_zenith_batch(
            lat=loc.latitude,
            lon=loc.longitude,
            timestamps=time_stamps,
            elevation_m=0,
            temperature_C=temperature,
            pressure_mbar=pressure,
        )
        lunar_zenith[li] = lz_arr   # shape (T,)
        solar_zenith[li] = sz_arr

        for ti in range(num_T):

            lz          = lz_arr[ti]
            sz          = sz_arr[ti]
            lunar_dist  = lunar_distances[ti]
            lunar_phase = lunar_phases[ti]

            cloud_val = float(cloud_array[li, ti]) if cloud_array is not None else float(cloud_mode)
            cloud_cover[li, ti] = cloud_val

            solar_direct[li, ti]  = get_direct_irradiance_spectrum(WAVELENGTHS_NM, sz, cloud_val)
            solar_diffuse[li, ti] = get_diffuse_irradiance_spectrum(WAVELENGTHS_NM, sz, cloud_val)

            lunar_direct[li, ti]  = lunar_direct_irradiance_spectrum(
                wavelengths_nm=WAVELENGTHS_NM,
                lza_deg=lz,
                cloud=cloud_val,
                lunar_distance_au=lunar_dist,
                lunar_phase_angle_deg=lunar_phase,
            )
            lunar_diffuse[li, ti] = lunar_diffuse_irradiance_spectrum(
                wavelengths_nm=WAVELENGTHS_NM,
                lza_deg=lz,
                cloud=cloud_val,
                lunar_distance_au=lunar_dist,
                lunar_phase_angle_deg=lunar_phase,
            )

            # ---------------------------------------------------
            # Spectrum → PAR and broadband totals
            # ---------------------------------------------------
            total_solar = solar_direct[li, ti] + solar_diffuse[li, ti]
            total_lunar = lunar_direct[li, ti] + lunar_diffuse[li, ti]
            total = total_solar + total_lunar

            spectral_par[li, ti] = spectral_par_bands(
                wavelengths_nm=WAVELENGTHS_NM,
                E_lambda_W_m2_nm=total,
                band_width_nm=10.0,
            )

            broadband_par_total[li, ti] = broadband_par(
                wavelengths_nm=WAVELENGTHS_NM,
                E_lambda_W_m2_nm=total,
            )

            broadband_irr_total[li, ti] = broadband_irradiance(total)

    # -----------------------------------------------------------
    # Below-surface transmission (optional)
    # -----------------------------------------------------------
    results = {
        "wavelengths_nm": WAVELENGTHS_NM,
        "solar_direct": solar_direct,
        "solar_diffuse": solar_diffuse,
        "lunar_direct": lunar_direct,
        "lunar_diffuse": lunar_diffuse,
        "spectral_par": spectral_par,
        "broadband_par": broadband_par_total,
        "broadband_irradiance": broadband_irr_total,
        "solar_zenith": solar_zenith,
        "lunar_zenith": lunar_zenith,
        "cloud_cover": cloud_cover,
        "locations": locations,
        "timestamps": time_stamps,
    }

    if not cfg.above_surface_only:
        if cfg.open_water_only:
            print("Open water mode: skipping sea ice data, applying Fresnel transmission only...")
            sic = np.zeros((num_L, num_T))
            sit = np.zeros((num_L, num_T))
            sd  = np.zeros((num_L, num_T))
        else:
            print("Fetching sea ice data from Copernicus Marine...")
            ice_data = fetch_ice_data(cfg.time, locations)
            sic = ice_data["sic"]
            sit = ice_data["sit"]
            sd  = ice_data["sd"]

        print("Applying BIFROST surface transmission...")
        apply_transmission(results, sic, sit, sd)

        if cfg.depth is not None:
            print("Applying depth attenuation...")
            apply_depth_attenuation(results, cfg.depth)

    return results


# ---------------------------------------------------------------
# RUN WHEN EXECUTED DIRECTLY
# ---------------------------------------------------------------
if __name__ == "__main__":
    results = run_model()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    os.makedirs("model_outputs", exist_ok=True)
    output_file = f"model_outputs/model_output_{timestamp}.nc"

    save_results_to_netcdf(results, output_file)

    print(f"Model run complete. Output saved to {output_file}")