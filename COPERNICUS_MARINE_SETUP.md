# Copernicus Marine Service Setup Guide

To use `above_surface_only: false` in HEIMDALL, you need a free Copernicus
Marine account and credentials stored on your machine. Follow the steps below.

---

## 1. Create a Copernicus Marine Account

1. Go to [https://marine.copernicus.eu](https://marine.copernicus.eu)
2. Click **Login** then **Register** and complete the registration form.
3. Verify your email address.

---

## 2. Store Your Credentials

With your virtual environment active, run the login command once:

```bash
copernicusmarine login
```

You will be prompted for your username and password. Your credentials are then
saved to `~/.copernicusmarine/.copernicusmarine-credentials` and reused
automatically on all future runs — you do not need to log in again.

---

## 3. Install the Copernicus Marine Package

With your HEIMDALL virtual environment active, run:

```bash
pip install copernicusmarine
```

Or install all dependencies at once:

```bash
pip install -r requirements.txt
```

---

## Troubleshooting

- **"Configuration file not found"** — run `copernicusmarine login` to create
  the credentials file.
- **"403 Forbidden"** — your account may not have access to this dataset. Ensure your account is active at [https://marine.copernicus.eu](https://marine.copernicus.eu).
- **"Dataset not found"** — verify the dataset ID. The current ID used by
  HEIMDALL is `cmems_mod_arc_phy_myint_topaz4_P1D-m`. If this has changed,
  search for `ARCTIC_MULTIYEAR_PHY_002_003` in the Copernicus Marine catalogue
  and update `DATASET_ID` in `fetch_ice_data.py`.

For further help see:
https://help.marine.copernicus.eu/en/collections/4060068-copernicus-marine-toolbox
