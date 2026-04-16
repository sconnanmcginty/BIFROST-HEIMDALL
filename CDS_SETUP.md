# Copernicus Climate Data Store (CDS) Setup Guide

To use `cloud_mode: 2` in HEIMDALL, you need a free CDS account and a credentials
file on your machine. Follow the steps below for your operating system.

---

## 1. Create a CDS Account

1. Go to [https://cds.climate.copernicus.eu](https://cds.climate.copernicus.eu)
2. Click **Login** then **Register** and complete the registration form.
3. Verify your email address.

---

## 2. Get Your API Key

1. Log in and go to your profile:
   [https://cds.climate.copernicus.eu/profile](https://cds.climate.copernicus.eu/profile)
2. Scroll to the **"API setup"** section.
3. Copy the two-line snippet shown — it will look like this:

```
url: https://cds.climate.copernicus.eu/api
key: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

---

## 3. Create Your `.cdsapirc` File

### macOS

Open Terminal and run:

```bash
touch ~/.cdsapirc
open -e ~/.cdsapirc
```

Paste your two-line snippet into the file, save, and close.

### Windows

1. Open Notepad.
2. Paste your two-line snippet.
3. Save the file as `.cdsapirc` (with the leading dot) in your home directory:
   `C:\Users\YourUsername\.cdsapirc`
   - In the Save dialog, set **Save as type** to **All Files** to prevent Notepad
     adding a `.txt` extension.

---

## 4. Install the CDS API Package

With your HEIMDALL virtual environment active, ensure you have run:

```bash
pip install -r requirements.txt
```

---

## Troubleshooting

- **"Missing/incomplete configuration file"** — your `.cdsapirc` is not in your
  home directory, or the `url`/`key` lines are missing.
- **"You are not authorised"** — ensure your CDS account is active and that you are able to log in at [https://cds.climate.copernicus.eu](https://cds.climate.copernicus.eu).
- **403 / authentication error** — double-check that you copied the full key from
  your CDS profile page with no extra spaces.

For further help see the official guide:
https://confluence.ecmwf.int/display/CKB/Climate+Data+Store+%28CDS%29+User+Guide
