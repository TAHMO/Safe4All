# Climate Risk Assessment (Floods and Drought)

**SAFE4ALL Climate Risk Assessment Workshop notebook**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TAHMO/Safe4All/blob/main/SAFE4ALL_Climate_Risk_Assessment_Workshop.ipynb)

Hands-on Jupyter notebook for **flood and drought risk mapping** in **Kenya**,
**Zimbabwe**, and **Ghana** (default: Ghana, whole country). It follows a
hazard x exposure x vulnerability workflow in Google Earth Engine and produces
classified risk maps and district risk statistics, with TAHMO station
observations for ground validation and TAMSAT for drought context.

## Run it — pick one

**Google Colab (default, no local setup):** click the badge above, or open
<https://colab.research.google.com/github/TAHMO/Safe4All/blob/main/SAFE4ALL_Climate_Risk_Assessment_Workshop.ipynb>,
then **Runtime -> Run all**. Section 0 of the notebook clones this repo into
the Colab runtime and installs anything missing — nothing else to configure.

**Local Jupyter or Anaconda:** clone this repo and run the notebook from its
root folder with **Run -> Run All Cells**. Section 0 detects that the
`safe4all` package is already available locally and skips the Colab-only
steps; any missing Python packages are installed automatically. A conda
environment file is provided if you prefer to pre-install everything instead:

```bash
git clone https://github.com/TAHMO/Safe4All.git
cd Safe4All
conda env create -f environment.yml
conda activate safe4all-climate-risk
python -m ipykernel install --user --name safe4all-climate-risk --display-name "Python (SAFE4ALL)"
jupyter lab
```

Open `SAFE4ALL_Climate_Risk_Assessment_Workshop.ipynb` and select the
**Python (SAFE4ALL)** kernel. All three environments (Colab, Jupyter,
Anaconda) run the same notebook and produce the same screening outputs.

To verify a local installation outside the notebook, run:

```bash
python -m unittest tests/test_project.py
```

## Workshop flow

Run `SAFE4ALL_Climate_Risk_Assessment_Workshop.ipynb` from top to bottom. It
follows ten steps, and each one has its own module in the `safe4all` package:

| Step | What it does | Module |
|---|---|---|
| 1-2 | Select country / region / district / river basin; load GAUL boundaries | `aoi.py` |
| 3 | Climate: CHIRPS, ERA5-Land, TAHMO QC and validation (optional), TAMSAT (optional) | `drought.py`, `tahmo.py`, `qc.py`, `validation.py` |
| 4 | Drought hazard: SPI-1/3/6, rainfall anomaly, NDVI anomaly, VCI, soil-moisture anomaly | `drought.py` |
| 5 | Flood hazard: extreme rainfall, Sentinel-1 flood frequency, DEM, slope, HAND, flow accumulation, distance from river | `flood.py` |
| 6 | Exposure: WorldPop, cropland, settlements, infrastructure | `exposure.py` |
| 7 | Vulnerability: socioeconomic indicators, land use, agricultural dependence | `vulnerability.py` |
| 8 | Risk = Hazard x Exposure x Vulnerability (flood, drought, multi-hazard) | `model.py` |
| 9 | Classify: Very Low, Low, Moderate, High, Very High | `model.py` |
| 10 | Export: GeoTIFF, CSV, Shapefile, interactive HTML maps, district statistics, PNG figures | `export.py` |

Study area, periods, weights, and classification method are set in Section 1
of the notebook. Per-country flood/dry/rainy-season months, default weights,
and land-use susceptibility scores are in `safe4all/config.py`.

## Outputs

- **Google Drive** (`SAFE4ALL_Flood_Drought_Risk`): GeoTIFFs of hazard,
  exposure, vulnerability, risk index and risk class for flood, drought and
  multi-hazard, the separate exposure layers (population per km2, cropland
  and settlement cover fractions, infrastructure index), plus the district
  statistics as CSV and Shapefile.
- **Workshop cache** (`outputs/<study area>/`): district risk statistics CSV
  (with population per km2, cropland %, settlement % and infrastructure index per district),
  zipped shapefile, interactive district map (self-contained HTML), interactive
  raster map (HTML, including population, cropland, settlement and
  infrastructure layers), area and population per risk class, district ranking
  chart and static risk maps (PNG), and one zip of everything.

## Data strategy

- **TAHMO stations:** local observations, station QC, and product validation.
- **TAMSAT:** primary Africa-focused rainfall and soil-moisture source for
  drought monitoring and long-term risk context.
- **CHIRPS:** independent long-term rainfall comparator available in Earth
  Engine.
- **IMERG:** half-hourly event-rainfall context for flood case studies.

The project caches every approved live extract. In Colab, set
`SAFE4ALL_SHARED_DRIVE` in the notebook to the exact Shared Drive name before
starting; the default is `SAFE4ALL-workshop`. If it cannot be found, the
notebook falls back to a workshop folder in the participant's MyDrive. Local
Jupyter/Anaconda use `.safe4all-cache/` inside the project folder.

## Google Earth Engine access

Earth Engine now requires every request to be attached to a Google Cloud
project. `EE_PROJECT` in Section 1 of the notebook is pre-set to the SAFE4ALL
workshop project; point it at a different project (with the Earth Engine API
enabled — free to create at <https://console.cloud.google.com/>) if you want
to run this against your own instead.

Authentication (tried in this order):

- **Shared Drive service-account key (default, recommended for the workshop):**
  the notebook reads `service-key.json` from the shared **`SAFE4ALL-workshop`**
  Google Drive folder, so every participant can "Run all" without an
  individual OAuth login. In Colab the notebook mounts your Drive and finds the
  folder whether it is a Shared Drive or a folder shared with you. If it was
  only shared with you, add it to your Drive first: *Shared with me ->
  right-click `SAFE4ALL-workshop` -> Organize -> Add shortcut -> My Drive*.
  Locally, Google Drive for Desktop must be syncing that folder. The service
  account needs Earth Engine access on `EE_PROJECT`.
- **Explicit key file:** set `EE_SERVICE_ACCOUNT_KEY` in Section 1 (or the
  `EE_SERVICE_ACCOUNT_KEY` / `GOOGLE_APPLICATION_CREDENTIALS` environment
  variable) to the local path of a service-account JSON key. This takes
  priority over the Shared Drive key.
- **Interactive:** if no key is found, the notebook prompts you to sign in
  with `ee.Authenticate()`.

**Never commit a key file to this repository.** Keep it in the shared Drive
folder, outside the project folder, or inside `.safe4all-cache/` (already
git-ignored). If a key is ever pasted into a chat, ticket, or shared document,
treat it as compromised and rotate it immediately in
[IAM & Admin -> Service Accounts](https://console.cloud.google.com/iam-admin/serviceaccounts),
then replace `service-key.json` in the `SAFE4ALL-workshop` folder.

## TAHMO API credentials

The notebook reads the TAHMO API key and secret from **`tahmo-credentials.txt`**
in the same shared `SAFE4ALL-workshop` Google Drive folder, so participants are
not asked to type them. Create the file with exactly these two lines:

```text
TAHMO_API_KEY=your-api-key
TAHMO_API_SECRET=your-api-secret
```

Blank lines and `#` comments are ignored. If `TAHMO_API_KEY` and
`TAHMO_API_SECRET` are already set as environment variables, they are used
instead. If neither is available, TAHMO is skipped and the risk assessment still
runs.

The project does not store credentials or API keys in code. Never commit
`tahmo-credentials.txt` or `service-key.json` to this repository, and share the
Drive folder only with workshop participants.

## Interpretation note

The flood and drought maps are **screening-level risk products**, not
operational forecasts or disaster warnings. Indices are relative to the
selected study area. Calibrate weights and thresholds with local experts and
validate the classes against observed flood extents, disaster records, and
drought impacts before using them for decisions.
