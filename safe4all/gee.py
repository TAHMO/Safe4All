"""Small, explicit Google Earth Engine helpers for workshop notebooks.

By default Earth Engine signs in with the workshop service-account key stored as
``service-key.json`` in the shared ``SAFE4ALL-workshop`` Google Drive folder, so
participants do not each need an OAuth login. The key is never stored in this
repository.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .config import country_settings
from .shared_drive import DEFAULT_FOLDER, find_shared_drive_file

SHARED_KEY_FILENAME = "service-key.json"


def initialise_earth_engine(project: str | None = None):
    """Authenticate and return the Earth Engine module.

    Since November 2024, Earth Engine requires every request to be attached
    to a Google Cloud project. Pass ``project`` explicitly or set the
    ``EE_PROJECT`` environment variable (the notebook's Section 1 does this).

    Authentication order:

    1. **Explicit key file:** ``EE_SERVICE_ACCOUNT_KEY`` (or the standard
       ``GOOGLE_APPLICATION_CREDENTIALS``) pointing to a service-account JSON
       key. It must exist.
    2. **Shared Drive key (default):** ``service-key.json`` in the shared
       ``SAFE4ALL-workshop`` Google Drive folder. In Colab, Drive is mounted
       automatically.
    3. **Interactive OAuth:** if no key is found, the usual
       ``ee.Authenticate()`` browser prompt is used.

    Key files must never be committed to the repository.
    """
    project = project or os.environ.get("EE_PROJECT")
    if not project:
        raise RuntimeError(
            "Earth Engine needs a Cloud project. Set EE_PROJECT in Section 1 "
            "of the notebook to a project where you have enabled the Earth "
            "Engine API (see https://console.cloud.google.com/)."
        )
    import ee

    key_path = os.environ.get("EE_SERVICE_ACCOUNT_KEY") or os.environ.get(
        "GOOGLE_APPLICATION_CREDENTIALS"
    )
    if key_path:
        key_path = Path(key_path)
        if not key_path.is_file():
            raise FileNotFoundError(f"Service account key not found: {key_path}")
    else:
        key_path = find_shared_drive_file(SHARED_KEY_FILENAME)
        if key_path is None:
            folder = os.environ.get("SAFE4ALL_SHARED_DRIVE", DEFAULT_FOLDER)
            print(
                f"{SHARED_KEY_FILENAME} was not found in the shared '{folder}' Drive folder; "
                "falling back to interactive Earth Engine sign-in. If the folder was shared "
                "with you, add it to your Drive (Shared with me -> right-click it -> "
                "Organize -> Add shortcut -> My Drive) and re-run."
            )

    if key_path:
        key_info = json.loads(Path(key_path).read_text())
        credentials = ee.ServiceAccountCredentials(key_info["client_email"], str(key_path))
        ee.Initialize(credentials, project=project)
        return _patient(ee)

    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)
    return _patient(ee)


# Retries for "429 Too Many Requests". Many participants share one project, and in
# restricted mode Earth Engine allows very few requests at once; the client waits
# about 1, 2, 4 ... 256 s between attempts (~8 min at most) instead of giving up
# after the default 5 short retries.
EE_MAX_RETRIES = 9


def _patient(ee):
    if hasattr(ee.data, "setMaxRetries"):
        ee.data.setMaxRetries(EE_MAX_RETRIES)
    return ee


def country_boundary(ee, country: str):
    """Return the GAUL level-0 boundary for a selected workshop country."""
    country_settings(country)  # Validate before making a remote request.
    return ee.FeatureCollection("FAO/GAUL/2015/level0").filter(
        ee.Filter.eq("ADM0_NAME", country)
    )
