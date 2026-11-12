"""Minimal access to TAHMO precipitation data.

Credentials are never stored in this repository. They come from environment
variables or from ``tahmo-credentials.txt`` in the shared SAFE4ALL-workshop
Google Drive folder.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests


API_ROOT = "https://datahub.tahmo.org"
COUNTRY_CODES = {"Kenya": "KE", "Zimbabwe": "ZW", "Ghana": "GH"}


def credentials() -> tuple[str, str]:
    """Read TAHMO credentials from environment variables, never from Git."""
    key = os.environ.get("TAHMO_API_KEY")
    secret = os.environ.get("TAHMO_API_SECRET")
    if not key or not secret:
        raise RuntimeError(
            "TAHMO credentials not found. Put tahmo-credentials.txt in the shared SAFE4ALL-workshop "
            "Drive folder, or set TAHMO_API_KEY and TAHMO_API_SECRET."
        )
    return key, secret


def _get(endpoint: str, params: dict | None = None) -> dict:
    key, secret = credentials()
    response = requests.get(
        f"{API_ROOT}/{endpoint}", params=params, auth=(key, secret), timeout=60
    )
    response.raise_for_status()
    return response.json()


def stations(country: str) -> pd.DataFrame:
    """Fetch station metadata for a workshop country."""
    if country not in COUNTRY_CODES:
        raise ValueError(f"Unsupported country: {country}")
    payload = _get("services/assets/v2/stations", {"sort": "code"})
    frame = pd.json_normalize(payload["data"])
    return frame.loc[
        frame["location.countrycode"].eq(COUNTRY_CODES[country])
        & ~frame["code"].astype(str).str.contains("TH", na=False)
    ].copy()


def station_precipitation(station_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch controlled precipitation observations for one station as long data.

    Quality flags are retained; filtering happens explicitly in the QC step.
    """
    payload = _get(
        f"services/measurements/v2/stations/{station_code}/measurements/controlled",
        {"start": f"{start_date}T00:00:00Z", "end": f"{end_date}T23:59:59Z", "variable": "pr"},
    )
    rows = []
    for result in payload.get("results", []):
        for series in result.get("series", []):
            columns = series.get("columns", [])
            for values in series.get("values", []):
                row = dict(zip(columns, values))
                if row.get("variable") == "pr":
                    row["station"] = station_code
                    rows.append(row)
    return pd.DataFrame(rows)


def save_station_metadata(path: Path, country: str) -> None:
    stations(country).to_csv(path, index=False)


def _skip_reason(error: Exception) -> str:
    status = getattr(getattr(error, "response", None), "status_code", None)
    if status in (401, 403):
        return "no access for this API account"
    if status == 404:
        return "station not found"
    return str(error) or type(error).__name__


def collect_station_rainfall(station_codes, start_date: str, end_date: str, max_stations: int,
                             cache_dir: Path | None = None, fetch=None, log=print):
    """Fetch rainfall for up to ``max_stations`` stations that actually return data.

    Stations are tried in order until ``max_stations`` of them return
    observations for the period, so stations the API account cannot access, or
    that have no data in the period, do not use up the station budget. Only
    non-empty results are cached in ``cache_dir`` (an empty or unreadable
    cached file is removed and fetched again).

    Returns ``(observations, used_codes, skipped)`` where ``skipped`` maps a
    station code to the reason it was not used.
    """
    fetch = fetch or station_precipitation
    frames, used, skipped = [], [], {}
    for code in station_codes:
        if len(used) >= max_stations:
            break
        cache_path = Path(cache_dir) / f"{code}_precip.csv" if cache_dir else None
        frame = None
        if cache_path and cache_path.is_file():
            try:
                frame = pd.read_csv(cache_path)
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                frame = None
            if frame is None or frame.empty:
                cache_path.unlink()  # an empty result cached by an older version
                frame = None
        if frame is None:
            try:
                frame = fetch(code, start_date, end_date)
            except Exception as error:
                skipped[code] = _skip_reason(error)
                continue
            if frame.empty:
                skipped[code] = "no rainfall data in the period"
                continue
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(cache_path, index=False)
        frames.append(frame)
        used.append(code)
    tried = len(used) + len(skipped)
    log(f"TAHMO: {len(used)} station(s) with data out of {tried} tried "
        f"(max {max_stations}).")
    if skipped:
        reasons = pd.Series(skipped).value_counts()
        for reason, count in reasons.items():
            log(f"  skipped {count}: {reason}")
    observations = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return observations, used, skipped
