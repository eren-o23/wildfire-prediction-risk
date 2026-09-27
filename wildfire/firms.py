"""Pull historical VIIRS hotspot detections for California from NASA FIRMS."""
import io
import os
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "firms"
CA_BBOX = "-124.5,32.5,-114.1,42.0"  # west,south,east,north
# why: _SP = standard processing, the corrected archive. Live serving (Phase 1) will use _NRT
# data, which FIRMS revises later. Recording which version we trained on explains any gap
# between offline and live accuracy (tracked in Phase 6).
SOURCES = ("VIIRS_SNPP_SP", "VIIRS_NOAA20_SP")
SEASONS = (2022, 2023, 2024)  # Open-Meteo historical forecasts start ~2022
URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{src}/{bbox}/{days}/{start}"
MAX_DAYS = 10  # FIRMS area API limit per request


def _fetch_chunk(key: str, src: str, start: date, days: int) -> pd.DataFrame:
    # why: each raw response is cached untouched, one file per request. Everything downstream is
    # recomputed from these files, so changing clustering or labels never needs a refetch, and
    # any result can be traced back to its source data (Phase 0 version of the staging table).
    path = RAW_DIR / f"{src}_{start}_{days}d.parquet"
    if path.exists():
        return pd.read_parquet(path)
    url = URL.format(key=key, src=src, bbox=CA_BBOX, days=days, start=start)
    for attempt in range(4):
        r = requests.get(url, timeout=120)
        if r.ok and r.text.startswith("latitude"):
            break
        # why: FIRMS delays and rate limits are routine, so back off and retry, don't crash
        time.sleep(5 * 2**attempt)
    else:
        raise RuntimeError(f"FIRMS {src} {start}: HTTP {r.status_code} {r.text[:200]}")
    df = pd.read_csv(io.StringIO(r.text), dtype={"acq_time": str, "confidence": str})
    df["source"] = src
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def pull(seasons=SEASONS, sources=SOURCES) -> None:
    """Download Jun-Nov of each season in 10-day chunks. Safe to re-run: cached chunks are skipped."""
    load_dotenv()
    key = os.environ["FIRMS_MAP_KEY"]
    for year in seasons:
        for src in sources:
            start, end = date(year, 6, 1), date(year, 11, 30)
            while start <= end:
                days = min(MAX_DAYS, (end - start).days + 1)
                n = len(_fetch_chunk(key, src, start, days))
                print(f"{src} {start} +{days}d: {n} detections")
                start += timedelta(days=days)


def load() -> pd.DataFrame:
    """All cached detections as one tidy frame, with a UTC acquisition timestamp."""
    df = pd.concat([pd.read_parquet(p) for p in sorted(RAW_DIR.glob("*.parquet"))], ignore_index=True)
    df["time"] = pd.to_datetime(df["acq_date"] + df["acq_time"].str.zfill(4), format="%Y-%m-%d%H%M", utc=True)
    df = df.rename(columns={"latitude": "lat", "longitude": "lon"})
    df = df[["time", "lat", "lon", "frp", "confidence", "daynight", "source"]]
    return df.drop_duplicates().sort_values("time", ignore_index=True)


if __name__ == "__main__":
    pull()
