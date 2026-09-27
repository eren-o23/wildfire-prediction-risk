"""Open-Meteo historical forecasts per fire location, cached per grid cell."""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "weather"
URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
HOURLY = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_gusts_10m", "wind_direction_10m"]
# why: fires within ~25 km share one weather lookup. Weather model grids are about this
# coarse anyway, and caching per cell keeps us inside Open-Meteo's free-tier limits
# (this becomes the Phase 5 weather cache).
GRID = 0.25


def to_cell(x: pd.Series) -> pd.Series:
    return (x / GRID).round() * GRID


def _fetch(lat: float, lon: float, start, end) -> pd.DataFrame:
    path = RAW_DIR / f"{lat:.2f}_{lon:.2f}_{start}_{end}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    params = dict(latitude=lat, longitude=lon, start_date=start, end_date=end, hourly=",".join(HOURLY), timezone="UTC")
    for attempt in range(5):
        try:
            r = requests.get(URL, params=params, timeout=60)
        except requests.RequestException as e:  # timeouts and dropped connections are transient too
            print(f"Open-Meteo {lat},{lon}: {type(e).__name__}, retrying")
            time.sleep(10 * 2**attempt)
            continue
        if r.ok:
            break
        time.sleep(30 * 2**attempt if r.status_code == 429 else 5)
    else:
        raise RuntimeError(f"Open-Meteo {lat},{lon}: gave up after retries")
    df = pd.DataFrame(r.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["cell_lat"], df["cell_lon"] = lat, lon
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def hourly_for(summary: pd.DataFrame) -> pd.DataFrame:
    """Hourly forecasts covering every fire's cell, from its first bin to 2 days after its last."""
    s = summary.assign(cell_lat=to_cell(summary["lat"]), cell_lon=to_cell(summary["lon"]), year=summary["bin"].dt.year)
    spans = s.groupby(["cell_lat", "cell_lon", "year"])["bin"].agg(["min", "max"]).reset_index()
    frames = []
    for i, row in enumerate(spans.itertuples()):
        start, end = row.min.date(), (row.max + pd.Timedelta("2D")).date()
        frames.append(_fetch(row.cell_lat, row.cell_lon, start, end))
        if i % 100 == 0:
            print(f"weather {i}/{len(spans)}")
    return pd.concat(frames, ignore_index=True).drop_duplicates(["cell_lat", "cell_lon", "time"])


def window_features(as_of: pd.Series, lat: pd.Series, lon: pd.Series, hourly: pd.DataFrame, hours=24) -> pd.DataFrame:
    """Weather over [as_of, as_of + hours) at each row's cell. Returns a frame aligned to `as_of`'s index."""
    # ponytail: the historical forecast API stitches together the first hours of each model
    # run, so its lead time is shorter than a real 24h-ahead forecast. That's still much
    # closer to serving than observed weather. Upgrade to the Previous Runs API
    # (day-1 lead) if it covers our years.
    rows = pd.DataFrame({"row": as_of.index, "t0": as_of.dt.floor("h"), "cell_lat": to_cell(lat), "cell_lon": to_cell(lon)})
    rows = rows.loc[rows.index.repeat(hours)]
    rows["time"] = rows["t0"] + pd.to_timedelta(np.tile(np.arange(hours), len(as_of)), unit="h")
    w = rows.merge(hourly, on=["cell_lat", "cell_lon", "time"], how="left")
    # vector-average wind direction (degrees wind comes from) so 350 and 10 average to 0, not 180
    rad = np.radians(w["wind_direction_10m"])
    w["u"], w["v"] = -w["wind_speed_10m"] * np.sin(rad), -w["wind_speed_10m"] * np.cos(rad)
    g = w.groupby("row")
    out = pd.DataFrame(
        {
            "wx_wind_max": g["wind_speed_10m"].max(),
            "wx_wind_mean": g["wind_speed_10m"].mean(),
            "wx_gust_max": g["wind_gusts_10m"].max(),
            "wx_rh_min": g["relative_humidity_2m"].min(),
            "wx_temp_max": g["temperature_2m"].max(),
        }
    )
    direction = np.arctan2(-g["u"].mean(), -g["v"].mean())
    out["wx_dir_sin"], out["wx_dir_cos"] = np.sin(direction), np.cos(direction)
    return out.reindex(as_of.index)
