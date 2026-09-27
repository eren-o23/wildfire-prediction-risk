"""Model features for each (fire_id, bin T).

why: this is the only place features are computed. The training notebook calls it now and the
scoring job will call it in Phase 4. Duplicated feature logic between training and serving
drifts apart silently, which is the most common production ML bug (training-serving skew).
"""
import numpy as np
import pandas as pd

from wildfire.fires import BIN
from wildfire.weather import window_features

def as_of(bins: pd.Series) -> pd.Series:
    """When the prediction for bin T is made: the end of the bin.

    why: point-in-time correctness. Features only use bins <= T and weather forecasts from
    this moment on, never anything that happened later. Passes sit mid-bin (~6h in), and we
    assume FIRMS publishes them within ~3h. The archive has no ingestion timestamps to check
    this against, so it's an assumption.
    """
    return bins + BIN


def fire_state(summary: pd.DataFrame) -> pd.DataFrame:
    """Fire size, intensity and recent growth as of each bin, using only bins <= T."""
    s = summary.sort_values(["fire_id", "bin"]).reset_index(drop=True)
    g = s.groupby("fire_id")

    def at_lag(hours):
        lag = s[["fire_id", "bin", "n_det", "frp_sum"]].assign(bin=lambda d: d["bin"] + pd.Timedelta(hours=hours))
        return s[["fire_id", "bin"]].merge(lag, on=["fire_id", "bin"], how="left")[["n_det", "frp_sum"]].fillna(0)

    prev12, prev24 = at_lag(12), at_lag(24)
    out = s[["fire_id", "bin"]].copy()
    out["n_det"] = s["n_det"]
    out["area_km2"] = s["area_km2"]
    out["frp_sum"] = s["frp_sum"]
    out["frp_mean"] = s["frp_mean"]
    out["conf_high"] = s["conf_high"]
    out["is_day"] = (s["daynight"] == "D").astype(int)
    # growth on a log scale; 24h compares the same time of day, so it has no day/night bias
    out["growth_n_24h"] = np.log1p(s["n_det"]) - np.log1p(prev24["n_det"])
    out["growth_frp_24h"] = np.log1p(s["frp_sum"]) - np.log1p(prev24["frp_sum"])
    out["growth_n_12h"] = np.log1p(s["n_det"]) - np.log1p(prev12["n_det"])
    out["age_h"] = (s["bin"] - g["bin"].transform("min")).dt.total_seconds() / 3600
    # long gaps flag passes where the fire was probably hidden by cloud or smoke
    out["hours_since_prev"] = (s["bin"] - g["bin"].shift()).dt.total_seconds().div(3600).fillna(0)
    out["n_det_cum"] = g["n_det"].cumsum()
    out["month"] = s["bin"].dt.month
    return out


def build(summary: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    """Fire state + next-24h weather forecast, one row per (fire_id, bin)."""
    s = summary.sort_values(["fire_id", "bin"]).reset_index(drop=True)
    wx = window_features(as_of(s["bin"]), s["lat"], s["lon"], hourly)
    return pd.concat([fire_state(s), wx], axis=1)


FEATURES = [
    "n_det", "area_km2", "frp_sum", "frp_mean", "conf_high", "is_day",
    "growth_n_24h", "growth_frp_24h", "growth_n_12h", "age_h", "hours_since_prev", "n_det_cum", "month",
    "wx_wind_max", "wx_wind_mean", "wx_gust_max", "wx_rh_min", "wx_temp_max", "wx_dir_sin", "wx_dir_cos",
]
