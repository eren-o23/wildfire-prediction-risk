"""Turn raw hotspot detections into tracked fire events with stable IDs."""
import numpy as np
import pandas as pd
from shapely import MultiPoint
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree

EARTH_KM = 6371.0
# VIIRS passes over California at about 09-10 UTC (night) and 20-21 UTC (day). Bins run
# 03-15 and 15-03 UTC so each pass sits in the middle of a bin, and SNPP and NOAA-20
# (~50 min apart) land in the same bin.
BIN = pd.Timedelta("12h")
BIN_OFFSET = pd.Timedelta("3h")


def assign_bins(times: pd.Series) -> pd.Series:
    return (times - BIN_OFFSET).dt.floor(BIN) + BIN_OFFSET


def track(det: pd.DataFrame, eps_km=1.5, link_km=3.0, memory=pd.Timedelta("72h")):
    """Cluster each 12h bin with DBSCAN, then link clusters to fires seen in the last `memory`.

    Returns (detections with `bin` and `fire_id`, merges log with bin/survivor/absorbed).
    """
    # why: entity resolution. DBSCAN per snapshot gives new cluster labels every pass; labels,
    # features, serving (GET /fire/{id}) and monitoring all need one ID that persists over time.
    det = det.assign(bin=assign_bins(det["time"])).sort_values("bin", ignore_index=True)
    coords = np.radians(det[["lat", "lon"]].to_numpy())
    bins = det["bin"].to_numpy()
    fire_ids = np.full(len(det), -1, dtype=np.int64)
    alias: dict[int, int] = {}  # absorbed fire -> survivor
    merges = []
    next_id = 0

    def resolve(f):
        while f in alias:
            f = alias[f]
        return f

    for b, idx in det.groupby("bin", sort=True).indices.items():
        labels = DBSCAN(eps=eps_km / EARTH_KM, min_samples=1, metric="haversine").fit_predict(coords[idx])
        # detections are sorted by bin, so the last `memory` of history is one contiguous slice
        recent = np.arange(np.searchsorted(bins, b - memory), idx[0])
        tree = BallTree(coords[recent], metric="haversine") if len(recent) else None
        for c in np.unique(labels):
            pts = idx[labels == c]
            matched = set()
            if tree is not None:
                hits = tree.query_radius(coords[pts], r=link_km / EARTH_KM)
                matched = {resolve(fire_ids[recent[h]]) for arr in hits for h in arr}
            if not matched:
                fid, next_id = next_id, next_id + 1
            else:
                # ponytail: nearest-detection linking; switch to hull overlap if merges look wrong
                fid = min(matched)  # lower id = older fire, it survives the merge
                for other in matched - {fid}:
                    alias[other] = fid
                    merges.append((b, fid, other))
            fire_ids[pts] = fid

    # why: past rows keep the fire_id they had at the time (point-in-time); merges are a log,
    # not a rewrite of history
    det["fire_id"] = fire_ids
    return det, pd.DataFrame(merges, columns=["bin", "survivor", "absorbed"])


def hull_area_km2(lat: np.ndarray, lon: np.ndarray) -> float:
    if len(lat) < 3:
        return 0.0
    # local equirectangular projection; plenty accurate at fire scale
    x = np.radians(lon) * EARTH_KM * np.cos(np.radians(lat.mean()))
    y = np.radians(lat) * EARTH_KM
    return MultiPoint(np.column_stack([x, y])).convex_hull.area


def summarize(det: pd.DataFrame) -> pd.DataFrame:
    """One row per (fire_id, bin): size and intensity of the fire in that pass."""
    g = det.groupby(["fire_id", "bin"])
    out = g.agg(
        n_det=("frp", "size"),
        frp_sum=("frp", "sum"),
        frp_mean=("frp", "mean"),
        conf_high=("confidence", lambda s: (s == "h").mean()),
        lat=("lat", "mean"),
        lon=("lon", "mean"),
        daynight=("daynight", "first"),
    )
    out["area_km2"] = g.apply(lambda d: hull_area_km2(d["lat"].to_numpy(), d["lon"].to_numpy()))
    return out.reset_index()
