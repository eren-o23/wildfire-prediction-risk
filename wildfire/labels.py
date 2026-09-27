"""Escalation label: did the fire get bigger or more intense 24h later?"""
import numpy as np
import pandas as pd

HORIZON = pd.Timedelta("24h")


def label(summary: pd.DataFrame, merges: pd.DataFrame, x: float = 0.5) -> pd.DataFrame:
    """Add `escalated` to each (fire_id, bin) row of `fires.summarize` output; drop unlabelable rows.

    escalated = 1 if detection count or total FRP at T+24h >= (1+x) * value at T.
    """
    # T+24h, not T+12h: the next bin flips between day and night, and night passes always
    # show less fire. Comparing the same time of day removes that diurnal bias.
    nxt = summary[["fire_id", "bin", "n_det", "frp_sum"]].assign(bin=lambda d: d["bin"] - HORIZON)
    out = summary.merge(nxt, on=["fire_id", "bin"], how="left", suffixes=("", "_next"))

    grew = (out["n_det_next"] >= (1 + x) * out["n_det"]) | (out["frp_sum_next"] >= (1 + x) * out["frp_sum"])
    seen_next = out["n_det_next"].notna()
    last_seen = out["fire_id"].map(summary.groupby("fire_id")["bin"].max())
    target = out["bin"] + HORIZON
    # Not seen at T+24h but seen after it: probably hidden by cloud or smoke, so we can't know.
    # Not seen at T+24h and never again: the fire really went quiet, so it's a genuine 0.
    out["escalated"] = np.where(seen_next, grew.astype(float), np.where(last_seen > target, np.nan, 0.0))

    # T+24h falls past the end of that season's data, so the outcome is unknown
    season_end = out["bin"].dt.year.map(summary.groupby(summary["bin"].dt.year)["bin"].max())
    out.loc[target > season_end, "escalated"] = np.nan
    # absorbed into another fire inside the window: its growth now shows up under another id
    absorbed_at = merges.groupby("absorbed")["bin"].min().reindex(out["fire_id"]).set_axis(out.index)
    out.loc[(absorbed_at > out["bin"]) & (absorbed_at <= target), "escalated"] = np.nan

    # why: the label is a modelling choice, so it's versioned. Results are only comparable
    # within one label version.
    out["label_version"] = f"h=24h|x={x}|count_or_frp|occluded=drop"
    return out.dropna(subset=["escalated"]).drop(columns=["n_det_next", "frp_sum_next"])
