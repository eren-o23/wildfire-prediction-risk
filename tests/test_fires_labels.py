import pandas as pd

from wildfire.fires import summarize, track

T0 = pd.Timestamp("2024-07-01 21:00", tz="UTC")  # a daytime pass, mid-bin


def det(rows):
    """rows: (hours after T0, lat, lon)"""
    return pd.DataFrame(
        [
            dict(time=T0 + pd.Timedelta(hours=h), lat=lat, lon=lon, frp=10.0, confidence="n", daynight="D")
            for h, lat, lon in rows
        ]
    )


def test_tracking_links_and_merges():
    d, merges = track(
        det(
            [
                (0, 38.00, -120.0),   # fire A
                (0, 38.04, -120.0),   # fire B, ~4.4 km away: separate cluster
                (12, 38.00, -120.0),  # A again next pass: same id
                (24, 38.02, -120.0),  # ~2.2 km from both A and B: merges them
                (0, 36.00, -118.0),   # unrelated fire far away
            ]
        )
    )
    fid = lambda mask: d.loc[mask, "fire_id"].item()
    a, b, far = fid((d.lat == 38.00) & (d.time == T0)), fid(d.lat == 38.04), fid(d.lat == 36.00)
    assert len({a, b, far}) == 3
    assert fid((d.lat == 38.00) & (d.time > T0)) == a
    assert fid(d.lat == 38.02) == min(a, b)
    assert merges[["survivor", "absorbed"]].values.tolist() == [[min(a, b), max(a, b)]]
    assert len(summarize(d)) == 5


def test_labels_growth_occlusion_and_closed():
    from wildfire.labels import label

    d, merges = track(
        det(
            [
                (0, 35.0, -117.0),  # E: 1 detection...
                (24, 35.0, -117.0), (24, 35.001, -117.0), (24, 35.002, -117.0),  # ...then 3 at T+24h
                (0, 34.0, -116.0),  # C: missing at T+24h, back at T+48h -> occluded
                (48, 34.0, -116.0),
                (0, 33.0, -115.0),  # D: never seen again -> genuinely went quiet
                (96, 40.0, -122.0),  # unrelated, just extends the season past the windows
            ]
        )
    )
    lab = label(summarize(d), merges, x=0.5)
    at_t0 = lab[lab.bin == lab.bin.min()].set_index("lat")["escalated"]
    assert at_t0.to_dict() == {33.0: 0.0, 35.0: 1.0}  # C (lat 34) dropped as occluded
