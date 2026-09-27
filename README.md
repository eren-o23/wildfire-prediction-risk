# Wildfire Escalation Risk Prediction

Given an active wildfire detected by satellite, predict the probability that it escalates over the next 24 hours.

## Phase 0: Feasibility (complete, conditional GO)

Before building any infrastructure, [the feasibility notebook](notebooks/00_feasibility.ipynb) checks whether a model can predict escalation better than a persistence rule ("it grew recently, so it keeps growing").

**Data:**
- NASA FIRMS VIIRS (NOAA-20) hotspots for California, Jun–Nov 2022–2024: 94k detections, which the tracker groups into 9.2k fires
- Open-Meteo historical weather forecasts

**Results on the 2024 test set** (trained on 2022–23; label: count or FRP up ≥50% at the same time next day):

| | PR-AUC | Brier |
|---|---|---|
| Base rate | 0.145 | – |
| Persistence baseline | 0.237 | 0.121 |
| XGBoost | **0.343** | **0.111** |

Checked against NIFC's official records, all 10 of the largest 2024 California fires were tracked from within half a day of discovery. 8 of the 10 kept ≥90% of their detections under one fire ID.

**Caveat:** 62% of rows are single-detection fires. On fires with ≥10 detections, the lift shrinks to 0.27 against a baseline of 0.25. Next step: tighten the label for small fires before Phase 1.

## How it works

```
FIRMS detections → cluster each satellite pass (DBSCAN) → track fires across passes (fire_id)
  → label: did it grow 24h later? → features: fire state + next-24h weather forecast
  → persistence baseline vs XGBoost, time split
```

| Module | What it does |
|---|---|
| `wildfire/firms.py` | Downloads and caches raw satellite detections |
| `wildfire/fires.py` | Clusters each pass and tracks fires across passes, logging merges |
| `wildfire/labels.py` | Versioned escalation label |
| `wildfire/features.py` | The single feature function used by training and future serving |
| `wildfire/weather.py` | Open-Meteo forecasts, cached per ~25 km grid cell |

## ML system design decisions

| Where | Decision | Why |
|---|---|---|
| `firms.py` | Raw API responses are cached untouched and everything is recomputed from them | Change clustering or labels without refetching; any result traces back to source data |
| `firms.py` | Train on the corrected (standard-processing) archive, knowing live data is revised later | Explains gaps between offline and live accuracy |
| `firms.py` | One satellite only (NOAA-20) | Mixing sensors fakes shrinkage whenever one drops out, which corrupts labels |
| `firms.py`, `weather.py` | Retry with backoff on timeouts and rate limits | Upstream delays are routine, not exceptional |
| `fires.py` | Stable `fire_id` across passes, with merges logged rather than rewritten | Labels, features, serving and monitoring all key on one persistent entity; rewriting history would leak future knowledge |
| `labels.py` | Label compares T with T+24h (same time of day), drops rows likely hidden by cloud or smoke, and is versioned | Removes day/night bias, keeps noisy labels out, and makes results comparable only within one label version |
| `features.py` | One feature function for training and serving | Prevents training-serving skew |
| `features.py` | Point-in-time features: only data available at prediction time | Using future data inflates offline scores |
| `weather.py` | Historical *forecasts*, not observed weather | Serving only ever has forecasts |
| `weather.py` | Cache per grid cell | Stays within free API limits; becomes the Phase 5 weather cache |
| Notebook | Time split plus `GroupKFold` by fire | Mirrors deployment and stops one fire leaking across folds |
| Notebook | Persistence baseline; PR-AUC, Brier score and a reliability curve | Justifies the model's complexity; handles class imbalance; checks the probabilities are honest |

## Run

```bash
cp .env.example .env   # add your FIRMS MAP_KEY
uv sync
uv run pytest
uv run jupyter lab notebooks/00_feasibility.ipynb
```

The first run downloads about 111 FIRMS chunks and about 1,400 weather lookups (roughly 1 hour). Later runs use the cache and take under a minute.
