# Wildfire Escalation Risk Prediction

Given an active wildfire detected by satellite, predict the probability that it escalates over the next 24 hours.

## Phase 0: Feasibility

Before building any infrastructure, a notebook checks whether a model can predict escalation better than a persistence rule ("it grew recently, so it keeps growing").

Data:
- NASA FIRMS VIIRS hotspots for California, Jun–Nov 2022–2024
- Open-Meteo historical weather forecasts

### Run

```bash
cp .env.example .env   # add your FIRMS MAP_KEY
uv sync
uv run pytest
uv run jupyter lab notebooks/00_feasibility.ipynb
```
