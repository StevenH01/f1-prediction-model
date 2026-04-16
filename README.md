
# F1 Race Winner – Probabilistic Prediction

Predict **win probabilities per driver** for Formula 1 races using two leak‑safe models:

- **Pre‑Qual** — uses information known before qualifying (driver/constructor Elo, rolling form, circuit affinity).
- **Post‑Qual** — adds qualifying/grid signals and (optionally) FastF1 practice/quali pace. This usually performs better.

Includes a console demo and optional API to show **past race** and **upcoming race** probabilities.

---

## Table of Contents

- [Features](#features)
- [Project Layout](#project-layout)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data Setup](#data-setup)
- [Build Features & Train](#build-features--train)
- [Usage](#usage)
  - [Predict a Past Race](#predict-a-past-race)
  - [Predict the Next Upcoming Race](#predict-the-next-upcoming-race)
  - [Get Race IDs](#get-race-ids)
- [Commands Quick Reference](#commands-quick-reference)
- [Configuration](#configuration)
  - [Driver Exclusions](#driver-exclusions)
  - [Team Names / Live Teams](#team-names--live-teams)
- [Notebooks](#notebooks)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [License & Acknowledgments](#license--acknowledgments)

---

## Features

- **Leak‑safe, time‑aware pipeline** for pre‑qual and post‑qual modes
- **Multiplayer Elo** for drivers and constructors
- Rolling form (points/finishes) & circuit affinity
- **Qualifying/grid features**, plus optional **FastF1** practice/quali pace
- **Calibrated probabilities** (isotonic) with yearly CV metrics
- Console output: `Driver | Grid | Win % | Team`
- Optional **FastAPI** server for a frontend

---

## Project Layout

```
f1-prediction-model/
  data_raw/                # Kaggle/Ergast CSVs you provide
  data_interim/            # generated features & extras
  models/                  # trained models & metrics
  reports/figures/         # charts (log loss, etc.)
  src/
    __init__.py
    config.py              # paths, RNG seed, driver exclusions
    ingest.py              # CSV loading / normalization
    ratings.py             # multiplayer Elo
    build_features.py      # constructs prequal/postqual tables
    train.py               # LightGBM + isotonic calibration + CV
    predict.py             # predictions for past races (by raceId)
    fastf1_ingest.py       # (optional) FastF1 practice/quali extras
    upcoming.py            # predictions for the next race (pre/post‑qual)
    serve.py               # (optional) FastAPI for UI
  console_demo.py          # pretty console table for past/next races
  notebooks/               # (optional) EDA, calibration, error analysis, etc.
```

---

## Prerequisites

- Python 3.10+
- Kaggle/Ergast CSVs (see **Data Setup**)
- (Optional) **FastF1** for current‑season practice/quali signals and live team names

---

## Installation

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# macOS/Linux
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Data Setup

Place the Kaggle/Ergast CSVs into `data_raw/`:

```
races.csv, results.csv, drivers.csv, constructors.csv, qualifying.csv, circuits.csv, status.csv
```

> `constructors.csv` is used to display **team names** for past races; upcoming races can use **FastF1** to show live teams.

---

## Build Features & Train

### Build feature tables
```powershell
python -m src.build_features --mode prequal
python -m src.build_features --mode postqual
```

### Train models (year filtering supported)
```powershell
# Train on a window (e.g., 2022–2024) for post‑qual
python -m src.train --mode postqual --year-min 2022 --year-max 2024

# Or an explicit list of seasons
python -m src.train --mode postqual --years 2022,2023,2024
```

Artifacts:
- `models/model_{mode}.pkl`
- `models/metrics_{mode}.csv`
- `reports/figures/logloss_{mode}.png`

> You can only train seasons you have **results** for. If current‑year results aren’t in your CSVs yet, you can still **predict** next races; training will skip those rows until results exist.

---

## Usage

### Predict a Past Race
```powershell
# Find a raceId to test (last 20 rows)
python - << 'PY'
import pandas as pd
df = pd.read_parquet("data_interim/features_postqual.parquet")
print(df[['year','round','raceId']].drop_duplicates().sort_values(['year','round']).tail(20).to_string(index=False))
PY

# Predict (replace 1133 with a raceId you saw)
python .\console_demo.py --mode postqual --race-id 1133
```

### Predict the Next Upcoming Race

Install **FastF1** once:
```powershell
pip install fastf1
```

(Optional) build practice/quali extras for modern seasons (improves post‑qual):
```powershell
python -m src.fastf1_ingest --years 2022-2025
python -m src.build_features --mode postqual
```

Now predict the **next** race (no raceId needed):
```powershell
# Before qualifying (no grid yet)
python .\console_demo.py --mode prequal

# After qualifying (shows grid + live team names)
python .\console_demo.py --mode postqual
```

### Get Race IDs
```powershell
python - << 'PY'
import pandas as pd
df = pd.read_parquet("data_interim/features_postqual.parquet")
print(df[['year','round','raceId']].drop_duplicates().sort_values(['year','round']).tail(40).to_string(index=False))
PY
```

---

## Commands Quick Reference

```powershell
# --- Setup ---
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# --- Data ---
# Put CSVs into data_raw/: races, results, drivers, constructors, qualifying, circuits, status

# --- Build Features ---
python -m src.build_features --mode prequal
python -m src.build_features --mode postqual

# --- Train ---
python -m src.train --mode prequal
python -m src.train --mode postqual
# with year filters
python -m src.train --mode postqual --year-min 2022 --year-max 2024
python -m src.train --mode postqual --years 2022,2023,2024

# --- Predict Past Race ---
python .\console_demo.py --mode postqual --race-id <RACE_ID>

# --- Predict Next Race ---
pip install fastf1
python -m src.fastf1_ingest --years 2022-2025   # optional, improves post‑qual
python -m src.build_features --mode postqual    # rebuild to include extras
python .\console_demo.py --mode prequal
python .\console_demo.py --mode postqual

# --- API (optional) ---
uvicorn src.serve:app --reload --port 8000
# GET /races?mode=postqual
# GET /predictions?race_id=<id>&mode=postqual
# GET /predict_next?mode=postqual
# GET /metrics?mode=postqual
```

---

## Configuration

### Driver Exclusions
Filter reserve/incorrect entries by name/code in `src/config.py`:
```python
EXCLUDED_DRIVER_NAMES = {
  "jack doohan", "jack dohan",
  "kevin magnussen",
  "daniel ricciardo", "daniel riccardo"
}
EXCLUDED_DRIVER_CODES = {"DOO","MAG","RIC"}
```

### Team Names / Live Teams
- **Past races**: team name comes from the features or `data_raw/constructors.csv`.
- **Upcoming race**: `upcoming.py` merges FastF1 qualifying results and prefers the **live team name** (e.g., Hamilton → Ferrari) when available.

---

## Notebooks

Suggested notebooks (in `/notebooks`):
- `03_model_dev_postqual.ipynb` – metrics by year, calibration, importances
- `05_upcoming_workbench.ipynb` – interactive “next race” table
- `01_eda_postqual.ipynb` – quali/grid vs win rate
- `02_feature_quality_and_leakage.ipynb` – missingness & leakage checks

Each notebook should start with:
```python
import sys, pathlib; PROJECT_ROOT = pathlib.Path.cwd().resolve()
if str(PROJECT_ROOT) not in sys.path: sys.path.append(str(PROJECT_ROOT))
```

---

## Troubleshooting

- **`FileNotFoundError: data_interim/...`** – Create the folder or rerun build steps:
  ```powershell
  mkdir data_interim
  python -m src.build_features --mode prequal
  ```
- **`CalibratedClassifierCV got unexpected keyword 'base_estimator'`** – Use `estimator=` (already in `train.py`).
- **Pandas `include_groups` error** – Your pandas doesn’t support it; `build_features.py` uses a compatible call.
- **LightGBM warning “X does not have valid feature names”** – Harmless; you can wrap the scaled array back into a DataFrame with column names before `predict_proba`.
- **FastF1 not installed / next race fails** – `pip install fastf1`, then retry `console_demo.py --mode prequal/postqual`.
- **Team looks wrong for upcoming race** – Ensure you’ve applied the FastF1 team merge in `upcoming.py` and you’re using `--mode postqual` after qualifying.

---

## Roadmap

- Weather buckets (dry/mixed/wet) from FastF1 weather stream
- Final grid (post‑penalties) ingestion
- MLflow experiment tracking
- Small Next.js UI (already supported by the optional API)

---

## License & Acknowledgments

Data courtesy of Ergast (historical) and FastF1 (timing/quali/roster).  
This project is for educational/non‑commercial use.
