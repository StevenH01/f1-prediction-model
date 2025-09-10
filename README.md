# f1-win-prob

Predict **win probability per driver** for each Formula 1 race using historical data (Kaggle/Ergast). 
Two models: **pre-qual** (features known before qualifying) and **post-qual** (adds qualifying/grid & FP summaries).

## Quickstart
1) Put Kaggle CSVs from the Formula 1 dataset into: `/mnt/data/f1-win-prob/data_raw`. 
   Required at minimum: `races.csv`, `results.csv`, `drivers.csv`, `constructors.csv`, `qualifying.csv`, `circuits.csv`, `status.csv`.
2) Build features:
   ```bash
   python -m src.build_features --mode prequal
   python -m src.build_features --mode postqual
   ```
3) Train & evaluate:
   ```bash
   python -m src.train --mode prequal
   python -m src.train --mode postqual
   ```
4) Predict a specific race (by `raceId` present in features):
   ```bash
   python -m src.predict --mode postqual --race-id 1106
   ```

## Repo structure
```
f1-win-prob/
  data_raw/            # put Kaggle CSVs here (not tracked)
  data_interim/        # cleaned & feature tables (auto-generated)
  models/              # trained models (.pkl)
  reports/figures/     # evaluation charts
  src/
    __init__.py
    config.py
    ingest.py
    ratings.py
    build_features.py
    train.py
    predict.py
```

## Notes
- Feature generation enforces **no data leakage**: each row only uses info available **before** the race (or before qualifying for post-qual mode).
- Training uses **time-aware folds** by season and reports log loss, Brier score, accuracy (top-1), and calibration plots.
