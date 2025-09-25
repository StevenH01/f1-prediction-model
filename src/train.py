# src/train.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, brier_score_loss
from lightgbm import LGBMClassifier

from .config import DATA_INTERIM, MODELS_DIR, RANDOM_SEED

# Columns we never train on
ID_OR_TEXT_COLS = {
    "raceId","resultId","driverId","constructorId","circuitId",
    "code","forename","surname","driverRef","constructor","name","status",
    "date"
}
# Outcome/leaky columns (must be excluded)
LEAKY_COLS = {
    "is_win","finish_pos","points","position","positionOrder","rank","laps","time","milliseconds","statusId","dnf"
}

def _load_features(mode: str) -> pd.DataFrame:
    path = DATA_INTERIM / f"features_{mode}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Build features first with: python -m src.build_features --mode {mode}")
    df = pd.read_parquet(path)
    if "is_win" not in df.columns:
        # Create binary target if missing
        if "finish_pos" in df.columns:
            df["is_win"] = (pd.to_numeric(df["finish_pos"], errors="coerce") == 1).astype(int)
        else:
            raise ValueError("Target column 'is_win' missing and no 'finish_pos' to derive it.")
    return df

def _year_filter(df: pd.DataFrame, year_min: int | None, year_max: int | None, years: str | None) -> pd.DataFrame:
    if years:
        ys = [int(y) for y in str(years).replace(" ", "").split(",") if y]
        return df[df["year"].isin(ys)].copy()
    lo = year_min if year_min is not None else int(df["year"].min())
    hi = year_max if year_max is not None else int(df["year"].max())
    return df[(df["year"] >= lo) & (df["year"] <= hi)].copy()

def _make_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    # Replace Ergast-style nulls like '\\N' with NaN across the board
    df = df.replace({"\\\\N": np.nan, "\\N": np.nan})
    # Coerce object-looking numerics to floats
    for c in df.columns:
        if c not in ID_OR_TEXT_COLS | LEAKY_COLS:
            if df[c].dtype == "object":
                df[c] = pd.to_numeric(df[c], errors="coerce")
    # Candidate numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove leaky columns explicitly
    feature_cols = [c for c in num_cols if c not in LEAKY_COLS]
    X = df[feature_cols].copy()
    return X, feature_cols

def train_and_evaluate(mode: str = "postqual", year_min: int | None = None, year_max: int | None = None, years: str | None = None):
    df = _load_features(mode)
    df = _year_filter(df, year_min, year_max, years)

    y = df["is_win"].astype(int).values

    X, feature_cols = _make_feature_frame(df)
    # Impute + scale
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    scaler = StandardScaler(with_mean=False)
    X_feat = scaler.fit_transform(X_imp)

    # Grouped CV by (year) to reduce leakage across races in same season
    groups = df["year"].values
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    base = LGBMClassifier(
        n_estimators=800,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        max_depth=-1,
        num_leaves=63,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        objective="binary",
    )
    calibrator = CalibratedClassifierCV(estimator=base, method="isotonic", cv=gkf)

    calibrator.fit(X_feat, y)

    # Evaluate per year log loss (rough, on fitted data to summarize; use held-out notebooks for real eval)
    df_pred = df[["year","raceId","driverId"]].copy()
    df_pred["p_win"] = calibrator.predict_proba(X_feat)[:,1]
    df_pred["y"] = y
    summary = df_pred.groupby("year").apply(lambda g: pd.Series({
        "n": len(g),
        "logloss": float(log_loss(g["y"], g["p_win"])),
        "brier": float(brier_score_loss(g["y"], g["p_win"])),
        "wins@1%+": int((g["p_win"]>=0.01).sum())
    })).reset_index()

    print("Yearly metrics:")
    with pd.option_context("display.max_rows", 200, "display.precision", 4):
        print(summary.to_string(index=False))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    outp = MODELS_DIR / f"model_{mode}.pkl"
    joblib.dump(
        {
            "task": "win_classifier",
            "mode": mode,
            "feature_cols": feature_cols,
            "imputer": imputer,
            "scaler": scaler,
            "model": calibrator,
        },
        outp
    )
    print(f"Saved model bundle → {outp}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["prequal","postqual"], default="postqual")
    ap.add_argument("--year-min", type=int, default=None)
    ap.add_argument("--year-max", type=int, default=None)
    ap.add_argument("--years", type=str, default=None, help="Comma-separated list, e.g. 2022,2023,2024")
    args = ap.parse_args()
    train_and_evaluate(mode=args.mode, year_min=args.year_min, year_max=args.year_max, years=args.years)