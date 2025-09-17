from __future__ import annotations
import argparse
import joblib
import numpy as np
import pandas as pd
from .config import DATA_INTERIM, MODELS_DIR, EXCLUDED_DRIVER_NAMES, EXCLUDED_DRIVER_CODES, DATA_RAW

def _apply_driver_filter(df: pd.DataFrame) -> pd.DataFrame:
    full = (df.get("forename", "").astype(str).str.strip() + " " +
            df.get("surname", "").astype(str).str.strip()).str.replace(r"\s+", " ", regex=True).str.lower()
    code = df.get("code", "").astype(str).str.upper()
    keep = ~(full.isin(EXCLUDED_DRIVER_NAMES) | code.isin(EXCLUDED_DRIVER_CODES))
    return df[keep]

def _attach_team_name(out: pd.DataFrame) -> pd.DataFrame:
    if "constructor" in out.columns:
        return out
    try:
        m = pd.read_csv(DATA_RAW / "constructors.csv", usecols=["constructorId","name"]).rename(columns={"name":"constructor"})
        return out.merge(m, on="constructorId", how="left")
    except Exception:
        return out

def predict_race(race_id: int, mode: str = "postqual") -> pd.DataFrame:
    feats = pd.read_parquet(DATA_INTERIM / f"features_{mode}.parquet")
    bundle = joblib.load(MODELS_DIR / f"model_{mode}.pkl")
    scaler = bundle["scaler"]
    model = bundle["model"]
    feat_cols = bundle["feature_cols"]

    df = feats[feats["raceId"] == race_id].copy()
    if df.empty:
        raise ValueError(f"raceId {race_id} not found in features_{mode}.parquet")

    X = df[feat_cols].copy()
    Xs = scaler.transform(X)
    proba = model.predict_proba(Xs)[:, 1]

    out_cols = [c for c in ["raceId","year","round","driverId","constructorId","constructor","code","forename","surname","grid_pos"] if c in df.columns]
    out = df[out_cols].copy()
    out["win_proba"] = proba
    out = _apply_driver_filter(out).sort_values("win_proba", ascending=False).reset_index(drop=True)
    out = _attach_team_name(out)
    # Pretty columns for display
    out["win_pct"] = (out["win_proba"] * 100).round(1)  # numeric 0.1% precision
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["prequal","postqual"], default="postqual")
    parser.add_argument("--race-id", type=int, required=True)
    args = parser.parse_args()
    df = predict_race(args.race_id, mode=args.mode)

    # Create a nice console view (xx.x% and grid if available)
    show = df.copy()
    show["Win %"] = show["win_pct"].map(lambda x: f"{x:.1f}%")
    if "grid_pos" in show.columns:
        show["Grid"] = show["grid_pos"].fillna("-").astype("Int64").astype(str).str.replace("<NA>", "-")
    name = (show.get("code","").fillna("").astype(str) + " " +
            show.get("forename","").fillna("").astype(str) + " " +
            show.get("surname","").fillna("").astype(str)).str.strip()
    cols = ["raceId","year","round","Grid","Win %","constructorId"] if "Grid" in show else ["raceId","year","round","Win %","constructorId"]
    show = pd.DataFrame({"Driver": name, **{c: show[c] for c in cols if c in show.columns}})
    with pd.option_context('display.max_rows', 50, 'display.max_columns', None):
        print(show.head(20).to_string(index=False))
