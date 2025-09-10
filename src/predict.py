from __future__ import annotations
import argparse
import joblib
import numpy as np
import pandas as pd
from .config import DATA_INTERIM, MODELS_DIR

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
    proba = model.predict_proba(Xs)[:,1]

    out = df[["raceId","year","round","driverId","constructorId","code","forename","surname"]].copy()
    out["win_proba"] = proba
    out = out.sort_values("win_proba", ascending=False).reset_index(drop=True)
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["prequal","postqual"], default="postqual")
    parser.add_argument("--race-id", type=int, required=True)
    args = parser.parse_args()
    df = predict_race(args.race_id, mode=args.mode)
    # Pretty print
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df.head(20).to_string(index=False))
