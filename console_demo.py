# console_demo.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from src.config import DATA_INTERIM, MODELS_DIR, DATA_RAW

# Try to import exclusions from config; fall back to built-ins if missing
try:
    from src.config import EXCLUDED_DRIVER_NAMES, EXCLUDED_DRIVER_CODES
except Exception:
    EXCLUDED_DRIVER_NAMES = {"jack doohan","jack dohan","kevin magnussen","daniel ricciardo","daniel riccardo"}
    EXCLUDED_DRIVER_CODES = {"DOO","MAG","RIC"}

def _filter_drivers(df: pd.DataFrame) -> pd.DataFrame:
    full = (df.get("forename","").astype(str).str.strip() + " " +
            df.get("surname","").astype(str).str.strip()).str.replace(r"\s+"," ",regex=True).str.lower()
    code = df.get("code","").astype(str).str.upper()
    keep = ~(full.isin(EXCLUDED_DRIVER_NAMES) | code.isin(EXCLUDED_DRIVER_CODES))
    return df[keep]

def _constructor_map():
    try:
        import pandas as pd
        from src.config import DATA_RAW
        m = pd.read_csv(DATA_RAW / "constructors.csv", usecols=["constructorId","name"])
        return dict(zip(m["constructorId"], m["name"]))
    except Exception:
        return {}

def _team_series(df: pd.DataFrame) -> pd.Series:
    # 1) Prefer live team from FastF1 (for upcoming race)
    if "team_live" in df.columns:
        live = df["team_live"].fillna("").astype(str)
        if live.str.strip().any():
            return live
    # 2) Else use constructor (name) if present in features
    if "constructor" in df.columns:
        return df["constructor"].fillna("").astype(str)
    # 3) Fallback: map constructorId -> name via Kaggle file
    cmap = _constructor_map()
    if "constructorId" in df.columns and cmap:
        return df["constructorId"].map(cmap).fillna(df["constructorId"].astype(str))
    # 4) Last resort: blank
    import pandas as pd
    return pd.Series([""] * len(df))


def _pretty_print(df: pd.DataFrame, limit: int = 20):
    name = (df.get("code","").fillna("").astype(str) + " " +
            df.get("forename","").fillna("").astype(str) + " " +
            df.get("surname","").fillna("").astype(str)).str.replace(r"\s+"," ",regex=True).str.strip()

    if "grid_pos" in df.columns:
        grid_num = pd.to_numeric(df.get("grid_pos"), errors="coerce").astype("Int64")
        grid = grid_num.astype(str).str.replace("<NA>", "-")
    else:
        grid = pd.Series(["-"] * len(df))

    team = _team_series(df)

    show = pd.DataFrame({
        "#": range(1, len(df)+1),
        "Driver": name,
        "Grid": grid,
        "Win %": df["win_pct"].map(lambda x: f"{x:.1f}%"),
        "Team": team,
    })
    print(show.head(limit).to_string(index=False))



def _predict_past(race_id: int, mode: str) -> pd.DataFrame:
    feats = pd.read_parquet(DATA_INTERIM / f"features_{mode}.parquet")
    bundle = joblib.load(MODELS_DIR / f"model_{mode}.pkl")
    scaler, model, feat_cols = bundle["scaler"], bundle["model"], bundle["feature_cols"]

    df = feats[feats["raceId"] == int(race_id)].copy()
    if df.empty:
        raise SystemExit(f"raceId {race_id} not found in features_{mode}.parquet")

    Xs = scaler.transform(df[feat_cols])
    proba = model.predict_proba(Xs)[:, 1]
    out_cols = [c for c in ["raceId","year","round","driverId","constructorId","code","forename","surname","grid_pos"] if c in df.columns]
    out = df[out_cols].copy()
    out["win_pct"] = (proba * 100).round(1)
    return _filter_drivers(out).sort_values("win_pct", ascending=False).reset_index(drop=True)

def _predict_next(mode: str) -> pd.DataFrame:
    # Requires src/upcoming.py to be present
    from src.upcoming import predict_next as _pn
    out = _pn(mode=mode).copy()
    if "win_pct" not in out.columns:
        if "win_proba" in out.columns:
            out["win_pct"] = (out["win_proba"] * 100).round(1)
        else:
            out["win_pct"] = np.nan
    return _filter_drivers(out).sort_values("win_pct", ascending=False).reset_index(drop=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Console view of F1 win probabilities")
    ap.add_argument("--mode", choices=["prequal","postqual"], default="postqual")
    ap.add_argument("--race-id", type=int, help="Past raceId to predict; omit to predict the NEXT race")
    args = ap.parse_args()

    if args.race_id is not None:
        df = _predict_past(args.race_id, args.mode)
    else:
        df = _predict_next(args.mode)  # uses FastF1 to find next event

    _pretty_print(df, limit=20)
