from __future__ import annotations
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib

from .config import DATA_INTERIM, MODELS_DIR

@dataclass
class EventKey:
    year: int
    round: int

def _fastf1_next_event(now_utc: datetime | None = None) -> EventKey:
    """Find the next event (year, round) using FastF1 schedules."""
    import fastf1
    now_utc = now_utc or datetime.now(timezone.utc)
    years = [now_utc.year - 1, now_utc.year, now_utc.year + 1]
    candidates: list[tuple[pd.Timestamp, int, int]] = []

    for y in years:
        try:
            sched = fastf1.get_event_schedule(y)
        except Exception:
            continue
        # Try to normalize date column name differences across FastF1 versions
        date_col = None
        for c in ["EventDate", "EventDateUtc", "EventDateUTC", "EventStartDate"]:
            if c in sched.columns:
                date_col = c
                break
        if date_col is None or "RoundNumber" not in sched.columns:
            continue

        df = sched.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
        future = df[df[date_col] >= pd.Timestamp(now_utc)]
        if future.empty:
            continue
        row = future.sort_values(date_col).iloc[0]
        candidates.append((row[date_col], int(row["RoundNumber"]), int(y)))

    if not candidates:
        raise RuntimeError("FastF1: could not determine next event from schedules.")
    date, rnd, year = sorted(candidates, key=lambda x: x[0])[0]
    return EventKey(year=year, round=rnd)

def _load_last_known_prequal_features(target_year: int) -> pd.DataFrame:
    """Load the latest available per-driver prequal features for the season.
    Falls back to the most recent season in the parquet if target_year not present.
    """
    path = DATA_INTERIM / "features_prequal.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Build with: python -m src.build_features --mode prequal")

    df = pd.read_parquet(path)
    if "year" not in df.columns or "driverId" not in df.columns or "round" not in df.columns:
        raise ValueError("features_prequal.parquet missing required columns: year, driverId, round")

    # Choose target season if present, otherwise last available season
    if (df["year"] == target_year).any():
        df_season = df[df["year"] == target_year].copy()
    else:
        last_year = int(df["year"].max())
        df_season = df[df["year"] == last_year].copy()

    # Take the latest row per driver in that season (most recent round)
    df_season = df_season.sort_values(["driverId", "round"])
    latest = df_season.groupby("driverId", as_index=False).tail(1).copy()
    return latest

def _ensure_quali_cols(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["quali_pos", "grid_pos", "teammate_grid_gap"]:
        if col not in df.columns:
            df[col] = np.nan
    return df

def _augment_with_fastf1_quali(event: EventKey, upcoming: pd.DataFrame) -> pd.DataFrame:
    """Merge FastF1 qualifying positions into upcoming features (best-effort).
    If quali isn't available yet, returns the input with NaN quali fields.
    """
    import fastf1
    try:
        sched = fastf1.get_event_schedule(event.year)
        ev_row = sched.loc[sched["RoundNumber"] == event.round]
        if ev_row.empty:
            return _ensure_quali_cols(upcoming)
        event_name = ev_row.iloc[0]["EventName"]
    except Exception:
        return _ensure_quali_cols(upcoming)

    try:
        sess_q = fastf1.get_session(event.year, event_name, "Q")
        sess_q.load(telemetry=False, laps=True, weather=False)
        res = sess_q.results
        if res is None or res.empty:
            return _ensure_quali_cols(upcoming)
        q = res[["Abbreviation", "Position"]].rename(
            columns={"Abbreviation": "code", "Position": "quali_pos"}
        )
        q["grid_pos"] = q["quali_pos"]  # approximation; final grid may include penalties
        out = upcoming.merge(q, on="code", how="left")
        if "constructorId" in out.columns:
            out["teammate_grid_gap"] = out.groupby("constructorId")["grid_pos"].transform(lambda s: s - s.min())
        else:
            out["teammate_grid_gap"] = np.nan
        return out
    except Exception:
        return _ensure_quali_cols(upcoming)

def build_upcoming_features(include_quali: bool = False, year: int | None = None, rnd: int | None = None) -> pd.DataFrame:
    """Construct a per-driver feature table for the next (or specified) race.

    Strategy:
    - take each driver's latest prequal feature row for the season (form, Elo, etc.)
    - update metadata to the target (year, round, synthetic raceId)
    - optionally merge FastF1 qualifying results (post-qual mode)
    """
    if year is None or rnd is None:
        ek = _fastf1_next_event()
    else:
        ek = EventKey(year=int(year), round=int(rnd))

    last = _load_last_known_prequal_features(ek.year)
    upcoming = last.copy()
    upcoming["year"] = ek.year
    upcoming["round"] = ek.round
    upcoming["raceId"] = ek.year * 100 + ek.round  # synthetic to avoid collisions

    if include_quali:
        upcoming = _augment_with_fastf1_quali(ek, upcoming)
    else:
        upcoming = _ensure_quali_cols(upcoming)

    return upcoming

def predict_next(mode: str = "postqual", year: int | None = None, rnd: int | None = None) -> pd.DataFrame:
    """Run the trained model on upcoming event features and return win probabilities.

    mode='prequal' -> uses rolling features only.
    mode='postqual' -> adds FastF1 qualifying-derived features if available.
    """
    bundle_path = MODELS_DIR / f"model_{mode}.pkl"
    if not bundle_path.exists():
        raise FileNotFoundError(f"{bundle_path} not found. Train with: python -m src.train --mode {mode}")
    bundle = joblib.load(bundle_path)
    feat_cols: list[str] = bundle["feature_cols"]
    model = bundle["model"]
    scaler = bundle["scaler"]

    include_quali = (mode == "postqual")
    feats = build_upcoming_features(include_quali=include_quali, year=year, rnd=rnd)

    # Ensure all expected feature columns exist (fill missing with NaN)
    missing = [c for c in feat_cols if c not in feats.columns]
    for c in missing:
        feats[c] = np.nan

    X = feats[feat_cols].copy()
    Xs = scaler.transform(X)
    proba = model.predict_proba(Xs)[:, 1]

    out_cols = [c for c in ["raceId","year","round","driverId","constructorId","code","forename","surname"] if c in feats.columns]
    out = feats[out_cols].copy()
    out["win_proba"] = proba
    out = out.sort_values("win_proba", ascending=False).reset_index(drop=True)
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["prequal","postqual"], default="postqual")
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--round", dest="rnd", type=int, default=None)
    args = ap.parse_args()

    df = predict_next(mode=args.mode, year=args.year, rnd=args.rnd)
    with pd.option_context('display.max_rows', 50, 'display.max_columns', None):
        print(df.head(20).to_string(index=False))
