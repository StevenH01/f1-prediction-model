from __future__ import annotations
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib

from .config import DATA_INTERIM, MODELS_DIR, EXCLUDED_DRIVER_NAMES, EXCLUDED_DRIVER_CODES, DATA_RAW

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
    """Merge FastF1 qualifying positions and LIVE team names into upcoming features.
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

        # Qualifying positions
        q = res[["Abbreviation", "Position"]].rename(
            columns={"Abbreviation": "code", "Position": "quali_pos"}
        )
        q["grid_pos"] = q["quali_pos"]  # best-effort; real grid may change with penalties

        out = upcoming.merge(q, on="code", how="left")

        # LIVE team name from session results (column may be 'TeamName' or 'Team' depending on FastF1 version)
        team_col = "TeamName" if "TeamName" in res.columns else ("Team" if "Team" in res.columns else None)
        if team_col:
            roster = res[["Abbreviation", team_col]].rename(columns={"Abbreviation": "code", team_col: "team_live"})
            out = out.merge(roster, on="code", how="left")
            # Prefer the live team name for display; keep historical constructor as fallback
            if "constructor" in out.columns:
                out["constructor"] = out["team_live"].combine_first(out["constructor"])
            else:
                out["constructor"] = out["team_live"]

        # Teammate grid gap (uses whatever constructor grouping we now have)
        if "constructorId" in out.columns:
            out["teammate_grid_gap"] = out.groupby("constructorId")["grid_pos"].transform(lambda s: s - s.min())
        else:
            # If we only have names, group by constructor string
            if "constructor" in out.columns:
                out["teammate_grid_gap"] = out.groupby("constructor")["grid_pos"].transform(lambda s: s - s.min())
            else:
                out["teammate_grid_gap"] = np.nan

        return out
    except Exception:
        return _ensure_quali_cols(upcoming)

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
    
def _event_name(event) -> str | None:
    import fastf1
    sched = fastf1.get_event_schedule(event.year)
    row = sched.loc[sched["RoundNumber"] == event.round]
    if row.empty:
        return None
    return row.iloc[0]["EventName"]

def _session_results(year: int, event_name: str, session_codes: list[str]):
    """Try sessions in order and return the first non-empty .results DataFrame."""
    import fastf1
    for sc in session_codes:
        try:
            s = fastf1.get_session(year, event_name, sc)
            s.load(telemetry=False, laps=False, weather=False)
            if s.results is not None and not s.results.empty:
                return s.results, sc
        except Exception:
            pass
    return None, None

def _find_roster(event) -> tuple[pd.DataFrame, str]:
    """Return a 20-driver roster DataFrame for the target event, plus the source session code used.
    Columns: code, team_live, forename, surname
    """
    import fastf1
    # 1) Try target event sessions (Q, Sprint, R, FP3, FP2, FP1)
    name = _event_name(event)
    res, used = (None, None)
    if name:
        res, used = _session_results(event.year, name, ["Q", "Sprint", "R", "FP3", "FP2", "FP1"])

    # 2) If not available yet (e.g., before the weekend), fall back to the previous round's Race
    if res is None:
        try:
            sched = fastf1.get_event_schedule(event.year)
            prev = sched[(sched["RoundNumber"] < event.round)]
            if not prev.empty:
                prev = prev.sort_values("RoundNumber").iloc[-1]
                res, used = _session_results(event.year, prev["EventName"], ["R", "Q", "Sprint"])
        except Exception:
            pass

    if res is None or res.empty:
        # As an ultimate fallback, return empty; caller will handle
        return pd.DataFrame(columns=["code","team_live","forename","surname"]), "none"

    # Extract robustly across FastF1 versions
    code = res["Abbreviation"]
    team_col = "TeamName" if "TeamName" in res.columns else ("Team" if "Team" in res.columns else None)
    team = res[team_col] if team_col else pd.Series([None] * len(res))
    # Names
    if "FullName" in res.columns:
        full = res["FullName"].astype(str)
        forename = full.str.split().str[:-1].str.join(" ")
        surname = full.str.split().str[-1]
    elif "BroadcastName" in res.columns:
        # BroadcastName often like "L HAM"; we’ll keep it simple and leave names blank if messy
        forename = pd.Series([""] * len(res))
        surname = pd.Series([""] * len(res))
    else:
        forename = pd.Series([""] * len(res))
        surname = pd.Series([""] * len(res))

    roster = pd.DataFrame({
        "code": code.astype(str),
        "team_live": team.astype(str),
        "forename_live": forename.astype(str),
        "surname_live": surname.astype(str),
    }).drop_duplicates(subset=["code"])
    return roster, used

def build_upcoming_features(include_quali: bool = False, year: int | None = None, rnd: int | None = None) -> pd.DataFrame:
    """Construct per-driver features for the next (or specified) race, ensuring 20 drivers.

    Steps:
    - Build a 20-driver roster from FastF1 (Q/Sprint/R/FPx; or previous round if the weekend hasn't started).
    - Left-join the roster onto the latest prequal features by driver code.
    - Optionally merge qualifying positions (post-qual).
    """
    # Determine target event
    if year is None or rnd is None:
        ek = _fastf1_next_event()
    else:
        ek = EventKey(year=int(year), round=int(rnd))

    # Latest features per driver (last available season)
    last = _load_last_known_prequal_features(ek.year)

    # Build roster (20 drivers) and merge
    roster, src = _find_roster(ek)
    if roster.empty:
        # If we truly can't get any roster, fall back to whatever 'last' has (may be <20)
        upcoming = last.copy()
    else:
        upcoming = roster.merge(last, on="code", how="left", suffixes=("_live", ""))
        # Prefer live names/team for display; keep historical as fallback
        if "forename_live" in upcoming.columns:
            upcoming["forename"] = upcoming["forename_live"].where(upcoming["forename_live"].str.strip() != "", upcoming.get("forename"))
        if "surname_live" in upcoming.columns:
            upcoming["surname"] = upcoming["surname_live"].where(upcoming["surname_live"].str.strip() != "", upcoming.get("surname"))
        if "team_live" in upcoming.columns:
            # Put live team into 'constructor' for display purposes
            upcoming["constructor"] = upcoming["team_live"].where(upcoming["team_live"].str.strip() != "", upcoming.get("constructor"))

    # Set target metadata
    upcoming["year"] = ek.year
    upcoming["round"] = ek.round
    upcoming["raceId"] = ek.year * 100 + ek.round  # synthetic

    # Qualifying/grid if requested
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

    out_cols = [c for c in ["raceId","year","round","driverId","constructorId","code","forename","surname","grid_pos"] if c in feats.columns]
    out = feats[out_cols].copy()
    out["win_proba"] = proba
    out = _apply_driver_filter(out).sort_values("win_proba", ascending=False).reset_index(drop=True)
    out = _attach_team_name(out)

    # percent for convenience in CLI/UI
    out["win_pct"] = (out["win_proba"] * 100).round(1)
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
