"""live_season.py — Fetch completed race results for seasons not yet in the Kaggle dataset.

FastF1's Session.results includes Ergast-compatible 'DriverId' (driverRef string)
and 'TeamId' (constructorRef string), letting us map drivers/constructors back to
their Ergast integer IDs without manual lookups.

New drivers/constructors not yet in the Kaggle CSVs receive synthetic IDs that are
persisted in data_interim/synthetic_id_map.json so they stay stable across runs.

Usage:
    # Fetch and cache all completed 2026 races
    python -m src.live_season --year 2026

    # Then rebuild features (live data is picked up automatically)
    python -m src.build_features --mode postqual
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from .config import DATA_RAW, DATA_INTERIM, PROJECT_ROOT, KAGGLE_DATA_MAX_YEAR

# ---------------------------------------------------------------------------
# Synthetic ID ranges (well above Ergast maximums)
# ---------------------------------------------------------------------------
_SYNTH_MAP_PATH = DATA_INTERIM / "synthetic_id_map.json"
_SYNTH_DRIVER_BASE = 10_000   # Ergast driverIds top out around 860
_SYNTH_CONSTRUCTOR_BASE = 5_000  # Ergast constructorIds top out around 215


# ---------------------------------------------------------------------------
# Persistent ID-cache helpers
# ---------------------------------------------------------------------------

def _load_id_cache() -> dict:
    if _SYNTH_MAP_PATH.exists():
        try:
            return json.loads(_SYNTH_MAP_PATH.read_text())
        except Exception:
            pass
    return {
        "drivers": {},
        "constructors": {},
        "next_driver": _SYNTH_DRIVER_BASE,
        "next_constructor": _SYNTH_CONSTRUCTOR_BASE,
    }


def _save_id_cache(cache: dict) -> None:
    _SYNTH_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SYNTH_MAP_PATH.write_text(json.dumps(cache, indent=2))


# ---------------------------------------------------------------------------
# Kaggle lookup tables
# ---------------------------------------------------------------------------

def _kaggle_driver_map() -> dict[str, int]:
    """driverRef (lower-case) → driverId from drivers.csv"""
    path = DATA_RAW / "drivers.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, usecols=["driverId", "driverRef"])
    return {str(ref).strip().lower(): int(did)
            for ref, did in zip(df["driverRef"], df["driverId"])
            if pd.notna(ref) and pd.notna(did)}


def _kaggle_constructor_map() -> dict[str, int]:
    """constructorRef (lower-case) → constructorId from constructors.csv"""
    path = DATA_RAW / "constructors.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, usecols=["constructorId", "constructorRef"])
    return {str(ref).strip().lower(): int(cid)
            for ref, cid in zip(df["constructorRef"], df["constructorId"])
            if pd.notna(ref) and pd.notna(cid)}


def _kaggle_circuit_df() -> pd.DataFrame:
    path = DATA_RAW / "circuits.csv"
    if not path.exists():
        return pd.DataFrame(columns=["circuitId", "name", "location", "country"])
    return pd.read_csv(path, usecols=["circuitId", "name", "location", "country"])


# ---------------------------------------------------------------------------
# ID resolution
# ---------------------------------------------------------------------------

def _resolve_driver_id(driver_ref: str, cache: dict, drv_map: dict[str, int]) -> int:
    key = str(driver_ref).strip().lower()
    if key in drv_map:
        return int(drv_map[key])
    if key not in cache["drivers"]:
        cache["drivers"][key] = cache["next_driver"]
        cache["next_driver"] += 1
    return int(cache["drivers"][key])


def _resolve_constructor_id(constructor_ref: str, cache: dict, con_map: dict[str, int]) -> int:
    key = str(constructor_ref).strip().lower()
    if key in con_map:
        return int(con_map[key])
    if key not in cache["constructors"]:
        cache["constructors"][key] = cache["next_constructor"]
        cache["next_constructor"] += 1
    return int(cache["constructors"][key])


def _resolve_circuit_id(event_name: str, country: str, circuits_df: pd.DataFrame) -> Optional[int]:
    if circuits_df.empty:
        return None
    country_l = str(country).strip().lower()
    event_l = str(event_name).strip().lower()

    # 1) Exact country match
    m = circuits_df[circuits_df["country"].str.lower().str.strip() == country_l]
    if not m.empty:
        return int(m.iloc[0]["circuitId"])

    # 2) Partial event-name match against circuit name or location
    for _, row in circuits_df.iterrows():
        if (str(row["name"]).lower() in event_l
                or str(row["location"]).lower() in event_l
                or str(row["country"]).lower() in event_l):
            return int(row["circuitId"])
    return None


# ---------------------------------------------------------------------------
# FastF1 race-result fetcher
# ---------------------------------------------------------------------------

def _enable_ff1_cache(cache_dir: Path) -> None:
    import fastf1
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        fastf1.Cache.enable_cache(str(cache_dir))
    except Exception:
        pass


def _is_sprint_weekend(event_row) -> bool:
    fmt = str(event_row.get("EventFormat", "")).lower()
    return "sprint" in fmt


def _fetch_race_result(year: int, event_name: str, cache_dir: Path) -> Optional[pd.DataFrame]:
    """Load the Race session results for one event. Returns None if not yet run."""
    import fastf1
    try:
        sess = fastf1.get_session(year, event_name, "R")
        sess.load(telemetry=False, laps=False, weather=False)
        if sess.results is None or sess.results.empty:
            return None
        return sess.results
    except Exception:
        return None


def _parse_position(val) -> int:
    """Convert FastF1 ClassifiedPosition/Position to int; DSQ/DNF/DNS → 20."""
    try:
        return int(val)
    except (TypeError, ValueError):
        return 20


# ---------------------------------------------------------------------------
# Core public API
# ---------------------------------------------------------------------------

def build_live_supplement(year: int, cache_dir: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """Fetch all completed race results for `year` and return an Ergast-compatible DataFrame.

    Columns produced (mirrors what build_features.py expects after the initial merge):
        raceId, year, round, circuitId,
        driverId, constructorId,
        positionOrder, points, grid,
        status, dnf, is_win,
        driverRef, code, forename, surname, constructor
    """
    import fastf1

    if cache_dir is None:
        cache_dir = PROJECT_ROOT / "data_cache" / "fastf1"
    _enable_ff1_cache(cache_dir)

    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
    except Exception as exc:
        raise RuntimeError(f"FastF1: cannot get {year} schedule — {exc}") from exc

    drv_map = _kaggle_driver_map()
    con_map = _kaggle_constructor_map()
    circuits_df = _kaggle_circuit_df()
    id_cache = _load_id_cache()

    now_utc = pd.Timestamp.now(tz="UTC")
    parts: list[pd.DataFrame] = []

    for _, ev in schedule.iterrows():
        rnd = int(ev.get("RoundNumber", 0))
        if rnd <= 0:
            continue

        event_name = str(ev.get("EventName", ""))

        # Skip future rounds
        date_col = next(
            (c for c in ["EventDate", "EventDateUtc", "EventDateUTC", "EventStartDate"]
             if c in ev.index),
            None,
        )
        if date_col:
            ev_date = pd.to_datetime(ev[date_col], utc=True, errors="coerce")
            if pd.isna(ev_date) or ev_date > now_utc:
                continue

        results = _fetch_race_result(year, event_name, cache_dir)
        if results is None or results.empty:
            continue

        country = str(ev.get("Country", ev.get("Location", "")))
        circuit_id = _resolve_circuit_id(event_name, country, circuits_df)
        race_id = int(f"9{year}{rnd:03d}")  # e.g. 92026001 — outside Ergast range
        is_sprint = int(_is_sprint_weekend(ev))

        rows: list[dict] = []
        for _, r in results.iterrows():
            # Ergast-compatible refs (available in FastF1 v3+)
            driver_ref = str(r.get("DriverId", "")).strip().lower()
            if not driver_ref or driver_ref == "nan":
                driver_ref = str(r.get("Abbreviation", "unknown")).strip().lower()

            constructor_ref = str(r.get("TeamId", "")).strip().lower()
            if not constructor_ref or constructor_ref == "nan":
                team_raw = r.get("TeamName", r.get("Team", "unknown"))
                constructor_ref = str(team_raw).lower().replace(" ", "_")

            driver_id = _resolve_driver_id(driver_ref, id_cache, drv_map)
            constructor_id = _resolve_constructor_id(constructor_ref, id_cache, con_map)

            pos = _parse_position(r.get("ClassifiedPosition", r.get("Position")))
            pts = float(r.get("Points", 0) or 0)
            status_str = str(r.get("Status", "Finished"))

            grid = r.get("GridPosition", np.nan)
            try:
                grid = int(grid)
            except (TypeError, ValueError):
                grid = None

            # Names — FastF1 v3 provides FirstName / LastName
            fname = str(r.get("FirstName", "")).strip()
            lname = str(r.get("LastName", "")).strip()
            if not fname and not lname and "FullName" in r.index:
                parts_name = str(r["FullName"]).split()
                fname = parts_name[0] if parts_name else ""
                lname = parts_name[-1] if len(parts_name) > 1 else ""

            dnf = 0 if (status_str.lower().startswith("finished") or "lap" in status_str.lower()) else 1

            rows.append({
                "raceId":          race_id,
                "year":            int(year),
                "round":           int(rnd),
                "circuitId":       circuit_id,
                "driverId":        driver_id,
                "constructorId":   constructor_id,
                "positionOrder":   pos,
                "points":          pts,
                "grid":            grid,
                "status":          status_str,
                "statusId":        None,
                "dnf":             dnf,
                "is_win":          int(pos == 1),
                "is_sprint_weekend": is_sprint,
                "driverRef":       driver_ref,
                "code":            str(r.get("Abbreviation", "")).upper(),
                "forename":        fname,
                "surname":         lname,
                "constructor":     str(r.get("TeamName", r.get("Team", constructor_ref))),
            })

        if rows:
            parts.append(pd.DataFrame(rows))

    _save_id_cache(id_cache)

    if not parts:
        return None

    return pd.concat(parts, ignore_index=True)


def save_live_supplement(year: int, cache_dir: Optional[Path] = None) -> Path:
    """Fetch `year` race results and persist to data_interim/live_results_{year}.parquet."""
    out_path = DATA_INTERIM / f"live_results_{year}.parquet"
    DATA_INTERIM.mkdir(parents=True, exist_ok=True)

    print(f"[live_season] Fetching {year} results from FastF1…")
    df = build_live_supplement(year, cache_dir)

    if df is None or df.empty:
        print(f"[live_season] No completed races found for {year}.")
        df = pd.DataFrame()

    df.to_parquet(out_path, index=False)
    races = df["raceId"].nunique() if not df.empty else 0
    n = len(df)
    print(f"[live_season] Saved → {out_path}  ({races} races, {n} rows)")
    return out_path


def load_live_supplements(after_year: int = KAGGLE_DATA_MAX_YEAR) -> list[pd.DataFrame]:
    """Return cached live-season DataFrames for every year > after_year.

    Called by build_features.py before Elo computation so that 2026+ race
    results feed into rolling statistics and Elo updates.
    """
    parts: list[pd.DataFrame] = []
    for path in sorted(DATA_INTERIM.glob("live_results_*.parquet")):
        try:
            year = int(path.stem.replace("live_results_", ""))
        except ValueError:
            continue
        if year <= after_year:
            continue
        df = pd.read_parquet(path)
        if not df.empty:
            parts.append(df)
            print(f"[live_season] Loaded supplement: {path.name}  ({len(df)} rows)")
    return parts


def get_live_supplement_status() -> list[dict]:
    """Return a summary of what live supplements are on disk (for the UI)."""
    rows = []
    for path in sorted(DATA_INTERIM.glob("live_results_*.parquet")):
        try:
            year = int(path.stem.replace("live_results_", ""))
            df = pd.read_parquet(path)
            races = int(df["raceId"].nunique()) if not df.empty else 0
            rows.append({"year": year, "races": races, "path": str(path)})
        except Exception:
            continue
    return rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Fetch live F1 race results via FastF1")
    ap.add_argument("--year", type=int, default=2026, help="Season year to fetch")
    ap.add_argument("--cache", type=str, default=None, help="FastF1 cache directory")
    args = ap.parse_args()
    save_live_supplement(args.year, Path(args.cache) if args.cache else None)
