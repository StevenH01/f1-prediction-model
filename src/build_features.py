from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from rich import print as rprint

from .config import DATA_INTERIM, DATA_RAW, KAGGLE_DATA_MAX_YEAR
from .ingest import load_kaggle_tables
from .ratings import compute_multiplayer_elo
from .config import year_to_reg_era
from .live_season import load_live_supplements


def _flag_dnf(status_str: str) -> int:
    if pd.isna(status_str):
        return 0
    s = str(status_str).lower()
    # Consider anything not 'finished' or '+N laps' as DNF
    if s.startswith("finished") or "lap" in s:
        return 0
    return 1


def _load_sprint_race_ids(tables: dict) -> set[int]:
    """Return the set of raceIds that had a sprint race (from sprint_results.csv)."""
    try:
        sprint = tables.get("sprint_results")
        if sprint is None or sprint.empty:
            return set()
        return set(sprint["raceId"].dropna().astype(int).unique())
    except Exception:
        return set()


def build_features(mode: str = "prequal", year_min: int | None = None) -> Path:
    assert mode in {"prequal", "postqual"}
    tables = load_kaggle_tables()

    races = tables["races"]
    results = tables["results"]
    drivers = tables["drivers"]
    constructors = tables["constructors"]
    qualifying = tables["qualifying"]
    circuits = tables["circuits"]
    status = tables["status"]

    # Merge result essentials
    res = results.merge(races[["raceId", "year", "round", "circuitId", "name"]], on="raceId", how="left")
    res = res.merge(drivers[["driverId", "driverRef", "code", "forename", "surname"]], on="driverId", how="left")
    res = res.merge(
        constructors[["constructorId", "name"]].rename(columns={"name": "constructor"}),
        on="constructorId", how="left",
    )
    res = res.merge(status, on="statusId", how="left")

    # Optional: restrict to a minimum year to focus on modern eras
    if year_min is not None:
        res = res[res["year"] >= int(year_min)].copy()

    # Basic filters & fields
    res = res[pd.notna(res["positionOrder"])].copy()
    res["positionOrder"] = res["positionOrder"].astype(int)
    res["is_win"] = (res["positionOrder"] == 1).astype(int)
    res["dnf"] = res["status"].apply(_flag_dnf)

    # ------------------------------------------------------------------
    # Live season supplement — inject races for years beyond Kaggle data
    # (e.g. 2026 races fetched via FastF1 using `python -m src.live_season`)
    # The supplement is appended BEFORE Elo computation so that in-season
    # results flow into Elo updates and rolling statistics.
    # ------------------------------------------------------------------
    max_kaggle_year = int(res["year"].max()) if not res.empty else KAGGLE_DATA_MAX_YEAR
    live_parts = load_live_supplements(after_year=max_kaggle_year)
    if live_parts:
        # Align columns: supplement has a subset of res columns; extras become NaN
        live_df = pd.concat(live_parts, ignore_index=True)
        # Ensure numeric types match
        for col in ["positionOrder", "driverId", "constructorId", "year", "round"]:
            if col in live_df.columns:
                live_df[col] = pd.to_numeric(live_df[col], errors="coerce")
        live_df["positionOrder"] = live_df["positionOrder"].fillna(20).astype(int)

        # If the supplement already carried sprint flag, keep it; otherwise NaN
        if "is_sprint_weekend" not in live_df.columns:
            live_df["is_sprint_weekend"] = 0

        res = pd.concat([res, live_df], ignore_index=True)
        rprint(
            f"[cyan]Live supplement:[/cyan] appended {len(live_df):,} rows "
            f"({live_df['year'].nunique()} season(s): "
            f"{sorted(live_df['year'].unique().tolist())})"
        )

    # ------------------------------------------------------------------
    # Regulation era (ordinal integer — captures power-unit / aero regime)
    # ------------------------------------------------------------------
    res["reg_era"] = res["year"].apply(year_to_reg_era)

    # ------------------------------------------------------------------
    # Sprint weekend flag (Kaggle sprint_results; live supplement sets its own)
    # ------------------------------------------------------------------
    sprint_race_ids = _load_sprint_race_ids(tables)
    # Only overwrite for rows that don't already have the flag set by live supplement
    kaggle_mask = res["raceId"].isin(sprint_race_ids)
    if "is_sprint_weekend" not in res.columns:
        res["is_sprint_weekend"] = kaggle_mask.astype(int)
    else:
        res["is_sprint_weekend"] = res["is_sprint_weekend"].fillna(0).astype(int)
        res.loc[kaggle_mask, "is_sprint_weekend"] = 1

    # Driver Elo pre-race
    elo_driver = compute_multiplayer_elo(
        res[["raceId", "year", "round", "driverId", "positionOrder"]].copy(), by="driverId"
    )
    res = res.merge(elo_driver, on=["raceId", "driverId"], how="left")

    # Constructor Elo pre-race
    tmp = res[["raceId", "year", "round", "constructorId", "positionOrder"]].dropna().copy()
    elo_cons = compute_multiplayer_elo(
        tmp.rename(columns={"constructorId": "entityId"})
           .assign(constructorId=lambda d: d["entityId"])
           .drop(columns=["entityId"]),
        by="constructorId",
    )
    res = res.merge(elo_cons, on=["raceId", "constructorId"], how="left")

    # Rolling aggregates (driver season to date)
    res = res.sort_values(["year", "round", "driverId"])
    res["points"] = res["points"].fillna(0.0)

    def add_driver_rolling(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["year", "round"])
        df["drv_points_season_to_date"] = (
            df.groupby("year")["points"].cumsum().shift(1).fillna(0.0)
        )
        df["drv_avg_finish_season"] = (
            df.groupby("year")["positionOrder"]
            .apply(lambda s: s.shift(1).expanding().mean())
            .values
        )
        df["drv_dnf_season"] = df.groupby("year")["dnf"].cumsum().shift(1).fillna(0.0)
        df["drv_last5_points"] = df["points"].shift(1).rolling(5, min_periods=1).sum().fillna(0.0)
        return df

    res = res.groupby("driverId", group_keys=False).apply(add_driver_rolling)

    # Constructor rolling
    def add_cons_rolling(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["year", "round"])
        df["cons_points_season_to_date"] = (
            df.groupby("year")["points"].cumsum().shift(1).fillna(0.0)
        )
        df["cons_dnf_season"] = df.groupby("year")["dnf"].cumsum().shift(1).fillna(0.0)
        df["cons_last5_points"] = df["points"].shift(1).rolling(5, min_periods=1).sum().fillna(0.0)
        return df

    res = res.groupby("constructorId", group_keys=False).apply(add_cons_rolling)

    # Track affinity (driver & constructor at this circuit)
    res = res.sort_values(["year", "round"])
    res["drv_circuit_prev_med_finish"] = (
        res.groupby(["driverId", "circuitId"])["positionOrder"]
        .apply(lambda s: s.shift(1).rolling(3, min_periods=1).median())
        .reset_index(level=[0, 1], drop=True)
    )
    res["cons_circuit_prev_med_finish"] = (
        res.groupby(["constructorId", "circuitId"])["positionOrder"]
        .apply(lambda s: s.shift(1).rolling(3, min_periods=1).median())
        .reset_index(level=[0, 1], drop=True)
    )

    # Qualifying/grid (post-qual only)
    if mode == "postqual":
        qbest = (
            qualifying.sort_values(["raceId", "driverId", "position"])
            .drop_duplicates(["raceId", "driverId"], keep="first")
        )
        qbest = qbest.rename(columns={"position": "quali_pos"})
        res = res.merge(qbest[["raceId", "driverId", "quali_pos"]], on=["raceId", "driverId"], how="left")

        if "grid" in res.columns:
            res["grid_pos"] = res["grid"].replace(0, pd.NA)
        else:
            res["grid_pos"] = pd.NA

        res["teammate_grid_gap"] = res.groupby(["raceId", "constructorId"])["grid_pos"].transform(
            lambda s: s - s.min()
        )

    # FastF1 extras (FP pace + quali raw times from FastF1)
    ff1_path = DATA_INTERIM / "fastf1_extras.parquet"
    if ff1_path.exists():
        ff1 = pd.read_parquet(ff1_path)
        if not ff1.empty and "driver_code" in ff1.columns:
            if "code" in res.columns:
                res = res.merge(
                    ff1.rename(columns={"driver_code": "code"}),
                    on=["year", "round", "code"],
                    how="left",
                )

    # ------------------------------------------------------------------
    # Final feature set
    # ------------------------------------------------------------------
    feature_cols = [
        "driverId", "constructorId", "raceId", "year", "round",
        "driverRef", "code", "forename", "surname", "constructor",
        # Elo
        "driverId_elo_pre", "constructorId_elo_pre",
        # Driver rolling
        "drv_points_season_to_date", "drv_avg_finish_season",
        "drv_dnf_season", "drv_last5_points",
        # Constructor rolling
        "cons_points_season_to_date", "cons_dnf_season", "cons_last5_points",
        # Circuit affinity
        "drv_circuit_prev_med_finish", "cons_circuit_prev_med_finish",
        # Regulation era & sprint flag
        "reg_era", "is_sprint_weekend",
    ]

    if mode == "postqual":
        feature_cols += ["quali_pos", "grid_pos", "teammate_grid_gap"]

    # FastF1 pace columns (added only if fastf1_extras was populated)
    ff1_pace_cols = [
        "bestlap_s_fp1", "bestlap_s_fp2", "bestlap_s_fp3",
        "median_s_fp1", "median_s_fp2", "median_s_fp3",
        "bestlap_pct_fp1", "bestlap_pct_fp2", "bestlap_pct_fp3",
        "quali_best_s", "quali_rank_ff1",
    ]
    feature_cols += [c for c in ff1_pace_cols if c in res.columns]

    # Drop columns not present (early eras may lack quali data, etc.)
    feature_cols = [c for c in feature_cols if c in res.columns]

    feats = res[feature_cols + ["is_win"]].copy()

    # Numeric cleaning
    str_cols = {"driverRef", "code", "forename", "surname", "constructor"}
    numeric_cols = [c for c in feats.columns if c not in str_cols]
    feats[numeric_cols] = feats[numeric_cols].apply(pd.to_numeric, errors="coerce")
    feats = feats.dropna(subset=["raceId", "driverId", "constructorId", "year", "round"])

    # Save
    out_path = DATA_INTERIM / f"features_{mode}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out_path, index=False)
    rprint(f"[green]Wrote features -->[/green] {out_path}  ({len(feats):,} rows)")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["prequal", "postqual"], default="prequal")
    parser.add_argument(
        "--year-min", type=int, default=None,
        help="Restrict training data to this year and later (e.g. 2014 for hybrid era only)",
    )
    args = parser.parse_args()
    build_features(mode=args.mode, year_min=args.year_min)
