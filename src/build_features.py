from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from rich import print as rprint

from .config import DATA_INTERIM
from .ingest import load_kaggle_tables
from .ratings import compute_multiplayer_elo

def _flag_dnf(status_str: str) -> int:
    if pd.isna(status_str):
        return 0
    s = str(status_str).lower()
    # Consider anything not 'finished' or '+N laps' as DNF
    if s.startswith("finished") or "lap" in s:
        return 0
    return 1

def build_features(mode: str = "prequal") -> Path:
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
    res = results.merge(races[["raceId","year","round","circuitId","name"]], on="raceId", how="left")
    res = res.merge(drivers[["driverId","driverRef","code","forename","surname"]], on="driverId", how="left")
    res = res.merge(constructors[["constructorId","name"]].rename(columns={"name":"constructor"}), on="constructorId", how="left")
    res = res.merge(status, on="statusId", how="left")

    # Basic filters & fields
    # Use positionOrder (numeric order), drop rows without it
    res = res[pd.notna(res["positionOrder"])].copy()
    res["positionOrder"] = res["positionOrder"].astype(int)
    res["is_win"] = (res["positionOrder"] == 1).astype(int)
    res["dnf"] = res["status"].apply(_flag_dnf)

    # Driver Elo pre-race
    elo_driver = compute_multiplayer_elo(res[["raceId","year","round","driverId","positionOrder"]].copy(), by="driverId")
    res = res.merge(elo_driver, on=["raceId","driverId"], how="left")

    # Constructor Elo pre-race
    tmp = res[["raceId","year","round","constructorId","positionOrder"]].dropna().copy()
    elo_cons = compute_multiplayer_elo(tmp.rename(columns={"constructorId":"entityId"}).assign(constructorId=lambda d: d["entityId"]).drop(columns=["entityId"]),
                                       by="constructorId")
    res = res.merge(elo_cons, on=["raceId","constructorId"], how="left")

    # Rolling aggregates (driver season to date)
    res = res.sort_values(["year","round","driverId"])
    res["points"] = res["points"].fillna(0.0)

    def add_driver_rolling(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["year","round"])
        df["drv_points_season_to_date"] = df.groupby("year")["points"].cumsum().shift(1).fillna(0.0)
        df["drv_avg_finish_season"] = df.groupby("year")["positionOrder"].apply(lambda s: s.shift(1).expanding().mean()).values
        df["drv_dnf_season"] = df.groupby("year")["dnf"].cumsum().shift(1).fillna(0.0)
        df["drv_last5_points"] = df["points"].shift(1).rolling(5, min_periods=1).sum().fillna(0.0)
        return df

    res = res.groupby("driverId", group_keys=False, include_groups=False).apply(add_driver_rolling)

    # Constructor rolling
    def add_cons_rolling(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["year","round"])
        df["cons_points_season_to_date"] = df.groupby("year")["points"].cumsum().shift(1).fillna(0.0)
        df["cons_dnf_season"] = df.groupby("year")["dnf"].cumsum().shift(1).fillna(0.0)
        df["cons_last5_points"] = df["points"].shift(1).rolling(5, min_periods=1).sum().fillna(0.0)
        return df

    res = res.groupby("constructorId", group_keys=False, include_groups=False).apply(add_cons_rolling)

    # Track affinity (driver & constructor at this circuit)
    res = res.sort_values(["year","round"])
    res["drv_circuit_prev_med_finish"] = (
        res.groupby(["driverId","circuitId"])["positionOrder"]
          .apply(lambda s: s.shift(1).rolling(3, min_periods=1).median())
          .reset_index(level=[0,1], drop=True)
    )
    res["cons_circuit_prev_med_finish"] = (
        res.groupby(["constructorId","circuitId"])["positionOrder"]
          .apply(lambda s: s.shift(1).rolling(3, min_periods=1).median())
          .reset_index(level=[0,1], drop=True)
    )

    # Qualifying/grid (post-qual only)
    if mode == "postqual":
        # Get best quali position per driver in race
        qbest = qualifying.sort_values(["raceId","driverId","position"]).drop_duplicates(["raceId","driverId"], keep="first")
        qbest = qbest.rename(columns={"position":"quali_pos"})
        res = res.merge(qbest[["raceId","driverId","quali_pos"]], on=["raceId","driverId"], how="left")

        # Use results.grid as known pre-race grid (may include penalties)
        if "grid" in res.columns:
            res["grid_pos"] = res["grid"].replace(0, pd.NA)  # 0 sometimes means pit-lane start; treat as NA
        else:
            res["grid_pos"] = pd.NA

        # Teammate grid gap
        res["teammate_grid_gap"] = res.groupby(["raceId","constructorId"])["grid_pos"].transform(lambda s: s - s.min())

    # Final feature set
    feature_cols = [
        "driverId","constructorId","raceId","year","round",
        "driverRef","code","forename","surname","constructor",
        "driverId_elo_pre","constructorId_elo_pre",
        "drv_points_season_to_date","drv_avg_finish_season","drv_dnf_season","drv_last5_points",
        "cons_points_season_to_date","cons_dnf_season","cons_last5_points",
        "drv_circuit_prev_med_finish","cons_circuit_prev_med_finish",
    ]

    if mode == "postqual":
        feature_cols += ["quali_pos","grid_pos","teammate_grid_gap"]

    # Some columns may not exist if no quali data present in early eras
    feature_cols = [c for c in feature_cols if c in res.columns]

    feats = res[feature_cols + ["is_win"]].copy()

    # Numeric cleaning
    numeric_cols = [c for c in feats.columns if c not in {"driverRef","code","forename","surname","constructor"}]
    feats[numeric_cols] = feats[numeric_cols].apply(pd.to_numeric, errors="coerce")
    feats = feats.dropna(subset=["raceId","driverId","constructorId","year","round"])

    # Save
    out_path = DATA_INTERIM / f"features_{mode}.parquet"
    feats.to_parquet(out_path, index=False)
    rprint(f"[green]Wrote features -->[/green] {out_path}")
    return out_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["prequal","postqual"], default="prequal")
    args = parser.parse_args()
    build_features(mode=args.mode)
