# src/build_features.py
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

from .config import DATA_RAW, DATA_INTERIM

# -----------------------------
# Helpers
# -----------------------------

def _time_to_seconds(x):
    """Convert 'M:SS.mmm' or 'SS.mmm' lap time strings to float seconds."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none"}:
        return np.nan
    try:
        if ":" in s:
            m, rest = s.split(":", 1)
            return int(m) * 60.0 + float(rest)
        return float(s)
    except Exception:
        return np.nan

def _ensure_dirs():
    DATA_INTERIM.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load & merge raw tables
# -----------------------------

def load_raw() -> dict[str, pd.DataFrame]:
    r = {}
    r["races"] = pd.read_csv(DATA_RAW / "races.csv")
    r["results"] = pd.read_csv(DATA_RAW / "results.csv")
    r["drivers"] = pd.read_csv(DATA_RAW / "drivers.csv")
    r["constructors"] = pd.read_csv(DATA_RAW / "constructors.csv")
    qpath = DATA_RAW / "qualifying.csv"
    r["qualifying"] = pd.read_csv(qpath) if qpath.exists() else pd.DataFrame()
    spath = DATA_RAW / "status.csv"
    r["status"] = pd.read_csv(spath) if spath.exists() else pd.DataFrame()
    return r

def make_base(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    races = raw["races"].copy()
    res = raw["results"].copy()
    drv = raw["drivers"].copy()
    cons = raw["constructors"].copy()

    races_small = races[["raceId","year","round","circuitId","name","date"]].copy()

    out = res.merge(races_small, on="raceId", how="left", validate="many_to_one")

    # driver info
    drv["forename"] = drv.get("forename", "").astype(str)
    drv["surname"] = drv.get("surname", "").astype(str)
    keep_drv = [c for c in ["driverId","driverRef","code","forename","surname","nationality"] if c in drv.columns]
    out = out.merge(drv[keep_drv], on="driverId", how="left", validate="many_to_one")

    # constructor info
    cons = cons.rename(columns={"name":"constructor"})
    keep_cons = [c for c in ["constructorId","constructor","nationality"] if c in cons.columns]
    out = out.merge(cons[keep_cons], on="constructorId", how="left", validate="many_to_one")

    # grid position (treat <=0 as NaN)
    if "grid" in out.columns:
        out["grid_pos"] = pd.to_numeric(out["grid"], errors="coerce")
        out.loc[out["grid_pos"] <= 0, "grid_pos"] = np.nan

    # finish position (ranking label)
    if "finish_pos" not in out.columns:
        if "positionOrder" in out.columns:
            out["finish_pos"] = pd.to_numeric(out["positionOrder"], errors="coerce")
        elif "position" in out.columns:
            out["finish_pos"] = pd.to_numeric(out["position"], errors="coerce")
        else:
            out["finish_pos"] = np.nan

    # binary win target
    out["is_win"] = (out["finish_pos"] == 1).astype(int)

    # status -> DNF
    if "status" not in out.columns and "statusId" in out.columns and not raw["status"].empty:
        status_map = raw["status"][ ["statusId","status"] ]
        out = out.merge(status_map, on="statusId", how="left")
    if "status" in out.columns:
        s = out["status"].astype(str).str.lower()
        finished = s.str.contains("finished") | s.str.contains("lap") | s.str.contains("laps")
        out["dnf"] = (~finished).astype(int)
    else:
        out["dnf"] = np.nan

    # qualifying merge
    qual = raw["qualifying"]
    if not qual.empty:
        q = qual.copy()
        if "position" in q.columns:
            qpos = (q.groupby(["raceId","driverId"])['position'].min()
                      .rename("quali_pos").reset_index())
        else:
            qpos = pd.DataFrame(columns=["raceId","driverId","quali_pos"])
        for col in ["q1","q2","q3"]:
            if col in q.columns:
                q[col+"_s"] = q[col].apply(_time_to_seconds)
        if any(col in q.columns for col in ["q1_s","q2_s","q3_s"]):
            q_times = (q.assign(best_q_s = q[[c for c in ["q1_s","q2_s","q3_s"] if c in q.columns]].min(axis=1))
                         .groupby(["raceId","driverId"])['best_q_s'].min().reset_index()
                         .rename(columns={"best_q_s":"quali_best_s"}))
        else:
            q_times = pd.DataFrame(columns=["raceId","driverId","quali_best_s"])
        out = (out.merge(qpos, on=["raceId","driverId"], how="left")
                  .merge(q_times, on=["raceId","driverId"], how="left"))

    if "code" in out.columns:
        out["code"] = out["code"].astype(str).str.upper().replace({"nan":""})

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.sort_values(["year","round","raceId","driverId"]).reset_index(drop=True)
    return out

# -----------------------------
# Rolling features (leak-safe) — use transform to preserve index
# -----------------------------

def _driver_rollings(df: pd.DataFrame) -> pd.DataFrame:
    g = df.sort_values(["year","round"]).copy()

    # season-to-date driver points BEFORE the current race
    if "points" in g.columns:
        pts = pd.to_numeric(g["points"], errors="coerce").fillna(0.0)
        # cumsum per (year, driver), then subtract current race's points
        cum = pts.groupby([g["year"], g["driverId"]]).cumsum()
        g["drv_points_season_to_date"] = (cum - pts).astype(float)
    else:
        g["drv_points_season_to_date"] = np.nan

    # average finish last 5 (exclude current)
    g["drv_avg_finish_last5"] = (
        g.groupby("driverId")["finish_pos"]
         .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )

    # podium rate last 5
    pod = (g["finish_pos"] <= 3).astype(float)
    g["drv_podium_rate_last5"] = (
        pod.groupby(g["driverId"]).transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )

    # dnf rate last 8
    if "dnf" in g.columns:
        g["drv_dnf_rate_last8"] = (
            g.groupby("driverId")["dnf"].transform(lambda s: s.shift(1).rolling(8, min_periods=1).mean())
        )
    else:
        g["drv_dnf_rate_last8"] = np.nan

    return g

def _constructor_rollings(df: pd.DataFrame) -> pd.DataFrame:
    g = df.sort_values(["year","round"]).copy()

    if "points" in g.columns:
        ptsc = pd.to_numeric(g["points"], errors="coerce").fillna(0.0)
        cumc = ptsc.groupby([g["year"], g["constructorId"]]).cumsum()
        g["cons_points_season_to_date"] = (cumc - ptsc).astype(float)
    else:
        g["cons_points_season_to_date"] = np.nan

    g["cons_avg_finish_last5"] = (
        g.groupby("constructorId")["finish_pos"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )

    if "dnf" in g.columns:
        g["cons_dnf_rate_last8"] = (
            g.groupby("constructorId")["dnf"].transform(lambda s: s.shift(1).rolling(8, min_periods=1).mean())
        )
    else:
        g["cons_dnf_rate_last8"] = np.nan

    return g

# -----------------------------
# Optional FastF1 extras
# -----------------------------

def _merge_fastf1_extras(df: pd.DataFrame) -> pd.DataFrame:
    path = DATA_INTERIM / "fastf1_extras.parquet"
    if not path.exists():
        return df
    ff1 = pd.read_parquet(path)
    if "driver_code" in ff1.columns:
        ff1 = ff1.rename(columns={"driver_code":"code"})
    keep = [c for c in ff1.columns if c in {"year","round","code","quali_best_s","bestlap_pct_fp1","bestlap_pct_fp2","bestlap_pct_fp3","quali_rank"}]
    ff1 = ff1[keep].copy()
    if all(c in df.columns for c in ["year","round","code"]):
        df = df.merge(ff1, on=["year","round","code"], how="left")
    return df

# -----------------------------
# Build features
# -----------------------------

def build_features(mode: str):
    assert mode in {"prequal","postqual"}
    _ensure_dirs()
    raw = load_raw()
    res = make_base(raw)

    # Rolling features (transform-based to preserve index)
    res = _driver_rollings(res)
    res = _constructor_rollings(res)

    # Optional: merge FastF1 extras
    res = _merge_fastf1_extras(res)

    # Leakage guard by mode
    if mode == "prequal":
        for c in ["quali_pos","quali_best_s","grid_pos"]:
            if c in res.columns:
                res[c] = np.nan

    # Cast numerics
    for c in ["quali_pos","quali_best_s","grid_pos","finish_pos",
              "drv_points_season_to_date","drv_avg_finish_last5","drv_podium_rate_last5","drv_dnf_rate_last8",
              "cons_points_season_to_date","cons_avg_finish_last5","cons_dnf_rate_last8"]:
        if c in res.columns:
            res[c] = pd.to_numeric(res[c], errors="coerce" )

    out_path = DATA_INTERIM / f"features_{mode}.parquet"
    res.to_parquet(out_path, index=False)
    print(f"Wrote {out_path} with {len(res)} rows and {res.shape[1]} columns.")

# -----------------------------
# CLI
# -----------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["prequal","postqual"], default="postqual")
    args = ap.parse_args()
    build_features(mode=args.mode)
