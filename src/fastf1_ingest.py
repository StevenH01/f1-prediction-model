from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

# Import your project config (assumes this file is placed in your src/ package)
from .config import DATA_INTERIM, PROJECT_ROOT

def _enable_cache(cache_dir: Path):
    import fastf1
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        fastf1.Cache.enable_cache(str(cache_dir))
    except Exception:
        pass

def _event_for_round(year: int, round_number: int):
    import fastf1
    schedule = fastf1.get_event_schedule(year)
    if "RoundNumber" not in schedule.columns:
        raise RuntimeError("FastF1 schedule schema unexpected; missing RoundNumber")
    row = schedule.loc[schedule["RoundNumber"] == int(round_number)]
    if row.empty:
        raise ValueError(f"No event for year={year}, round={round_number}")
    return row.iloc[0]

def _lap_seconds(series):
    try:
        return series.dt.total_seconds()
    except Exception:
        return pd.to_numeric(series, errors="coerce")

def _fp_summary(year: int, round_number: int) -> pd.DataFrame:
    """Return per-driver FP1/FP2/FP3 pace features.

    Outputs columns keyed by (year, round, driver_code):
      - bestlap_s_fp1/fp2/fp3
      - median_s_fp1/fp2/fp3
      - bestlap_pct_fp1/fp2/fp3  (lower is faster)
    """
    import fastf1
    ev = _event_for_round(year, round_number)
    feats = []
    for sess in ["FP1", "FP2", "FP3"]:
        try:
            s = fastf1.get_session(year, ev["EventName"], sess)
            s.load(telemetry=False, laps=True, weather=True)
        except Exception:
            continue
        laps = s.laps
        if laps is None or laps.empty:
            continue
        laps = laps.pick_quicklaps()
        if laps.empty:
            continue
        df = laps[["Driver", "LapTime"]].copy()
        df["lap_time_s"] = _lap_seconds(df["LapTime"])
        g = df.groupby("Driver", as_index=False)["lap_time_s"]
        agg = g.agg(bestlap_s=("lap_time_s", "min"),
                    median_s=("lap_time_s", "median"))
        agg["bestlap_pct"] = agg["bestlap_s"].rank(pct=True, method="min")
        agg["session"] = sess.lower()
        agg["year"] = int(year)
        agg["round"] = int(round_number)
        agg = agg.rename(columns={"Driver": "driver_code"})
        feats.append(agg)

    if not feats:
        return pd.DataFrame(columns=["year","round","driver_code"])

    allfp = pd.concat(feats, ignore_index=True)
    wide = allfp.pivot_table(index=["year","round","driver_code"],
                             columns="session",
                             values=["bestlap_s","median_s","bestlap_pct"])
    wide.columns = [f"{a}_{b}" for a,b in wide.columns]
    wide = wide.reset_index()
    return wide

def _quali_summary(year: int, round_number: int) -> pd.DataFrame:
    """Return per-driver qualifying best time and rank from FastF1 results.

    Columns: year, round, driver_code, quali_best_s, quali_rank_ff1
    """
    import fastf1
    ev = _event_for_round(year, round_number)
    try:
        s = fastf1.get_session(year, ev["EventName"], "Q")
        s.load(telemetry=False, laps=True, weather=False)
    except Exception:
        return pd.DataFrame(columns=["year","round","driver_code","quali_best_s","quali_rank_ff1"])

    res = s.results
    if res is None or res.empty:
        return pd.DataFrame(columns=["year","round","driver_code","quali_best_s","quali_rank_ff1"])

    out = res[["DriverNumber","Abbreviation","Q1","Q2","Q3","Position"]].copy()
    for col in ["Q1","Q2","Q3"]:
        out[col] = pd.to_timedelta(out[col], errors="coerce").dt.total_seconds()
    out["quali_best_s"] = out[["Q1","Q2","Q3"]].min(axis=1, skipna=True)
    out = out.rename(columns={"Abbreviation":"driver_code", "Position":"quali_rank_ff1"})
    out["year"] = int(year); out["round"] = int(round_number)
    return out[["year","round","driver_code","quali_best_s","quali_rank_ff1"]]

def build_fastf1_extras(years: list[int], cache_dir: Path | None = None) -> Path:
    """Build a parquet with FP and qualifying summaries from FastF1.
    Timing/telemetry are most reliable from ~2018 onward.
    """
    cache_dir = (PROJECT_ROOT / "data_cache" / "fastf1") if cache_dir is None else Path(cache_dir)
    _enable_cache(cache_dir)

    parts = []
    try:
        import fastf1
    except Exception as e:
        raise RuntimeError("fastf1 is not installed. Run `pip install fastf1`.") from e

    for y in years:
        try:
            sched = fastf1.get_event_schedule(y)
            rounds = sorted(sched["RoundNumber"].dropna().astype(int).unique().tolist())
        except Exception:
            rounds = list(range(1, 26))
        for rnd in rounds:
            try:
                fp = _fp_summary(y, rnd)
                q = _quali_summary(y, rnd)
                df = fp.merge(q, on=["year","round","driver_code"], how="outer")
                if not df.empty:
                    parts.append(df)
            except Exception:
                continue
    extras = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["year","round","driver_code"])
    out = DATA_INTERIM / "fastf1_extras.parquet"
    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    extras.to_parquet(out, index=False)
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=str, default="2018-2025", help="Year list or range, e.g. 2019,2020 or 2018-2025")
    ap.add_argument("--cache", type=str, default=None, help="Cache directory for FastF1")
    args = ap.parse_args()

    years = []
    for part in args.years.split(","):
        part = part.strip()
        if "-" in part:
            a,b = part.split("-",1)
            years.extend(range(int(a), int(b)+1))
        else:
            years.append(int(part))
    path = build_fastf1_extras(years, cache_dir=Path(args.cache) if args.cache else None)
    print(f"Wrote {path}")