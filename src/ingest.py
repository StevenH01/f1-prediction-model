from __future__ import annotations
from pathlib import Path
import pandas as pd
from .config import DATA_RAW

REQUIRED = [
    "races.csv",
    "results.csv",
    "drivers.csv",
    "constructors.csv",
    "qualifying.csv",
    "circuits.csv",
    "status.csv",
]

def _read_csv(name: str, base_dir: Path) -> pd.DataFrame:
    path = base_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Expected file not found: {path}")
    df = pd.read_csv(path)
    return df

def load_kaggle_tables(base_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Load Kaggle/Ergast F1 CSVs into dataframes.
    Returns a dict keyed by logical table name.
    """
    base_dir = DATA_RAW if base_dir is None else Path(base_dir)
    for req in REQUIRED:
        if not (base_dir / req).exists():
            raise FileNotFoundError(f"Missing required file: {base_dir / req}")

    races = _read_csv("races.csv", base_dir)
    results = _read_csv("results.csv", base_dir)
    drivers = _read_csv("drivers.csv", base_dir)
    constructors = _read_csv("constructors.csv", base_dir)
    qualifying = _read_csv("qualifying.csv", base_dir)
    circuits = _read_csv("circuits.csv", base_dir)
    status = _read_csv("status.csv", base_dir)

    # Normalize dtypes & keys
    for df in [races, results, drivers, constructors, qualifying, circuits, status]:
        for c in [col for col in ["raceId","driverId","constructorId","statusId"] if col in df.columns]:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    # Ensure race ordering exists
    if "year" not in races.columns or "round" not in races.columns:
        raise ValueError("races.csv must include 'year' and 'round'.")

    races["race_order"] = races.sort_values(["year","round"]).groupby("year").cumcount()
    races["season_order"] = races.sort_values(["year","round"]).reset_index().index

    # Sprint results (optional — only present from 2021 onward)
    sprint_path = base_dir / "sprint_results.csv"
    if sprint_path.exists():
        sprint_results = pd.read_csv(sprint_path)
        for c in ["raceId", "driverId", "constructorId"] :
            if c in sprint_results.columns:
                sprint_results[c] = pd.to_numeric(sprint_results[c], errors="coerce").astype("Int64")
    else:
        sprint_results = pd.DataFrame(columns=["raceId", "driverId", "constructorId", "positionOrder"])

    return {
        "races": races,
        "results": results,
        "drivers": drivers,
        "constructors": constructors,
        "qualifying": qualifying,
        "circuits": circuits,
        "status": status,
        "sprint_results": sprint_results,
    }
