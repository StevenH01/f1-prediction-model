from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data_raw"
DATA_INTERIM = PROJECT_ROOT / "data_interim"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# ---------------------------------------------------------------------------
# 2025 driver exclusion list
# Drivers who are NOT on the 2025 F1 grid (retired/departed between seasons).
# Jack Doohan (DOO) is ACTIVE at Alpine 2025 — removed from exclusions.
# Kimi Antonelli (ANT), Isack Hadjar (HAD), Gabriel Bortoleto (BOR),
# Oliver Bearman (BEA), Liam Lawson (LAW) are 2025 rookies — not excluded.
# ---------------------------------------------------------------------------
EXCLUDED_DRIVER_NAMES = {
    # Departed after 2024
    "Kevin Magnussen",       # replaced by Bearman at Haas
    "Daniel Ricciardo",      # departed mid-2024
    "Daniel Riccardo",       # alternate spelling in Ergast data
    "Sergio Pérez",          # replaced by Lawson at Red Bull
    "Sergio Perez",          # alternate spelling without accent
    "Valtteri Bottas",       # departed (Sauber)
    "Logan Sargeant",        # departed mid-2024 (Williams)
    "Guanyu Zhou",           # departed (Sauber)
}

EXCLUDED_DRIVER_CODES = {
    "MAG",  # Kevin Magnussen
    "RIC",  # Daniel Ricciardo
    "PER",  # Sergio Pérez
    "BOT",  # Valtteri Bottas
    "SAR",  # Logan Sargeant
    "ZHO",  # Guanyu Zhou
}

# ---------------------------------------------------------------------------
# Regulation eras — used as an ordinal feature in build_features.py
# Encodes the aerodynamic/power-unit regime a race was run under.
# Higher integer = more recent era.
# ---------------------------------------------------------------------------
REG_ERA_BREAKS = [
    (1976, 0, "pre_ground_effect"),     # ≤ 1976
    (1982, 1, "ground_effect_v1"),       # 1977–1982
    (1988, 2, "turbo_v1"),               # 1983–1988
    (2008, 3, "naturally_aspirated"),    # 1989–2008
    (2013, 4, "kers_drs"),               # 2009–2013
    (2021, 5, "turbo_hybrid"),           # 2014–2021
    (2025, 6, "ground_effect_2022"),     # 2022–2025 (Newey-era ground effect)
    (9999, 7, "new_pu_active_aero"),     # 2026+ (new PU architecture + active aero)
]

# Human-readable era names keyed by era id — useful for display/logging
REG_ERA_NAMES: dict[int, str] = {e: name for _, e, name in REG_ERA_BREAKS}

def year_to_reg_era(year: int) -> int:
    """Map a calendar year to an ordinal regulation-era integer."""
    for cutoff, era_id, _ in REG_ERA_BREAKS:
        if year <= cutoff:
            return era_id
    return 7  # fallback: latest era

# ---------------------------------------------------------------------------
# Kaggle/Ergast dataset coverage — update this when a new CSV export is used.
# build_features.py uses this to know when to look for live FastF1 supplements.
# ---------------------------------------------------------------------------
KAGGLE_DATA_MAX_YEAR = 2024

# Misc
RANDOM_SEED = 42
