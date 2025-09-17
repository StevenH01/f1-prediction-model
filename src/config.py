from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data_raw"
DATA_INTERIM = PROJECT_ROOT / "data_interim"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# driver exlusion list
EXCLUDED_DRIVER_NAMES = {
  "Jack Doohan",
  "Kevin Magnussen",
  "Daniel Riccardo",
  "Sergio Pérez",
  "Valtteri Bottas",
  "Logan Sargeant",
  "Guanyu Zhou"
}

EXCLUDED_DRIVER_CODES = {
  "DOO",
  "MAG",
  "RIC",
  "PER",
  "BOT",
  "SAR",
  "ZHO"
}

# Misc
RANDOM_SEED = 42
