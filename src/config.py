from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data_raw"
DATA_INTERIM = PROJECT_ROOT / "data_interim"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Misc
RANDOM_SEED = 42
