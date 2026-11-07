import os
from pathlib import Path

# BACKEND_DIR is the directory that contains both `app/` and `artifacts/`.
# Locally that is <repo>/backend; on Vercel (service root = backend/) it is the
# function bundle root. Resolving from here keeps paths valid in both layouts.
BACKEND_DIR = Path(__file__).resolve().parents[2]
ROOT = BACKEND_DIR.parent  # repository root locally (training/explore scripts only)
RAW = ROOT / "data" / "raw"  # raw datasets: local training only, never needed for inference
ARTIFACTS = BACKEND_DIR / "artifacts"  # committed, read-only at inference time
RANDOM_STATE = 42

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]

