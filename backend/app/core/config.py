from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"
ARTIFACTS = ROOT / "backend" / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
