from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
MODEL_DIR = PROJECT_ROOT / "models"


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)
