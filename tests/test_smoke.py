from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_model_manifest_hashes() -> None:
    manifest = json.loads((ROOT / "models/MANIFEST.json").read_text(encoding="utf-8"))
    for filename, metadata in manifest["models"].items():
        path = ROOT / "models" / filename
        assert path.stat().st_size == metadata["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["sha256"]


def test_scene_config_has_one_selected_variant() -> None:
    config = json.loads((ROOT / "configs/scene_hybrid.json").read_text(encoding="utf-8"))
    assert len(config["variants"]) == 1
    assert "thresholds" in config
