from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    config["_config_path"] = str(config_path)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    required = [
        "run_id",
        "scene_root",
        "annotation_dir",
        "feature_table",
        "prediction_files",
        "outputs",
        "thresholds",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise KeyError(f"Missing config keys: {missing}")

    input_paths = [config["annotation_dir"], config["feature_table"]]
    input_paths.extend(config["prediction_files"].values())
    missing_paths = [str(p) for p in input_paths if not Path(p).exists()]
    if missing_paths:
        raise FileNotFoundError("Missing required input paths: " + ", ".join(missing_paths))

    for output_path in config["outputs"].values():
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)


def output_path(config: Mapping[str, Any], name: str) -> Path:
    try:
        return Path(config["outputs"][name])
    except KeyError as exc:
        raise KeyError(f"Unknown output path key: {name}") from exc


def path_stat(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    stat = p.stat()
    return {
        "path": str(p),
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
