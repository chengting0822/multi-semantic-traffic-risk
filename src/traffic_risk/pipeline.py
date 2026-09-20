"""Public orchestration API for the frozen single-vehicle and scene models."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .paths import CONFIG_DIR
from .scene.runner import run_scene_pipeline
from .single_vehicle.runner import run_single_vehicle_pipeline


def run_risk_pipeline(
    *,
    c4o_features: Path,
    trajectory_features: Path,
    timestamp_dir: Path,
    output_dir: Path,
    annotation_dir: Path,
    lane_map: Path = CONFIG_DIR / "lane_map.json",
    stop_lines: Path = CONFIG_DIR / "stop_lines.json",
    ipm: Path = CONFIG_DIR / "ipm.json",
    device: str = "auto",
    overwrite_cache: bool = False,
) -> Path:
    """Run semantic rebuilding, single-vehicle risk, and scene risk."""

    output_dir.mkdir(parents=True, exist_ok=True)
    single_dir = output_dir / "single_vehicle"
    scene_dir = output_dir / "scene"
    single = run_single_vehicle_pipeline(
        c4o_features=c4o_features,
        trajectory_features=trajectory_features,
        timestamp_dir=timestamp_dir,
        lane_map=lane_map,
        stop_lines=stop_lines,
        output_dir=single_dir,
        device=device,
        overwrite_cache=overwrite_cache,
    )
    trajectory = pd.read_csv(trajectory_features, low_memory=False)
    scene = run_scene_pipeline(
        single_vehicle=single.final_windows,
        trajectory_features=trajectory,
        timestamp_dir=timestamp_dir,
        annotation_dir=annotation_dir,
        output_dir=scene_dir,
        ipm=ipm,
        lane_map=lane_map,
        device=device,
    )
    final_path = output_dir / "traffic_risk.csv"
    scene.predictions.to_csv(final_path, index=False)
    manifest = {
        "schema_version": "traffic_risk.portfolio_pipeline/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "c4o_features": str(c4o_features),
            "trajectory_features": str(trajectory_features),
            "timestamp_dir": str(timestamp_dir),
            "annotation_dir": str(annotation_dir),
            "lane_map": str(lane_map),
            "stop_lines": str(stop_lines),
            "ipm": str(ipm),
        },
        "outputs": {
            "single_vehicle_risk": str(single_dir / "single_vehicle_risk.csv"),
            "scene_risk": str(scene_dir / "scene_risk.csv"),
            "traffic_risk": str(final_path),
        },
        "counts": {
            "single_vehicle_windows": int(len(single.final_windows)),
            "scene_windows": int(len(scene.predictions)),
        },
        "device": device,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return final_path


__all__ = ["run_risk_pipeline"]
