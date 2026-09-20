"""Build and run the frozen scene-risk pipeline from single-vehicle output."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from traffic_risk.paths import CONFIG_DIR

from .dataset import build_scene_dataset_v3
from .interaction import build_interaction_sidecar_from_frames
from .pipeline import SceneRiskPipeline
from .tensor_augmentation import INTERACTION_GLOBAL_FEATURE_COLUMNS, build_interaction_global_features


@dataclass(frozen=True)
class SceneRunResult:
    scene_windows: pd.DataFrame
    scene_tokens: pd.DataFrame
    interaction_features: pd.DataFrame
    predictions: pd.DataFrame


def _prediction_frame(single_vehicle: pd.DataFrame) -> pd.DataFrame:
    required = ["case_key", "video_id", "track_id", "ts_window_idx", "risk_level"]
    missing = [column for column in required if column not in single_vehicle.columns]
    if missing:
        raise ValueError(f"single-vehicle output is missing columns: {missing}")
    out = single_vehicle.copy()
    out["y_pred_argmax"] = pd.to_numeric(out["risk_level"], errors="coerce").fillna(0).astype(int)
    if "pred_gru_raw" in out.columns:
        out["y_pred_raw_argmax"] = pd.to_numeric(out["pred_gru_raw"], errors="coerce").fillna(0).astype(int)
    else:
        out["y_pred_raw_argmax"] = out["y_pred_argmax"]
    if "pred_v37e" in out.columns:
        out["pred_v22_candidate"] = pd.to_numeric(out["pred_v37e"], errors="coerce").fillna(0).astype(int)
    else:
        out["pred_v22_candidate"] = out["y_pred_argmax"]
    return out


def _fps_table(timestamp_dir: Path, video_ids: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for video_id in video_ids:
        path = timestamp_dir / f"{video_id}.csv"
        if not path.is_file():
            rows.append({"video_id": video_id, "fps": 0.0, "status": "missing"})
            continue
        sample = pd.read_csv(path, usecols=lambda column: column in {"frame", "timestamp_sec", "fps"}, low_memory=False)
        fps = 0.0
        if "fps" in sample.columns:
            values = pd.to_numeric(sample["fps"], errors="coerce").dropna()
            if len(values):
                fps = float(values.median())
        if fps <= 0 and {"frame", "timestamp_sec"}.issubset(sample.columns):
            frame = pd.to_numeric(sample["frame"], errors="coerce")
            time = pd.to_numeric(sample["timestamp_sec"], errors="coerce")
            valid = frame.notna() & time.notna()
            if valid.sum() >= 2:
                frame_span = float(frame[valid].max() - frame[valid].min())
                time_span = float(time[valid].max() - time[valid].min())
                fps = frame_span / time_span if time_span > 0 else 0.0
        rows.append({"video_id": video_id, "fps": fps, "status": "ok" if fps > 0 else "invalid"})
    return pd.DataFrame(rows)


def _dataset_config(
    *,
    feature_table: Path,
    predictions: Path,
    annotation_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "run_id": "portfolio_current_run",
        "feature_table": str(feature_table),
        "prediction_files": {"inference": str(predictions)},
        "annotation_dir": str(annotation_dir),
        "outputs": {},
        "scene": {
            "top_k": 16,
            "scene_stride_sec": 1.0 / 3.0,
            "history_decay": 0.85,
            "history_window_count": 8,
            "label_strategy": "annotation_first_last_max_risk_boundary_clean",
            "risk1_positive": True,
            "train_predictions_kind": "current_run",
            "feature_schema": "scene_feature_schema_v3",
            "boundary_label_policy": {
                "enabled": True,
                "name": "exclude_low_overlap_boundary_windows_v1",
                "hard_min_overlap_ratio": 0.33,
                "dominant_guard_max_overlap_ratio": 0.33,
                "require_non_primary_window": True,
                "affected_risks": [1, 2, 3],
                "exclude_from_training_and_primary_eval": True,
            },
            "single_vehicle_floor_source": {"mode": "window_pred_schemaC"},
            "single_vehicle_temporal_floor_overlap": {
                "enabled": True,
                "name": "single_final_risk_time_overlap_floor_v1",
                "min_overlap_sec": 0.20,
                "min_scene_overlap_ratio": 0.55,
                "min_token_overlap_ratio": 0.55,
                "require_track_first_window": True,
                "max_track_window_idx": 0,
                "only_prior_scene_windows": True,
                "max_scene_window_idx_gap": 1,
            },
        },
        "thresholds": {},
        "output_dir": str(output_dir),
    }


def run_scene_pipeline(
    *,
    single_vehicle: pd.DataFrame,
    trajectory_features: pd.DataFrame,
    timestamp_dir: Path,
    annotation_dir: Path,
    output_dir: Path,
    ipm: Path = CONFIG_DIR / "ipm.json",
    lane_map: Path = CONFIG_DIR / "lane_map.json",
    device: str = "auto",
) -> SceneRunResult:
    """Aggregate up to 16 vehicles, compute interactions, and infer scene risk."""

    output_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    features_path = output_dir / "scene_source_features.csv"
    predictions_path = output_dir / "scene_single_vehicle_predictions.csv"
    single_vehicle.to_csv(features_path, index=False)
    _prediction_frame(single_vehicle).to_csv(predictions_path, index=False)

    dataset = build_scene_dataset_v3(
        _dataset_config(
            feature_table=features_path,
            predictions=predictions_path,
            annotation_dir=annotation_dir,
            output_dir=output_dir,
        )
    )
    scene_windows = dataset["scene_windows"]
    scene_tokens = dataset["scene_tokens"]
    payload = dataset["tensor_payload"]

    video_ids = sorted(scene_windows["video_id"].astype(str).unique().tolist())
    fps = _fps_table(timestamp_dir, video_ids)
    interactions, interaction_manifest = build_interaction_sidecar_from_frames(
        scene_windows=scene_windows,
        scene_tokens=scene_tokens,
        fps_table=fps,
        bbox_dir=timestamp_dir,
        ipm_json=ipm,
        lane_map_json=lane_map,
        trajectory_sidecar=trajectory_features,
    )
    extra = build_interaction_global_features(interactions, row_count=len(scene_windows))
    payload["global_features"] = np.concatenate(
        [
            payload["global_features"].astype(np.float32, copy=False),
            extra[INTERACTION_GLOBAL_FEATURE_COLUMNS].to_numpy(dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    base_columns = [str(value) for value in payload["global_feature_columns"].tolist()]
    payload["global_feature_columns"] = np.asarray(base_columns + INTERACTION_GLOBAL_FEATURE_COLUMNS, dtype=object)

    tensor_path = output_dir / "scene_features.npz"
    np.savez_compressed(tensor_path, **payload)
    scene_windows.to_csv(output_dir / "scene_windows.csv", index=False)
    scene_tokens.to_csv(output_dir / "scene_tokens.csv", index=False)
    interactions.to_csv(output_dir / "interaction_features.csv", index=False)
    fps.to_csv(output_dir / "video_fps.csv", index=False)
    (output_dir / "interaction_manifest.json").write_text(
        json.dumps(interaction_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    predictions = SceneRiskPipeline(device=device).predict(
        dataset=tensor_path,
        scene_windows=scene_windows,
        scene_tokens=scene_tokens,
        interaction_features=interactions,
        labeled_only=False,
    )
    predictions.to_csv(output_dir / "scene_risk.csv", index=False)
    return SceneRunResult(scene_windows, scene_tokens, interactions, predictions)


__all__ = ["SceneRunResult", "run_scene_pipeline"]
