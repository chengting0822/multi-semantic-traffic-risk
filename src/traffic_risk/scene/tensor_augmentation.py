#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


INTERACTION_GLOBAL_FEATURE_COLUMNS = [
    "interaction_risk_vehicle_count",
    "interaction_other_vehicle_count",
    "interaction_has_risk_and_other",
    "interaction_pair_observation_log1p",
    "interaction_common_frame_ratio",
    "interaction_close_frame_ratio",
    "interaction_overlap_frame_ratio",
    "interaction_norm_center_closeness",
    "interaction_iou_max",
    "interaction_risk2_near_other_flag",
    "interaction_world_has_pair",
    "interaction_world_close_frame_ratio",
    "interaction_world_quality_frame_ratio",
    "interaction_world_jitter_frame_ratio",
    "interaction_world_edge_low_quality_frame_ratio",
    "interaction_world_same_lane_ratio",
    "interaction_world_lateral_close_ratio",
    "interaction_world_closing_ratio",
    "interaction_world_cpa_frame_ratio",
    "interaction_world_distance_closeness",
    "interaction_world_median_distance_closeness",
    "interaction_world_lateral_closeness",
    "interaction_world_longitudinal_closeness",
    "interaction_world_closing_speed_norm",
    "interaction_world_ttc_urgency",
    "interaction_world_cpa_distance_closeness",
    "interaction_world_cpa_ttc_urgency",
    "interaction_world_cpa_relative_speed_norm",
    "interaction_world_cpa_closing_speed_norm",
    "interaction_world_cpa_bearing_closure_score",
    "interaction_world_cpa_angle_norm",
    "interaction_world_cpa_crossing_angle_flag",
    "interaction_world_cpa_collision_course_score",
    "interaction_world_collision_course_flag",
    "interaction_world_risk2_near_other_flag",
    "interaction_world_quality_safe_score",
    "interaction_world_cpa_quality_weighted_score",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Append interaction sidecar features to SceneTokenPool global tensor.")
    parser.add_argument("--base-npz", required=True, type=Path)
    parser.add_argument("--interaction-sidecar-csv", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--manifest-json", required=True, type=Path)
    parser.add_argument("--output-feature-csv", type=Path)
    args = parser.parse_args()

    payload = load_npz(args.base_npz)
    sidecar = pd.read_csv(args.interaction_sidecar_csv, low_memory=False)
    extra = build_interaction_global_features(sidecar, row_count=len(payload["global_features"]))

    old_global = payload["global_features"].astype(np.float32, copy=False)
    old_columns = [str(value) for value in payload["global_feature_columns"].tolist()]
    extra_values = extra[INTERACTION_GLOBAL_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    payload["global_features"] = np.concatenate([old_global, extra_values], axis=1).astype(np.float32)
    payload["global_feature_columns"] = np.asarray(old_columns + INTERACTION_GLOBAL_FEATURE_COLUMNS, dtype=object)

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **payload)

    if args.output_feature_csv:
        args.output_feature_csv.parent.mkdir(parents=True, exist_ok=True)
        extra.to_csv(args.output_feature_csv, index=False)

    manifest = {
        "schema_version": "scene_risk.interaction_tensor_augment/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_npz": str(args.base_npz),
        "interaction_sidecar_csv": str(args.interaction_sidecar_csv),
        "output_npz": str(args.output_npz),
        "output_feature_csv": str(args.output_feature_csv) if args.output_feature_csv else None,
        "base_global_dim": int(old_global.shape[1]),
        "added_global_dim": len(INTERACTION_GLOBAL_FEATURE_COLUMNS),
        "new_global_dim": int(payload["global_features"].shape[1]),
        "row_count": int(len(extra)),
        "feature_columns": INTERACTION_GLOBAL_FEATURE_COLUMNS,
        "feature_stats": numeric_stats(extra, INTERACTION_GLOBAL_FEATURE_COLUMNS),
        "notes": [
            "This augmentation does not use scene labels.",
            "Distance/TTC columns are converted to bounded closeness or urgency features, with explicit has-pair flags.",
            "Raw track ids and case keys are not included in learned input.",
        ],
    }
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"output={args.output_npz} rows={len(extra)} "
        f"global_dim={manifest['base_global_dim']}+{manifest['added_global_dim']}->{manifest['new_global_dim']}"
    )


def load_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def build_interaction_global_features(sidecar: pd.DataFrame, *, row_count: int) -> pd.DataFrame:
    required = ["scene_row_id"]
    missing = [col for col in required if col not in sidecar.columns]
    if missing:
        raise KeyError(f"sidecar missing columns: {missing}")

    frame = sidecar.copy()
    frame["scene_row_id"] = pd.to_numeric(frame["scene_row_id"], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["scene_row_id"]).copy()
    frame["scene_row_id"] = frame["scene_row_id"].astype(int)
    frame = frame.drop_duplicates("scene_row_id", keep="first").set_index("scene_row_id")
    frame = frame.reindex(range(row_count))

    def col(name: str) -> pd.Series:
        if name not in frame.columns:
            return pd.Series(np.zeros(row_count, dtype=np.float32), index=frame.index)
        return pd.to_numeric(frame[name], errors="coerce").fillna(0.0).astype(float)

    risk_count = col("interaction_risk_vehicle_count")
    other_count = col("interaction_other_vehicle_count")
    common = col("interaction_common_frame_count")
    pair_obs = col("interaction_pair_observation_count")
    close = col("interaction_close_frame_count")
    overlap = col("interaction_overlap_frame_count")
    min_norm_dist = col("interaction_min_norm_center_distance")

    world_common = col("interaction_world_common_frame_count")
    world_pair_obs = col("interaction_world_pair_observation_count")
    world_close = col("interaction_world_close_frame_count")
    world_lane_known = col("interaction_world_lane_known_frame_count")
    world_same_lane = col("interaction_world_same_lane_frame_count")
    world_lateral_close = col("interaction_world_lateral_close_frame_count")
    world_closing = col("interaction_world_closing_frame_count")
    world_quality = col("interaction_world_quality_frame_ratio").clip(0.0, 1.0)
    world_jitter = col("interaction_world_jitter_frame_ratio").clip(0.0, 1.0)
    world_edge = col("interaction_world_edge_low_quality_frame_ratio").clip(0.0, 1.0)
    cpa_score = col("interaction_world_cpa_max_collision_course_score").clip(0.0, 1.0)
    angle = col("interaction_world_cpa_best_crossing_angle_deg").clip(0.0, 180.0)
    collision_flag = (col("interaction_world_collision_course_flag") > 0).astype(float)

    out = pd.DataFrame(index=frame.index)
    out["scene_row_id"] = np.arange(row_count, dtype=np.int64)
    out["interaction_risk_vehicle_count"] = risk_count.clip(0.0, 8.0)
    out["interaction_other_vehicle_count"] = other_count.clip(0.0, 32.0)
    out["interaction_has_risk_and_other"] = ((risk_count > 0) & (other_count > 0)).astype(float)
    out["interaction_pair_observation_log1p"] = np.log1p(pair_obs.clip(lower=0.0))
    out["interaction_common_frame_ratio"] = safe_ratio(common, pair_obs)
    out["interaction_close_frame_ratio"] = safe_ratio(close, common)
    out["interaction_overlap_frame_ratio"] = safe_ratio(overlap, common)
    out["interaction_norm_center_closeness"] = closeness(min_norm_dist, common > 0)
    out["interaction_iou_max"] = col("interaction_max_iou").clip(0.0, 1.0)
    out["interaction_risk2_near_other_flag"] = (col("interaction_risk2_near_other_flag") > 0).astype(float)

    out["interaction_world_has_pair"] = (world_pair_obs > 0).astype(float)
    out["interaction_world_close_frame_ratio"] = col("interaction_world_close_frame_ratio").clip(0.0, 1.0)
    out["interaction_world_quality_frame_ratio"] = world_quality
    out["interaction_world_jitter_frame_ratio"] = world_jitter
    out["interaction_world_edge_low_quality_frame_ratio"] = world_edge
    out["interaction_world_same_lane_ratio"] = safe_ratio(world_same_lane, world_lane_known)
    out["interaction_world_lateral_close_ratio"] = safe_ratio(world_lateral_close, world_common)
    out["interaction_world_closing_ratio"] = safe_ratio(world_closing, world_common)
    out["interaction_world_cpa_frame_ratio"] = col("interaction_world_cpa_frame_ratio").clip(0.0, 1.0)
    out["interaction_world_distance_closeness"] = closeness(col("interaction_world_min_distance_m"), world_pair_obs > 0)
    out["interaction_world_median_distance_closeness"] = closeness(col("interaction_world_median_frame_min_distance_m"), world_pair_obs > 0)
    out["interaction_world_lateral_closeness"] = closeness(col("interaction_world_min_abs_lateral_m"), world_pair_obs > 0)
    out["interaction_world_longitudinal_closeness"] = closeness(col("interaction_world_min_abs_longitudinal_m"), world_pair_obs > 0)
    out["interaction_world_closing_speed_norm"] = speed_norm(col("interaction_world_max_closing_speed_mps"))
    out["interaction_world_ttc_urgency"] = urgency(col("interaction_world_min_ttc_sec"), world_pair_obs > 0)
    out["interaction_world_cpa_distance_closeness"] = closeness(col("interaction_world_cpa_min_distance_m"), world_pair_obs > 0)
    out["interaction_world_cpa_ttc_urgency"] = urgency(col("interaction_world_cpa_min_collision_ttc_sec"), world_pair_obs > 0)
    out["interaction_world_cpa_relative_speed_norm"] = speed_norm(col("interaction_world_cpa_max_relative_speed_mps"))
    out["interaction_world_cpa_closing_speed_norm"] = speed_norm(col("interaction_world_cpa_max_closing_speed_mps"))
    out["interaction_world_cpa_bearing_closure_score"] = col("interaction_world_cpa_best_bearing_closure_score").clip(0.0, 1.0)
    out["interaction_world_cpa_angle_norm"] = angle / 180.0
    out["interaction_world_cpa_crossing_angle_flag"] = (angle >= 15.0).astype(float)
    out["interaction_world_cpa_collision_course_score"] = cpa_score
    out["interaction_world_collision_course_flag"] = collision_flag
    out["interaction_world_risk2_near_other_flag"] = (col("interaction_world_risk2_near_other_flag") > 0).astype(float)
    out["interaction_world_quality_safe_score"] = (world_quality * (1.0 - world_jitter) * (1.0 - world_edge)).clip(0.0, 1.0)
    out["interaction_world_cpa_quality_weighted_score"] = (
        cpa_score * out["interaction_world_quality_safe_score"] * out["interaction_world_cpa_crossing_angle_flag"]
    ).clip(0.0, 1.0)

    for name in INTERACTION_GLOBAL_FEATURE_COLUMNS:
        out[name] = pd.to_numeric(out[name], errors="coerce").fillna(0.0).astype(np.float32)
    return out[["scene_row_id"] + INTERACTION_GLOBAL_FEATURE_COLUMNS].reset_index(drop=True)


def safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    num_arr = num.to_numpy(dtype=float)
    den_arr = den.to_numpy(dtype=float)
    out = np.zeros(len(num_arr), dtype=np.float32)
    mask = den_arr > 0
    out[mask] = np.clip(num_arr[mask] / den_arr[mask], 0.0, 1.0)
    return pd.Series(out, index=num.index)


def closeness(distance_m: pd.Series, valid: pd.Series) -> pd.Series:
    values = distance_m.to_numpy(dtype=float)
    out = np.zeros(len(values), dtype=np.float32)
    mask = valid.to_numpy(dtype=bool) & np.isfinite(values) & (values >= 0)
    out[mask] = 1.0 / (1.0 + values[mask])
    return pd.Series(out, index=distance_m.index)


def urgency(ttc_s: pd.Series, valid: pd.Series) -> pd.Series:
    values = ttc_s.to_numpy(dtype=float)
    out = np.zeros(len(values), dtype=np.float32)
    mask = valid.to_numpy(dtype=bool) & np.isfinite(values) & (values > 0)
    out[mask] = 1.0 / (1.0 + values[mask])
    return pd.Series(out, index=ttc_s.index)


def speed_norm(speed_mps: pd.Series) -> pd.Series:
    return (speed_mps.clip(lower=0.0, upper=30.0) / 30.0).astype(np.float32)


def numeric_stats(frame: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for col in columns:
        values = pd.to_numeric(frame[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        stats[col] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "nonzero_rate": float((values != 0).mean()),
        }
    return stats


if __name__ == "__main__":
    main()
