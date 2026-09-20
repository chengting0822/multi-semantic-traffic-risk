#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


LABELS_4CLS = [0, 1, 2, 3]
LABELS_SEVERITY = [0, 1, 2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SceneHybridV1 variants on learned scene predictions.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "scene_hybrid_v1.json"))
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = _read_json(config_path)
    learned = pd.read_csv(config["inputs"]["learned_predictions_csv"], low_memory=False)
    scene_windows = pd.read_csv(config["inputs"]["scene_windows_csv"], low_memory=False)
    scene_tokens = None
    if config["inputs"].get("scene_tokens_csv"):
        scene_tokens = pd.read_csv(config["inputs"]["scene_tokens_csv"], low_memory=False)
    interaction_sidecar = None
    if config["inputs"].get("scene_interaction_sidecar_csv"):
        interaction_sidecar = pd.read_csv(config["inputs"]["scene_interaction_sidecar_csv"], low_memory=False)
    temporal_combo_sidecar = None
    if config["inputs"].get("scene_temporal_combo_sidecar_csv"):
        temporal_combo_sidecar = pd.read_csv(config["inputs"]["scene_temporal_combo_sidecar_csv"], low_memory=False)
    single_vehicle_causal_floor = None
    if config["inputs"].get("single_vehicle_causal_floor_csv"):
        single_vehicle_causal_floor = pd.read_csv(config["inputs"]["single_vehicle_causal_floor_csv"], low_memory=False)
    predictions = build_hybrid_predictions(
        learned,
        scene_windows,
        scene_tokens,
        interaction_sidecar,
        temporal_combo_sidecar,
        single_vehicle_causal_floor,
        config,
    )
    metrics = evaluate_variants(predictions, config["variants"])

    outputs = config["outputs"]
    prediction_path = Path(outputs["prediction_csv"])
    metrics_json_path = Path(outputs["metrics_json"])
    metrics_md_path = Path(outputs["metrics_md"])
    manifest_path = Path(outputs["manifest_json"])
    for path in [prediction_path, metrics_json_path, metrics_md_path, manifest_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(prediction_path, index=False)
    metrics_json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics_md_path.write_text(render_metrics_md(metrics), encoding="utf-8")
    manifest = {
        "schema_version": "scene_risk.scene_hybrid_v1.manifest/v1",
        "run_id": config["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "inputs": config["inputs"],
        "outputs": outputs,
        "variants": config["variants"],
        "thresholds": config.get("thresholds", {}),
        "row_count": int(len(predictions)),
        "metrics_summary": {
            variant: {
                split: {
                    "binary_f1": item["binary"]["f1"],
                    "four_class_macro_f1": item["four_class"]["macro_f1"],
                    "severity_macro_f1": item["severity"]["macro_f1"],
                    "error_counts": item["error_counts"],
                }
                for split, item in variant_metrics["splits"].items()
            }
            for variant, variant_metrics in metrics["variants"].items()
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"predictions={prediction_path} rows={len(predictions)}")
    for variant in config["variants"]:
        val = metrics["variants"][variant]["splits"]["val"]
        test = metrics["variants"][variant]["splits"]["test"]
        print(
            f"{variant}: val bin={val['binary']['f1']:.4f} 4cls={val['four_class']['macro_f1']:.4f} sev={val['severity']['macro_f1']:.4f}; "
            f"test bin={test['binary']['f1']:.4f} 4cls={test['four_class']['macro_f1']:.4f} sev={test['severity']['macro_f1']:.4f}"
        )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_hybrid_predictions(
    learned: pd.DataFrame,
    scene_windows: pd.DataFrame,
    scene_tokens: pd.DataFrame | None,
    interaction_sidecar: pd.DataFrame | None,
    temporal_combo_sidecar: pd.DataFrame | None,
    single_vehicle_causal_floor: pd.DataFrame | None,
    config: dict[str, Any],
) -> pd.DataFrame:
    required_prediction_cols = [
        "row_idx",
        "split",
        "video_id",
        "scene_window_idx",
        "y_4cls",
        "pred_4cls",
        "prob_binary_positive",
        "prob4_2",
        "prob4_3",
        "prob_severity_2",
    ]
    required_scene_cols = [
        "scene_row_id",
        "scene_y_original",
        "scene_y_binary",
        "scene_y_schemaC",
        "single_vehicle_floor_risk",
        "strict_policy_floor_risk",
        "support_floor_risk",
        "default_floor_risk",
        "count_single_vehicle_risk2",
        "count_single_vehicle_risk3",
        "count_strict_policy_risk2",
        "count_strict_policy_risk3",
        "count_medium_risk2_support",
        "count_strong_risk3_support",
        "max_overspeed_event_score",
        "max_redlight_event_score",
        "max_active_family_count",
        "max_cooccurrence_strength",
        "current_event_score",
        "active_track_count",
        "case_keys",
        "track_ids",
    ]
    _require_columns(learned, required_prediction_cols, "learned_predictions")
    _require_columns(scene_windows, required_scene_cols, "scene_windows")

    optional_scene_cols = [
        "single_vehicle_temporal_overlap_floor_risk",
        "single_vehicle_temporal_overlap_risk2_track_ids",
        "single_vehicle_temporal_overlap_risk3_track_ids",
        "single_vehicle_temporal_overlap_applied",
    ]
    scene_keep_cols = required_scene_cols + [col for col in optional_scene_cols if col in scene_windows.columns]
    scene_keep = scene_windows[scene_keep_cols].copy()
    merged = learned.merge(
        scene_keep,
        left_on="row_idx",
        right_on="scene_row_id",
        how="left",
        validate="one_to_one",
    )
    if merged["scene_row_id"].isna().any():
        missing = int(merged["scene_row_id"].isna().sum())
        raise RuntimeError(f"learned predictions could not be matched to scene rows: {missing}")
    merged = refresh_labels_from_scene_windows(merged)

    for col in [
        "single_vehicle_floor_risk",
        "strict_policy_floor_risk",
        "support_floor_risk",
        "default_floor_risk",
        "count_single_vehicle_risk2",
        "count_single_vehicle_risk3",
        "count_strict_policy_risk2",
        "count_strict_policy_risk3",
        "count_medium_risk2_support",
        "count_strong_risk3_support",
    ]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0).astype(int)
    for col in [
        "max_overspeed_event_score",
        "max_redlight_event_score",
        "max_active_family_count",
        "max_cooccurrence_strength",
        "current_event_score",
        "active_track_count",
    ]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0).astype(float)
    if scene_tokens is not None:
        merged = merged.merge(summarize_pure_overspeed_tokens(scene_tokens, config), on="scene_row_id", how="left")
        merged = merged.merge(summarize_risk_source_tokens(scene_tokens, config), on="scene_row_id", how="left")
        merged = merged.merge(
            summarize_temporal_overspeed_trajectory_tokens(scene_tokens, config),
            on="scene_row_id",
            how="left",
        )
    else:
        merged["pure_overspeed_rescue_token_count"] = 0
        merged["pure_overspeed_rescue_max_score"] = 0.0
        merged["pure_overspeed_rescue_case_keys"] = ""
        merged["pure_overspeed_rescue_track_ids"] = ""
        merged = add_empty_risk_source_columns(merged)
        merged = add_empty_temporal_overspeed_trajectory_columns(merged)
    for col in ["pure_overspeed_rescue_token_count", "pure_overspeed_rescue_max_score"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
    for col in ["pure_overspeed_rescue_case_keys", "pure_overspeed_rescue_track_ids"]:
        merged[col] = merged[col].fillna("")
    merged = fill_risk_source_columns(merged)
    merged = merge_temporal_overlap_risk_sources(merged)
    if single_vehicle_causal_floor is not None:
        merged = merge_single_vehicle_causal_floor(merged, single_vehicle_causal_floor, config)
    merged = fill_temporal_overspeed_trajectory_columns(merged)
    if interaction_sidecar is not None:
        merged = merged.merge(summarize_interaction_sidecar(interaction_sidecar), on="scene_row_id", how="left")
    else:
        merged = add_empty_interaction_columns(merged)
    merged = fill_interaction_columns(merged)
    if temporal_combo_sidecar is not None:
        merged = merged.merge(summarize_temporal_combo_sidecar(temporal_combo_sidecar), on="scene_row_id", how="left")
    else:
        merged = add_empty_temporal_combo_columns(merged)
    merged = fill_temporal_combo_columns(merged)

    learned_pred = merged["pred_4cls"].astype(int).clip(0, 3)
    single_floor = merged["single_vehicle_floor_risk"].astype(int).clip(0, 3)
    default_floor = merged["default_floor_risk"].astype(int).clip(0, 3)

    merged["scene_pred_learned_only"] = learned_pred
    merged["scene_pred_single_floor"] = np.maximum(learned_pred, single_floor).astype(int)
    merged["scene_pred_default_floor"] = np.maximum(learned_pred, default_floor).astype(int)

    single_strict_pred, single_strict_reason = apply_severity_gate(
        merged,
        learned_pred,
        single_floor,
        config,
        soft=False,
        promote_from_policy_floor=False,
    )
    single_soft_pred, single_soft_reason = apply_severity_gate(
        merged,
        learned_pred,
        single_floor,
        config,
        soft=True,
        promote_from_policy_floor=False,
    )
    strict_pred, strict_reason = apply_severity_gate(
        merged,
        learned_pred,
        default_floor,
        config,
        soft=False,
        promote_from_policy_floor=True,
    )
    soft_pred, soft_reason = apply_severity_gate(
        merged,
        learned_pred,
        default_floor,
        config,
        soft=True,
        promote_from_policy_floor=True,
    )
    merged["scene_pred_single_floor_severity_gate_strict"] = single_strict_pred
    pure_overspeed_pred, pure_overspeed_reason = apply_pure_overspeed_rescue_gate(
        merged,
        single_strict_pred,
        config,
    )
    merged["scene_pred_single_floor_severity_gate_strict_pure_overspeed_rescue"] = pure_overspeed_pred
    trajectory_moving_interaction_pred, trajectory_moving_interaction_reason = apply_trajectory_moving_interaction_gate(
        merged,
        pure_overspeed_pred,
        config,
    )
    contextual_base_pred = (
        trajectory_moving_interaction_pred
        if bool(config.get("thresholds", {}).get("trajectory_moving_interaction_feed_final_variants", False))
        else pure_overspeed_pred
    )
    temporal_combo_pred, temporal_combo_reason = apply_temporal_combo_track_aware_gate(
        merged,
        contextual_base_pred,
        config,
    )
    cpa_onset_pred, cpa_onset_reason = apply_cpa_onset_track_aware_gate(
        merged,
        temporal_combo_pred,
        config,
    )
    temporal_cpa_learned_interaction_guarded_pred, temporal_cpa_learned_interaction_guarded_reason = (
        apply_learned_interaction_guarded_gate(
            merged,
            cpa_onset_pred,
            config,
        )
    )
    temporal_cpa_learned_interaction_track_sticky_pred, temporal_cpa_learned_interaction_track_sticky_reason = (
        apply_learned_interaction_track_aware_sticky_gate(
            merged,
            cpa_onset_pred,
            config,
        )
    )
    interaction_guarded_pred, interaction_guarded_reason = apply_interaction_guarded_gate(
        merged,
        contextual_base_pred,
        config,
    )
    interaction_track_sticky_pred, interaction_track_sticky_reason = apply_interaction_track_aware_sticky_gate(
        merged,
        contextual_base_pred,
        config,
    )
    interaction_world_guarded_pred, interaction_world_guarded_reason = apply_interaction_world_guarded_gate(
        merged,
        contextual_base_pred,
        config,
    )
    interaction_world_track_sticky_pred, interaction_world_track_sticky_reason = apply_interaction_world_track_aware_sticky_gate(
        merged,
        contextual_base_pred,
        config,
    )
    (
        interaction_world_track_sticky_cpa_collision_pred,
        interaction_world_track_sticky_cpa_collision_reason,
    ) = apply_cpa_collision_course_rescue_gate(
        merged,
        interaction_world_track_sticky_pred,
        config,
    )
    learned_interaction_guarded_pred, learned_interaction_guarded_reason = apply_learned_interaction_guarded_gate(
        merged,
        contextual_base_pred,
        config,
    )
    learned_interaction_track_sticky_pred, learned_interaction_track_sticky_reason = apply_learned_interaction_track_aware_sticky_gate(
        merged,
        contextual_base_pred,
        config,
    )
    (
        learned_interaction_track_sticky_cpa_onset_calibrated_pred,
        learned_interaction_track_sticky_cpa_onset_calibrated_reason,
    ) = apply_cpa_onset_track_aware_gate(
        merged,
        learned_interaction_track_sticky_pred,
        config,
    )
    (
        learned_interaction_track_sticky_cpa_onset_high_confidence_pred,
        learned_interaction_track_sticky_cpa_onset_high_confidence_reason,
    ) = apply_high_confidence_cpa_onset_gate(
        merged,
        learned_interaction_track_sticky_pred,
        config,
    )
    (
        learned_interaction_track_sticky_cpa_onset_high_confidence_temporal_overspeed_trajectory_pred,
        learned_interaction_track_sticky_cpa_onset_high_confidence_temporal_overspeed_trajectory_reason,
    ) = apply_temporal_overspeed_trajectory_combo_gate(
        merged,
        learned_interaction_track_sticky_cpa_onset_high_confidence_pred,
        config,
    )
    merged["scene_pred_single_floor_severity_gate_strict_pure_overspeed_rescue_temporal_combo"] = temporal_combo_pred
    merged[
        "scene_pred_single_floor_severity_gate_strict_pure_overspeed_rescue_trajectory_moving_interaction"
    ] = trajectory_moving_interaction_pred
    merged["scene_pred_single_floor_severity_gate_strict_pure_overspeed_rescue_temporal_combo_cpa_onset"] = cpa_onset_pred
    merged[
        "scene_pred_single_floor_severity_gate_strict_pure_overspeed_rescue_temporal_combo_cpa_onset_learned_interaction_guarded"
    ] = temporal_cpa_learned_interaction_guarded_pred
    merged[
        "scene_pred_single_floor_severity_gate_strict_pure_overspeed_rescue_temporal_combo_cpa_onset_learned_interaction_track_sticky"
    ] = temporal_cpa_learned_interaction_track_sticky_pred
    merged["scene_pred_single_floor_severity_gate_strict_pure_overspeed_rescue_interaction_guarded"] = interaction_guarded_pred
    merged["scene_pred_single_floor_severity_gate_strict_pure_overspeed_rescue_interaction_track_sticky"] = interaction_track_sticky_pred
    merged["scene_pred_single_floor_severity_gate_strict_pure_overspeed_rescue_interaction_world_guarded"] = interaction_world_guarded_pred
    merged["scene_pred_single_floor_severity_gate_strict_pure_overspeed_rescue_interaction_world_track_sticky"] = interaction_world_track_sticky_pred
    merged[
        "scene_pred_single_floor_severity_gate_strict_pure_overspeed_rescue_interaction_world_track_sticky_cpa_collision"
    ] = interaction_world_track_sticky_cpa_collision_pred
    merged["scene_pred_single_floor_severity_gate_strict_pure_overspeed_rescue_learned_interaction_guarded"] = learned_interaction_guarded_pred
    merged["scene_pred_single_floor_severity_gate_strict_pure_overspeed_rescue_learned_interaction_track_sticky"] = learned_interaction_track_sticky_pred
    merged[
        "scene_pred_single_floor_severity_gate_strict_pure_overspeed_rescue_learned_interaction_track_sticky_cpa_onset_calibrated"
    ] = learned_interaction_track_sticky_cpa_onset_calibrated_pred
    merged[
        "scene_pred_single_floor_severity_gate_strict_pure_overspeed_rescue_learned_interaction_track_sticky_cpa_onset_high_confidence"
    ] = learned_interaction_track_sticky_cpa_onset_high_confidence_pred
    merged[
        "scene_pred_single_floor_severity_gate_strict_pure_overspeed_rescue_learned_interaction_track_sticky_cpa_onset_high_confidence_temporal_overspeed_trajectory_combo"
    ] = learned_interaction_track_sticky_cpa_onset_high_confidence_temporal_overspeed_trajectory_pred
    merged["scene_pred_single_floor_severity_gate_strict_pure_overspeed_rescue_interaction_sticky"] = interaction_track_sticky_pred
    merged["scene_pred_single_floor_severity_gate_soft"] = single_soft_pred
    merged["scene_pred_severity_gate_strict"] = strict_pred
    merged["scene_pred_severity_gate_soft"] = soft_pred
    merged["single_floor_severity_gate_strict_reason"] = single_strict_reason
    merged["single_floor_severity_gate_strict_pure_overspeed_rescue_reason"] = pure_overspeed_reason
    merged[
        "single_floor_severity_gate_strict_pure_overspeed_rescue_trajectory_moving_interaction_reason"
    ] = trajectory_moving_interaction_reason
    merged["single_floor_severity_gate_strict_pure_overspeed_rescue_temporal_combo_reason"] = temporal_combo_reason
    merged["single_floor_severity_gate_strict_pure_overspeed_rescue_temporal_combo_cpa_onset_reason"] = cpa_onset_reason
    merged[
        "single_floor_severity_gate_strict_pure_overspeed_rescue_temporal_combo_cpa_onset_learned_interaction_guarded_reason"
    ] = temporal_cpa_learned_interaction_guarded_reason
    merged[
        "single_floor_severity_gate_strict_pure_overspeed_rescue_temporal_combo_cpa_onset_learned_interaction_track_sticky_reason"
    ] = temporal_cpa_learned_interaction_track_sticky_reason
    merged["single_floor_severity_gate_strict_pure_overspeed_rescue_interaction_guarded_reason"] = interaction_guarded_reason
    merged["single_floor_severity_gate_strict_pure_overspeed_rescue_interaction_track_sticky_reason"] = interaction_track_sticky_reason
    merged["single_floor_severity_gate_strict_pure_overspeed_rescue_interaction_world_guarded_reason"] = interaction_world_guarded_reason
    merged["single_floor_severity_gate_strict_pure_overspeed_rescue_interaction_world_track_sticky_reason"] = interaction_world_track_sticky_reason
    merged[
        "single_floor_severity_gate_strict_pure_overspeed_rescue_interaction_world_track_sticky_cpa_collision_reason"
    ] = interaction_world_track_sticky_cpa_collision_reason
    merged["single_floor_severity_gate_strict_pure_overspeed_rescue_learned_interaction_guarded_reason"] = learned_interaction_guarded_reason
    merged["single_floor_severity_gate_strict_pure_overspeed_rescue_learned_interaction_track_sticky_reason"] = learned_interaction_track_sticky_reason
    merged[
        "single_floor_severity_gate_strict_pure_overspeed_rescue_learned_interaction_track_sticky_cpa_onset_calibrated_reason"
    ] = learned_interaction_track_sticky_cpa_onset_calibrated_reason
    merged[
        "single_floor_severity_gate_strict_pure_overspeed_rescue_learned_interaction_track_sticky_cpa_onset_high_confidence_reason"
    ] = learned_interaction_track_sticky_cpa_onset_high_confidence_reason
    merged[
        "single_floor_severity_gate_strict_pure_overspeed_rescue_learned_interaction_track_sticky_cpa_onset_high_confidence_temporal_overspeed_trajectory_combo_reason"
    ] = learned_interaction_track_sticky_cpa_onset_high_confidence_temporal_overspeed_trajectory_reason
    merged["single_floor_severity_gate_strict_pure_overspeed_rescue_interaction_sticky_reason"] = interaction_track_sticky_reason
    merged["single_floor_severity_gate_soft_reason"] = single_soft_reason
    merged["severity_gate_strict_reason"] = strict_reason
    merged["severity_gate_soft_reason"] = soft_reason
    return merged


def refresh_labels_from_scene_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Use current labels when present and keep ``-1`` for live inference.

    Labels are needed for evaluation, not for the hybrid decision itself.  The
    historical evaluator rejected unlabeled rows here, which prevented the
    same frozen rules from being used on a new video.
    """
    out = df.copy()
    label_sources = {
        "y_4cls": "scene_y_original",
        "y_binary": "scene_y_binary",
        "y_severity": "scene_y_schemaC",
    }
    for target, source in label_sources.items():
        out[target] = pd.to_numeric(out[source], errors="coerce").fillna(-1).astype(int)
    return out


def summarize_interaction_sidecar(sidecar: pd.DataFrame) -> pd.DataFrame:
    required = [
        "scene_row_id",
        "interaction_status",
        "interaction_risk_track_ids",
        "interaction_risk_case_keys",
        "interaction_risk_vehicle_count",
        "interaction_other_vehicle_count",
        "interaction_common_frame_count",
        "interaction_close_frame_count",
        "interaction_overlap_frame_count",
        "interaction_pair_observation_count",
        "interaction_min_norm_center_distance",
        "interaction_median_frame_min_norm_center_distance",
        "interaction_min_center_distance_px",
        "interaction_max_iou",
        "interaction_nearest_risk_track_id",
        "interaction_nearest_other_track_id",
        "interaction_risk2_near_other_flag",
    ]
    _require_columns(sidecar, required, "scene_interaction_sidecar")
    optional = [
        "interaction_world_common_frame_count",
        "interaction_world_close_frame_count",
        "interaction_world_close_frame_ratio",
        "interaction_world_quality_frame_ratio",
        "interaction_world_jitter_frame_ratio",
        "interaction_world_edge_low_quality_frame_ratio",
        "interaction_world_other_moving_frame_ratio",
        "interaction_world_other_stationary_frame_ratio",
        "interaction_world_same_lane_frame_count",
        "interaction_world_lateral_close_frame_count",
        "interaction_world_lane_known_frame_count",
        "interaction_world_closing_frame_count",
        "interaction_world_cpa_frame_count",
        "interaction_world_cpa_frame_ratio",
        "interaction_world_pair_observation_count",
        "interaction_world_min_distance_m",
        "interaction_world_median_frame_min_distance_m",
        "interaction_world_min_abs_lateral_m",
        "interaction_world_min_abs_longitudinal_m",
        "interaction_world_nearest_risk_speed_kmh",
        "interaction_world_nearest_other_speed_kmh",
        "interaction_world_max_closing_speed_mps",
        "interaction_world_min_ttc_sec",
        "interaction_world_cpa_min_distance_m",
        "interaction_world_cpa_best_time_sec",
        "interaction_world_cpa_min_time_sec",
        "interaction_world_cpa_min_collision_ttc_sec",
        "interaction_world_cpa_best_collision_radius_m",
        "interaction_world_cpa_max_relative_speed_mps",
        "interaction_world_cpa_max_closing_speed_mps",
        "interaction_world_cpa_best_bearing_closure_score",
        "interaction_world_cpa_best_crossing_angle_deg",
        "interaction_world_cpa_max_collision_course_score",
        "interaction_world_cpa_nearest_risk_speed_kmh",
        "interaction_world_cpa_nearest_other_speed_kmh",
        "interaction_world_nearest_risk_track_id",
        "interaction_world_nearest_other_track_id",
        "interaction_world_cpa_nearest_risk_track_id",
        "interaction_world_cpa_nearest_other_track_id",
        "interaction_world_collision_course_flag",
        "interaction_world_risk2_near_other_flag",
    ]
    keep = required + [col for col in optional if col in sidecar.columns]
    frame = sidecar[keep].copy()
    frame["scene_row_id"] = pd.to_numeric(frame["scene_row_id"], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["scene_row_id"]).copy()
    frame["scene_row_id"] = frame["scene_row_id"].astype(int)
    return frame.drop_duplicates("scene_row_id", keep="first")


def add_empty_interaction_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [
        "interaction_risk_vehicle_count",
        "interaction_other_vehicle_count",
        "interaction_common_frame_count",
        "interaction_close_frame_count",
        "interaction_overlap_frame_count",
        "interaction_pair_observation_count",
        "interaction_min_norm_center_distance",
        "interaction_median_frame_min_norm_center_distance",
        "interaction_min_center_distance_px",
        "interaction_max_iou",
        "interaction_risk2_near_other_flag",
        "interaction_world_common_frame_count",
        "interaction_world_close_frame_count",
        "interaction_world_close_frame_ratio",
        "interaction_world_quality_frame_ratio",
        "interaction_world_jitter_frame_ratio",
        "interaction_world_edge_low_quality_frame_ratio",
        "interaction_world_other_moving_frame_ratio",
        "interaction_world_other_stationary_frame_ratio",
        "interaction_world_same_lane_frame_count",
        "interaction_world_lateral_close_frame_count",
        "interaction_world_lane_known_frame_count",
        "interaction_world_closing_frame_count",
        "interaction_world_cpa_frame_count",
        "interaction_world_cpa_frame_ratio",
        "interaction_world_pair_observation_count",
        "interaction_world_min_distance_m",
        "interaction_world_median_frame_min_distance_m",
        "interaction_world_min_abs_lateral_m",
        "interaction_world_min_abs_longitudinal_m",
        "interaction_world_nearest_risk_speed_kmh",
        "interaction_world_nearest_other_speed_kmh",
        "interaction_world_max_closing_speed_mps",
        "interaction_world_min_ttc_sec",
        "interaction_world_cpa_min_distance_m",
        "interaction_world_cpa_best_time_sec",
        "interaction_world_cpa_min_time_sec",
        "interaction_world_cpa_min_collision_ttc_sec",
        "interaction_world_cpa_best_collision_radius_m",
        "interaction_world_cpa_max_relative_speed_mps",
        "interaction_world_cpa_max_closing_speed_mps",
        "interaction_world_cpa_best_bearing_closure_score",
        "interaction_world_cpa_best_crossing_angle_deg",
        "interaction_world_cpa_max_collision_course_score",
        "interaction_world_cpa_nearest_risk_speed_kmh",
        "interaction_world_cpa_nearest_other_speed_kmh",
        "interaction_world_collision_course_flag",
        "interaction_world_risk2_near_other_flag",
    ]:
        out[col] = 0.0
    for col in [
        "interaction_status",
        "interaction_risk_track_ids",
        "interaction_risk_case_keys",
        "interaction_nearest_risk_track_id",
        "interaction_nearest_other_track_id",
        "interaction_world_nearest_risk_track_id",
        "interaction_world_nearest_other_track_id",
        "interaction_world_cpa_nearest_risk_track_id",
        "interaction_world_cpa_nearest_other_track_id",
    ]:
        out[col] = ""
    return out


def fill_interaction_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = [
        "interaction_risk_vehicle_count",
        "interaction_other_vehicle_count",
        "interaction_common_frame_count",
        "interaction_close_frame_count",
        "interaction_overlap_frame_count",
        "interaction_pair_observation_count",
        "interaction_min_norm_center_distance",
        "interaction_median_frame_min_norm_center_distance",
        "interaction_min_center_distance_px",
        "interaction_max_iou",
        "interaction_risk2_near_other_flag",
        "interaction_world_common_frame_count",
        "interaction_world_close_frame_count",
        "interaction_world_close_frame_ratio",
        "interaction_world_quality_frame_ratio",
        "interaction_world_jitter_frame_ratio",
        "interaction_world_edge_low_quality_frame_ratio",
        "interaction_world_other_moving_frame_ratio",
        "interaction_world_other_stationary_frame_ratio",
        "interaction_world_same_lane_frame_count",
        "interaction_world_lateral_close_frame_count",
        "interaction_world_lane_known_frame_count",
        "interaction_world_closing_frame_count",
        "interaction_world_cpa_frame_count",
        "interaction_world_cpa_frame_ratio",
        "interaction_world_pair_observation_count",
        "interaction_world_min_distance_m",
        "interaction_world_median_frame_min_distance_m",
        "interaction_world_min_abs_lateral_m",
        "interaction_world_min_abs_longitudinal_m",
        "interaction_world_nearest_risk_speed_kmh",
        "interaction_world_nearest_other_speed_kmh",
        "interaction_world_max_closing_speed_mps",
        "interaction_world_min_ttc_sec",
        "interaction_world_cpa_min_distance_m",
        "interaction_world_cpa_best_time_sec",
        "interaction_world_cpa_min_time_sec",
        "interaction_world_cpa_min_collision_ttc_sec",
        "interaction_world_cpa_best_collision_radius_m",
        "interaction_world_cpa_max_relative_speed_mps",
        "interaction_world_cpa_max_closing_speed_mps",
        "interaction_world_cpa_best_bearing_closure_score",
        "interaction_world_cpa_best_crossing_angle_deg",
        "interaction_world_cpa_max_collision_course_score",
        "interaction_world_cpa_nearest_risk_speed_kmh",
        "interaction_world_cpa_nearest_other_speed_kmh",
        "interaction_world_collision_course_flag",
        "interaction_world_risk2_near_other_flag",
    ]
    for col in numeric_cols:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    for col in [
        "interaction_status",
        "interaction_risk_track_ids",
        "interaction_risk_case_keys",
        "interaction_nearest_risk_track_id",
        "interaction_nearest_other_track_id",
        "interaction_world_nearest_risk_track_id",
        "interaction_world_nearest_other_track_id",
        "interaction_world_cpa_nearest_risk_track_id",
        "interaction_world_cpa_nearest_other_track_id",
    ]:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("")
    return out


def summarize_temporal_combo_sidecar(sidecar: pd.DataFrame) -> pd.DataFrame:
    required = [
        "scene_row_id",
        "temporal_combo_risk3_direct_flag",
        "temporal_combo_risk3_track_ids",
        "temporal_combo_risk3_case_keys",
        "temporal_combo_recent_trajectory_score",
        "temporal_combo_redlight_score",
        "temporal_combo_current_trajectory_score",
        "temporal_combo_recent_trajectory_hit_count",
        "temporal_combo_context_ok_count",
    ]
    _require_columns(sidecar, required, "scene_temporal_combo_sidecar")
    frame = sidecar[required].copy()
    frame["scene_row_id"] = pd.to_numeric(frame["scene_row_id"], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["scene_row_id"]).copy()
    frame["scene_row_id"] = frame["scene_row_id"].astype(int)
    return frame.drop_duplicates("scene_row_id", keep="first")


def add_empty_temporal_combo_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [
        "temporal_combo_risk3_direct_flag",
        "temporal_combo_recent_trajectory_score",
        "temporal_combo_redlight_score",
        "temporal_combo_current_trajectory_score",
        "temporal_combo_recent_trajectory_hit_count",
        "temporal_combo_context_ok_count",
    ]:
        out[col] = 0.0
    for col in ["temporal_combo_risk3_track_ids", "temporal_combo_risk3_case_keys"]:
        out[col] = ""
    return out


def fill_temporal_combo_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [
        "temporal_combo_risk3_direct_flag",
        "temporal_combo_recent_trajectory_score",
        "temporal_combo_redlight_score",
        "temporal_combo_current_trajectory_score",
        "temporal_combo_recent_trajectory_hit_count",
        "temporal_combo_context_ok_count",
    ]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    for col in ["temporal_combo_risk3_track_ids", "temporal_combo_risk3_case_keys"]:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str)
    return out


def summarize_temporal_overspeed_trajectory_tokens(
    scene_tokens: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    thresholds = (config or {}).get("thresholds", {})
    required = [
        "scene_row_id",
        "case_key",
        "video_id",
        "track_id",
        "scene_window_idx",
        "ts_window_idx",
        "token_single_vehicle_floor_risk",
        "token_strict_policy_floor_risk",
        "token_support_floor_risk",
        "overspeed_event_score",
        "active_family_count",
        "cooccurrence_strength",
    ]
    _require_columns(scene_tokens, required, "scene_tokens")
    frame = scene_tokens[required].copy()
    frame["scene_row_id"] = pd.to_numeric(frame["scene_row_id"], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["scene_row_id"]).copy()
    frame["scene_row_id"] = frame["scene_row_id"].astype(int)
    for col in [
        "scene_window_idx",
        "ts_window_idx",
        "track_id",
        "token_single_vehicle_floor_risk",
        "token_strict_policy_floor_risk",
        "token_support_floor_risk",
        "overspeed_event_score",
        "active_family_count",
        "cooccurrence_strength",
    ]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    frame["video_key"] = frame["video_id"].astype(str)
    frame["track_key"] = frame["track_id"].astype(int).astype(str)
    frame = frame.sort_values(["video_key", "track_key", "scene_window_idx", "ts_window_idx"], kind="mergesort")

    memory_windows = max(1, int(thresholds.get("temporal_overspeed_trajectory_memory_windows", 6)))
    frame["temporal_overspeed_trajectory_recent_overspeed_score"] = frame.groupby(
        ["video_key", "track_key"], sort=False
    )["overspeed_event_score"].transform(lambda s: s.rolling(window=memory_windows, min_periods=1).max())

    if bool(thresholds.get("temporal_overspeed_trajectory_require_policy_risk3", True)):
        support_risk3 = (frame["token_strict_policy_floor_risk"] >= 3) | (frame["token_support_floor_risk"] >= 3)
    else:
        support_risk3 = pd.Series(True, index=frame.index)

    direct = (
        (frame["token_single_vehicle_floor_risk"] >= float(thresholds.get("temporal_overspeed_trajectory_min_single_floor_risk", 2)))
        & (frame["token_single_vehicle_floor_risk"] < 3)
        & support_risk3
        & (
            frame["temporal_overspeed_trajectory_recent_overspeed_score"]
            >= float(thresholds.get("temporal_overspeed_trajectory_min_recent_overspeed_score", 0.98))
        )
        & (
            frame["overspeed_event_score"]
            >= float(thresholds.get("temporal_overspeed_trajectory_min_current_overspeed_score", 0.80))
        )
        & (
            frame["active_family_count"]
            >= float(thresholds.get("temporal_overspeed_trajectory_min_active_family_count", 2.0))
        )
        & (
            frame["cooccurrence_strength"]
            >= float(thresholds.get("temporal_overspeed_trajectory_min_cooccurrence_strength", 0.95))
        )
    )
    frame["temporal_overspeed_trajectory_direct_flag"] = direct.astype(int)

    columns = [
        "scene_row_id",
        "temporal_overspeed_trajectory_direct_flag",
        "temporal_overspeed_trajectory_track_ids",
        "temporal_overspeed_trajectory_case_keys",
        "temporal_overspeed_trajectory_recent_overspeed_score",
        "temporal_overspeed_trajectory_current_overspeed_score",
        "temporal_overspeed_trajectory_active_family_count",
        "temporal_overspeed_trajectory_cooccurrence_strength",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for scene_row_id, group in frame.groupby("scene_row_id", sort=False):
        direct_group = group[group["temporal_overspeed_trajectory_direct_flag"] > 0]
        rows.append(
            {
                "scene_row_id": int(scene_row_id),
                "temporal_overspeed_trajectory_direct_flag": int(len(direct_group) > 0),
                "temporal_overspeed_trajectory_track_ids": _join_int_ids(direct_group["track_id"].tolist()),
                "temporal_overspeed_trajectory_case_keys": _join_str_values(direct_group["case_key"].tolist()),
                "temporal_overspeed_trajectory_recent_overspeed_score": (
                    float(direct_group["temporal_overspeed_trajectory_recent_overspeed_score"].max())
                    if len(direct_group) > 0
                    else 0.0
                ),
                "temporal_overspeed_trajectory_current_overspeed_score": (
                    float(direct_group["overspeed_event_score"].max()) if len(direct_group) > 0 else 0.0
                ),
                "temporal_overspeed_trajectory_active_family_count": (
                    float(direct_group["active_family_count"].max()) if len(direct_group) > 0 else 0.0
                ),
                "temporal_overspeed_trajectory_cooccurrence_strength": (
                    float(direct_group["cooccurrence_strength"].max()) if len(direct_group) > 0 else 0.0
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def add_empty_temporal_overspeed_trajectory_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [
        "temporal_overspeed_trajectory_direct_flag",
        "temporal_overspeed_trajectory_recent_overspeed_score",
        "temporal_overspeed_trajectory_current_overspeed_score",
        "temporal_overspeed_trajectory_active_family_count",
        "temporal_overspeed_trajectory_cooccurrence_strength",
    ]:
        out[col] = 0.0
    for col in ["temporal_overspeed_trajectory_track_ids", "temporal_overspeed_trajectory_case_keys"]:
        out[col] = ""
    return out


def fill_temporal_overspeed_trajectory_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [
        "temporal_overspeed_trajectory_direct_flag",
        "temporal_overspeed_trajectory_recent_overspeed_score",
        "temporal_overspeed_trajectory_current_overspeed_score",
        "temporal_overspeed_trajectory_active_family_count",
        "temporal_overspeed_trajectory_cooccurrence_strength",
    ]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    for col in ["temporal_overspeed_trajectory_track_ids", "temporal_overspeed_trajectory_case_keys"]:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str)
    return out


def summarize_risk_source_tokens(scene_tokens: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    thresholds = (config or {}).get("thresholds", {})
    floor_mode = str(thresholds.get("risk_source_floor_mode", "max_all"))
    required = [
        "scene_row_id",
        "case_key",
        "track_id",
        "token_single_vehicle_floor_risk",
        "token_strict_policy_floor_risk",
        "token_support_floor_risk",
    ]
    _require_columns(scene_tokens, required, "scene_tokens")
    frame = scene_tokens[required].copy()
    frame["track_id_int"] = pd.to_numeric(frame["track_id"], errors="coerce")
    frame = frame[frame["track_id_int"].notna()].copy()
    frame["track_id_int"] = frame["track_id_int"].astype(int)
    for col in ["token_single_vehicle_floor_risk", "token_strict_policy_floor_risk", "token_support_floor_risk"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0).astype(int)
    if floor_mode == "single_vehicle_only":
        frame["token_source_floor_risk"] = frame["token_single_vehicle_floor_risk"]
    elif floor_mode == "single_strict_only":
        frame["token_source_floor_risk"] = frame[
            ["token_single_vehicle_floor_risk", "token_strict_policy_floor_risk"]
        ].max(axis=1)
    else:
        frame["token_source_floor_risk"] = frame[
            ["token_single_vehicle_floor_risk", "token_strict_policy_floor_risk", "token_support_floor_risk"]
        ].max(axis=1)

    columns = [
        "scene_row_id",
        "risk2_source_track_ids",
        "risk2_source_case_keys",
        "risk2_source_track_count",
        "risk3_source_track_ids",
        "risk3_source_case_keys",
        "risk3_source_track_count",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for scene_row_id, group in frame.groupby("scene_row_id", sort=False):
        risk2 = group[group["token_source_floor_risk"] >= 2]
        risk3 = group[group["token_source_floor_risk"] >= 3]
        rows.append(
            {
                "scene_row_id": int(scene_row_id),
                "risk2_source_track_ids": _join_int_ids(risk2["track_id_int"].tolist()),
                "risk2_source_case_keys": _join_str_values(risk2["case_key"].tolist()),
                "risk2_source_track_count": int(risk2["track_id_int"].nunique()),
                "risk3_source_track_ids": _join_int_ids(risk3["track_id_int"].tolist()),
                "risk3_source_case_keys": _join_str_values(risk3["case_key"].tolist()),
                "risk3_source_track_count": int(risk3["track_id_int"].nunique()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def add_empty_risk_source_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["risk2_source_track_count", "risk3_source_track_count"]:
        out[col] = 0
    for col in [
        "risk2_source_track_ids",
        "risk2_source_case_keys",
        "risk3_source_track_ids",
        "risk3_source_case_keys",
    ]:
        out[col] = ""
    return out


def fill_risk_source_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["risk2_source_track_count", "risk3_source_track_count"]:
        if col not in out.columns:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    for col in [
        "risk2_source_track_ids",
        "risk2_source_case_keys",
        "risk3_source_track_ids",
        "risk3_source_case_keys",
    ]:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str)
    return out


def merge_temporal_overlap_risk_sources(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    source_pairs = [
        ("risk2_source_track_ids", "single_vehicle_temporal_overlap_risk2_track_ids", "risk2_source_track_count"),
        ("risk3_source_track_ids", "single_vehicle_temporal_overlap_risk3_track_ids", "risk3_source_track_count"),
    ]
    for target_col, overlap_col, count_col in source_pairs:
        if overlap_col not in out.columns:
            continue
        merged_values: list[str] = []
        for base, overlap in zip(out[target_col], out[overlap_col], strict=False):
            merged_values.append(_format_track_ids(_parse_track_ids(base) | _parse_track_ids(overlap)))
        out[target_col] = merged_values
        out[count_col] = [_count_track_ids(value) for value in merged_values]
    return out


def merge_single_vehicle_causal_floor(
    df: pd.DataFrame,
    sidecar: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    required = [
        "scene_row_id",
        "single_vehicle_causal_floor_risk",
        "single_vehicle_causal_floor_schemaC",
        "single_vehicle_causal_risk2_track_ids",
        "single_vehicle_causal_risk2_case_keys",
        "single_vehicle_causal_risk3_track_ids",
        "single_vehicle_causal_risk3_case_keys",
        "single_vehicle_causal_risk2_track_count",
        "single_vehicle_causal_risk3_track_count",
    ]
    _require_columns(sidecar, required, "single_vehicle_causal_floor")
    keep_cols = [
        col
        for col in sidecar.columns
        if col == "scene_row_id" or col.startswith("single_vehicle_causal_")
    ]
    out = df.merge(
        sidecar[keep_cols],
        on="scene_row_id",
        how="left",
        validate="one_to_one",
    )
    for col in [
        "single_vehicle_causal_floor_risk",
        "single_vehicle_causal_floor_schemaC",
        "single_vehicle_causal_risk2_track_count",
        "single_vehicle_causal_risk3_track_count",
        "single_vehicle_causal_source_count",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    for col in [
        "single_vehicle_causal_max_overlap_sec",
        "single_vehicle_causal_max_scene_overlap_ratio",
        "single_vehicle_causal_max_token_overlap_ratio",
        "single_vehicle_causal_latest_source_end_sec",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).astype(float)
    for col in [
        "single_vehicle_causal_risk2_track_ids",
        "single_vehicle_causal_risk2_case_keys",
        "single_vehicle_causal_risk3_track_ids",
        "single_vehicle_causal_risk3_case_keys",
    ]:
        if col in out.columns:
            out[col] = out[col].fillna("").astype(str)

    thresholds = config.get("thresholds", {})
    if str(thresholds.get("single_vehicle_floor_mode", "")).lower() != "causal_v1":
        return out

    out["single_vehicle_floor_risk"] = out["single_vehicle_causal_floor_risk"].astype(int).clip(0, 3)
    out["single_vehicle_floor_schemaC"] = out["single_vehicle_causal_floor_schemaC"].astype(int).clip(0, 2)
    out["count_single_vehicle_risk2"] = out["single_vehicle_causal_risk2_track_count"].astype(int)
    out["count_single_vehicle_risk3"] = out["single_vehicle_causal_risk3_track_count"].astype(int)
    out["has_single_vehicle_risk2"] = (out["count_single_vehicle_risk2"] > 0).astype(int)
    out["has_single_vehicle_risk3"] = (out["count_single_vehicle_risk3"] > 0).astype(int)
    out["risk2_source_track_ids"] = out["single_vehicle_causal_risk2_track_ids"].fillna("").astype(str)
    out["risk2_source_case_keys"] = out["single_vehicle_causal_risk2_case_keys"].fillna("").astype(str)
    out["risk2_source_track_count"] = out["count_single_vehicle_risk2"].astype(int)
    out["risk3_source_track_ids"] = out["single_vehicle_causal_risk3_track_ids"].fillna("").astype(str)
    out["risk3_source_case_keys"] = out["single_vehicle_causal_risk3_case_keys"].fillna("").astype(str)
    out["risk3_source_track_count"] = out["count_single_vehicle_risk3"].astype(int)
    out["default_floor_risk"] = np.maximum(
        out["single_vehicle_floor_risk"].astype(int),
        out["strict_policy_floor_risk"].astype(int),
    ).astype(int)
    return out


def _count_track_ids(value: Any) -> int:
    return len(_parse_track_ids(value))


def summarize_pure_overspeed_tokens(scene_tokens: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    thresholds = config.get("thresholds", {})
    required = [
        "scene_row_id",
        "case_key",
        "track_id",
        "overspeed_event_score",
        "redlight_event_score",
        "r46_risk2_policy",
        "r46_risk3_policy",
        "r32_overspeed_midband_strength",
        "r32_overspeed_ge2_duration_strength",
        "r32_overspeed_reliable_signal",
        "r32_overspeed_rescue_signal",
    ]
    _require_columns(scene_tokens, required, "scene_tokens")
    frame = scene_tokens[required].copy()
    numeric_cols = [col for col in required if col not in {"scene_row_id", "case_key", "track_id"}]
    for col in numeric_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)

    mask = (
        (frame["overspeed_event_score"] >= float(thresholds.get("pure_overspeed_min_event_score", 0.50)))
        & (frame["r46_risk2_policy"] >= float(thresholds.get("pure_overspeed_min_risk2_policy", 1.0)))
        & (frame["r46_risk3_policy"] <= float(thresholds.get("pure_overspeed_max_risk3_policy", 0.0)))
        & (frame["redlight_event_score"] <= float(thresholds.get("pure_overspeed_max_redlight_event_score", 0.0)))
        & (frame["r32_overspeed_midband_strength"] >= float(thresholds.get("pure_overspeed_min_midband_strength", 0.50)))
        & (frame["r32_overspeed_ge2_duration_strength"] >= float(thresholds.get("pure_overspeed_min_ge2_duration_strength", 0.85)))
        & (frame["r32_overspeed_reliable_signal"] >= float(thresholds.get("pure_overspeed_min_reliable_signal", 0.95)))
        & (frame["r32_overspeed_rescue_signal"] >= float(thresholds.get("pure_overspeed_min_rescue_signal", 0.40)))
    )
    rescue = frame[mask].copy()
    columns = [
        "scene_row_id",
        "pure_overspeed_rescue_token_count",
        "pure_overspeed_rescue_max_score",
        "pure_overspeed_rescue_case_keys",
        "pure_overspeed_rescue_track_ids",
    ]
    if rescue.empty:
        return pd.DataFrame(columns=columns)

    grouped = rescue.groupby("scene_row_id", sort=False)
    summary = grouped.agg(
        pure_overspeed_rescue_token_count=("track_id", "size"),
        pure_overspeed_rescue_max_score=("overspeed_event_score", "max"),
        pure_overspeed_rescue_case_keys=("case_key", lambda values: ",".join(sorted({str(v) for v in values if str(v)}))),
        pure_overspeed_rescue_track_ids=("track_id", lambda values: ",".join(sorted({str(v) for v in values if str(v)}))),
    ).reset_index()
    return summary[columns]


def apply_pure_overspeed_rescue_gate(
    df: pd.DataFrame,
    base_pred: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    thresholds = config.get("thresholds", {})
    pred = np.asarray(base_pred, dtype=int).copy().clip(0, 3)
    rescue_mask = (
        (df["pure_overspeed_rescue_token_count"].to_numpy(dtype=float) > 0)
        & (df["prob_binary_positive"].to_numpy(dtype=float) >= float(thresholds.get("pure_overspeed_min_prob_binary_positive", 0.35)))
        & (df["prob4_2"].to_numpy(dtype=float) >= float(thresholds.get("pure_overspeed_min_prob4_2", 0.15)))
    )
    reasons: list[str] = []
    for idx in range(len(df)):
        if rescue_mask[idx] and pred[idx] < 2:
            pred[idx] = 2
            reasons.append("risk2_promoted_by_pure_overspeed_rescue")
        else:
            reasons.append("unchanged")
    return pred.astype(int), reasons


def apply_trajectory_moving_interaction_gate(
    df: pd.DataFrame,
    base_pred: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    thresholds = config.get("thresholds", {})
    pred = np.asarray(base_pred, dtype=int).copy().clip(0, 3)
    promoted = pred.copy()
    reasons = np.full(len(df), "unchanged", dtype=object)
    if not bool(thresholds.get("trajectory_moving_interaction_enabled", False)):
        return promoted.astype(int), reasons.tolist()

    quality_safe = (
        df["interaction_world_quality_frame_ratio"].to_numpy(dtype=float)
        * (1.0 - df["interaction_world_jitter_frame_ratio"].to_numpy(dtype=float))
        * (1.0 - df["interaction_world_edge_low_quality_frame_ratio"].to_numpy(dtype=float))
    )
    other_speed = np.maximum(
        np.nan_to_num(df["interaction_world_nearest_other_speed_kmh"].to_numpy(dtype=float), nan=0.0),
        np.nan_to_num(df["interaction_world_cpa_nearest_other_speed_kmh"].to_numpy(dtype=float), nan=0.0),
    )
    nearest_other_speed = np.nan_to_num(
        df["interaction_world_nearest_other_speed_kmh"].to_numpy(dtype=float),
        nan=0.0,
    )
    risk_speed = np.maximum(
        np.nan_to_num(df["interaction_world_nearest_risk_speed_kmh"].to_numpy(dtype=float), nan=0.0),
        np.nan_to_num(df["interaction_world_cpa_nearest_risk_speed_kmh"].to_numpy(dtype=float), nan=0.0),
    )
    other_dynamic_ok = (
        (df["interaction_world_other_moving_frame_ratio"].to_numpy(dtype=float) >= float(thresholds.get("trajectory_moving_interaction_min_other_moving_ratio", 0.20)))
        | (other_speed >= float(thresholds.get("trajectory_moving_interaction_min_other_speed_kmh", 5.0)))
    )
    trajectory_like = (
        (df["max_overspeed_event_score"].to_numpy(dtype=float) <= float(thresholds.get("trajectory_moving_interaction_max_overspeed_event_score", 0.05)))
        & (df["max_redlight_event_score"].to_numpy(dtype=float) <= float(thresholds.get("trajectory_moving_interaction_max_redlight_event_score", 0.05)))
        & (df["max_active_family_count"].to_numpy(dtype=float) <= float(thresholds.get("trajectory_moving_interaction_max_active_family_count", 1.25)))
    )
    direct_mask = (
        (pred >= int(thresholds.get("trajectory_moving_interaction_min_base_pred", 2)))
        & (pred < 3)
        & (df["single_vehicle_floor_risk"].to_numpy(dtype=int) >= 2)
        & trajectory_like
        & (df["prob_binary_positive"].to_numpy(dtype=float) >= float(thresholds.get("trajectory_moving_interaction_min_prob_binary_positive", 0.90)))
        & (df["interaction_risk2_near_other_flag"].to_numpy(dtype=float) > 0)
        & (df["interaction_close_frame_count"].to_numpy(dtype=float) >= float(thresholds.get("trajectory_moving_interaction_min_close_frame_count", 4.0)))
        & (df["interaction_overlap_frame_count"].to_numpy(dtype=float) >= float(thresholds.get("trajectory_moving_interaction_min_overlap_frame_count", 3.0)))
        & (df["interaction_common_frame_count"].to_numpy(dtype=float) >= float(thresholds.get("trajectory_moving_interaction_min_common_frame_count", 10.0)))
        & (
            df["interaction_min_norm_center_distance"].to_numpy(dtype=float)
            <= float(thresholds.get("trajectory_moving_interaction_max_norm_center_distance", 0.80))
        )
        & (
            df["interaction_world_min_distance_m"].to_numpy(dtype=float)
            <= float(thresholds.get("trajectory_moving_interaction_max_world_distance_m", 5.0))
        )
        & (quality_safe >= float(thresholds.get("trajectory_moving_interaction_min_quality_safe_score", 0.65)))
        & (
            df["interaction_world_edge_low_quality_frame_ratio"].to_numpy(dtype=float)
            <= float(thresholds.get("trajectory_moving_interaction_max_edge_low_quality_frame_ratio", 0.25))
        )
        & (risk_speed >= float(thresholds.get("trajectory_moving_interaction_min_risk_speed_kmh", 0.0)))
        & (
            nearest_other_speed
            >= float(thresholds.get("trajectory_moving_interaction_min_nearest_other_speed_kmh_hard", 0.0))
        )
        & other_dynamic_ok
    )

    direct_source_ids: list[set[int]] = []
    for idx, is_direct in enumerate(direct_mask):
        source_ids = _parse_track_ids(df["interaction_nearest_risk_track_id"].iloc[idx])
        if not source_ids:
            source_ids = _parse_track_ids(df["interaction_world_nearest_risk_track_id"].iloc[idx])
        if not source_ids:
            source_ids = _parse_track_ids(df["risk2_source_track_ids"].iloc[idx])
        direct_source_ids.append(source_ids)
        if bool(is_direct):
            promoted[idx] = 3
            other_ids = _parse_track_ids(df["interaction_nearest_other_track_id"].iloc[idx])
            if not other_ids:
                other_ids = _parse_track_ids(df["interaction_world_nearest_other_track_id"].iloc[idx])
            if source_ids and other_ids:
                reasons[idx] = (
                    f"risk3_promoted_by_trajectory_moving_interaction_pair="
                    f"{_format_track_ids(source_ids)}->{_format_track_ids(other_ids)}"
                )
            else:
                reasons[idx] = "risk3_promoted_by_trajectory_moving_interaction"

    if not bool(thresholds.get("trajectory_moving_interaction_track_sticky_enabled", True)):
        return promoted.astype(int), reasons.tolist()

    min_current_pred = int(thresholds.get("trajectory_moving_interaction_track_sticky_min_current_pred", 2))
    order_frame = df[["video_id", "scene_window_idx", "risk2_source_track_ids"]].copy()
    order_frame["_pos"] = np.arange(len(df), dtype=int)
    order_frame["scene_window_idx"] = pd.to_numeric(order_frame["scene_window_idx"], errors="coerce").fillna(-1).astype(int)
    sorted_order = order_frame.sort_values(["video_id", "scene_window_idx"], kind="mergesort")
    for _video_id, group in sorted_order.groupby("video_id", sort=False):
        active_sources: set[int] = set()
        for _video_id2, _scene_window_idx, risk2_ids_raw, pos in group[
            ["video_id", "scene_window_idx", "risk2_source_track_ids", "_pos"]
        ].itertuples(index=False, name=None):
            pos = int(pos)
            current_risk2_ids = _parse_track_ids(risk2_ids_raw)
            active_sources &= current_risk2_ids
            if direct_mask[pos]:
                source_ids = direct_source_ids[pos] & current_risk2_ids
                if not source_ids:
                    source_ids = direct_source_ids[pos]
                active_sources |= source_ids
                continue
            if active_sources and promoted[pos] >= min_current_pred and promoted[pos] < 3:
                promoted[pos] = 3
                reasons[pos] = f"risk3_trajectory_moving_interaction_track_sticky_sources={_format_track_ids(active_sources)}"
            if not current_risk2_ids or promoted[pos] <= 0:
                active_sources.clear()
    return promoted.astype(int), reasons.tolist()


def apply_temporal_combo_track_aware_gate(
    df: pd.DataFrame,
    base_pred: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    thresholds = config.get("thresholds", {})
    pred = np.asarray(base_pred, dtype=int).copy().clip(0, 3)
    promoted = pred.copy()
    reasons = np.full(len(df), "unchanged", dtype=object)
    min_base_pred = int(thresholds.get("temporal_combo_min_base_pred", 2))
    direct_mask = (
        (pred >= min_base_pred)
        & (pred < 3)
        & (df["temporal_combo_risk3_direct_flag"].to_numpy(dtype=float) > 0)
        & (
            df["temporal_combo_recent_trajectory_score"].to_numpy(dtype=float)
            >= float(thresholds.get("temporal_combo_min_recent_trajectory_score", 0.55))
        )
        & (
            df["temporal_combo_redlight_score"].to_numpy(dtype=float)
            >= float(thresholds.get("temporal_combo_min_redlight_score", 0.60))
        )
    )

    direct_source_ids: list[set[int]] = []
    for idx, is_direct in enumerate(direct_mask):
        source_ids = _parse_track_ids(df["temporal_combo_risk3_track_ids"].iloc[idx])
        if not source_ids:
            source_ids = _parse_track_ids(df["risk2_source_track_ids"].iloc[idx])
        direct_source_ids.append(source_ids)
        if bool(is_direct):
            promoted[idx] = 3
            if source_ids:
                reasons[idx] = f"risk3_promoted_by_temporal_combo_track={_format_track_ids(source_ids)}"
            else:
                reasons[idx] = "risk3_promoted_by_temporal_combo"

    if not bool(thresholds.get("temporal_combo_track_sticky_enabled", True)):
        return promoted.astype(int), reasons.tolist()

    min_current_pred = int(thresholds.get("temporal_combo_track_sticky_min_current_pred", 2))
    order_frame = df[["video_id", "scene_window_idx", "risk2_source_track_ids"]].copy()
    order_frame["_pos"] = np.arange(len(df), dtype=int)
    order_frame["scene_window_idx"] = pd.to_numeric(order_frame["scene_window_idx"], errors="coerce").fillna(-1).astype(int)
    sorted_order = order_frame.sort_values(["video_id", "scene_window_idx"], kind="mergesort")
    for _video_id, group in sorted_order.groupby("video_id", sort=False):
        active_sources: set[int] = set()
        for _video_id2, _scene_window_idx, risk2_ids_raw, pos in group[
            ["video_id", "scene_window_idx", "risk2_source_track_ids", "_pos"]
        ].itertuples(index=False, name=None):
            pos = int(pos)
            current_risk2_ids = _parse_track_ids(risk2_ids_raw)
            active_sources &= current_risk2_ids
            if direct_mask[pos]:
                source_ids = direct_source_ids[pos] & current_risk2_ids
                if not source_ids:
                    source_ids = direct_source_ids[pos]
                active_sources |= source_ids
                continue
            if active_sources and promoted[pos] >= min_current_pred and promoted[pos] < 3:
                promoted[pos] = 3
                reasons[pos] = f"risk3_temporal_combo_track_sticky_sources={_format_track_ids(active_sources)}"
            if not current_risk2_ids or promoted[pos] <= 0:
                active_sources.clear()
    return promoted.astype(int), reasons.tolist()


def apply_temporal_overspeed_trajectory_combo_gate(
    df: pd.DataFrame,
    base_pred: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    thresholds = config.get("thresholds", {})
    pred = np.asarray(base_pred, dtype=int).copy().clip(0, 3)
    promoted = pred.copy()
    reasons = np.full(len(df), "unchanged", dtype=object)
    if not bool(thresholds.get("temporal_overspeed_trajectory_enabled", False)):
        return promoted.astype(int), reasons.tolist()

    direct_mask = (
        (pred >= int(thresholds.get("temporal_overspeed_trajectory_min_base_pred", 2)))
        & (pred < 3)
        & (df["temporal_overspeed_trajectory_direct_flag"].to_numpy(dtype=float) > 0)
        & (
            df["temporal_overspeed_trajectory_recent_overspeed_score"].to_numpy(dtype=float)
            >= float(thresholds.get("temporal_overspeed_trajectory_min_recent_overspeed_score", 0.98))
        )
        & (
            df["temporal_overspeed_trajectory_current_overspeed_score"].to_numpy(dtype=float)
            >= float(thresholds.get("temporal_overspeed_trajectory_min_current_overspeed_score", 0.80))
        )
        & (
            df["temporal_overspeed_trajectory_active_family_count"].to_numpy(dtype=float)
            >= float(thresholds.get("temporal_overspeed_trajectory_min_active_family_count", 2.0))
        )
        & (
            df["temporal_overspeed_trajectory_cooccurrence_strength"].to_numpy(dtype=float)
            >= float(thresholds.get("temporal_overspeed_trajectory_min_cooccurrence_strength", 0.95))
        )
    )
    for idx, is_direct in enumerate(direct_mask):
        if not bool(is_direct):
            continue
        promoted[idx] = 3
        case_keys = str(df["temporal_overspeed_trajectory_case_keys"].iloc[idx])
        track_ids = str(df["temporal_overspeed_trajectory_track_ids"].iloc[idx])
        if case_keys or track_ids:
            reasons[idx] = f"risk3_promoted_by_temporal_overspeed_trajectory_combo_cases={case_keys};tracks={track_ids}"
        else:
            reasons[idx] = "risk3_promoted_by_temporal_overspeed_trajectory_combo"
    return promoted.astype(int), reasons.tolist()


def apply_cpa_onset_track_aware_gate(
    df: pd.DataFrame,
    base_pred: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    thresholds = config.get("thresholds", {})
    pred = np.asarray(base_pred, dtype=int).copy().clip(0, 3)
    candidate_mask, quality_score = build_cpa_onset_candidate_mask(df, pred, thresholds)
    promoted = pred.copy()
    reasons = np.full(len(df), "unchanged", dtype=object)
    min_consecutive = int(thresholds.get("cpa_onset_min_consecutive_windows", 2))
    min_current_pred = int(thresholds.get("cpa_onset_track_sticky_min_current_pred", 2))
    sticky_enabled = bool(thresholds.get("cpa_onset_track_sticky_enabled", True))

    order_frame = df[["video_id", "scene_window_idx", "risk2_source_track_ids"]].copy()
    order_frame["_pos"] = np.arange(len(df), dtype=int)
    order_frame["scene_window_idx"] = pd.to_numeric(order_frame["scene_window_idx"], errors="coerce").fillna(-1).astype(int)
    sorted_order = order_frame.sort_values(["video_id", "scene_window_idx"], kind="mergesort")
    for _video_id, group in sorted_order.groupby("video_id", sort=False):
        candidate_counts: dict[int, int] = {}
        active_sources: set[int] = set()
        for _video_id2, _scene_window_idx, risk2_ids_raw, pos in group[
            ["video_id", "scene_window_idx", "risk2_source_track_ids", "_pos"]
        ].itertuples(index=False, name=None):
            pos = int(pos)
            current_risk2_ids = _parse_track_ids(risk2_ids_raw)
            active_sources &= current_risk2_ids
            candidate_counts = {track_id: count for track_id, count in candidate_counts.items() if track_id in current_risk2_ids}

            source_ids: set[int] = set()
            if candidate_mask[pos]:
                source_ids = _parse_track_ids(df["interaction_world_cpa_nearest_risk_track_id"].iloc[pos])
                if not source_ids:
                    source_ids = _parse_track_ids(df["interaction_world_nearest_risk_track_id"].iloc[pos])
                if not source_ids:
                    source_ids = current_risk2_ids
                source_ids &= current_risk2_ids if current_risk2_ids else source_ids
                for track_id in source_ids:
                    candidate_counts[track_id] = candidate_counts.get(track_id, 0) + 1
                for track_id in list(candidate_counts):
                    if track_id not in source_ids:
                        candidate_counts.pop(track_id, None)
            else:
                candidate_counts.clear()

            eligible_sources = {track_id for track_id, count in candidate_counts.items() if count >= min_consecutive}
            if eligible_sources and promoted[pos] < 3:
                promoted[pos] = 3
                active_sources |= eligible_sources
                reasons[pos] = (
                    f"risk3_promoted_by_cpa_onset_tracks={_format_track_ids(eligible_sources)}"
                    f"_q={float(quality_score[pos]):.3f}"
                )
                continue

            if sticky_enabled and active_sources and promoted[pos] >= min_current_pred and promoted[pos] < 3:
                promoted[pos] = 3
                reasons[pos] = f"risk3_cpa_onset_track_sticky_sources={_format_track_ids(active_sources)}"
            if not current_risk2_ids or promoted[pos] <= 0:
                active_sources.clear()
    return promoted.astype(int), reasons.tolist()


def build_cpa_onset_candidate_mask(
    df: pd.DataFrame,
    pred: np.ndarray,
    thresholds: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    quality = df["interaction_world_quality_frame_ratio"].to_numpy(dtype=float)
    jitter = df["interaction_world_jitter_frame_ratio"].to_numpy(dtype=float)
    edge = df["interaction_world_edge_low_quality_frame_ratio"].to_numpy(dtype=float)
    quality_safe = np.clip(quality * (1.0 - jitter) * (1.0 - edge), 0.0, 1.0)
    cpa_distance = df["interaction_world_cpa_min_distance_m"].to_numpy(dtype=float)
    current_world_distance = df["interaction_world_min_distance_m"].to_numpy(dtype=float)
    cpa_ttc = df["interaction_world_cpa_min_collision_ttc_sec"].to_numpy(dtype=float)
    cpa_score = np.clip(df["interaction_world_cpa_max_collision_course_score"].to_numpy(dtype=float), 0.0, 1.0)
    angle = df["interaction_world_cpa_best_crossing_angle_deg"].to_numpy(dtype=float)
    crossing_angle_ok = angle >= float(thresholds.get("cpa_onset_min_crossing_angle_deg", 45.0))
    near_zero_angle_but_strong = (
        (angle <= float(thresholds.get("cpa_onset_max_zero_angle_deg", 5.0)))
        & (cpa_score >= float(thresholds.get("cpa_onset_min_zero_angle_collision_score", 0.85)))
        & (cpa_distance > 0)
        & (cpa_distance <= float(thresholds.get("cpa_onset_max_zero_angle_cpa_distance_m", 0.25)))
    )
    angle_ok = crossing_angle_ok | near_zero_angle_but_strong
    distance_ok = (
        (cpa_distance > 0)
        & (cpa_distance <= float(thresholds.get("cpa_onset_max_cpa_distance_m", 1.0)))
    )
    ttc_ok = (
        (cpa_ttc > 0)
        & (cpa_ttc <= float(thresholds.get("cpa_onset_max_ttc_sec", 1.0)))
    )
    quality_score = np.clip(cpa_score * quality_safe * angle_ok.astype(float), 0.0, 1.0)
    candidate = (
        (pred >= int(thresholds.get("cpa_onset_min_base_pred", 2)))
        & (pred < 3)
        & (df["prob_binary_positive"].to_numpy(dtype=float) >= float(thresholds.get("cpa_onset_min_prob_binary_positive", 0.75)))
        & (df["prob4_3"].to_numpy(dtype=float) >= float(thresholds.get("cpa_onset_min_prob4_3", 0.15)))
        & (df["prob_severity_2"].to_numpy(dtype=float) >= float(thresholds.get("cpa_onset_min_prob_severity_2", 0.10)))
        & (quality_safe >= float(thresholds.get("cpa_onset_min_quality_safe_score", 0.70)))
        & (edge <= float(thresholds.get("cpa_onset_max_edge_low_quality_frame_ratio", 0.35)))
        & distance_ok
        & ttc_ok
        & (cpa_score >= float(thresholds.get("cpa_onset_min_collision_course_score", 0.70)))
        & (quality_score >= float(thresholds.get("cpa_onset_min_quality_weighted_score", 0.65)))
    )
    return candidate, quality_score


def apply_high_confidence_cpa_onset_gate(
    df: pd.DataFrame,
    base_pred: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    thresholds = config.get("thresholds", {})
    pred = np.asarray(base_pred, dtype=int).copy().clip(0, 3)
    candidate_mask, quality_score = build_high_confidence_cpa_onset_mask(df, pred, thresholds)
    promoted = pred.copy()
    reasons = np.full(len(df), "unchanged", dtype=object)
    for idx, is_candidate in enumerate(candidate_mask):
        if not bool(is_candidate) or promoted[idx] >= 3:
            continue
        promoted[idx] = 3
        source_ids = _parse_track_ids(df["interaction_world_cpa_nearest_risk_track_id"].iloc[idx])
        other_ids = _parse_track_ids(df["interaction_world_cpa_nearest_other_track_id"].iloc[idx])
        if source_ids and other_ids:
            reasons[idx] = (
                f"risk3_promoted_by_high_confidence_cpa_pair="
                f"{_format_track_ids(source_ids)}->{_format_track_ids(other_ids)}"
                f"_q={float(quality_score[idx]):.3f}"
            )
        else:
            reasons[idx] = f"risk3_promoted_by_high_confidence_cpa_q={float(quality_score[idx]):.3f}"
    return promoted.astype(int), reasons.tolist()


def build_high_confidence_cpa_onset_mask(
    df: pd.DataFrame,
    pred: np.ndarray,
    thresholds: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    risk2_source_ids = [_parse_track_ids(value) for value in df["risk2_source_track_ids"]]
    cpa_risk_ids = [_parse_track_ids(value) for value in df["interaction_world_cpa_nearest_risk_track_id"]]
    cpa_other_ids = [_parse_track_ids(value) for value in df["interaction_world_cpa_nearest_other_track_id"]]
    source_matches_risk2 = np.array(
        [bool(risk2_ids and risk_ids and (risk2_ids & risk_ids)) for risk2_ids, risk_ids in zip(risk2_source_ids, cpa_risk_ids, strict=False)],
        dtype=bool,
    )
    other_is_not_risk2_source = np.array(
        [bool(other_ids and (other_ids - risk2_ids)) for risk2_ids, other_ids in zip(risk2_source_ids, cpa_other_ids, strict=False)],
        dtype=bool,
    )

    quality = df["interaction_world_quality_frame_ratio"].to_numpy(dtype=float)
    jitter = df["interaction_world_jitter_frame_ratio"].to_numpy(dtype=float)
    edge = df["interaction_world_edge_low_quality_frame_ratio"].to_numpy(dtype=float)
    quality_safe = np.clip(quality * (1.0 - jitter) * (1.0 - edge), 0.0, 1.0)
    cpa_distance = df["interaction_world_cpa_min_distance_m"].to_numpy(dtype=float)
    current_world_distance = df["interaction_world_min_distance_m"].to_numpy(dtype=float)
    cpa_ttc = df["interaction_world_cpa_min_collision_ttc_sec"].to_numpy(dtype=float)
    cpa_score = np.clip(df["interaction_world_cpa_max_collision_course_score"].to_numpy(dtype=float), 0.0, 1.0)
    angle = df["interaction_world_cpa_best_crossing_angle_deg"].to_numpy(dtype=float)
    quality_score = np.clip(cpa_score * quality_safe, 0.0, 1.0)
    collision_flag_ok = np.ones(len(df), dtype=bool)
    if bool(thresholds.get("high_conf_cpa_require_collision_course_flag", True)):
        collision_flag_ok = df["interaction_world_collision_course_flag"].to_numpy(dtype=float) > 0

    candidate = (
        (pred >= int(thresholds.get("high_conf_cpa_min_base_pred", 2)))
        & (pred < 3)
        & source_matches_risk2
        & other_is_not_risk2_source
        & collision_flag_ok
        & (df["prob_binary_positive"].to_numpy(dtype=float) >= float(thresholds.get("high_conf_cpa_min_prob_binary_positive", 0.95)))
        & (quality_safe >= float(thresholds.get("high_conf_cpa_min_quality_safe_score", 0.60)))
        & (edge <= float(thresholds.get("high_conf_cpa_max_edge_low_quality_frame_ratio", 0.40)))
        & (cpa_distance > 0)
        & (cpa_distance <= float(thresholds.get("high_conf_cpa_max_cpa_distance_m", 0.25)))
        & (current_world_distance > 0)
        & (
            current_world_distance
            <= float(thresholds.get("high_conf_cpa_max_current_world_distance_m", 7.5))
        )
        & (cpa_ttc > 0)
        & (cpa_ttc <= float(thresholds.get("high_conf_cpa_max_ttc_sec", 0.50)))
        & (cpa_score >= float(thresholds.get("high_conf_cpa_min_collision_course_score", 0.93)))
        & (angle >= float(thresholds.get("high_conf_cpa_min_crossing_angle_deg", 150.0)))
        & (quality_score >= float(thresholds.get("high_conf_cpa_min_quality_weighted_score", 0.55)))
    )
    return candidate, quality_score


def apply_cpa_collision_course_rescue_gate(
    df: pd.DataFrame,
    base_pred: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    thresholds = config.get("thresholds", {})
    pred = np.asarray(base_pred, dtype=int).copy().clip(0, 3)
    candidate_mask, quality_score = build_cpa_collision_course_rescue_mask(df, pred, thresholds)
    promoted = pred.copy()
    reasons = np.full(len(df), "unchanged", dtype=object)
    for idx, is_candidate in enumerate(candidate_mask):
        if not bool(is_candidate) or promoted[idx] >= 3:
            continue
        promoted[idx] = 3
        source_ids = _parse_track_ids(df["interaction_world_cpa_nearest_risk_track_id"].iloc[idx])
        if not source_ids:
            source_ids = _parse_track_ids(df["interaction_world_nearest_risk_track_id"].iloc[idx])
        other_ids = _parse_track_ids(df["interaction_world_cpa_nearest_other_track_id"].iloc[idx])
        if not other_ids:
            other_ids = _parse_track_ids(df["interaction_world_nearest_other_track_id"].iloc[idx])
        if source_ids and other_ids:
            reasons[idx] = (
                f"risk3_promoted_by_cpa_collision_course_pair="
                f"{_format_track_ids(source_ids)}->{_format_track_ids(other_ids)}"
                f"_q={float(quality_score[idx]):.3f}"
            )
        else:
            reasons[idx] = f"risk3_promoted_by_cpa_collision_course_q={float(quality_score[idx]):.3f}"
    return promoted.astype(int), reasons.tolist()


def build_cpa_collision_course_rescue_mask(
    df: pd.DataFrame,
    pred: np.ndarray,
    thresholds: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    risk2_source_ids = [_parse_track_ids(value) for value in df["risk2_source_track_ids"]]
    risk3_source_ids = [_parse_track_ids(value) for value in df["risk3_source_track_ids"]]
    cpa_risk_ids = [_parse_track_ids(value) for value in df["interaction_world_cpa_nearest_risk_track_id"]]
    cpa_other_ids = [_parse_track_ids(value) for value in df["interaction_world_cpa_nearest_other_track_id"]]

    source_matches_risk2 = np.array(
        [bool(risk2_ids and risk_ids and (risk2_ids & risk_ids)) for risk2_ids, risk_ids in zip(risk2_source_ids, cpa_risk_ids, strict=False)],
        dtype=bool,
    )
    other_is_not_source = np.array(
        [bool(other_ids and (other_ids - risk2_ids)) for risk2_ids, other_ids in zip(risk2_source_ids, cpa_other_ids, strict=False)],
        dtype=bool,
    )
    # If the same row already has a risk3 source, the ordinary single-floor path should
    # handle it; this gate is for risk2 vehicles whose collision course upgrades scene severity.
    no_existing_risk3_source = np.array([not bool(ids) for ids in risk3_source_ids], dtype=bool)

    quality = df["interaction_world_quality_frame_ratio"].to_numpy(dtype=float)
    jitter = df["interaction_world_jitter_frame_ratio"].to_numpy(dtype=float)
    edge = df["interaction_world_edge_low_quality_frame_ratio"].to_numpy(dtype=float)
    quality_safe = np.clip(quality * (1.0 - jitter) * (1.0 - edge), 0.0, 1.0)
    cpa_distance = df["interaction_world_cpa_min_distance_m"].to_numpy(dtype=float)
    current_world_distance = df["interaction_world_min_distance_m"].to_numpy(dtype=float)
    cpa_ttc = df["interaction_world_cpa_min_collision_ttc_sec"].to_numpy(dtype=float)
    cpa_score = np.clip(df["interaction_world_cpa_max_collision_course_score"].to_numpy(dtype=float), 0.0, 1.0)
    angle = df["interaction_world_cpa_best_crossing_angle_deg"].to_numpy(dtype=float)
    other_moving_ratio = df["interaction_world_other_moving_frame_ratio"].to_numpy(dtype=float)
    cpa_other_speed = np.nan_to_num(df["interaction_world_cpa_nearest_other_speed_kmh"].to_numpy(dtype=float), nan=0.0)
    nearest_other_speed = np.nan_to_num(df["interaction_world_nearest_other_speed_kmh"].to_numpy(dtype=float), nan=0.0)
    other_speed = np.maximum(cpa_other_speed, nearest_other_speed)
    if bool(thresholds.get("cpa_collision_require_single_floor_risk2", False)):
        source_floor_ok = df["single_vehicle_floor_risk"].to_numpy(dtype=float) >= 2
    else:
        source_floor_ok = np.ones(len(df), dtype=bool)

    angle_ok = (
        angle >= float(thresholds.get("cpa_collision_min_crossing_angle_deg", 120.0))
    ) | (
        (angle <= float(thresholds.get("cpa_collision_max_parallel_angle_deg", 8.0)))
        & (cpa_distance <= float(thresholds.get("cpa_collision_parallel_max_cpa_distance_m", 0.20)))
        & (cpa_score >= float(thresholds.get("cpa_collision_parallel_min_score", 0.65)))
    )
    dynamic_other_ok = (
        (other_moving_ratio >= float(thresholds.get("cpa_collision_min_other_moving_ratio", 0.20)))
        | (other_speed >= float(thresholds.get("cpa_collision_min_other_speed_kmh", 5.0)))
    )
    quality_score = np.clip(cpa_score * quality_safe * angle_ok.astype(float), 0.0, 1.0)
    candidate = (
        (pred >= int(thresholds.get("cpa_collision_min_base_pred", 2)))
        & (pred < 3)
        & source_matches_risk2
        & other_is_not_source
        & no_existing_risk3_source
        & source_floor_ok
        & dynamic_other_ok
        & (quality_safe >= float(thresholds.get("cpa_collision_min_quality_safe_score", 0.70)))
        & (edge <= float(thresholds.get("cpa_collision_max_edge_low_quality_frame_ratio", 0.40)))
        & (cpa_distance > 0)
        & (cpa_distance <= float(thresholds.get("cpa_collision_max_cpa_distance_m", 0.75)))
        & (current_world_distance > 0)
        & (current_world_distance <= float(thresholds.get("cpa_collision_max_current_world_distance_m", 8.0)))
        & (cpa_ttc > 0)
        & (cpa_ttc <= float(thresholds.get("cpa_collision_max_ttc_sec", 1.50)))
        & (cpa_score >= float(thresholds.get("cpa_collision_min_collision_course_score", 0.65)))
        & angle_ok
        & (quality_score >= float(thresholds.get("cpa_collision_min_quality_weighted_score", 0.55)))
    )
    return candidate, quality_score


def apply_interaction_guarded_gate(
    df: pd.DataFrame,
    base_pred: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    thresholds = config.get("thresholds", {})
    pred = np.asarray(base_pred, dtype=int).copy().clip(0, 3)
    direct_mask = build_interaction_direct_mask(df, pred, thresholds)
    reasons = np.full(len(df), "unchanged", dtype=object)
    promoted = pred.copy()
    for idx, is_direct in enumerate(direct_mask):
        if bool(is_direct):
            promoted[idx] = 3
            reasons[idx] = "risk3_promoted_by_risk_vehicle_proximity"
    return promoted.astype(int), reasons.tolist()


def apply_interaction_track_aware_sticky_gate(
    df: pd.DataFrame,
    base_pred: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    thresholds = config.get("thresholds", {})
    pred = np.asarray(base_pred, dtype=int).copy().clip(0, 3)
    direct_mask = build_interaction_direct_mask(df, pred, thresholds)
    promoted = pred.copy()
    reasons = np.full(len(df), "unchanged", dtype=object)
    direct_source_ids: list[set[int]] = []

    for idx, is_direct in enumerate(direct_mask):
        source_ids = _parse_track_ids(df["interaction_nearest_risk_track_id"].iloc[idx])
        if not source_ids:
            source_ids = _parse_track_ids(df["interaction_risk_track_ids"].iloc[idx])
        direct_source_ids.append(source_ids)
        if bool(is_direct):
            promoted[idx] = 3
            if source_ids:
                reasons[idx] = f"risk3_promoted_by_risk_vehicle_proximity_track={_format_track_ids(source_ids)}"
            else:
                reasons[idx] = "risk3_promoted_by_risk_vehicle_proximity"

    if not bool(thresholds.get("track_sticky_enabled", True)):
        return promoted.astype(int), reasons.tolist()

    start_from_interaction = bool(thresholds.get("track_sticky_start_from_interaction", True))
    start_from_floor_risk3 = bool(thresholds.get("track_sticky_start_from_floor_risk3", False))
    min_current_pred = int(thresholds.get("track_sticky_min_current_pred", 1))

    order_frame = df[
        [
            "video_id",
            "scene_window_idx",
            "risk2_source_track_ids",
            "risk3_source_track_ids",
        ]
    ].copy()
    order_frame["_pos"] = np.arange(len(df), dtype=int)
    order_frame["scene_window_idx"] = pd.to_numeric(order_frame["scene_window_idx"], errors="coerce").fillna(-1).astype(int)
    sorted_order = order_frame.sort_values(["video_id", "scene_window_idx"], kind="mergesort")
    for _video_id, group in sorted_order.groupby("video_id", sort=False):
        active_sources: set[int] = set()
        for video_id, scene_window_idx, risk2_ids_raw, risk3_ids_raw, pos in group[
            ["video_id", "scene_window_idx", "risk2_source_track_ids", "risk3_source_track_ids", "_pos"]
        ].itertuples(index=False, name=None):
            _ = video_id, scene_window_idx
            pos = int(pos)
            current_risk2_ids = _parse_track_ids(risk2_ids_raw)
            current_risk3_ids = _parse_track_ids(risk3_ids_raw)
            active_sources &= current_risk2_ids

            if start_from_floor_risk3 and promoted[pos] >= 3 and current_risk3_ids:
                active_sources |= current_risk3_ids

            if start_from_interaction and direct_mask[pos]:
                source_ids = direct_source_ids[pos] & current_risk2_ids
                if not source_ids:
                    source_ids = direct_source_ids[pos]
                active_sources |= source_ids
                continue

            if active_sources and promoted[pos] >= min_current_pred and promoted[pos] < 3:
                promoted[pos] = 3
                reasons[pos] = f"risk3_track_sticky_sources={_format_track_ids(active_sources)}"
            if not current_risk2_ids or promoted[pos] <= 0:
                active_sources.clear()
    return promoted.astype(int), reasons.tolist()


def apply_interaction_world_guarded_gate(
    df: pd.DataFrame,
    base_pred: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    thresholds = config.get("thresholds", {})
    pred = np.asarray(base_pred, dtype=int).copy().clip(0, 3)
    direct_mask = build_interaction_world_direct_mask(df, pred, thresholds)
    reasons = np.full(len(df), "unchanged", dtype=object)
    promoted = pred.copy()
    for idx, is_direct in enumerate(direct_mask):
        if bool(is_direct):
            promoted[idx] = 3
            source_ids = _parse_track_ids(df["interaction_world_nearest_risk_track_id"].iloc[idx])
            other_ids = _parse_track_ids(df["interaction_world_nearest_other_track_id"].iloc[idx])
            if source_ids and other_ids:
                reasons[idx] = (
                    f"risk3_promoted_by_world_proximity_pair="
                    f"{_format_track_ids(source_ids)}->{_format_track_ids(other_ids)}"
                )
            else:
                reasons[idx] = "risk3_promoted_by_world_proximity"
    return promoted.astype(int), reasons.tolist()


def apply_interaction_world_track_aware_sticky_gate(
    df: pd.DataFrame,
    base_pred: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    thresholds = config.get("thresholds", {})
    pred = np.asarray(base_pred, dtype=int).copy().clip(0, 3)
    direct_mask = build_interaction_world_direct_mask(df, pred, thresholds)
    promoted = pred.copy()
    reasons = np.full(len(df), "unchanged", dtype=object)
    direct_source_ids: list[set[int]] = []

    for idx, is_direct in enumerate(direct_mask):
        source_ids = _parse_track_ids(df["interaction_world_nearest_risk_track_id"].iloc[idx])
        if not source_ids:
            source_ids = _parse_track_ids(df["interaction_risk_track_ids"].iloc[idx])
        direct_source_ids.append(source_ids)
        if bool(is_direct):
            promoted[idx] = 3
            other_ids = _parse_track_ids(df["interaction_world_nearest_other_track_id"].iloc[idx])
            if source_ids and other_ids:
                reasons[idx] = (
                    f"risk3_promoted_by_world_proximity_pair="
                    f"{_format_track_ids(source_ids)}->{_format_track_ids(other_ids)}"
                )
            else:
                reasons[idx] = "risk3_promoted_by_world_proximity"

    if not bool(thresholds.get("world_track_sticky_enabled", True)):
        return promoted.astype(int), reasons.tolist()

    start_from_world_interaction = bool(thresholds.get("world_track_sticky_start_from_interaction", True))
    min_current_pred = int(thresholds.get("world_track_sticky_min_current_pred", 1))
    order_frame = df[["video_id", "scene_window_idx", "risk2_source_track_ids"]].copy()
    order_frame["_pos"] = np.arange(len(df), dtype=int)
    order_frame["scene_window_idx"] = pd.to_numeric(order_frame["scene_window_idx"], errors="coerce").fillna(-1).astype(int)
    sorted_order = order_frame.sort_values(["video_id", "scene_window_idx"], kind="mergesort")
    for _video_id, group in sorted_order.groupby("video_id", sort=False):
        active_sources: set[int] = set()
        for video_id, scene_window_idx, risk2_ids_raw, pos in group[
            ["video_id", "scene_window_idx", "risk2_source_track_ids", "_pos"]
        ].itertuples(index=False, name=None):
            _ = video_id, scene_window_idx
            pos = int(pos)
            current_risk2_ids = _parse_track_ids(risk2_ids_raw)
            active_sources &= current_risk2_ids
            if start_from_world_interaction and direct_mask[pos]:
                source_ids = direct_source_ids[pos] & current_risk2_ids
                if not source_ids:
                    source_ids = direct_source_ids[pos]
                active_sources |= source_ids
                continue
            if active_sources and promoted[pos] >= min_current_pred and promoted[pos] < 3:
                promoted[pos] = 3
                reasons[pos] = f"risk3_world_track_sticky_sources={_format_track_ids(active_sources)}"
            if not current_risk2_ids or promoted[pos] <= 0:
                active_sources.clear()
    return promoted.astype(int), reasons.tolist()


def apply_learned_interaction_guarded_gate(
    df: pd.DataFrame,
    base_pred: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    thresholds = config.get("thresholds", {})
    pred = np.asarray(base_pred, dtype=int).copy().clip(0, 3)
    direct_mask, _quality_score = build_learned_interaction_direct_mask(df, pred, thresholds)
    require_source_match = bool(thresholds.get("learned_interaction_require_source_in_risk2_sources", False))
    promoted = pred.copy()
    reasons = np.full(len(df), "unchanged", dtype=object)
    for idx, is_direct in enumerate(direct_mask):
        if bool(is_direct):
            source_ids = _parse_track_ids(df["interaction_world_cpa_nearest_risk_track_id"].iloc[idx])
            if not source_ids:
                source_ids = _parse_track_ids(df["interaction_world_nearest_risk_track_id"].iloc[idx])
            risk2_source_ids = _parse_track_ids(df["risk2_source_track_ids"].iloc[idx])
            matched_source_ids = source_ids & risk2_source_ids
            if require_source_match and not matched_source_ids:
                continue
            if matched_source_ids:
                source_ids = matched_source_ids
            promoted[idx] = 3
            other_ids = _parse_track_ids(df["interaction_world_cpa_nearest_other_track_id"].iloc[idx])
            if not other_ids:
                other_ids = _parse_track_ids(df["interaction_world_nearest_other_track_id"].iloc[idx])
            if source_ids and other_ids:
                reasons[idx] = (
                    f"risk3_promoted_by_learned_interaction_pair="
                    f"{_format_track_ids(source_ids)}->{_format_track_ids(other_ids)}"
                )
            else:
                reasons[idx] = "risk3_promoted_by_learned_interaction"
    return promoted.astype(int), reasons.tolist()


def apply_learned_interaction_track_aware_sticky_gate(
    df: pd.DataFrame,
    base_pred: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    thresholds = config.get("thresholds", {})
    pred = np.asarray(base_pred, dtype=int).copy().clip(0, 3)
    direct_mask, _quality_score = build_learned_interaction_direct_mask(df, pred, thresholds)
    require_source_match = bool(thresholds.get("learned_interaction_require_source_in_risk2_sources", False))
    promoted = pred.copy()
    reasons = np.full(len(df), "unchanged", dtype=object)
    direct_source_ids: list[set[int]] = []

    for idx, is_direct in enumerate(direct_mask):
        source_ids = _parse_track_ids(df["interaction_world_cpa_nearest_risk_track_id"].iloc[idx])
        if not source_ids:
            source_ids = _parse_track_ids(df["interaction_world_nearest_risk_track_id"].iloc[idx])
        if not source_ids:
            source_ids = _parse_track_ids(df["interaction_risk_track_ids"].iloc[idx])
        risk2_source_ids = _parse_track_ids(df["risk2_source_track_ids"].iloc[idx])
        matched_source_ids = source_ids & risk2_source_ids
        if matched_source_ids:
            source_ids = matched_source_ids
        direct_source_ids.append(source_ids)
        if bool(is_direct):
            if require_source_match and not matched_source_ids:
                continue
            promoted[idx] = 3
            other_ids = _parse_track_ids(df["interaction_world_cpa_nearest_other_track_id"].iloc[idx])
            if not other_ids:
                other_ids = _parse_track_ids(df["interaction_world_nearest_other_track_id"].iloc[idx])
            if source_ids and other_ids:
                reasons[idx] = (
                    f"risk3_promoted_by_learned_interaction_pair="
                    f"{_format_track_ids(source_ids)}->{_format_track_ids(other_ids)}"
                )
            else:
                reasons[idx] = "risk3_promoted_by_learned_interaction"

    if not bool(thresholds.get("learned_interaction_track_sticky_enabled", True)):
        return promoted.astype(int), reasons.tolist()

    min_current_pred = int(thresholds.get("learned_interaction_track_sticky_min_current_pred", 1))
    order_frame = df[["video_id", "scene_window_idx", "risk2_source_track_ids"]].copy()
    order_frame["_pos"] = np.arange(len(df), dtype=int)
    order_frame["scene_window_idx"] = pd.to_numeric(order_frame["scene_window_idx"], errors="coerce").fillna(-1).astype(int)
    sorted_order = order_frame.sort_values(["video_id", "scene_window_idx"], kind="mergesort")
    for _video_id, group in sorted_order.groupby("video_id", sort=False):
        active_sources: set[int] = set()
        for video_id, scene_window_idx, risk2_ids_raw, pos in group[
            ["video_id", "scene_window_idx", "risk2_source_track_ids", "_pos"]
        ].itertuples(index=False, name=None):
            _ = video_id, scene_window_idx
            pos = int(pos)
            current_risk2_ids = _parse_track_ids(risk2_ids_raw)
            active_sources &= current_risk2_ids
            if direct_mask[pos]:
                source_ids = direct_source_ids[pos] & current_risk2_ids
                if not source_ids:
                    if require_source_match:
                        continue
                    source_ids = direct_source_ids[pos]
                active_sources |= source_ids
                continue
            if active_sources and promoted[pos] >= min_current_pred and promoted[pos] < 3:
                promoted[pos] = 3
                reasons[pos] = f"risk3_learned_interaction_track_sticky_sources={_format_track_ids(active_sources)}"
            if not current_risk2_ids or promoted[pos] <= 0:
                active_sources.clear()
    return promoted.astype(int), reasons.tolist()


def build_interaction_direct_mask(df: pd.DataFrame, pred: np.ndarray, thresholds: dict[str, Any]) -> np.ndarray:
    return (
        (pred >= int(thresholds.get("interaction_promote_min_base_pred", 2)))
        & (pred < 3)
        & (df["interaction_risk2_near_other_flag"].to_numpy(dtype=float) > 0)
        & (df["prob4_3"].to_numpy(dtype=float) >= float(thresholds.get("interaction_min_prob4_3", 0.20)))
        & (df["prob_severity_2"].to_numpy(dtype=float) >= float(thresholds.get("interaction_min_prob_severity_2", 0.20)))
        & (
            df["interaction_min_norm_center_distance"].to_numpy(dtype=float)
            <= float(thresholds.get("interaction_max_norm_center_distance", 1.05))
        )
        & (
            df["interaction_close_frame_count"].to_numpy(dtype=float)
            >= float(thresholds.get("interaction_min_close_frame_count", 3))
        )
        & (
            df["interaction_overlap_frame_count"].to_numpy(dtype=float)
            >= float(thresholds.get("interaction_min_overlap_frame_count", 0))
        )
        & (
            df["interaction_common_frame_count"].to_numpy(dtype=float)
            >= float(thresholds.get("interaction_min_common_frame_count", 3))
        )
    )


def build_learned_interaction_direct_mask(
    df: pd.DataFrame,
    pred: np.ndarray,
    thresholds: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    quality = df["interaction_world_quality_frame_ratio"].to_numpy(dtype=float)
    jitter = df["interaction_world_jitter_frame_ratio"].to_numpy(dtype=float)
    edge = df["interaction_world_edge_low_quality_frame_ratio"].to_numpy(dtype=float)
    quality_safe = np.clip(quality * (1.0 - jitter) * (1.0 - edge), 0.0, 1.0)
    cpa_score = np.clip(df["interaction_world_cpa_max_collision_course_score"].to_numpy(dtype=float), 0.0, 1.0)
    angle = df["interaction_world_cpa_best_crossing_angle_deg"].to_numpy(dtype=float)
    crossing = angle >= float(thresholds.get("learned_interaction_min_cpa_crossing_angle_deg", 15.0))
    quality_weighted_score = np.clip(cpa_score * quality_safe * crossing.astype(float), 0.0, 1.0)
    cpa_distance = df["interaction_world_cpa_min_distance_m"].to_numpy(dtype=float)
    world_distance = df["interaction_world_min_distance_m"].to_numpy(dtype=float)
    cpa_ttc = df["interaction_world_cpa_min_collision_ttc_sec"].to_numpy(dtype=float)
    cpa_distance_ok = cpa_distance <= float(thresholds.get("learned_interaction_max_cpa_distance_m", 2.2))
    world_distance_ok = world_distance <= float(thresholds.get("learned_interaction_max_world_distance_m", 6.0))
    ttc_ok = (
        (cpa_ttc > 0)
        & (cpa_ttc <= float(thresholds.get("learned_interaction_max_cpa_ttc_sec", 1.5)))
    )
    cpa_score_ok = cpa_score >= float(thresholds.get("learned_interaction_min_cpa_collision_course_score", 0.45))
    edge_ok = edge <= float(thresholds.get("learned_interaction_max_edge_low_quality_frame_ratio", 1.0))
    if bool(thresholds.get("learned_interaction_require_dynamic_other_or_strong_static", False)):
        other_moving_ratio = df["interaction_world_other_moving_frame_ratio"].to_numpy(dtype=float)
        cpa_other_speed = df["interaction_world_cpa_nearest_other_speed_kmh"].to_numpy(dtype=float)
        nearest_other_speed = df["interaction_world_nearest_other_speed_kmh"].to_numpy(dtype=float)
        other_speed = np.maximum(
            np.nan_to_num(cpa_other_speed, nan=0.0),
            np.nan_to_num(nearest_other_speed, nan=0.0),
        )
        other_dynamic_ok = (
            (other_moving_ratio >= float(thresholds.get("learned_interaction_min_other_moving_frame_ratio", 0.20)))
            | (other_speed >= float(thresholds.get("learned_interaction_min_other_speed_kmh", 5.0)))
        )
        static_strong_ok = (
            (df["interaction_world_collision_course_flag"].to_numpy(dtype=float) > 0)
            & (world_distance > 0)
            & (world_distance <= float(thresholds.get("learned_interaction_static_max_world_distance_m", 4.0)))
            & (cpa_distance > 0)
            & (cpa_distance <= float(thresholds.get("learned_interaction_static_max_cpa_distance_m", 0.75)))
            & (cpa_ttc > 0)
            & (cpa_ttc <= float(thresholds.get("learned_interaction_static_max_cpa_ttc_sec", 1.0)))
            & (cpa_score >= float(thresholds.get("learned_interaction_static_min_cpa_collision_course_score", 0.75)))
        )
        dynamic_context_ok = other_dynamic_ok | static_strong_ok
    else:
        dynamic_context_ok = np.ones(len(df), dtype=bool)
    if bool(thresholds.get("learned_interaction_block_static_weak_cpa", False)):
        other_moving_ratio = df["interaction_world_other_moving_frame_ratio"].to_numpy(dtype=float)
        cpa_other_speed = df["interaction_world_cpa_nearest_other_speed_kmh"].to_numpy(dtype=float)
        nearest_other_speed = df["interaction_world_nearest_other_speed_kmh"].to_numpy(dtype=float)
        other_speed = np.maximum(
            np.nan_to_num(cpa_other_speed, nan=0.0),
            np.nan_to_num(nearest_other_speed, nan=0.0),
        )
        static_other = (
            (other_moving_ratio <= float(thresholds.get("learned_interaction_static_weak_max_other_moving_ratio", 0.05)))
            & (other_speed <= float(thresholds.get("learned_interaction_static_weak_max_other_speed_kmh", 3.0)))
        )
        weak_cpa = (
            (df["interaction_world_collision_course_flag"].to_numpy(dtype=float) <= 0)
            & (cpa_score < float(thresholds.get("learned_interaction_static_weak_max_cpa_collision_course_score", 0.55)))
            & (cpa_distance >= float(thresholds.get("learned_interaction_static_weak_min_cpa_distance_m", 1.2)))
            & (
                (cpa_ttc <= 0)
                | (cpa_ttc >= float(thresholds.get("learned_interaction_static_weak_min_cpa_ttc_sec", 1.5)))
            )
        )
        static_weak_cpa_ok = ~(static_other & weak_cpa)
    else:
        static_weak_cpa_ok = np.ones(len(df), dtype=bool)
    require_near_flag = bool(thresholds.get("learned_interaction_require_world_risk2_near_other_flag", False))
    if require_near_flag:
        near_ok = df["interaction_world_risk2_near_other_flag"].to_numpy(dtype=float) > 0
    else:
        near_ok = np.ones(len(df), dtype=bool)
    direct = (
        (pred >= int(thresholds.get("learned_interaction_promote_min_base_pred", 2)))
        & (pred < 3)
        & near_ok
        & (df["prob4_3"].to_numpy(dtype=float) >= float(thresholds.get("learned_interaction_min_prob4_3", 0.35)))
        & (df["prob_severity_2"].to_numpy(dtype=float) >= float(thresholds.get("learned_interaction_min_prob_severity_2", 0.30)))
        & (df["prob_binary_positive"].to_numpy(dtype=float) >= float(thresholds.get("learned_interaction_min_prob_binary_positive", 0.45)))
        & (quality_weighted_score >= float(thresholds.get("learned_interaction_min_cpa_quality_weighted_score", 0.08)))
        & (df["interaction_world_close_frame_ratio"].to_numpy(dtype=float) >= float(thresholds.get("learned_interaction_min_world_close_frame_ratio", 0.25)))
        & (quality_safe >= float(thresholds.get("learned_interaction_min_quality_safe_score", 0.45)))
        & edge_ok
        & dynamic_context_ok
        & static_weak_cpa_ok
        & (cpa_distance_ok | world_distance_ok)
        & (ttc_ok | cpa_score_ok)
    )
    return direct, quality_weighted_score


def build_interaction_world_direct_mask(df: pd.DataFrame, pred: np.ndarray, thresholds: dict[str, Any]) -> np.ndarray:
    return (
        (pred >= int(thresholds.get("interaction_world_promote_min_base_pred", 2)))
        & (pred < 3)
        & (df["interaction_world_risk2_near_other_flag"].to_numpy(dtype=float) > 0)
        & (df["prob4_3"].to_numpy(dtype=float) >= float(thresholds.get("interaction_world_min_prob4_3", 0.20)))
        & (df["prob_severity_2"].to_numpy(dtype=float) >= float(thresholds.get("interaction_world_min_prob_severity_2", 0.20)))
        & (
            df["interaction_world_min_distance_m"].to_numpy(dtype=float)
            <= float(thresholds.get("interaction_world_max_distance_m", 7.5))
        )
        & (
            df["interaction_world_close_frame_ratio"].to_numpy(dtype=float)
            >= float(thresholds.get("interaction_world_min_close_frame_ratio", 0.35))
        )
        & (
            df["interaction_world_quality_frame_ratio"].to_numpy(dtype=float)
            >= float(thresholds.get("interaction_world_min_quality_frame_ratio", 0.70))
        )
        & (
            df["interaction_world_jitter_frame_ratio"].to_numpy(dtype=float)
            <= float(thresholds.get("interaction_world_max_jitter_frame_ratio", 0.25))
        )
        & (
            df["interaction_world_common_frame_count"].to_numpy(dtype=float)
            >= float(thresholds.get("interaction_world_min_common_frame_count", 3))
        )
    )


def _parse_track_ids(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, float) and np.isnan(value):
        return set()
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return set()
    ids: set[int] = set()
    for item in text.replace(";", ",").split(","):
        token = item.strip()
        if not token or token.lower() in {"nan", "none", "<na>"}:
            continue
        try:
            ids.add(int(float(token)))
        except ValueError:
            continue
    return ids


def _format_track_ids(values: set[int]) -> str:
    return ",".join(str(value) for value in sorted(values))


def _join_int_ids(values: list[Any]) -> str:
    ids: set[int] = set()
    for value in values:
        try:
            if pd.notna(value):
                ids.add(int(value))
        except (TypeError, ValueError):
            continue
    return ",".join(str(value) for value in sorted(ids))


def _join_str_values(values: list[Any]) -> str:
    return ",".join(sorted({str(value) for value in values if pd.notna(value) and str(value)}))


def apply_severity_gate(
    df: pd.DataFrame,
    learned_pred: pd.Series,
    default_floor: pd.Series,
    config: dict[str, Any],
    *,
    soft: bool,
    promote_from_policy_floor: bool,
) -> tuple[np.ndarray, list[str]]:
    thresholds = config.get("thresholds", {})
    pred = np.maximum(learned_pred.astype(int).to_numpy(), default_floor.astype(int).to_numpy()).clip(0, 3)
    strong_mode = str(thresholds.get("severity_gate_strong_risk3_mode", "all_support"))
    medium_mode = str(thresholds.get("severity_gate_medium_risk2_mode", "all_support"))

    single_strong_risk3 = df["single_vehicle_floor_risk"].to_numpy(dtype=int) >= 3
    single_medium_risk2 = df["single_vehicle_floor_risk"].to_numpy(dtype=int) >= 2
    single_strict_strong_risk3 = (
        (df["single_vehicle_floor_risk"].to_numpy(dtype=int) >= 3)
        | (df["strict_policy_floor_risk"].to_numpy(dtype=int) >= 3)
    )
    single_strict_medium_risk2 = (
        (df["single_vehicle_floor_risk"].to_numpy(dtype=int) >= 2)
        | (df["strict_policy_floor_risk"].to_numpy(dtype=int) >= 2)
    )
    all_strong_risk3 = (
        single_strict_strong_risk3
        | (df["support_floor_risk"].to_numpy(dtype=int) >= 3)
        | (df["count_strong_risk3_support"].to_numpy(dtype=int) > 0)
    )
    all_medium_risk2 = (
        single_strict_medium_risk2
        | (df["support_floor_risk"].to_numpy(dtype=int) >= 2)
        | (df["count_medium_risk2_support"].to_numpy(dtype=int) > 0)
    )
    if strong_mode == "single_vehicle_only":
        strong_floor_risk3 = single_strong_risk3
    elif strong_mode == "single_strict_only":
        strong_floor_risk3 = single_strict_strong_risk3
    else:
        strong_floor_risk3 = all_strong_risk3
    if medium_mode == "single_vehicle_only":
        medium_floor_risk2 = single_medium_risk2
    elif medium_mode == "single_strict_only":
        medium_floor_risk2 = single_strict_medium_risk2
    else:
        medium_floor_risk2 = all_medium_risk2
    learned_soft_risk3 = (
        (df["prob4_3"].to_numpy(dtype=float) >= float(thresholds.get("soft_risk3_min_prob4_3", 0.70)))
        & (df["prob_severity_2"].to_numpy(dtype=float) >= float(thresholds.get("soft_risk3_min_prob_severity_2", 0.60)))
        & (
            (df["max_overspeed_event_score"].to_numpy(dtype=float) >= float(thresholds.get("soft_risk3_min_overspeed_event_score", 0.80)))
            | (df["max_redlight_event_score"].to_numpy(dtype=float) >= float(thresholds.get("soft_risk3_min_redlight_event_score", 0.80)))
            | (df["max_active_family_count"].to_numpy(dtype=float) >= float(thresholds.get("soft_risk3_min_active_family_count", 2.0)))
            | (df["max_cooccurrence_strength"].to_numpy(dtype=float) >= float(thresholds.get("soft_risk3_min_cooccurrence_strength", 0.40)))
        )
    )
    risk3_allowed = strong_floor_risk3 | (learned_soft_risk3 if soft else False)

    reasons: list[str] = []
    for idx in range(len(df)):
        reason = "unchanged"
        if pred[idx] >= 3 and not risk3_allowed[idx]:
            pred[idx] = 2 if medium_floor_risk2[idx] or learned_pred.iloc[idx] > 0 else 0
            reason = "risk3_capped_no_risk3_evidence"
        elif promote_from_policy_floor and pred[idx] < 3 and strong_floor_risk3[idx]:
            pred[idx] = 3
            reason = "risk3_promoted_by_floor"
        elif promote_from_policy_floor and pred[idx] < 2 and medium_floor_risk2[idx]:
            pred[idx] = 2
            reason = "risk2_promoted_by_floor"
        elif soft and learned_soft_risk3[idx] and pred[idx] < 3:
            pred[idx] = 3
            reason = "risk3_promoted_by_soft_learned_evidence"
        reasons.append(reason)
    return pred.astype(int), reasons


def evaluate_variants(predictions: pd.DataFrame, variants: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": "scene_risk.scene_hybrid_v1.metrics/v1",
        "row_count": int(len(predictions)),
        "variants": {},
    }
    for variant in variants:
        pred_col = f"scene_pred_{variant}"
        _require_columns(predictions, [pred_col], f"predictions[{variant}]")
        result["variants"][variant] = {
            "prediction_column": pred_col,
            "splits": {
                split: evaluate_split(predictions[predictions["split"].eq(split)].copy(), pred_col)
                for split in ["train", "val", "test"]
            },
        }
        result["variants"][variant]["splits"]["all"] = evaluate_split(predictions, pred_col)
    return result


def evaluate_split(df: pd.DataFrame, pred_col: str) -> dict[str, Any]:
    if df.empty:
        return _empty_split_metrics()
    y4 = df["y_4cls"].to_numpy(dtype=int)
    pred4 = df[pred_col].to_numpy(dtype=int).clip(0, 3)
    ybin = (y4 > 0).astype(int)
    pbin = (pred4 > 0).astype(int)
    ysev = original_to_severity(y4)
    psev = original_to_severity(pred4)
    return {
        "row_count": int(len(df)),
        "binary": binary_metrics(ybin, pbin),
        "four_class": multiclass_metrics(y4, pred4, LABELS_4CLS),
        "severity": multiclass_metrics(ysev, psev, LABELS_SEVERITY),
        "error_counts": error_counts(y4, pred4),
        "error_samples": error_samples(df, pred_col),
    }


def original_to_severity(values: np.ndarray) -> np.ndarray:
    out = np.zeros(values.shape[0], dtype=np.int64)
    out[values == 2] = 1
    out[values == 3] = 2
    return out


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> dict[str, Any]:
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for true, pred in zip(y_true.astype(int), y_pred.astype(int), strict=False):
        if int(true) in label_to_idx and int(pred) in label_to_idx:
            cm[label_to_idx[int(true)], label_to_idx[int(pred)]] += 1
    per_class: dict[str, dict[str, float | int]] = {}
    for label, idx in label_to_idx.items():
        tp = int(cm[idx, idx])
        fp = int(cm[:, idx].sum() - tp)
        fn = int(cm[idx, :].sum() - tp)
        support = int(cm[idx, :].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[str(label)] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": support,
        }
    return {
        "macro_f1": float(np.mean([per_class[str(label)]["f1"] for label in labels])),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def error_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    return {
        "false_positive_binary": int(((y_true == 0) & (y_pred > 0)).sum()),
        "false_negative_binary": int(((y_true > 0) & (y_pred == 0)).sum()),
        "risk1_to_0": int(((y_true == 1) & (y_pred == 0)).sum()),
        "risk2_to_0": int(((y_true == 2) & (y_pred == 0)).sum()),
        "risk3_to_0": int(((y_true == 3) & (y_pred == 0)).sum()),
        "risk2_to_3": int(((y_true == 2) & (y_pred == 3)).sum()),
        "risk3_to_2": int(((y_true == 3) & (y_pred == 2)).sum()),
        "normal_to_risk2_or_3": int(((y_true == 0) & (y_pred >= 2)).sum()),
    }


def error_samples(df: pd.DataFrame, pred_col: str, limit: int = 20) -> dict[str, list[dict[str, Any]]]:
    y = df["y_4cls"].astype(int)
    p = df[pred_col].astype(int)
    masks = {
        "false_positive_binary": (y == 0) & (p > 0),
        "false_negative_binary": (y > 0) & (p == 0),
        "risk2_to_3": (y == 2) & (p == 3),
        "risk3_to_2": (y == 3) & (p == 2),
    }
    cols = [
        "video_id",
        "scene_window_idx",
        "start_sec",
        "end_sec",
        "y_4cls",
        pred_col,
        "pred_4cls",
        "single_vehicle_floor_risk",
        "strict_policy_floor_risk",
        "support_floor_risk",
        "max_overspeed_event_score",
        "max_redlight_event_score",
        "max_active_family_count",
        "case_keys",
        "track_ids",
    ]
    out: dict[str, list[dict[str, Any]]] = {}
    for name, mask in masks.items():
        sample = df.loc[mask, cols].head(limit)
        out[name] = sample.to_dict("records")
    return out


def render_metrics_md(metrics: dict[str, Any]) -> str:
    lines = [
        "# SceneHybridV1 Metrics",
        "",
        f"- rows: {metrics['row_count']}",
        "",
        "## Summary",
        "",
        "| variant | split | rows | binary F1 | 4cls macro F1 | severity macro F1 | FP | FN | risk2->3 | risk3->2 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, variant_metrics in metrics["variants"].items():
        for split in ["train", "val", "test"]:
            item = variant_metrics["splits"][split]
            err = item["error_counts"]
            lines.append(
                f"| {variant} | {split} | {item['row_count']} | {item['binary']['f1']:.4f} | "
                f"{item['four_class']['macro_f1']:.4f} | {item['severity']['macro_f1']:.4f} | "
                f"{err['false_positive_binary']} | {err['false_negative_binary']} | {err['risk2_to_3']} | {err['risk3_to_2']} |"
            )

    lines.extend(["", "## Val/Test Per-Class F1", ""])
    for variant, variant_metrics in metrics["variants"].items():
        lines.append(f"### {variant}")
        lines.append("")
        lines.append("| split | class | precision | recall | F1 | support |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for split in ["val", "test"]:
            for cls, item in variant_metrics["splits"][split]["four_class"]["per_class"].items():
                lines.append(
                    f"| {split} | {cls} | {item['precision']:.4f} | {item['recall']:.4f} | {item['f1']:.4f} | {item['support']} |"
                )
        lines.append("")
    return "\n".join(lines) + "\n"


def _empty_split_metrics() -> dict[str, Any]:
    return {
        "row_count": 0,
        "binary": binary_metrics(np.asarray([], dtype=int), np.asarray([], dtype=int)),
        "four_class": multiclass_metrics(np.asarray([], dtype=int), np.asarray([], dtype=int), LABELS_4CLS),
        "severity": multiclass_metrics(np.asarray([], dtype=int), np.asarray([], dtype=int), LABELS_SEVERITY),
        "error_counts": error_counts(np.asarray([], dtype=int), np.asarray([], dtype=int)),
        "error_samples": {},
    }


def _require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"{name} missing columns: {missing}")


if __name__ == "__main__":
    main()
