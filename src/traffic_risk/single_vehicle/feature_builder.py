"""Build the semantic sidecars required by the frozen single-vehicle model.

This module is the public, path-independent form of the accepted v92/v93
feature preparation.  It deliberately builds trajectory context from upstream
windows and tracking CSV files; old predictions and evaluation labels are not
inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .pipeline import PolicyInputs, normalize_keys
from .policies.lane_assignment import build_clean_v28_lane_sidecar
from .policies.lane_candidates import build_clean_v28_frame_lane_candidate_cache
from .policies.lane_family_features import build_clean_v41_lane_family_sidecar
from .policies.redlight_zone_features import build_clean_redlight_zone_movement_sidecar
from .policies.tail_lane_features import build_clean_v48_tail_lane_sidecar
from .policies.trajectory_context import (
    KEY_COLUMNS,
    V24F_SIDECAR_COLUMNS,
    V24F_TEXT_COLUMNS,
    add_context_features,
    build_window_lane_sidecar,
    load_lanes,
)
from .policies.trajectory_semantics import (
    HISTORICAL_V37_SIDECAR_COLUMNS,
    build_clean_v37_trajectory_sidecar,
    overwrite_historical_v37_trajectory_aliases,
)
from .policies.wrongway_tail import build_v88_last5_sidecar


@dataclass(frozen=True)
class SemanticFeatureBundle:
    """Model-ready windows plus every sidecar used by the policy chain."""

    model_features: pd.DataFrame
    policy_inputs: PolicyInputs
    frame_cache_summary: dict[str, int]


def _strict_v24f_base(c4o_features: pd.DataFrame, trajectory_features: pd.DataFrame) -> pd.DataFrame:
    """Create the fold-clean trajectory-context input used by formal v93."""

    labels = normalize_keys(c4o_features)
    label_keep = KEY_COLUMNS + [
        "start_sec",
        "end_sec",
        "redlight_event_score",
        "overspeed_event_score",
        "speed_kmh_p95",
        "s33_current_local_semantic_support",
        "s33_prefix_state_confirmed",
    ]
    missing = [column for column in KEY_COLUMNS + ["start_sec", "end_sec"] if column not in labels.columns]
    if missing:
        raise ValueError(f"upstream feature table is missing columns: {missing}")
    base = labels[[column for column in label_keep if column in labels.columns]].copy()
    # Formal v93 uses a case-level causal order.  It does not leak the original
    # train/validation/test split into semantic feature generation.
    base["split"] = "all"

    sidecar = normalize_keys(trajectory_features)
    side_keep = KEY_COLUMNS + [
        "start_sec",
        "end_sec",
        "trajectory_subtype_primary",
        "trajectory_subtype_peak",
        "lane_family_switch_count",
        "opposite_lane_run_length",
        "opposite_lane_occupancy_ratio",
        "reversed_flag",
        "artifact_bbox_area_cv",
        "artifact_max_center_step",
        "osc_h20_score",
        "osc_h30_score",
        "osc_h20_lateral_sign_change_count",
        "osc_h30_lateral_sign_change_count",
        "osc_h20_two_sided_ratio",
        "osc_h30_two_sided_ratio",
    ]
    missing = [column for column in KEY_COLUMNS if column not in sidecar.columns]
    if missing:
        raise ValueError(f"trajectory feature table is missing columns: {missing}")
    sidecar = sidecar[[column for column in side_keep if column in sidecar.columns]].copy()
    merged = base.merge(sidecar, on=KEY_COLUMNS, how="left", suffixes=("", "_trajectory"), validate="one_to_one")
    if "start_sec_trajectory" in merged.columns:
        merged["start_sec"] = merged["start_sec"].fillna(merged["start_sec_trajectory"])
        merged["end_sec"] = merged["end_sec"].fillna(merged["end_sec_trajectory"])
        merged = merged.drop(columns=["start_sec_trajectory", "end_sec_trajectory"])
    return merged


def build_strict_trajectory_context(
    c4o_features: pd.DataFrame,
    trajectory_features: pd.DataFrame,
    *,
    timestamp_dir: Path,
    lane_map: Path,
) -> pd.DataFrame:
    """Build formal v93 trajectory context without historical predictions."""

    base = _strict_v24f_base(c4o_features, trajectory_features)
    lane_sidecar = normalize_keys(
        build_window_lane_sidecar(
            base[KEY_COLUMNS + ["start_sec", "end_sec"]],
            timestamp_dir,
            load_lanes(lane_map),
        )
    )
    merged = base.merge(lane_sidecar, on=KEY_COLUMNS, how="left", validate="one_to_one")
    merged = add_context_features(
        merged.sort_values(
            ["case_key", "video_id", "track_id", "ts_window_idx"],
            kind="mergesort",
        ).reset_index(drop=True)
    )
    for column in V24F_SIDECAR_COLUMNS:
        if column not in merged.columns:
            merged[column] = "" if column in V24F_TEXT_COLUMNS else 0.0
    forbidden = {"split", "pred_v22_candidate", "prob_class0", "prob_class1", "prob_class2", "y_schemaC"}
    keep = [
        *KEY_COLUMNS,
        "start_sec",
        "end_sec",
        *[
            column
            for column in V24F_SIDECAR_COLUMNS
            if column not in forbidden and column not in KEY_COLUMNS and column not in {"start_sec", "end_sec"}
        ],
    ]
    return merged[keep].copy()


def build_double_yellow_features(trajectory_features: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the accepted cautious double-yellow pre-confirm signal."""

    source = normalize_keys(trajectory_features)

    def number(column: str, default: float = 0.0) -> pd.Series:
        if column not in source.columns:
            return pd.Series(default, index=source.index, dtype=float)
        return pd.to_numeric(source[column], errors="coerce").fillna(default).astype(float)

    contact = (
        number("window_distance_to_double_yellow_min", 999.0).le(20.0)
        & number("selected_lane_known", 1.0).eq(0.0)
        & number("flow_cos").gt(0.90)
        & number("flow_conflict_score").lt(0.20)
        & number("artifact_bbox_area_cv").lt(0.25)
        & number("artifact_max_center_step").lt(10.0)
        & number("tracking_quality_bad").eq(0.0)
    )
    out = source[KEY_COLUMNS].copy()
    out["trajectory_double_yellow_contact_preconfirm_v71"] = contact.astype(int)
    out["trajectory_double_yellow_contact_preconfirm_risk2_support_v71"] = contact.astype(float)
    out["v72_double_yellow_contact_preconfirm_risk2_applied"] = contact.astype(float)
    out["trajectory_double_yellow_distance_min_v71"] = number("window_distance_to_double_yellow_min", 999.0)
    out["trajectory_double_yellow_cross_count_v71"] = number("window_double_yellow_cross_count")
    out["trajectory_double_yellow_crossed_once_v71"] = number("window_double_yellow_crossed_once")
    out["trajectory_double_yellow_contact_reason_v71"] = "none"
    out.loc[contact, "trajectory_double_yellow_contact_reason_v71"] = "near_unknown_lane_stable_quality"
    return out.drop_duplicates(KEY_COLUMNS, keep="last")


def _model_features(c4o_features: pd.DataFrame, trajectory_sidecar: pd.DataFrame) -> pd.DataFrame:
    base = normalize_keys(c4o_features)
    aliases = normalize_keys(overwrite_historical_v37_trajectory_aliases(trajectory_sidecar))
    feature_columns = list(HISTORICAL_V37_SIDECAR_COLUMNS.values())
    payload = aliases[KEY_COLUMNS + feature_columns].drop_duplicates(KEY_COLUMNS, keep="last")
    base = base.drop(columns=[column for column in feature_columns if column in base.columns], errors="ignore")
    return base.merge(payload, on=KEY_COLUMNS, how="left", validate="one_to_one")


def build_semantic_feature_bundle(
    *,
    c4o_features: pd.DataFrame,
    trajectory_features: pd.DataFrame,
    timestamp_dir: Path,
    lane_map: Path,
    stop_lines: Path,
    cache_dir: Path,
    overwrite_cache: bool = False,
) -> SemanticFeatureBundle:
    """Build all frozen single-vehicle features from current-run evidence."""

    source = normalize_keys(c4o_features)
    trajectory = normalize_keys(trajectory_features)
    required_paths = [timestamp_dir, lane_map, stop_lines]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"semantic feature inputs do not exist: {missing_paths}")

    frame_cache = cache_dir / "frame_lane_candidates"
    summary = build_clean_v28_frame_lane_candidate_cache(
        source[["video_id", "track_id"]].drop_duplicates(),
        lane_map=lane_map,
        timestamp_dir=timestamp_dir,
        cache_dir=frame_cache,
        overwrite=overwrite_cache,
    )
    lane = build_clean_v28_lane_sidecar(source, lane_map=lane_map, frame_cache_dir=frame_cache)
    redlight_zone = build_clean_redlight_zone_movement_sidecar(
        source,
        source_dir=timestamp_dir,
        stopline_json=stop_lines,
    )
    trajectory_context = build_strict_trajectory_context(
        source,
        trajectory,
        timestamp_dir=timestamp_dir,
        lane_map=lane_map,
    )
    clean_trajectory = build_clean_v37_trajectory_sidecar(
        source,
        v28_sidecar=lane,
        v24f_sidecar=trajectory_context,
        redlight_zone_sidecar=redlight_zone,
    )
    lane_family = build_clean_v41_lane_family_sidecar(source, cache_dir=frame_cache, lane_map=lane_map)
    tail_lane = build_clean_v48_tail_lane_sidecar(source, lane, frame_cache_dir=frame_cache)
    wrongway_tail = build_v88_last5_sidecar(source, cache_dir=frame_cache, lane_map=lane_map)
    double_yellow = build_double_yellow_features(trajectory)

    policy_inputs = PolicyInputs(
        source_features=source,
        trajectory_features=clean_trajectory,
        lane_features=lane,
        lane_family_features=lane_family,
        redlight_zone_features=redlight_zone,
        tail_lane_features=tail_lane,
        wrongway_tail_features=wrongway_tail,
        double_yellow_features=double_yellow,
    )
    return SemanticFeatureBundle(
        model_features=_model_features(source, clean_trajectory),
        policy_inputs=policy_inputs,
        frame_cache_summary=summary,
    )


def write_semantic_feature_bundle(bundle: SemanticFeatureBundle, output_dir: Path) -> None:
    """Write inspectable intermediates without making them runtime inputs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    bundle.model_features.to_csv(output_dir / "model_features.csv", index=False)
    sidecars = {
        "source_features.csv": bundle.policy_inputs.source_features,
        "trajectory_features.csv": bundle.policy_inputs.trajectory_features,
        "lane_features.csv": bundle.policy_inputs.lane_features,
        "lane_family_features.csv": bundle.policy_inputs.lane_family_features,
        "redlight_zone_features.csv": bundle.policy_inputs.redlight_zone_features,
        "tail_lane_features.csv": bundle.policy_inputs.tail_lane_features,
        "wrongway_tail_features.csv": bundle.policy_inputs.wrongway_tail_features,
        "double_yellow_features.csv": bundle.policy_inputs.double_yellow_features,
    }
    for filename, frame in sidecars.items():
        frame.to_csv(output_dir / filename, index=False)


__all__ = [
    "SemanticFeatureBundle",
    "build_double_yellow_features",
    "build_semantic_feature_bundle",
    "build_strict_trajectory_context",
    "write_semantic_feature_bundle",
]
