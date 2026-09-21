from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


SOURCE_STATUS_IMPLEMENTED = "implemented"
SOURCE_STATUS_SIDECAR_ONLY = "sidecar_only"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    dtype: str
    min_value: float | int | None
    max_value: float | int | None
    missing_policy: str
    normalization: str
    capping: str
    source_status: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


JOIN_KEY_FIELDNAMES = (
    "case_key",
    "source_type",
    "source_id",
    "video_id",
    "track_id",
    "ts_window_idx",
    "start_sec",
    "end_sec",
)

MODULE_V0_FIELD_ORDER = (
    "redlight_confirmed",
    "redlight_score",
    "redlight_signal_valid",
    "redlight_failure_reason",
    "tl_state_window",
    "tl_red_ratio",
    "tl_prob_red_mean",
    "tl_prob_red_max",
    "stopline_crossed_confirmed",
    "forward_motion_after_stopline",
    "redlight_event_candidate",
)

GRU_V0_FEATURE_ORDER = (
    "redlight_confirmed",
    "redlight_score",
    "redlight_duration_sec_proxy",
    "redlight_signal_valid",
)

MODULE_V0_SIDECAR_FIELD_ORDER = (
    "source_csv",
    "timestamp_column_used",
    "fps",
    "window_point_count",
    "timestamp_valid",
    "tracking_geometry_valid",
    "tl_state_sequence",
    "tl_valid_state_count",
    "tl_known_state_count",
    "tl_frame_count",
    "line_side_value_start",
    "line_side_value_end",
    "line_side_value_min",
    "line_side_value_max",
    "signed_distance_px_start",
    "signed_distance_px_end",
    "signed_distance_px_min",
    "signed_distance_px_max",
    "stopline_id_start",
    "stopline_id_end",
    "dominant_stopline_id",
    "crossing_stopline_id",
    "stopline_id_unique_count",
    "stopline_id_switch_count",
    "stopline_multi_candidate_flag",
    "stopline_identity_confidence",
    "crossing_event_in_window",
    "crossing_frame",
    "crossing_timestamp_sec",
    "crossing_red_signal",
    "motion_after_crossing_px",
    "motion_after_crossing_world",
    "bbox_bottom_center_x",
    "bbox_bottom_center_y",
    "stopline_geometry_source",
    "stopline_geometry_valid",
    "primary_evidence_summary",
)

CONFIRMED_VID_TID_FIELD_ORDER = (
    "video_id",
    "track_id",
    "case_key",
    "first_confirmed_ts_window_idx",
    "first_confirmed_start_sec",
    "first_confirmed_end_sec",
    "last_confirmed_start_sec",
    "last_confirmed_end_sec",
    "confirmed_window_count",
    "max_redlight_score",
    "max_redlight_duration_sec_proxy",
    "tl_red_ratio_at_first_confirmed",
    "stopline_crossed_confirmed",
    "forward_motion_after_stopline",
    "redlight_signal_valid",
    "source_csv",
    "primary_evidence_summary",
)

FORBIDDEN_LEAKAGE_COLUMNS = (
    "label",
    "final_risk",
    "tcn_pred",
    "probs",
    "rl_floor_applied",
    "rl_floor",
    "redlight_violation",
    "audit_verdict",
    "false_positive_flag",
    "false_negative_flag",
)

GRU_FORBIDDEN_RAW_EVIDENCE_COLUMNS = (
    "tl_state",
    "tl_state_window",
    "tl_prob_red",
    "tl_prob_green",
    "tl_red_ratio",
    "tl_prob_red_mean",
    "tl_prob_red_max",
    "crossed_stop_line",
    "line_side_value",
    "raw_bbox_geometry",
    "bbox_bottom_center_x",
    "bbox_bottom_center_y",
    "raw_pixel_movement",
    "stopline_geometry",
    "stopline_geometry_source",
    "redlight_violation",
)

FEATURE_SPECS = (
    FeatureSpec(
        name="redlight_confirmed",
        dtype="int8",
        min_value=0,
        max_value=1,
        missing_policy="fill_zero",
        normalization="identity",
        capping="clip[0,1]",
        source_status=SOURCE_STATUS_IMPLEMENTED,
        notes="Event-type semantic confirmation from red signal, stop-line crossing, forward motion, and validity gates.",
    ),
    FeatureSpec(
        name="redlight_score",
        dtype="float32",
        min_value=0.0,
        max_value=1.0,
        missing_policy="fill_zero",
        normalization="identity",
        capping="clip[0.0,1.0]",
        source_status=SOURCE_STATUS_IMPLEMENTED,
        notes="Normalized semantic score, not a direct tl_prob_red passthrough.",
    ),
    FeatureSpec(
        name="redlight_duration_sec_proxy",
        dtype="float32",
        min_value=0.0,
        max_value=6.0,
        missing_policy="fill_zero",
        normalization="divide_by_6_then_clip01",
        capping="clip[0.0,6.0]",
        source_status=SOURCE_STATUS_IMPLEMENTED,
        notes="Causal consecutive confirmed red-light windows times stride_sec, capped at 6 seconds.",
    ),
    FeatureSpec(
        name="redlight_signal_valid",
        dtype="int8",
        min_value=0,
        max_value=1,
        missing_policy="fill_zero",
        normalization="identity",
        capping="clip[0,1]",
        source_status=SOURCE_STATUS_IMPLEMENTED,
        notes="Quality gate covering traffic-light state, stop-line geometry, timestamp, and tracking geometry validity.",
    ),
    FeatureSpec(
        name="tl_state_window",
        dtype="str",
        min_value=None,
        max_value=None,
        missing_policy="unknown",
        normalization="sidecar_only",
        capping="none",
        source_status=SOURCE_STATUS_SIDECAR_ONLY,
        notes="Module evidence field only; keep out of GRU main input.",
    ),
)

