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
    "speed_kmh",
    "speed_limit_kmh",
    "overspeed_ratio",
    "overspeed_confirmed",
    "overspeed_level",
    "overspeed_reliability",
    "overspeed_failure_reason",
    "timestamp_delta_valid",
    "ipm_source_valid",
    "track_motion_valid",
)

GRU_V0_FEATURE_ORDER = (
    "overspeed_score",
    "overspeed_level",
    "overspeed_duration_sec_proxy",
    "overspeed_reliability",
)

MODULE_V0_SIDECAR_FIELD_ORDER = (
    "source_csv",
    "timestamp_column_used",
    "fps",
    "window_point_count",
    "window_valid_speed_count",
    "window_valid_pair_count",
    "window_valid_speed_ratio",
    "window_valid_pair_ratio",
    "window_ipm_valid_ratio",
    "window_extreme_speed_count",
    "timestamp_delta_sec_p50",
    "timestamp_delta_sec_p95",
    "ground_distance_m_p95",
    "ipm_x_last",
    "ipm_y_last",
    "bbox_bottom_center_x_last",
    "bbox_bottom_center_y_last",
    "speed_raw_kmh_p95",
    "speed_smoothed_kmh_p95",
    "track_display_status",
    "track_failure_reason",
)

FORBIDDEN_LEAKAGE_COLUMNS = (
    "label",
    "final_risk",
    "tcn_pred",
    "probs",
    "overspeed_floor_applied",
    "audit_verdict",
    "false_positive_flag",
    "false_negative_flag",
)

FEATURE_SPECS = (
    FeatureSpec(
        name="overspeed_score",
        dtype="float32",
        min_value=0.0,
        max_value=1.0,
        missing_policy="fill_zero",
        normalization="identity",
        capping="clip[0.0,1.0]",
        source_status=SOURCE_STATUS_IMPLEMENTED,
        notes="Normalized semantic score using clip01((speed_kmh - 70) / 40).",
    ),
    FeatureSpec(
        name="overspeed_level",
        dtype="int8",
        min_value=0,
        max_value=3,
        missing_policy="fill_zero",
        normalization="divide_by_3_then_clip01",
        capping="clip[0,3]",
        source_status=SOURCE_STATUS_IMPLEMENTED,
        notes="Uses v0 bins: <70=>0, 70-<90=>1, 90-<=110=>2, >110=>3.",
    ),
    FeatureSpec(
        name="overspeed_duration_sec_proxy",
        dtype="float32",
        min_value=0.0,
        max_value=6.0,
        missing_policy="fill_zero",
        normalization="divide_by_6_then_clip01",
        capping="clip[0.0,6.0]",
        source_status=SOURCE_STATUS_IMPLEMENTED,
        notes="Causal consecutive overspeed windows times stride_sec, capped at 6 seconds.",
    ),
    FeatureSpec(
        name="overspeed_reliability",
        dtype="float32",
        min_value=0.0,
        max_value=1.0,
        missing_policy="fill_zero",
        normalization="identity",
        capping="clip[0.0,1.0]",
        source_status=SOURCE_STATUS_IMPLEMENTED,
        notes="Quality gate derived from valid speed coverage and timestamp confidence.",
    ),
    FeatureSpec(
        name="overspeed_confirmed",
        dtype="int8",
        min_value=0,
        max_value=1,
        missing_policy="fill_zero",
        normalization="identity",
        capping="clip[0,1]",
        source_status=SOURCE_STATUS_SIDECAR_ONLY,
        notes="Module output field and sidecar-friendly confirmation; keep out of GRU main input.",
    ),
)
