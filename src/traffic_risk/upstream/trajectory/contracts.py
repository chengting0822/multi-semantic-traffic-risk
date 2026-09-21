"""Stable contracts shared by the trajectory materializer and semantic stages."""

WINDOW_SEC = 0.6667
STRIDE_SEC = 0.3333
SAMPLE_COUNT = 20
POLICY_ID = "window06667_stride03333_s3_fixed20"

KEY_COLUMNS = ["case_key", "source_type", "source_id", "video_id", "track_id", "ts_window_idx"]
FEATURE_COLUMNS = [
    "cx", "cy", "w", "h", "vx", "vy", "speed", "ax", "ay",
    "dw", "dh", "heading_change", "aspect_ratio", "conf",
]


def stat_feature_names() -> list[str]:
    names: list[str] = []
    for column in FEATURE_COLUMNS:
        names.extend([f"{column}_mean", f"{column}_std"])
        if column in {"vx", "vy", "speed", "ax", "ay", "dw", "dh", "heading_change"}:
            names.append(f"{column}_absmax")
    return [*names, "total_displacement", "conf_min"]


OFFICIAL_40D_FEATURES = [*stat_feature_names(), "heading_cos_flow", "perspective_size_ratio"]
B5_FEATURES = [
    *OFFICIAL_40D_FEATURES,
    "flow_cos",
    "reversed_flag",
    "opposite_lane_occupancy_ratio",
    "opposite_lane_after_crossing_run_length",
    "window_double_yellow_crossed_once",
    "window_distance_to_double_yellow_min",
    "tracking_quality_bad",
]

TRAJECTORY_OUTPUT_COLUMNS = [
    "trajectory_wrong_way_flag", "trajectory_wrong_way_score", "trajectory_wrong_way_confidence",
    "trajectory_wrong_way_duration", "trajectory_weaving_flag", "trajectory_weaving_score",
    "trajectory_weaving_confidence", "trajectory_weaving_duration",
    "trajectory_double_yellow_crossing_flag", "trajectory_double_yellow_crossing_score",
    "trajectory_double_yellow_crossing_confidence", "trajectory_double_yellow_crossing_duration",
    "trajectory_opposite_lane_driving_flag", "trajectory_opposite_lane_driving_score",
    "trajectory_opposite_lane_driving_confidence", "trajectory_opposite_lane_driving_duration",
    "trajectory_illegal_flow_direction_flag", "trajectory_illegal_flow_direction_score",
    "trajectory_illegal_flow_direction_confidence", "trajectory_illegal_flow_direction_duration",
    "trajectory_heading_change_peak", "trajectory_lateral_drift_score",
    "trajectory_reverse_like_motion_score", "trajectory_abrupt_stop_go_score",
    "trajectory_cut_in_score", "trajectory_subtype_peak", "trajectory_subtype_entropy",
    "trajectory_subtype_switch_count", "trajectory_repeated_severe_burst_count",
    "trajectory_post_peak_persistence",
]

OSCILLATION_HORIZONS_SEC = {"h20": 2.0, "h30": 3.0}
OSCILLATION_METRICS = [
    "sample_count", "disp_px", "amp_px", "two_sided_amp_px", "two_sided_ratio",
    "lateral_sign_change_count", "velocity_sign_change_count", "area_cv", "conf_min", "score",
]
OSCILLATION_COLUMNS = [
    f"osc_{horizon}_{metric}"
    for horizon in OSCILLATION_HORIZONS_SEC
    for metric in OSCILLATION_METRICS
]
