"""Accepted trajectory-v8 formulas operating on current-run fixed20 evidence."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .contracts import KEY_COLUMNS as WINDOW_JOIN_KEYS
from .contracts import OSCILLATION_COLUMNS as OSCILLATION_SIDECAR_COLUMNS
from .contracts import OSCILLATION_HORIZONS_SEC, TRAJECTORY_OUTPUT_COLUMNS


def numeric_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype(float)


def binary_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return (numeric_series(frame, column) >= 0.5).astype(float)


def clip_nonnegative(series: pd.Series) -> pd.Series:
    return series.astype(float).clip(lower=0.0)


def clip01(series: pd.Series) -> pd.Series:
    return series.astype(float).clip(lower=0.0, upper=1.0)


def safe_divide(numerator: Any, denominator: Any, default: float = 0.0) -> pd.Series:
    left = numerator if isinstance(numerator, pd.Series) else pd.Series(numerator)
    right = denominator if isinstance(denominator, pd.Series) else pd.Series(denominator, index=left.index)
    right_values = right.to_numpy(dtype=float)
    values = np.divide(left.to_numpy(dtype=float), right_values, out=np.full(len(left), default), where=np.abs(right_values) > 1e-12)
    return pd.Series(values, index=left.index, dtype=float)


def sigmoid_clip(series: pd.Series, *, center: float, scale: float) -> pd.Series:
    values = np.clip((series.astype(float) - center) / max(float(scale), 1e-6), -60.0, 60.0)
    return pd.Series(1.0 / (1.0 + np.exp(-values)), index=series.index, dtype=float)


def window_duration(frame: pd.DataFrame) -> pd.Series:
    return clip_nonnegative(numeric_series(frame, "end_sec") - numeric_series(frame, "start_sec"))


def approx_sample_dt(frame: pd.DataFrame) -> pd.Series:
    return safe_divide(window_duration(frame), numeric_series(frame, "sample_count", 20.0).clip(lower=1.0), default=0.0)


def entropy_from_scores(score_frame: pd.DataFrame) -> pd.Series:
    values = np.clip(score_frame.to_numpy(dtype=float), 0.0, None)
    totals = values.sum(axis=1, keepdims=True)
    probabilities = np.divide(values, totals, out=np.zeros_like(values), where=totals > 1e-9)
    safe = np.where(probabilities > 1e-9, probabilities, 1.0)
    entropy = -np.sum(np.where(probabilities > 1e-9, probabilities * np.log(safe), 0.0), axis=1)
    return pd.Series(entropy / max(math.log(values.shape[1]), 1e-9), index=score_frame.index).fillna(0.0)


def _count_sign_changes(values: np.ndarray, deadband: float) -> int:
    signs = np.where(values > deadband, 1, np.where(values < -deadband, -1, 0))
    signs = signs[signs != 0]
    return int(np.count_nonzero(signs[1:] != signs[:-1])) if signs.size >= 2 else 0


def _sigmoid_scalar(value: float, center: float, scale: float) -> float:
    return float(1.0 / (1.0 + np.exp(-(float(value) - center) / max(scale, 1e-6))))


def compute_oscillation_metrics(sample_slice: pd.DataFrame) -> dict[str, float]:
    if sample_slice.shape[0] < 4:
        return {"sample_count": float(len(sample_slice)), "disp_px": 0.0, "amp_px": 0.0, "two_sided_amp_px": 0.0, "two_sided_ratio": 0.0, "lateral_sign_change_count": 0.0, "velocity_sign_change_count": 0.0, "area_cv": 1.0, "conf_min": 0.0, "score": 0.0}
    x = sample_slice["interpolated_center_x"].to_numpy(dtype=float); y = sample_slice["interpolated_center_y"].to_numpy(dtype=float)
    area = sample_slice["interpolated_area"].to_numpy(dtype=float); conf = sample_slice["interpolated_confidence"].to_numpy(dtype=float)
    dx, dy = x[-1] - x[0], y[-1] - y[0]; disp = float(np.hypot(dx, dy))
    residual = y - y[0] if disp < 1e-6 else ((x - x[0]) * (-dy) + (y - y[0]) * dx) / disp
    amp = float(np.nanmax(residual) - np.nanmin(residual)); positive = max(float(np.nanmax(residual)), 0.0); negative = max(-float(np.nanmin(residual)), 0.0)
    two_sided = 2.0 * min(positive, negative); ratio = float(np.clip(two_sided / amp, 0.0, 1.0)) if amp > 1e-6 else 0.0
    deadband = max(disp / 40.0, 1.0); lateral = _count_sign_changes(residual, deadband); velocity = _count_sign_changes(np.diff(residual), max(deadband * 0.5, 0.5))
    finite_area = area[np.isfinite(area) & (area > 0.0)]; area_cv = float(np.std(finite_area) / max(np.mean(finite_area), 1e-6)) if finite_area.size else 1.0
    finite_conf = conf[np.isfinite(conf)]; conf_min = float(np.min(finite_conf)) if finite_conf.size else 0.0
    score = float(np.clip(0.35 * _sigmoid_scalar(two_sided, 20.0, 6.0) + 0.25 * ratio + 0.25 * min(lateral / 2.0, 1.0) + 0.15 * min(velocity / 2.0, 1.0), 0.0, 1.0))
    if area_cv > 0.55: score *= float(np.clip((0.75 - area_cv) / 0.20, 0.0, 1.0))
    if conf_min < 0.15: score *= float(np.clip(conf_min / 0.15, 0.0, 1.0))
    return {"sample_count": float(len(sample_slice)), "disp_px": disp, "amp_px": amp, "two_sided_amp_px": two_sided, "two_sided_ratio": ratio, "lateral_sign_change_count": float(lateral), "velocity_sign_change_count": float(velocity), "area_cv": area_cv, "conf_min": conf_min, "score": score}


def normalize_window_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in ["case_key", "source_type", "source_id", "video_id", "track_id"]:
        if column in out.columns:
            out[column] = out[column].astype("string").fillna("")
    if "window_id" in out.columns and "ts_window_idx" not in out.columns:
        out = out.rename(columns={"window_id": "ts_window_idx"})
    if "ts_window_idx" in out.columns:
        out["ts_window_idx"] = pd.to_numeric(out["ts_window_idx"], errors="coerce").fillna(-1).astype(int)
    for column in ["start_sec", "end_sec", "sample_timestamp_sec", "interpolated_center_x", "interpolated_center_y", "interpolated_area", "interpolated_confidence"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def build_oscillation_sidecar_from_samples(
    *,
    window_frame: pd.DataFrame,
    sample_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Build v8 oscillation sidecar rows from in-memory fixed20 samples.

    Parameters
    ----------
    window_frame:
        Rows keyed by `WINDOW_JOIN_KEYS` and containing `start_sec` / `end_sec`.
        This is typically the dirty-tail window set that needs recomputation.
    sample_frame:
        Fixed20 sample rows for the same tracks.  The frame must include enough
        history before each `end_sec`; for the current v8 contract that means at
        least 3 seconds for `osc_h30_*`.
    """

    if window_frame.empty:
        return pd.DataFrame(columns=WINDOW_JOIN_KEYS + OSCILLATION_SIDECAR_COLUMNS)

    output = normalize_window_keys(window_frame[WINDOW_JOIN_KEYS + ["start_sec", "end_sec"]]).copy()
    for column in OSCILLATION_SIDECAR_COLUMNS:
        output[column] = 1.0 if column.endswith("_area_cv") else 0.0

    samples = normalize_window_keys(sample_frame)
    if samples.empty:
        return output[WINDOW_JOIN_KEYS + OSCILLATION_SIDECAR_COLUMNS].copy()
    window_key_set = set(map(tuple, output[WINDOW_JOIN_KEYS].itertuples(index=False, name=None)))
    if window_key_set:
        sample_mask = [tuple(row) in window_key_set for row in samples[WINDOW_JOIN_KEYS].itertuples(index=False, name=None)]
        samples = samples.loc[sample_mask].copy()
    if "valid_sample" in samples.columns:
        numeric_valid = pd.to_numeric(samples["valid_sample"], errors="coerce")
        text_valid = samples["valid_sample"].astype("string").str.strip().str.lower().isin({"true", "1", "1.0", "yes"})
        valid = numeric_valid.fillna(0.0).ge(0.5) | text_valid
        samples = samples.loc[valid].copy()
    samples = samples.dropna(subset=["sample_timestamp_sec", "interpolated_center_x", "interpolated_center_y"])
    if samples.empty:
        return output[WINDOW_JOIN_KEYS + OSCILLATION_SIDECAR_COLUMNS].copy()

    track_keys = ["case_key", "source_type", "source_id", "video_id", "track_id"]
    sample_groups = {
        key: group.sort_values("sample_timestamp_sec", kind="mergesort").reset_index(drop=True)
        for key, group in samples.groupby(track_keys, sort=False)
    }
    for key, window_group in output.groupby(track_keys, sort=False):
        sample_group = sample_groups.get(key)
        if sample_group is None or sample_group.empty:
            continue
        times = sample_group["sample_timestamp_sec"].to_numpy(dtype=float)
        ordered_windows = window_group.sort_values(["ts_window_idx", "start_sec"], kind="mergesort")
        for row in ordered_windows.itertuples(index=True):
            end_sec = float(row.end_sec)
            for horizon_name, horizon_sec in OSCILLATION_HORIZONS_SEC.items():
                left = int(np.searchsorted(times, end_sec - float(horizon_sec), side="left"))
                right = int(np.searchsorted(times, end_sec, side="right"))
                metrics = compute_oscillation_metrics(sample_group.iloc[left:right])
                for metric_name, metric_value in metrics.items():
                    output.at[row.Index, f"osc_{horizon_name}_{metric_name}"] = float(metric_value)

    return output[WINDOW_JOIN_KEYS + OSCILLATION_SIDECAR_COLUMNS].copy()


def build_trajectory_v8_frames_from_merged(
    *,
    merged_frame: pd.DataFrame,
    geometry_source_path: str = "",
    geometry_source_present: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute trajectory v8 main and sidecar frames from an already merged frame.

    The formula body mirrors `features.build_trajectory_policy_outputs_v8.main`.
    Keep this function narrow and parity-tested; it is the bridge toward a
    stateful trajectory v8 calculator that avoids the CSV builder.
    """

    merged = normalize_window_keys(merged_frame).copy()
    if merged.empty:
        return (
            pd.DataFrame(columns=WINDOW_JOIN_KEYS + ["start_sec", "end_sec"] + TRAJECTORY_OUTPUT_COLUMNS),
            pd.DataFrame(columns=WINDOW_JOIN_KEYS + ["start_sec", "end_sec", *OSCILLATION_SIDECAR_COLUMNS]),
        )
    for column in OSCILLATION_SIDECAR_COLUMNS:
        if column not in merged.columns:
            merged[column] = 1.0 if column.endswith("_area_cv") else 0.0
    for field in TRAJECTORY_OUTPUT_COLUMNS:
        merged[field] = 0.0

    duration_sec = window_duration(merged)
    sample_dt = approx_sample_dt(merged)
    traj_reliability = clip01(numeric_series(merged, "trajectory_anomaly_reliability", default=0.5))
    semantic_support = clip01(numeric_series(merged, "semantic_support_flag"))
    selected_lane_known = binary_series(merged, "selected_lane_known")
    tracking_quality_bad = clip01(numeric_series(merged, "tracking_quality_bad"))
    tracking_quality_good = 1.0 - tracking_quality_bad

    flow_cos = numeric_series(merged, "flow_cos")
    occ_present = "occupied_lane_flow_cos" in merged.columns and pd.to_numeric(
        merged["occupied_lane_flow_cos"], errors="coerce"
    ).notna().any()
    if occ_present:
        occupied_lane_flow_cos = pd.to_numeric(merged["occupied_lane_flow_cos"], errors="coerce").fillna(flow_cos)
        reversed_flag = (occupied_lane_flow_cos < -0.5).astype(float)
        flow_conflict_score = clip01(1.0 - occupied_lane_flow_cos)
        negative_flow_margin = clip_nonnegative(-occupied_lane_flow_cos)
    else:
        occupied_lane_flow_cos = flow_cos.copy()
        reversed_flag = binary_series(merged, "reversed_flag")
        flow_conflict_score = clip01(numeric_series(merged, "flow_conflict_score"))
        negative_flow_margin = clip_nonnegative(numeric_series(merged, "negative_flow_margin"))
    merged["reversed_flag"] = reversed_flag.values

    opposite_lane_occupancy_ratio = clip01(numeric_series(merged, "opposite_lane_occupancy_ratio"))
    opposite_lane_run_length = clip_nonnegative(numeric_series(merged, "opposite_lane_run_length"))
    opposite_lane_after_crossing_flag = binary_series(merged, "opposite_lane_after_crossing_flag")
    opposite_lane_after_crossing_run_length = clip_nonnegative(numeric_series(merged, "opposite_lane_after_crossing_run_length"))
    heading_change_peak = clip_nonnegative(numeric_series(merged, "heading_change_absmax"))
    heading_change_std = clip_nonnegative(numeric_series(merged, "heading_change_std"))
    lane_family_switch_count = clip_nonnegative(numeric_series(merged, "lane_family_switch_count"))
    speed_std = clip_nonnegative(numeric_series(merged, "speed_std"))
    speed_absmax = clip_nonnegative(numeric_series(merged, "speed_absmax"))
    ax_absmax = clip_nonnegative(numeric_series(merged, "ax_absmax"))
    ay_absmax = clip_nonnegative(numeric_series(merged, "ay_absmax"))
    double_yellow_distance_min = clip_nonnegative(numeric_series(merged, "window_distance_to_double_yellow_min", default=999.0))
    double_yellow_cross_count = clip_nonnegative(numeric_series(merged, "window_double_yellow_cross_count"))
    double_yellow_crossed_once = binary_series(merged, "window_double_yellow_crossed_once")
    double_yellow_side_switch_count = clip_nonnegative(numeric_series(merged, "window_double_yellow_side_switch_count"))

    occ_negative_margin = clip_nonnegative(-occupied_lane_flow_cos)
    wrong_dir_strength = clip01(1.0 - sigmoid_clip(occupied_lane_flow_cos, center=0.25, scale=0.10))
    negative_margin_strength = clip01(sigmoid_clip(occ_negative_margin, center=0.10, scale=0.04))
    flow_conflict_strength = clip01(sigmoid_clip(flow_conflict_score, center=0.35, scale=0.12))
    opposite_lane_strength = clip01(sigmoid_clip(opposite_lane_occupancy_ratio, center=0.35, scale=0.12))
    opposite_lane_run_strength = clip01(sigmoid_clip(opposite_lane_run_length, center=3.0, scale=1.0))
    reverse_flow_support = clip01(0.40 * reversed_flag + 0.35 * wrong_dir_strength + 0.25 * negative_margin_strength)

    artifact_cv_series = clip01(numeric_series(merged, "artifact_bbox_area_cv", default=0.0))
    bbox_direction_gate = clip01(1.0 - sigmoid_clip(artifact_cv_series, center=0.35, scale=0.05))
    speed_mean_series = clip_nonnegative(numeric_series(merged, "speed_mean"))
    heading_speed_gate = clip01(sigmoid_clip(speed_mean_series, center=0.05, scale=0.02))
    artifact_max_center_step = clip_nonnegative(numeric_series(merged, "artifact_max_center_step"))
    heading_center_motion_gate = clip01(sigmoid_clip(artifact_max_center_step, center=1.0, scale=0.20))
    heading_motion_gate = heading_speed_gate * heading_center_motion_gate
    heading_change_peak_gated = heading_change_peak * heading_motion_gate
    heading_change_std_gated = heading_change_std * heading_motion_gate

    wrong_way_score = clip01(
        bbox_direction_gate
        * (
            0.35 * wrong_dir_strength
            + 0.25 * negative_margin_strength
            + 0.20 * reversed_flag
            + 0.15 * flow_conflict_strength
        )
        + 0.05 * opposite_lane_strength
    )
    wrong_way_confidence = clip01(
        wrong_way_score
        * (
            0.42
            + 0.18 * selected_lane_known
            + 0.10 * semantic_support
            + 0.15 * traj_reliability
            + 0.10 * reverse_flow_support * selected_lane_known
            + 0.05 * reversed_flag * selected_lane_known
        )
    )
    wrong_way_flag = ((wrong_way_confidence >= 0.60) & ((reversed_flag >= 1.0) | (wrong_dir_strength >= 0.6))).astype(float)
    wrong_way_duration = np.maximum(
        opposite_lane_run_length.to_numpy(dtype=float) * sample_dt.to_numpy(dtype=float),
        duration_sec.to_numpy(dtype=float) * wrong_way_score.to_numpy(dtype=float) * reversed_flag.to_numpy(dtype=float),
    )

    weaving_switch_gate = (lane_family_switch_count > 0).astype(float)
    weaving_score = clip01(
        0.40 * clip01(sigmoid_clip(heading_change_peak_gated, center=2.5, scale=0.30))
        + 0.30 * weaving_switch_gate * clip01(sigmoid_clip(lane_family_switch_count, center=1.0, scale=0.45))
        + 0.15 * clip01(sigmoid_clip(heading_change_std_gated, center=0.30, scale=0.10))
        + 0.15 * clip01(sigmoid_clip(speed_std, center=0.20, scale=0.08))
    )
    weaving_confidence = clip01(
        weaving_score * (0.50 + 0.20 * semantic_support + 0.20 * traj_reliability + 0.10 * tracking_quality_good)
    )
    weaving_flag = ((weaving_confidence >= 0.58) & ((heading_change_peak >= 1.0) | (lane_family_switch_count >= 2.0))).astype(float)
    weaving_heading_duration_gate = ((artifact_max_center_step >= 1.0) & (heading_change_peak >= 0.25)).astype(float)
    weaving_duration = duration_sec * clip01(
        0.60 * weaving_heading_duration_gate * clip01(sigmoid_clip(heading_change_peak_gated, center=1.0, scale=0.30))
        + 0.40 * weaving_switch_gate * clip01(sigmoid_clip(lane_family_switch_count, center=1.0, scale=0.45))
    )

    yellow_proximity = clip01(1.0 - sigmoid_clip(double_yellow_distance_min, center=30.0, scale=10.0))
    dy_cross_gate = (double_yellow_cross_count > 0).astype(float)
    dy_side_gate = (double_yellow_side_switch_count > 0).astype(float)
    double_yellow_score = clip01(
        0.35 * double_yellow_crossed_once
        + 0.25 * dy_cross_gate * clip01(sigmoid_clip(double_yellow_cross_count, center=1.0, scale=0.50))
        + 0.20 * yellow_proximity
        + 0.20 * dy_side_gate * clip01(sigmoid_clip(double_yellow_side_switch_count, center=1.0, scale=0.50))
    )
    double_yellow_confidence = clip01(double_yellow_score * (0.55 + 0.20 * semantic_support + 0.25 * traj_reliability))
    double_yellow_flag = ((double_yellow_confidence >= 0.55) & ((double_yellow_crossed_once >= 1.0) | (double_yellow_cross_count >= 1.0))).astype(float)
    double_yellow_duration = np.maximum(
        double_yellow_cross_count.to_numpy(dtype=float) * sample_dt.to_numpy(dtype=float),
        duration_sec.to_numpy(dtype=float)
        * double_yellow_score.to_numpy(dtype=float)
        * np.where(double_yellow_cross_count.to_numpy(dtype=float) > 0.0, 0.5, 0.0),
    )

    opp_occ_gate = (opposite_lane_occupancy_ratio > 0).astype(float)
    opp_run_gate = (opposite_lane_run_length > 0).astype(float)
    opp_after_run_gate = (opposite_lane_after_crossing_run_length > 0).astype(float)
    opposite_lane_score = clip01(
        0.45 * opp_occ_gate * opposite_lane_strength
        + 0.25 * opp_run_gate * opposite_lane_run_strength
        + 0.15 * opposite_lane_after_crossing_flag
        + 0.15 * opp_after_run_gate * clip01(sigmoid_clip(opposite_lane_after_crossing_run_length, center=2.0, scale=1.0))
    )
    opposite_lane_confidence = clip01(
        opposite_lane_score * (0.45 + 0.25 * selected_lane_known + 0.10 * semantic_support + 0.20 * traj_reliability)
    )
    opposite_lane_flag = (
        (opposite_lane_confidence >= 0.58)
        & (selected_lane_known >= 1.0)
        & ((opposite_lane_occupancy_ratio >= 0.25) | (opposite_lane_run_length >= 2.0))
    ).astype(float)
    opposite_lane_duration = np.maximum(
        np.maximum(opposite_lane_run_length.to_numpy(dtype=float), opposite_lane_after_crossing_run_length.to_numpy(dtype=float))
        * sample_dt.to_numpy(dtype=float),
        duration_sec.to_numpy(dtype=float) * opposite_lane_occupancy_ratio.to_numpy(dtype=float),
    )

    illegal_flow_score = clip01(
        bbox_direction_gate
        * (0.34 * wrong_dir_strength + 0.24 * negative_margin_strength + 0.18 * flow_conflict_strength + 0.14 * reversed_flag)
        + 0.05 * selected_lane_known
        + 0.05 * semantic_support
    )
    illegal_flow_confidence = clip01(
        illegal_flow_score
        * (
            0.42
            + 0.20 * selected_lane_known
            + 0.08 * semantic_support
            + 0.15 * traj_reliability
            + 0.10 * reverse_flow_support * selected_lane_known
        )
    )
    illegal_flow_flag = (
        (illegal_flow_confidence >= 0.60)
        & (selected_lane_known >= 1.0)
        & ((wrong_dir_strength >= 0.55) | (negative_margin_strength >= 0.55))
    ).astype(float)
    illegal_flow_duration = np.maximum(
        opposite_lane_run_length.to_numpy(dtype=float) * sample_dt.to_numpy(dtype=float),
        duration_sec.to_numpy(dtype=float) * illegal_flow_score.to_numpy(dtype=float) * illegal_flow_flag.to_numpy(dtype=float) * 0.75,
    )

    merged["trajectory_wrong_way_flag"] = wrong_way_flag
    merged["trajectory_wrong_way_score"] = wrong_way_score
    merged["trajectory_wrong_way_confidence"] = wrong_way_confidence
    merged["trajectory_wrong_way_duration"] = clip_nonnegative(pd.Series(wrong_way_duration, index=merged.index))
    merged["trajectory_weaving_flag"] = weaving_flag
    merged["trajectory_weaving_score"] = weaving_score
    merged["trajectory_weaving_confidence"] = weaving_confidence
    merged["trajectory_weaving_duration"] = clip_nonnegative(weaving_duration)
    merged["trajectory_double_yellow_crossing_flag"] = double_yellow_flag
    merged["trajectory_double_yellow_crossing_score"] = double_yellow_score
    merged["trajectory_double_yellow_crossing_confidence"] = double_yellow_confidence
    merged["trajectory_double_yellow_crossing_duration"] = clip_nonnegative(pd.Series(double_yellow_duration, index=merged.index))
    merged["trajectory_opposite_lane_driving_flag"] = opposite_lane_flag
    merged["trajectory_opposite_lane_driving_score"] = opposite_lane_score
    merged["trajectory_opposite_lane_driving_confidence"] = opposite_lane_confidence
    merged["trajectory_opposite_lane_driving_duration"] = clip_nonnegative(pd.Series(opposite_lane_duration, index=merged.index))
    merged["trajectory_illegal_flow_direction_flag"] = illegal_flow_flag
    merged["trajectory_illegal_flow_direction_score"] = illegal_flow_score
    merged["trajectory_illegal_flow_direction_confidence"] = illegal_flow_confidence
    merged["trajectory_illegal_flow_direction_duration"] = clip_nonnegative(pd.Series(illegal_flow_duration, index=merged.index))
    merged["trajectory_heading_change_peak"] = heading_change_peak
    merged["trajectory_lateral_drift_score"] = clip01(
        0.60 * clip01(sigmoid_clip(lane_family_switch_count, center=1.0, scale=0.45))
        + 0.40 * clip01(sigmoid_clip(heading_change_std, center=0.30, scale=0.10))
    )
    merged["trajectory_reverse_like_motion_score"] = clip01(0.45 * wrong_dir_strength + 0.30 * negative_margin_strength + 0.25 * reversed_flag)
    merged["trajectory_abrupt_stop_go_score"] = clip01(
        0.35 * clip01(sigmoid_clip(speed_std, center=0.20, scale=0.08))
        + 0.325 * clip01(sigmoid_clip(ax_absmax, center=0.50, scale=0.04))
        + 0.325 * clip01(sigmoid_clip(ay_absmax, center=0.50, scale=0.04))
    )
    merged["trajectory_cut_in_score"] = clip01(
        0.50 * clip01(sigmoid_clip(lane_family_switch_count, center=1.0, scale=0.45))
        + 0.30 * clip01(sigmoid_clip(heading_change_peak, center=1.0, scale=0.30))
        + 0.20 * clip01(sigmoid_clip(speed_absmax, center=0.30, scale=0.10))
    )

    subtype_score_frame = merged[
        [
            "trajectory_wrong_way_score",
            "trajectory_weaving_score",
            "trajectory_double_yellow_crossing_score",
            "trajectory_opposite_lane_driving_score",
            "trajectory_illegal_flow_direction_score",
        ]
    ].copy()
    merged["trajectory_subtype_peak"] = subtype_score_frame.max(axis=1)
    merged["trajectory_subtype_entropy"] = entropy_from_scores(subtype_score_frame)
    merged["trajectory_subtype_switch_count"] = lane_family_switch_count
    merged["trajectory_repeated_severe_burst_count"] = double_yellow_cross_count + (lane_family_switch_count >= 2.0).astype(float)
    peak_duration = pd.concat(
        [
            merged["trajectory_wrong_way_duration"],
            merged["trajectory_weaving_duration"],
            merged["trajectory_double_yellow_crossing_duration"],
            merged["trajectory_opposite_lane_driving_duration"],
            merged["trajectory_illegal_flow_direction_duration"],
        ],
        axis=1,
    ).max(axis=1)
    merged["trajectory_post_peak_persistence"] = clip01(safe_divide(peak_duration, duration_sec.replace(0.0, np.nan), default=0.0))
    merged["heading_motion_gate"] = heading_motion_gate

    output_columns = WINDOW_JOIN_KEYS + ["start_sec", "end_sec"] + TRAJECTORY_OUTPUT_COLUMNS
    output_frame = merged[output_columns].copy()
    sidecar_columns = WINDOW_JOIN_KEYS + [
        "start_sec",
        "end_sec",
        "flow_cos",
        "occupied_lane_flow_cos",
        "reversed_flag",
        "flow_conflict_score",
        "negative_flow_margin",
        "selected_lane_known",
        "semantic_support_flag",
        "heading_change_absmax",
        "heading_change_std",
        "lane_family_switch_count",
        "opposite_lane_occupancy_ratio",
        "opposite_lane_run_length",
        "opposite_lane_after_crossing_flag",
        "opposite_lane_after_crossing_run_length",
        "window_distance_to_double_yellow_min",
        "window_double_yellow_cross_count",
        "window_double_yellow_crossed_once",
        "window_double_yellow_side_switch_count",
        "trajectory_subtype_peak",
        "trajectory_subtype_entropy",
        "artifact_bbox_area_cv",
        "artifact_max_center_step",
        "heading_motion_gate",
        *OSCILLATION_SIDECAR_COLUMNS,
    ]
    sidecar_frame = merged[[column for column in sidecar_columns if column in merged.columns]].copy()
    sidecar_frame["trajectory_subtype_primary"] = subtype_score_frame.idxmax(axis=1)
    sidecar_frame["trajectory_geometry_source_path"] = str(geometry_source_path or "")
    sidecar_frame["trajectory_geometry_source_present"] = bool(geometry_source_present)
    return output_frame, sidecar_frame

__all__ = ["build_oscillation_sidecar_from_samples", "build_trajectory_v8_frames_from_merged"]

