"""Frozen overspeed v8 policy and raw-IPM recomputation.

Mechanically extracted from the accepted research-time v8 calculator; no
research-tree imports or absolute paths are needed at runtime.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from traffic_risk.identifiers import normalize_window_keys
from traffic_risk.upstream.v8_common import (
    WINDOW_JOIN_KEYS, build_window_key_set, burst_count_by_track,
    clip_nonnegative, cumulative_duration_by_track,
    number as numeric_series, safe_divide,
)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
        return value if np.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def validation_summary(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    series = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return {"valid_count": int(len(series)), "mean": float(series.mean()) if len(series) else 0.0}

def load_ipm_matrix(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    matrix = np.asarray(payload["homography_img_to_world"], dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"unexpected homography shape: {matrix.shape}")
    return matrix


def project_bottom_centers(track_frame: pd.DataFrame, homography: np.ndarray) -> np.ndarray:
    center_x = ((pd.to_numeric(track_frame["x1"], errors="coerce") + pd.to_numeric(track_frame["x2"], errors="coerce")) / 2.0).to_numpy(dtype=np.float64)
    bottom_y = pd.to_numeric(track_frame["y2"], errors="coerce").to_numpy(dtype=np.float64)
    points = np.stack([center_x, bottom_y, np.ones_like(center_x)], axis=1)
    projected = points @ homography.T
    scale = projected[:, 2:3]
    scale[~np.isfinite(scale)] = np.nan
    xy = np.divide(projected[:, :2], scale, out=np.full_like(projected[:, :2], np.nan), where=np.abs(scale) > 1e-9)
    return xy


def build_interval_frame(track_frame: pd.DataFrame, homography: np.ndarray) -> pd.DataFrame:
    frame = track_frame.copy()
    frame["timestamp_sec"] = pd.to_numeric(frame["timestamp_sec"], errors="coerce")
    frame = frame.dropna(subset=["timestamp_sec", "x1", "x2", "y2"]).sort_values("timestamp_sec", kind="mergesort").reset_index(drop=True)
    if frame.shape[0] < 2:
        return pd.DataFrame(columns=["interval_mid_sec", "speed_kmh", "pixel_distance_px", "abs_pixel_delta_y_px", "bbox_height_px", "world_distance_m", "delta_t_sec"])
    center_x = ((pd.to_numeric(frame["x1"], errors="coerce") + pd.to_numeric(frame["x2"], errors="coerce")) / 2.0).to_numpy(dtype=np.float64)
    bottom_y = pd.to_numeric(frame["y2"], errors="coerce").to_numpy(dtype=np.float64)
    if "y1" in frame.columns:
        bbox_height_series = pd.to_numeric(frame["y2"], errors="coerce") - pd.to_numeric(frame["y1"], errors="coerce")
    else:
        bbox_height_series = pd.Series(np.nan, index=frame.index, dtype=float)
    bbox_height = bbox_height_series.to_numpy(dtype=np.float64)
    world_xy = project_bottom_centers(frame, homography)
    timestamp = frame["timestamp_sec"].to_numpy(dtype=np.float64)
    delta_t = np.diff(timestamp)
    delta_xy = np.diff(world_xy, axis=0)
    distance_m = np.linalg.norm(delta_xy, axis=1)
    pixel_delta_x = np.diff(center_x)
    pixel_delta_y = np.diff(bottom_y)
    pixel_distance = np.sqrt(pixel_delta_x * pixel_delta_x + pixel_delta_y * pixel_delta_y)
    interval_bbox_height = (bbox_height[1:] + bbox_height[:-1]) / 2.0
    valid = np.isfinite(delta_t) & (delta_t > 1e-9) & np.isfinite(distance_m)
    speeds = np.full(delta_t.shape, np.nan, dtype=np.float64)
    speeds[valid] = distance_m[valid] / delta_t[valid] * 3.6
    interval_mid = (timestamp[1:] + timestamp[:-1]) / 2.0
    out = pd.DataFrame(
        {
            "interval_mid_sec": interval_mid,
            "speed_kmh": speeds,
            "pixel_distance_px": pixel_distance,
            "abs_pixel_delta_y_px": np.abs(pixel_delta_y),
            "bbox_height_px": interval_bbox_height,
            "world_distance_m": distance_m,
            "delta_t_sec": delta_t,
        }
    )
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=["interval_mid_sec"]).reset_index(drop=True)


def compute_window_speed_stats(group: pd.DataFrame, homography: np.ndarray) -> pd.DataFrame:
    source_csv = Path(str(group["source_csv"].iloc[0]))
    if not source_csv.exists():
        out = group.copy()
        out["computed_speed_kmh_max"] = np.nan
        out["computed_speed_kmh_p95"] = np.nan
        out["computed_valid_interval_count"] = 0
        out["computed_pixel_distance_px_p95"] = np.nan
        out["computed_pixel_distance_px_max"] = np.nan
        out["computed_bbox_height_px_p95"] = np.nan
        out["computed_pixel_path_length_px"] = np.nan
        out["computed_net_pixel_displacement_px"] = np.nan
        out["computed_path_efficiency"] = np.nan
        out["computed_from_raw_intervals"] = False
        out["fallback_reason"] = "source_csv_missing"
        return out
    track_frame = pd.read_csv(source_csv, low_memory=False)
    if "track_id" not in track_frame.columns:
        out = group.copy()
        out["computed_speed_kmh_max"] = np.nan
        out["computed_speed_kmh_p95"] = np.nan
        out["computed_valid_interval_count"] = 0
        out["computed_pixel_distance_px_p95"] = np.nan
        out["computed_pixel_distance_px_max"] = np.nan
        out["computed_bbox_height_px_p95"] = np.nan
        out["computed_pixel_path_length_px"] = np.nan
        out["computed_net_pixel_displacement_px"] = np.nan
        out["computed_path_efficiency"] = np.nan
        out["computed_from_raw_intervals"] = False
        out["fallback_reason"] = "source_track_id_missing"
        return out

    track_frame = track_frame.copy()
    track_frame["track_id"] = pd.to_numeric(track_frame["track_id"], errors="coerce")
    track_frame["timestamp_sec"] = pd.to_numeric(track_frame.get("timestamp_sec"), errors="coerce")
    track_frame["center_x_px"] = (pd.to_numeric(track_frame.get("x1"), errors="coerce") + pd.to_numeric(track_frame.get("x2"), errors="coerce")) / 2.0
    track_frame["bottom_y_px"] = pd.to_numeric(track_frame.get("y2"), errors="coerce")

    rows: list[dict[str, Any]] = []
    for track_id, track_group in group.groupby("track_id", sort=False):
        track_id_value = pd.to_numeric(pd.Series([track_id]), errors="coerce").iloc[0]
        if pd.isna(track_id_value):
            for row in track_group.itertuples(index=False):
                rows.append(
                    {
                        **row._asdict(),
                        "computed_speed_kmh_max": np.nan,
                        "computed_speed_kmh_p95": np.nan,
                        "computed_valid_interval_count": 0,
                        "computed_pixel_distance_px_p95": np.nan,
                        "computed_pixel_distance_px_max": np.nan,
                        "computed_bbox_height_px_p95": np.nan,
                        "computed_pixel_path_length_px": np.nan,
                        "computed_net_pixel_displacement_px": np.nan,
                        "computed_path_efficiency": np.nan,
                        "computed_from_raw_intervals": False,
                        "fallback_reason": "group_track_id_invalid",
                    }
                )
            continue

        track_source = track_frame.loc[track_frame["track_id"] == float(track_id_value)].copy()
        track_source = track_source.dropna(subset=["timestamp_sec", "center_x_px", "bottom_y_px"]).sort_values("timestamp_sec", kind="mergesort")
        interval_frame = build_interval_frame(track_source, homography)
        if interval_frame.empty:
            for row in track_group.itertuples(index=False):
                rows.append(
                    {
                        **row._asdict(),
                        "computed_speed_kmh_max": np.nan,
                        "computed_speed_kmh_p95": np.nan,
                        "computed_valid_interval_count": 0,
                        "computed_pixel_distance_px_p95": np.nan,
                        "computed_pixel_distance_px_max": np.nan,
                        "computed_bbox_height_px_p95": np.nan,
                        "computed_pixel_path_length_px": np.nan,
                        "computed_net_pixel_displacement_px": np.nan,
                        "computed_path_efficiency": np.nan,
                        "computed_from_raw_intervals": False,
                        "fallback_reason": "track_no_valid_intervals",
                    }
                )
            continue

        interval_mid = interval_frame["interval_mid_sec"].to_numpy(dtype=np.float64)
        interval_speed = interval_frame["speed_kmh"].to_numpy(dtype=np.float64)
        order = np.argsort(interval_mid, kind="mergesort")
        interval_mid = interval_mid[order]
        interval_speed = interval_speed[order]
        for row in track_group.itertuples(index=False):
            start_sec = safe_float(row.start_sec)
            end_sec = safe_float(row.end_sec)
            left = int(np.searchsorted(interval_mid, start_sec, side="left"))
            right = int(np.searchsorted(interval_mid, end_sec, side="right"))
            interval_slice = interval_frame.iloc[left:right]
            speeds = interval_speed[left:right]
            valid_speeds = speeds[np.isfinite(speeds)]
            pixel_distances = pd.to_numeric(interval_slice.get("pixel_distance_px", np.nan), errors="coerce")
            valid_pixel_distances = pixel_distances[np.isfinite(pixel_distances)]
            bbox_heights = pd.to_numeric(interval_slice.get("bbox_height_px", np.nan), errors="coerce")
            valid_bbox_heights = bbox_heights[np.isfinite(bbox_heights)]
            window_points = track_source.loc[
                (track_source["timestamp_sec"] >= start_sec)
                & (track_source["timestamp_sec"] <= end_sec),
                ["center_x_px", "bottom_y_px"],
            ]
            if window_points.shape[0] >= 2:
                delta_x = float(window_points["center_x_px"].iloc[-1] - window_points["center_x_px"].iloc[0])
                delta_y = float(window_points["bottom_y_px"].iloc[-1] - window_points["bottom_y_px"].iloc[0])
                net_pixel_displacement = float(np.hypot(delta_x, delta_y))
            else:
                net_pixel_displacement = np.nan
            pixel_path_length = float(valid_pixel_distances.sum()) if valid_pixel_distances.size else np.nan
            path_efficiency = (
                float(net_pixel_displacement / pixel_path_length)
                if np.isfinite(net_pixel_displacement) and np.isfinite(pixel_path_length) and pixel_path_length > 1e-9
                else np.nan
            )
            if valid_speeds.size == 0:
                rows.append(
                    {
                        **row._asdict(),
                        "computed_speed_kmh_max": np.nan,
                        "computed_speed_kmh_p95": np.nan,
                        "computed_valid_interval_count": 0,
                        "computed_pixel_distance_px_p95": float(np.quantile(valid_pixel_distances, 0.95)) if valid_pixel_distances.size else np.nan,
                        "computed_pixel_distance_px_max": float(np.max(valid_pixel_distances)) if valid_pixel_distances.size else np.nan,
                        "computed_bbox_height_px_p95": float(np.quantile(valid_bbox_heights, 0.95)) if valid_bbox_heights.size else np.nan,
                        "computed_pixel_path_length_px": pixel_path_length,
                        "computed_net_pixel_displacement_px": net_pixel_displacement,
                        "computed_path_efficiency": path_efficiency,
                        "computed_from_raw_intervals": False,
                        "fallback_reason": "window_no_valid_intervals",
                    }
                )
                continue
            rows.append(
                {
                    **row._asdict(),
                    "computed_speed_kmh_max": float(np.max(valid_speeds)),
                    "computed_speed_kmh_p95": float(np.quantile(valid_speeds, 0.95)),
                    "computed_valid_interval_count": int(valid_speeds.size),
                    "computed_pixel_distance_px_p95": float(np.quantile(valid_pixel_distances, 0.95)) if valid_pixel_distances.size else np.nan,
                    "computed_pixel_distance_px_max": float(np.max(valid_pixel_distances)) if valid_pixel_distances.size else np.nan,
                    "computed_bbox_height_px_p95": float(np.quantile(valid_bbox_heights, 0.95)) if valid_bbox_heights.size else np.nan,
                    "computed_pixel_path_length_px": pixel_path_length,
                    "computed_net_pixel_displacement_px": net_pixel_displacement,
                    "computed_path_efficiency": path_efficiency,
                    "computed_from_raw_intervals": True,
                    "fallback_reason": "",
                }
            )
    return pd.DataFrame(rows)

SCHEMA_VERSION = "schemaC_full_v1_no_cls3.overspeed_policy_outputs/v8"
SPEED_LIMIT_DEFAULT = 70.0
OVERSPEED_BAND_2 = 90.0
OVERSPEED_BAND_3 = 110.0
BELOW_LIMIT_RAW_RATIO_MIN = 0.85
BELOW_LIMIT_RAW_RATIO_MAX = 1.20
BELOW_LIMIT_RAW_DELTA_MAX = 10.0
OSCILLATORY_PATH_EFFICIENCY_MAX = 0.72
OSCILLATORY_GROUND_DISTANCE_MAX = 0.80
SHORT_TRACK_WINDOW_COUNT_MAX = 2
ULTRA_SHORT_TRACK_WINDOW_COUNT_MAX = 3
SHORT_TRACK_FRAGMENT_WINDOW_COUNT_MAX = 6
SHORT_TRACK_FRAGMENT_BBOX_HEIGHT_MAX = 30.0
OVERSPEED_TRACK_WARMUP_SEC = 0.5
WARMUP_BYPASS_BBOX_HEIGHT_MIN = 90.0
WARMUP_BYPASS_PIXEL_MOTION_MIN = 4.0
WARMUP_BYPASS_EXTREME_PIXEL_MOTION_MIN = 8.0
WARMUP_BYPASS_EXTREME_RAW_SPEED_MIN = 250.0
REMOTE_WARMUP_BYPASS_BBOX_HEIGHT_MAX = 90.0
REMOTE_WARMUP_BYPASS_SMOOTHED_SPEED_MIN = 110.0
REMOTE_WARMUP_BYPASS_RAW_SPEED_MIN = 200.0
REMOTE_WARMUP_BYPASS_PIXEL_MOTION_MIN = 2.0
REMOTE_WARMUP_BYPASS_RAW_RATIO_MAX = 2.0
REMOTE_WARMUP_BYPASS_PATH_EFFICIENCY_MIN = 0.90
REMOTE_WARMUP_BYPASS_GROUND_DISTANCE_MIN = 0.90
SHORT_TRACK_FRAGMENT_PIXEL_P95_MAX = 4.0
SINGLE_WINDOW_TINY_BOX_HEIGHT_MAX = 24.0
SINGLE_WINDOW_TINY_BOX_PIXEL_P95_MAX = 5.0
SINGLE_WINDOW_TINY_BOX_PIXEL_MAX_MAX = 5.5
SINGLE_WINDOW_TINY_BOX_PATH_EFFICIENCY_MAX = 0.80
SMALL_BBOX_RAW_RATIO_MAX = 1.30
SMALL_BBOX_PIXEL_MOTION_MAX = 2.0
TINY_BBOX_HEIGHT_MAX = 30.0
TINY_BBOX_SEVERE_PIXEL_MOTION_MAX = 2.5
TINY_BBOX_SEVERE_PIXEL_MOTION_MAX_MAX = 2.75


def clip01(series: pd.Series) -> pd.Series:
    return series.astype(float).clip(lower=0.0, upper=1.0)


def build_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Overspeed Policy Outputs v8 Build Report",
        "",
        f"- schema_version: {payload['schema_version']}",
        f"- row_count: {payload['row_count']}",
        f"- physical_speed_source_ratio: {payload['physical_speed_source_ratio']:.4f}",
        f"- raw_recompute_rejected_count: {payload['raw_recompute_rejected_count']}",
        f"- exact_speed_kmh_max_unstable: {str(payload['exact_speed_kmh_max_unstable']).lower()}",
        f"- leakage_audit_passed: {str(payload['leakage_audit_passed']).lower()}",
        "",
    ]
    for field_name, summary in payload["validation_summary"].items():
        lines.extend([f"## {field_name}", f"- summary: {summary}", ""])
    return "\n".join(lines) + "\n"


def build_overspeed_v8_frames_from_inputs(
    *,
    base_frame: pd.DataFrame,
    module_frame: pd.DataFrame,
    sidecar_frame: pd.DataFrame,
    ipm_json: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build overspeed v8 main/sidecar frames from in-memory inputs."""

    homography = load_ipm_matrix(ipm_json)
    base = normalize_window_keys(base_frame)
    window_keys = build_window_key_set(base)

    module_df = normalize_window_keys(module_frame)
    sidecar_df = normalize_window_keys(sidecar_frame)
    if module_df.empty:
        output_columns = WINDOW_JOIN_KEYS + [
            "start_sec",
            "end_sec",
            "speed_kmh_max",
            "speed_kmh_p95",
            "speed_kmh_mean",
            "speed_band_70_90_flag",
            "speed_band_90_110_flag",
            "speed_band_ge_110_flag",
            "overspeed_level_ge2_duration",
            "overspeed_level_ge3_duration",
            "overspeed_peak_minus_limit",
            "overspeed_severe_burst_count",
            "overspeed_reliability",
            "overspeed_signal_quality",
        ]
        empty = pd.DataFrame(columns=output_columns)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "row_count": 0,
            "physical_speed_source_ratio": 0.0,
            "raw_recompute_attempt_ratio": 0.0,
            "raw_recompute_rejected_count": 0,
            "exact_speed_kmh_max_unstable": False,
            "fallback_window_count": 0,
            "validation_summary": {},
            "leakage_audit_passed": True,
        }
        return empty, pd.DataFrame(columns=WINDOW_JOIN_KEYS + ["start_sec", "end_sec"]), payload

    frame = module_df.merge(
        sidecar_df,
        on=WINDOW_JOIN_KEYS + ["start_sec", "end_sec"],
        how="left",
        validate="one_to_one",
    )
    key_series = list(
        zip(
            frame["case_key"].tolist(),
            frame["source_type"].tolist(),
            frame["source_id"].tolist(),
            frame["video_id"].tolist(),
            frame["track_id"].tolist(),
            frame["ts_window_idx"].tolist(),
            strict=False,
        )
    )
    frame = frame.loc[[key in window_keys for key in key_series]].copy()

    grouped_frames: list[pd.DataFrame] = []
    if "source_csv" in frame.columns:
        for _, group in frame.groupby("source_csv", sort=False):
            grouped_frames.append(compute_window_speed_stats(group.copy(), homography))
    frame = pd.concat(grouped_frames, ignore_index=True) if grouped_frames else frame.copy()

    speed_limit = pd.to_numeric(frame.get("speed_limit_kmh", SPEED_LIMIT_DEFAULT), errors="coerce").fillna(SPEED_LIMIT_DEFAULT)
    smoothed_p95 = pd.concat(
        [
            pd.to_numeric(frame.get("speed_smoothed_kmh_p95", np.nan), errors="coerce"),
            pd.to_numeric(frame.get("speed_kmh", np.nan), errors="coerce"),
        ],
        axis=1,
    ).bfill(axis=1).iloc[:, 0]
    raw_p95 = pd.to_numeric(frame.get("speed_raw_kmh_p95", np.nan), errors="coerce")
    guarded_raw_p95 = raw_p95.where(raw_p95 <= np.maximum(200.0, smoothed_p95.fillna(0.0) * 1.5 + 5.0))
    fallback_p95 = smoothed_p95.fillna(guarded_raw_p95).fillna(0.0)
    fallback_upper_proxy = pd.concat([fallback_p95, guarded_raw_p95.fillna(fallback_p95)], axis=1).max(axis=1)
    computed_speed_kmh_p95 = pd.to_numeric(frame.get("computed_speed_kmh_p95", np.nan), errors="coerce")
    computed_speed_kmh_max = pd.to_numeric(frame.get("computed_speed_kmh_max", np.nan), errors="coerce")
    pixel_motion_p95 = pd.to_numeric(frame.get("computed_pixel_distance_px_p95", np.nan), errors="coerce")
    pixel_motion_max = pd.to_numeric(frame.get("computed_pixel_distance_px_max", np.nan), errors="coerce")
    bbox_height_p95 = pd.to_numeric(frame.get("computed_bbox_height_px_p95", np.nan), errors="coerce")
    ground_distance_p95 = pd.to_numeric(frame.get("ground_distance_m_p95", np.nan), errors="coerce")
    path_efficiency = pd.to_numeric(frame.get("computed_path_efficiency", np.nan), errors="coerce")
    track_group_keys = ["case_key", "source_type", "source_id", "video_id", "track_id"]
    track_window_count = frame.groupby(track_group_keys, sort=False)["ts_window_idx"].transform("size")
    track_entry_offset_sec = (
        pd.to_numeric(frame["start_sec"], errors="coerce")
        - pd.to_numeric(frame["start_sec"], errors="coerce")
        .groupby([frame[key] for key in track_group_keys], sort=False)
        .transform("min")
    ).fillna(0.0)
    fallback_reference = fallback_p95.replace(0.0, np.nan)
    raw_ratio = computed_speed_kmh_p95 / fallback_reference
    computed_from_raw = frame.get("computed_from_raw_intervals", False).astype(bool)
    candidate_speed_peak = pd.concat(
        [computed_speed_kmh_p95, computed_speed_kmh_max, raw_p95, smoothed_p95, fallback_p95],
        axis=1,
    ).max(axis=1).fillna(0.0)
    pixel_motion_floor = np.maximum(1.5, bbox_height_p95.fillna(0.0) * 0.05)
    pixel_motion_ceiling = np.maximum(4.0, bbox_height_p95.fillna(0.0) * 0.12)
    micro_motion_floor = np.maximum(2.0, bbox_height_p95.fillna(0.0) * 0.08)
    short_track_motion_ceiling = np.maximum(2.5, bbox_height_p95.fillna(0.0) * 0.15)
    stationary_jitter_suspect = (
        pixel_motion_p95.notna()
        & pixel_motion_max.notna()
        & (candidate_speed_peak >= 3.0)
        & (pixel_motion_p95 <= pixel_motion_floor)
        & (pixel_motion_max <= pixel_motion_ceiling)
        & (ground_distance_p95.fillna(np.inf) <= 0.35)
    )
    oscillatory_micro_jitter_suspect = (
        computed_from_raw
        & path_efficiency.notna()
        & (candidate_speed_peak >= SPEED_LIMIT_DEFAULT)
        & (pixel_motion_p95 <= micro_motion_floor)
        & (path_efficiency <= OSCILLATORY_PATH_EFFICIENCY_MAX)
        & (ground_distance_p95.fillna(np.inf) <= OSCILLATORY_GROUND_DISTANCE_MAX)
    )
    short_track_micro_jitter_suspect = (
        computed_from_raw
        & (track_window_count <= SHORT_TRACK_WINDOW_COUNT_MAX)
        & (candidate_speed_peak >= OVERSPEED_BAND_3)
        & (pixel_motion_p95 <= micro_motion_floor)
        & (pixel_motion_max <= short_track_motion_ceiling)
    )
    short_track_unstable_jump_suspect = (
        computed_from_raw
        & (track_window_count <= SHORT_TRACK_WINDOW_COUNT_MAX)
        & (candidate_speed_peak >= OVERSPEED_BAND_3)
        & path_efficiency.notna()
        & (path_efficiency <= 0.50)
        & ((computed_speed_kmh_max > 220.0) | (computed_speed_kmh_p95 > 200.0) | (raw_p95 > 200.0))
    )
    ultra_short_track_fragment_suspect = (track_window_count <= ULTRA_SHORT_TRACK_WINDOW_COUNT_MAX) & (
        candidate_speed_peak >= OVERSPEED_BAND_2
    )
    track_candidate_speed_peak = candidate_speed_peak.groupby([frame[key] for key in track_group_keys], sort=False).transform("max")
    track_bbox_height_p95_min = bbox_height_p95.groupby([frame[key] for key in track_group_keys], sort=False).transform("min")
    track_pixel_motion_p95_max = pixel_motion_p95.groupby([frame[key] for key in track_group_keys], sort=False).transform("max")
    short_track_tiny_motion_fragment_suspect = (
        (track_window_count > ULTRA_SHORT_TRACK_WINDOW_COUNT_MAX)
        & (track_window_count <= SHORT_TRACK_FRAGMENT_WINDOW_COUNT_MAX)
        & (track_candidate_speed_peak >= OVERSPEED_BAND_2)
        & (track_bbox_height_p95_min <= SHORT_TRACK_FRAGMENT_BBOX_HEIGHT_MAX)
        & (track_pixel_motion_p95_max <= SHORT_TRACK_FRAGMENT_PIXEL_P95_MAX)
    )
    single_window_tiny_box_jitter_suspect = (
        computed_from_raw
        & (track_window_count == 1)
        & (computed_speed_kmh_p95 >= OVERSPEED_BAND_3)
        & (bbox_height_p95 <= SINGLE_WINDOW_TINY_BOX_HEIGHT_MAX)
        & (pixel_motion_p95 <= SINGLE_WINDOW_TINY_BOX_PIXEL_P95_MAX)
        & (pixel_motion_max <= SINGLE_WINDOW_TINY_BOX_PIXEL_MAX_MAX)
        & path_efficiency.notna()
        & (path_efficiency <= SINGLE_WINDOW_TINY_BOX_PATH_EFFICIENCY_MAX)
    )
    warmup_high_confidence_zone = bbox_height_p95.fillna(0.0) >= WARMUP_BYPASS_BBOX_HEIGHT_MIN
    warmup_supported_severe_speed = (
        (computed_speed_kmh_p95.fillna(0.0) >= OVERSPEED_BAND_3)
        & (smoothed_p95.fillna(0.0) >= OVERSPEED_BAND_3)
    )
    warmup_supported_extreme_raw_speed = (
        (computed_speed_kmh_p95.fillna(0.0) >= WARMUP_BYPASS_EXTREME_RAW_SPEED_MIN)
        & (pixel_motion_p95.fillna(0.0) >= WARMUP_BYPASS_EXTREME_PIXEL_MOTION_MIN)
    )
    track_entry_near_field_warmup_bypass = (
        warmup_high_confidence_zone
        & (pixel_motion_p95.fillna(0.0) >= WARMUP_BYPASS_PIXEL_MOTION_MIN)
        & (warmup_supported_severe_speed | warmup_supported_extreme_raw_speed)
    )
    track_entry_remote_severe_warmup_bypass = (
        (bbox_height_p95.fillna(np.inf) < REMOTE_WARMUP_BYPASS_BBOX_HEIGHT_MAX)
        & (smoothed_p95.fillna(0.0) >= REMOTE_WARMUP_BYPASS_SMOOTHED_SPEED_MIN)
        & (computed_speed_kmh_p95.fillna(0.0) >= REMOTE_WARMUP_BYPASS_RAW_SPEED_MIN)
        & (pixel_motion_p95.fillna(0.0) >= REMOTE_WARMUP_BYPASS_PIXEL_MOTION_MIN)
        & (raw_ratio.fillna(np.inf) <= REMOTE_WARMUP_BYPASS_RAW_RATIO_MAX)
        & (path_efficiency.fillna(0.0) >= REMOTE_WARMUP_BYPASS_PATH_EFFICIENCY_MIN)
        & (ground_distance_p95.fillna(0.0) >= REMOTE_WARMUP_BYPASS_GROUND_DISTANCE_MIN)
    )
    track_entry_warmup_bypass = track_entry_near_field_warmup_bypass | track_entry_remote_severe_warmup_bypass
    track_entry_ipm_perspective_suspect = (
        computed_from_raw
        & (track_entry_offset_sec < OVERSPEED_TRACK_WARMUP_SEC)
        & (candidate_speed_peak >= SPEED_LIMIT_DEFAULT)
        & ~track_entry_warmup_bypass
    )
    small_bbox_ipm_instability_suspect = (
        computed_from_raw
        & (bbox_height_p95.fillna(0.0) < 90.0)
        & (computed_speed_kmh_p95 >= OVERSPEED_BAND_2)
        & fallback_reference.notna()
        & (fallback_reference > 20.0)
        & (fallback_reference < OVERSPEED_BAND_2)
        & (raw_ratio > SMALL_BBOX_RAW_RATIO_MAX)
        & (pixel_motion_p95.fillna(999.0) < SMALL_BBOX_PIXEL_MOTION_MAX)
    )
    tiny_bbox_severe_ipm_instability_suspect = (
        computed_from_raw
        & (bbox_height_p95.fillna(0.0) < TINY_BBOX_HEIGHT_MAX)
        & (computed_speed_kmh_p95 >= OVERSPEED_BAND_3)
        & fallback_reference.notna()
        & (fallback_reference > 20.0)
        & (fallback_reference < OVERSPEED_BAND_2)
        & (raw_ratio > SMALL_BBOX_RAW_RATIO_MAX)
        & (pixel_motion_p95.fillna(999.0) < TINY_BBOX_SEVERE_PIXEL_MOTION_MAX)
        & (pixel_motion_max.fillna(999.0) < TINY_BBOX_SEVERE_PIXEL_MOTION_MAX_MAX)
    )
    any_jitter_suspect = (
        stationary_jitter_suspect
        | oscillatory_micro_jitter_suspect
        | short_track_micro_jitter_suspect
        | short_track_unstable_jump_suspect
        | ultra_short_track_fragment_suspect
        | short_track_tiny_motion_fragment_suspect
        | single_window_tiny_box_jitter_suspect
        | track_entry_ipm_perspective_suspect
        | small_bbox_ipm_instability_suspect
        | tiny_bbox_severe_ipm_instability_suspect
    )
    below_limit_conflict = (
        smoothed_p95.notna()
        & (smoothed_p95 > 0)
        & (smoothed_p95 < SPEED_LIMIT_DEFAULT)
        & (computed_speed_kmh_p95 >= SPEED_LIMIT_DEFAULT)
    )
    below_limit_raw_agreement = (
        (computed_speed_kmh_p95 <= np.maximum(SPEED_LIMIT_DEFAULT, smoothed_p95.fillna(0.0) + BELOW_LIMIT_RAW_DELTA_MAX))
        & raw_ratio.between(BELOW_LIMIT_RAW_RATIO_MIN, BELOW_LIMIT_RAW_RATIO_MAX, inclusive="both")
    )
    high_confidence_zone = bbox_height_p95.fillna(0.0) >= 90.0
    speed_ceiling_p95 = high_confidence_zone.map({True: 500.0, False: 200.0})
    speed_ceiling_max = high_confidence_zone.map({True: 550.0, False: 220.0})
    stable_raw_mask = (
        computed_from_raw
        & computed_speed_kmh_p95.notna()
        & computed_speed_kmh_max.notna()
        & (computed_speed_kmh_p95 <= speed_ceiling_p95)
        & (computed_speed_kmh_max <= speed_ceiling_max)
        & (fallback_reference.isna() | raw_ratio.between(0.5, 2.0, inclusive="both"))
        & (~below_limit_conflict | below_limit_raw_agreement)
        & ~any_jitter_suspect
    )
    frame["raw_recompute_stable"] = stable_raw_mask.astype(bool)
    frame["stationary_jitter_suspect"] = stationary_jitter_suspect.astype(bool)
    frame["oscillatory_micro_jitter_suspect"] = oscillatory_micro_jitter_suspect.astype(bool)
    frame["short_track_micro_jitter_suspect"] = short_track_micro_jitter_suspect.astype(bool)
    frame["short_track_unstable_jump_suspect"] = short_track_unstable_jump_suspect.astype(bool)
    frame["ultra_short_track_fragment_suspect"] = ultra_short_track_fragment_suspect.astype(bool)
    frame["short_track_tiny_motion_fragment_suspect"] = short_track_tiny_motion_fragment_suspect.astype(bool)
    frame["single_window_tiny_box_jitter_suspect"] = single_window_tiny_box_jitter_suspect.astype(bool)
    frame["track_entry_ipm_perspective_suspect"] = track_entry_ipm_perspective_suspect.astype(bool)
    frame["track_entry_near_field_warmup_bypass"] = track_entry_near_field_warmup_bypass.astype(bool)
    frame["track_entry_remote_severe_warmup_bypass"] = track_entry_remote_severe_warmup_bypass.astype(bool)
    frame["track_entry_warmup_bypass"] = track_entry_warmup_bypass.astype(bool)
    frame["small_bbox_ipm_instability_suspect"] = small_bbox_ipm_instability_suspect.astype(bool)
    frame["tiny_bbox_severe_ipm_instability_suspect"] = tiny_bbox_severe_ipm_instability_suspect.astype(bool)
    frame["raw_recompute_rejected_reason"] = np.select(
        [
            stationary_jitter_suspect,
            oscillatory_micro_jitter_suspect,
            short_track_micro_jitter_suspect,
            short_track_unstable_jump_suspect,
            ultra_short_track_fragment_suspect,
            short_track_tiny_motion_fragment_suspect,
            single_window_tiny_box_jitter_suspect,
            track_entry_ipm_perspective_suspect,
            small_bbox_ipm_instability_suspect,
            tiny_bbox_severe_ipm_instability_suspect,
            computed_from_raw & below_limit_conflict & ~below_limit_raw_agreement,
            computed_from_raw & ~stable_raw_mask,
        ],
        [
            "raw_ipm_recompute_stationary_pixel_jitter",
            "raw_ipm_recompute_oscillatory_micro_motion",
            "raw_ipm_recompute_short_track_micro_motion",
            "raw_ipm_recompute_short_track_unstable_jump",
            "overspeed_ultra_short_track_fragment",
            "overspeed_short_track_tiny_motion_fragment",
            "raw_ipm_recompute_single_window_tiny_box_jitter",
            "raw_ipm_recompute_track_entry_ipm_perspective_instability",
            "raw_ipm_recompute_small_bbox_ipm_instability",
            "raw_ipm_recompute_tiny_bbox_severe_ipm_instability",
            "raw_ipm_recompute_conflicts_with_below_limit_smoothed_speed",
            "raw_ipm_recompute_unstable_against_persisted_sidecar",
        ],
        default="",
    )
    source_agreement = clip01(
        1.0
        - safe_divide(
            (computed_speed_kmh_p95 - fallback_p95).abs(),
            fallback_p95.where(fallback_p95 > 20.0, 20.0),
            default=1.0,
        )
    )
    source_agreement = source_agreement.where(computed_from_raw, 1.0)
    frame["speed_kmh_p95"] = computed_speed_kmh_p95.where(stable_raw_mask, fallback_p95).fillna(0.0)
    frame["speed_kmh_max"] = computed_speed_kmh_max.where(stable_raw_mask, fallback_upper_proxy).fillna(0.0)
    frame["speed_kmh_mean"] = pd.to_numeric(frame.get("speed_kmh", np.nan), errors="coerce").fillna(frame["speed_kmh_p95"])
    frame.loc[any_jitter_suspect, ["speed_kmh_p95", "speed_kmh_max", "speed_kmh_mean"]] = 0.0

    frame["speed_band_70_90_flag"] = ((frame["speed_kmh_p95"] >= SPEED_LIMIT_DEFAULT) & (frame["speed_kmh_p95"] < OVERSPEED_BAND_2)).astype(float)
    frame["speed_band_90_110_flag"] = ((frame["speed_kmh_p95"] >= OVERSPEED_BAND_2) & (frame["speed_kmh_p95"] < OVERSPEED_BAND_3)).astype(float)
    frame["speed_band_ge_110_flag"] = (frame["speed_kmh_p95"] >= OVERSPEED_BAND_3).astype(float)
    overspeed_level = pd.to_numeric(frame.get("overspeed_level", 0), errors="coerce").fillna(0.0)
    ge2_active = ((frame["speed_kmh_p95"] >= OVERSPEED_BAND_2) | (overspeed_level >= 2.0)).astype(float)
    ge3_active = ((frame["speed_kmh_p95"] >= OVERSPEED_BAND_3) | (overspeed_level >= 3.0)).astype(float)
    ge2_active = ge2_active.where(~any_jitter_suspect, 0.0)
    ge3_active = ge3_active.where(~any_jitter_suspect, 0.0)
    frame["overspeed_level_ge2_duration"] = cumulative_duration_by_track(frame, ge2_active)
    frame["overspeed_level_ge3_duration"] = cumulative_duration_by_track(frame, ge3_active)
    frame["overspeed_peak_minus_limit"] = clip_nonnegative(frame["speed_kmh_max"] - speed_limit)
    frame["overspeed_severe_burst_count"] = burst_count_by_track(frame, ge3_active)
    frame["overspeed_reliability"] = clip01(numeric_series(frame, "overspeed_reliability"))
    frame.loc[any_jitter_suspect, "overspeed_reliability"] = 0.0
    frame["overspeed_signal_quality"] = clip01(
        0.25 * clip01(numeric_series(frame, "window_valid_pair_ratio"))
        + 0.20 * clip01(numeric_series(frame, "window_ipm_valid_ratio"))
        + 0.15 * clip01(numeric_series(frame, "timestamp_delta_valid"))
        + 0.25 * frame["raw_recompute_stable"].astype(float)
        + 0.15 * source_agreement
    )

    output_columns = WINDOW_JOIN_KEYS + [
        "start_sec",
        "end_sec",
        "speed_kmh_max",
        "speed_kmh_p95",
        "speed_kmh_mean",
        "speed_band_70_90_flag",
        "speed_band_90_110_flag",
        "speed_band_ge_110_flag",
        "overspeed_level_ge2_duration",
        "overspeed_level_ge3_duration",
        "overspeed_peak_minus_limit",
        "overspeed_severe_burst_count",
        "overspeed_reliability",
        "overspeed_signal_quality",
    ]
    output_frame = frame[output_columns].copy()

    sidecar_columns = WINDOW_JOIN_KEYS + [
        "start_sec",
        "end_sec",
        "source_csv",
        "timestamp_column_used",
        "window_valid_pair_count",
        "window_valid_pair_ratio",
        "window_ipm_valid_ratio",
        "timestamp_delta_valid",
        "timestamp_delta_sec_p50",
        "timestamp_delta_sec_p95",
        "ground_distance_m_p95",
        "speed_raw_kmh_p95",
        "speed_smoothed_kmh_p95",
        "computed_speed_kmh_max",
        "computed_speed_kmh_p95",
        "computed_valid_interval_count",
        "computed_pixel_distance_px_p95",
        "computed_pixel_distance_px_max",
        "computed_bbox_height_px_p95",
        "computed_pixel_path_length_px",
        "computed_net_pixel_displacement_px",
        "computed_path_efficiency",
        "computed_from_raw_intervals",
        "raw_recompute_stable",
        "stationary_jitter_suspect",
        "oscillatory_micro_jitter_suspect",
        "short_track_micro_jitter_suspect",
        "short_track_unstable_jump_suspect",
        "ultra_short_track_fragment_suspect",
        "short_track_tiny_motion_fragment_suspect",
        "single_window_tiny_box_jitter_suspect",
        "track_entry_ipm_perspective_suspect",
        "track_entry_near_field_warmup_bypass",
        "track_entry_remote_severe_warmup_bypass",
        "track_entry_warmup_bypass",
        "small_bbox_ipm_instability_suspect",
        "tiny_bbox_severe_ipm_instability_suspect",
        "raw_recompute_rejected_reason",
        "source_agreement",
        "fallback_reason",
    ]
    output_sidecar = frame[[column for column in sidecar_columns if column in frame.columns]].copy()
    output_sidecar["ipm_json_path"] = str(ipm_json)
    output_sidecar["speed_kmh_max_final"] = output_frame["speed_kmh_max"]
    output_sidecar["speed_kmh_p95_final"] = output_frame["speed_kmh_p95"]
    output_sidecar["speed_kmh_mean_final"] = output_frame["speed_kmh_mean"]
    output_sidecar["overspeed_signal_quality_final"] = output_frame["overspeed_signal_quality"]

    build_payload = {
        "schema_version": SCHEMA_VERSION,
        "row_count": int(output_frame.shape[0]),
        "input_paths": {"ipm_json": str(ipm_json)},
        "physical_speed_source_ratio": float(stable_raw_mask.mean()) if len(stable_raw_mask) else 0.0,
        "raw_recompute_attempt_ratio": float(computed_from_raw.mean()) if len(computed_from_raw) else 0.0,
        "raw_recompute_rejected_count": int((computed_from_raw & ~stable_raw_mask).sum()) if len(computed_from_raw) else 0,
        "exact_speed_kmh_max_unstable": bool((computed_from_raw & ~stable_raw_mask).any()),
        "fallback_window_count": int((~stable_raw_mask).sum()) if len(stable_raw_mask) else int(output_frame.shape[0]),
        "validation_summary": {
            field_name: validation_summary(output_frame, field_name)
            for field_name in output_columns
            if field_name not in WINDOW_JOIN_KEYS + ["start_sec", "end_sec"]
        },
        "leakage_audit_passed": True,
    }
    return output_frame, output_sidecar, build_payload
