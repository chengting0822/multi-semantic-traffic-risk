"""Accepted IPM speed-profile formulas used by the research pipeline."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


GUI_HOMOGRAPHY_KEY = "homography_img_to_world"
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov", ".ts")
DEFAULT_ANOMALY_FPS = 30.0
SPEED_LIMIT_KMH_GUI = 70.0
OVERSPEED_BINS_GUI = (70.0, 90.0, 110.0)
OVERSPEED_MIN_RATIO_GUI = 0.15
OVERSPEED_ROBUST_QUANTILE_GUI = 0.95
SPEED_SMOOTH_MEDIAN_WINDOW_GUI = 5
SPEED_SMOOTH_EMA_TAU_SEC_GUI = 0.45
STOPPED_SPEED_THR_GUI = 2.0
MAX_REASONABLE_SPEED_KMH = 220.0


def clip(value: float, lower: float, upper: float) -> float:
    return max(float(lower), min(float(value), float(upper)))


def clip01(value: float) -> float:
    return clip(value, 0.0, 1.0)


def gui_overspeed_level(speed_value: float, overspeed_ratio: float) -> int:
    if overspeed_ratio < OVERSPEED_MIN_RATIO_GUI:
        return 0
    if speed_value <= OVERSPEED_BINS_GUI[0]:
        return 0
    if speed_value <= OVERSPEED_BINS_GUI[1]:
        return 1
    if speed_value <= OVERSPEED_BINS_GUI[2]:
        return 2
    return 3


def overspeed_level_from_speed_kmh(speed_value: float) -> int:
    if not math.isfinite(float(speed_value)):
        return 0
    speed_value = float(speed_value)
    if speed_value < SPEED_LIMIT_KMH_GUI:
        return 0
    if speed_value < OVERSPEED_BINS_GUI[1]:
        return 1
    if speed_value <= OVERSPEED_BINS_GUI[2]:
        return 2
    return 3


def rolling_median_valid(values: np.ndarray, window_size: int) -> np.ndarray:
    if len(values) == 0 or window_size <= 1:
        return values.copy()
    radius = max(window_size // 2, 0)
    out = np.empty_like(values, dtype=np.float64)
    for index in range(len(values)):
        lo = max(0, index - radius)
        hi = min(len(values), index + radius + 1)
        out[index] = float(np.median(values[lo:hi]))
    return out


def smooth_speed_profile_kmh(
    raw_speed_kmh: np.ndarray,
    timestamps_sec: np.ndarray,
    *,
    median_window: int = SPEED_SMOOTH_MEDIAN_WINDOW_GUI,
    ema_tau_sec: float = SPEED_SMOOTH_EMA_TAU_SEC_GUI,
) -> np.ndarray:
    smoothed = np.full_like(raw_speed_kmh, np.nan, dtype=np.float64)
    valid_idx = np.flatnonzero(np.isfinite(raw_speed_kmh) & np.isfinite(timestamps_sec))
    if len(valid_idx) == 0:
        return smoothed

    valid_speed = raw_speed_kmh[valid_idx].astype(np.float64)
    valid_ts = timestamps_sec[valid_idx].astype(np.float64)
    median_speed = rolling_median_valid(valid_speed, median_window)
    ema_speed = np.empty_like(median_speed, dtype=np.float64)
    ema_speed[0] = median_speed[0]

    tau_sec = max(float(ema_tau_sec), 1e-3)
    for index in range(1, len(median_speed)):
        dt_sec = max(float(valid_ts[index] - valid_ts[index - 1]), 1e-3)
        alpha = 1.0 - np.exp(-dt_sec / tau_sec)
        ema_speed[index] = alpha * median_speed[index] + (1.0 - alpha) * ema_speed[index - 1]

    smoothed[valid_idx] = ema_speed
    return smoothed


def canonical_video_id(video_id: Any) -> str:
    return str(video_id or "").strip()


def find_video_path(video_dir: Any, video_id: Any) -> Path | None:
    directory = Path(video_dir) if video_dir else None
    if directory is None or not directory.is_dir():
        return None
    stem = canonical_video_id(video_id)
    if not stem:
        return None
    for extension in VIDEO_EXTENSIONS:
        candidate = directory / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    return None


def read_video_metadata(video_path: Any) -> dict[str, Any]:
    try:
        import cv2
    except Exception as exc:
        return {
            "fps": DEFAULT_ANOMALY_FPS,
            "frame_count_if_available": "",
            "duration_sec_if_available": "",
            "fps_read_success": False,
            "fallback_used": True,
            "notes": f"cv2 import failed: {exc}",
        }

    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {
            "fps": DEFAULT_ANOMALY_FPS,
            "frame_count_if_available": "",
            "duration_sec_if_available": "",
            "fps_read_success": False,
            "fallback_used": True,
            "notes": "video open failed",
        }
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        ok = math.isfinite(fps) and fps > 0
        fps_value = fps if ok else DEFAULT_ANOMALY_FPS
        duration = frame_count / fps_value if frame_count > 0 and fps_value > 0 else ""
        return {
            "fps": float(fps_value),
            "frame_count_if_available": frame_count if frame_count > 0 else "",
            "duration_sec_if_available": duration,
            "fps_read_success": bool(ok),
            "fallback_used": not ok,
            "notes": "" if ok else "invalid fps from video",
        }
    finally:
        cap.release()


def resolve_csv_fps(
    csv_path: Any,
    *,
    video_dir: Any = None,
    video_id: Any = "",
    fps_from_video: bool = True,
) -> tuple[float, dict[str, Any]]:
    resolved_video_id = canonical_video_id(video_id or Path(csv_path).stem)
    video_path = find_video_path(video_dir, resolved_video_id)
    if fps_from_video and video_path is not None:
        metadata = read_video_metadata(video_path)
        metadata["video_id"] = resolved_video_id
        metadata["video_path"] = str(video_path)
        return float(metadata["fps"]), metadata

    return DEFAULT_ANOMALY_FPS, {
        "video_id": resolved_video_id,
        "video_path": str(video_path) if video_path else "",
        "fps_read_success": False,
        "fallback_used": True,
        "notes": "fps_from_video disabled or video not found",
    }


def ensure_timestamp_sec(df: pd.DataFrame, fps: float) -> tuple[pd.DataFrame, str]:
    out = df.copy()
    if "timestamp_sec" in out.columns:
        out["timestamp_sec"] = pd.to_numeric(out["timestamp_sec"], errors="coerce")
        return out, "timestamp_sec"
    if "frame" not in out.columns:
        raise ValueError("CSV must contain frame when timestamp_sec is absent")
    fps_value = float(fps) if fps and fps > 0 else DEFAULT_ANOMALY_FPS
    out["timestamp_sec"] = pd.to_numeric(out["frame"], errors="coerce") / fps_value
    return out, "frame_over_fps"


def load_homography_img_to_world(
    ipm_json_path: Any,
    *,
    homography_key: str = GUI_HOMOGRAPHY_KEY,
) -> tuple[np.ndarray | None, str]:
    path = Path(ipm_json_path)
    if not path.is_file():
        return None, "無法計算：找不到 IPM0327.json"
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        return None, f"無法計算：IPM0327.json 讀取失敗：{exc}"
    homography = np.asarray(payload.get(homography_key, []), dtype=np.float64)
    if homography.shape != (3, 3) or not np.all(np.isfinite(homography)):
        return None, f"無法計算：{homography_key} 缺失"
    return homography, ""


def _failure_stats(
    *,
    video_id: str,
    track_id: int,
    ipm_source_path: str,
    ipm_source_type: str,
    homography_key: str,
    failure_reason: str,
    timestamp_column_used: str,
    timestamp_count: int = 0,
    invalid_delta_t_count: int = 0,
) -> dict[str, Any]:
    return {
        "loaded_ipm_path": ipm_source_path,
        "ipm_source_type": ipm_source_type,
        "homography_key": homography_key,
        "timestamp_column_used": timestamp_column_used,
        "frame_column_used": "frame",
        "speed_formula_mode": "timestamp_based",
        "track_id": int(track_id),
        "video_id": video_id,
        "timestamp_count": int(timestamp_count),
        "valid_delta_t_count": 0,
        "invalid_delta_t_count": int(invalid_delta_t_count),
        "speed_kmh_min": "",
        "speed_kmh_max": "",
        "overspeed_ratio": "",
        "overspeed_level": "",
        "display_status": "failed",
        "failure_reason": failure_reason,
    }


def compute_track_speed_profile(
    track_df: pd.DataFrame,
    H_img_to_world: np.ndarray | None,
    *,
    video_id: Any,
    track_id: Any,
    ipm_source_path: str,
    ipm_source_type: str = "IPM0327.json",
    homography_key: str = GUI_HOMOGRAPHY_KEY,
    timestamp_column_used: str = "timestamp_sec",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required_columns = {"frame", "x1", "x2", "y2", "timestamp_sec"}
    missing = sorted(required_columns - set(track_df.columns))
    if missing:
        raise ValueError(f"track_df missing required columns: {missing}")

    ordered = track_df.sort_values("frame").reset_index(drop=True).copy()
    frames = ordered["frame"].to_numpy(dtype=np.int32)
    center_x = ((ordered["x1"].to_numpy(dtype=np.float64) + ordered["x2"].to_numpy(dtype=np.float64)) * 0.5)
    foot_y = ordered["y2"].to_numpy(dtype=np.float64)
    timestamps = pd.to_numeric(ordered["timestamp_sec"], errors="coerce").to_numpy(dtype=np.float64)
    timestamp_count = int(np.isfinite(timestamps).sum())
    video_id_text = canonical_video_id(video_id)
    track_id_int = int(track_id)

    evidence = pd.DataFrame(
        {
            "frame": frames,
            "timestamp_sec": timestamps,
            "center_x": center_x,
            "foot_y": foot_y,
            "ground_x_m": np.nan,
            "ground_y_m": np.nan,
            "delta_t_sec": np.nan,
            "ground_distance_m": np.nan,
            "raw_speed_kmh": np.nan,
            "speed_kmh": np.nan,
            "valid_pair": 0,
            "ipm_point_valid": 0,
            "extreme_speed_flag": 0,
            "speed_reason": "",
        }
    )
    if len(evidence) > 0:
        evidence.loc[evidence.index[0], "speed_reason"] = "無法計算：前一個有效點不存在"

    if H_img_to_world is None:
        stats = _failure_stats(
            video_id=video_id_text,
            track_id=track_id_int,
            ipm_source_path=ipm_source_path,
            ipm_source_type=ipm_source_type,
            homography_key=homography_key,
            failure_reason="無法計算：找不到 IPM0327.json 或 homography_img_to_world 缺失",
            timestamp_column_used=timestamp_column_used,
            timestamp_count=timestamp_count,
        )
        evidence["speed_reason"] = stats["failure_reason"]
        return evidence, stats

    if len(frames) < 2 or timestamp_count < 2:
        stats = _failure_stats(
            video_id=video_id_text,
            track_id=track_id_int,
            ipm_source_path=ipm_source_path,
            ipm_source_type=ipm_source_type,
            homography_key=homography_key,
            failure_reason="無法計算：track 點數不足",
            timestamp_column_used=timestamp_column_used,
            timestamp_count=timestamp_count,
        )
        evidence["speed_reason"] = stats["failure_reason"]
        return evidence, stats

    points_xy = np.stack([center_x, foot_y], axis=1)
    homogeneous_points = np.concatenate(
        [points_xy, np.ones((len(points_xy), 1), dtype=np.float64)],
        axis=1,
    )
    projected = homogeneous_points @ H_img_to_world.T
    denominator = projected[:, 2:3]
    denominator = np.where(np.abs(denominator) < 1e-9, np.nan, denominator)
    ground_xy = projected[:, :2] / denominator
    evidence["ground_x_m"] = ground_xy[:, 0]
    evidence["ground_y_m"] = ground_xy[:, 1]
    evidence["ipm_point_valid"] = (
        np.isfinite(ground_xy[:, 0]) & np.isfinite(ground_xy[:, 1])
    ).astype(np.int8)

    delta_ground = np.diff(ground_xy, axis=0)
    delta_time = np.diff(timestamps)
    ground_distance = np.linalg.norm(delta_ground, axis=1)
    valid_pairs = np.isfinite(ground_distance) & np.isfinite(delta_time) & (delta_time > 1e-8)
    invalid_delta_count = int((np.isfinite(delta_time) & (delta_time <= 1e-8)).sum())
    delta_t_values = np.full(len(frames), np.nan, dtype=np.float64)
    delta_t_values[1:] = delta_time
    ground_distance_values = np.full(len(frames), np.nan, dtype=np.float64)
    ground_distance_values[1:] = ground_distance
    valid_pair_values = np.zeros(len(frames), dtype=np.int8)
    valid_pair_values[1:] = valid_pairs.astype(np.int8)
    evidence["delta_t_sec"] = delta_t_values
    evidence["ground_distance_m"] = ground_distance_values
    evidence["valid_pair"] = valid_pair_values

    pair_speed_kmh = np.full(len(delta_time), np.nan, dtype=np.float64)
    pair_speed_kmh[valid_pairs] = ground_distance[valid_pairs] / delta_time[valid_pairs] * 3.6
    raw_speed = np.full(len(frames), np.nan, dtype=np.float64)
    raw_speed[1:] = pair_speed_kmh
    evidence["raw_speed_kmh"] = raw_speed

    if not valid_pairs.any():
        failure_reason = (
            "無法計算：delta_t_sec <= 0"
            if invalid_delta_count
            else "無法計算：前一個有效點不存在"
        )
        stats = _failure_stats(
            video_id=video_id_text,
            track_id=track_id_int,
            ipm_source_path=ipm_source_path,
            ipm_source_type=ipm_source_type,
            homography_key=homography_key,
            failure_reason=failure_reason,
            timestamp_column_used=timestamp_column_used,
            timestamp_count=timestamp_count,
            invalid_delta_t_count=invalid_delta_count,
        )
        evidence["speed_reason"] = failure_reason
        return evidence, stats

    for index in range(1, len(frames)):
        if np.isfinite(raw_speed[index]):
            continue
        if not np.isfinite(timestamps[index]) or not np.isfinite(timestamps[index - 1]):
            evidence.at[index, "speed_reason"] = "無法計算：缺少 timestamp_sec"
        elif timestamps[index] - timestamps[index - 1] <= 1e-8:
            evidence.at[index, "speed_reason"] = "無法計算：delta_t_sec <= 0"
        else:
            evidence.at[index, "speed_reason"] = "無法計算：IPM ground point 無效"

    smoothed_speed = smooth_speed_profile_kmh(raw_speed, timestamps)
    smooth_clean_speed = smoothed_speed[np.isfinite(smoothed_speed)]
    raw_clean_speed = pair_speed_kmh[np.isfinite(pair_speed_kmh)]
    if len(smooth_clean_speed) == 0:
        smoothed_speed = raw_speed.copy()
        smooth_clean_speed = raw_clean_speed
    evidence["speed_kmh"] = smoothed_speed
    extreme_speed_mask = np.isfinite(smoothed_speed) & (smoothed_speed > MAX_REASONABLE_SPEED_KMH)
    evidence["extreme_speed_flag"] = extreme_speed_mask.astype(np.int8)

    valid_speed_mask = np.isfinite(smoothed_speed)
    evidence.loc[valid_speed_mask, "speed_reason"] = ""

    overspeed_ratio = float(np.mean(smooth_clean_speed > SPEED_LIMIT_KMH_GUI))
    robust_speed = float(np.quantile(smooth_clean_speed, OVERSPEED_ROBUST_QUANTILE_GUI))
    stopped_ratio = float(np.mean(smooth_clean_speed <= STOPPED_SPEED_THR_GUI))
    stats = {
        "loaded_ipm_path": ipm_source_path,
        "ipm_source_type": ipm_source_type,
        "homography_key": homography_key,
        "timestamp_column_used": timestamp_column_used,
        "frame_column_used": "frame",
        "speed_formula_mode": "timestamp_based",
        "track_id": track_id_int,
        "video_id": video_id_text,
        "timestamp_count": timestamp_count,
        "valid_delta_t_count": int(valid_pairs.sum()),
        "invalid_delta_t_count": invalid_delta_count,
        "speed_smoothing_strategy": "rolling_median_then_time_ema",
        "speed_smoothing_median_window": SPEED_SMOOTH_MEDIAN_WINDOW_GUI,
        "speed_smoothing_ema_tau_sec": SPEED_SMOOTH_EMA_TAU_SEC_GUI,
        "raw_speed_kmh_min": float(np.min(raw_clean_speed)),
        "raw_speed_kmh_max": float(np.max(raw_clean_speed)),
        "speed_kmh_min": float(np.min(smooth_clean_speed)),
        "speed_kmh_max": float(np.max(smooth_clean_speed)),
        "speed_kmh_p95": robust_speed,
        "overspeed_ratio": overspeed_ratio,
        "overspeed_level": gui_overspeed_level(robust_speed, overspeed_ratio),
        "stopped_ratio": stopped_ratio,
        "valid_speed_ratio": float(np.mean(valid_speed_mask)),
        "ipm_point_valid_ratio": float(np.mean(evidence["ipm_point_valid"].to_numpy(dtype=np.float64))),
        "extreme_speed_sample_count": int(extreme_speed_mask.sum()),
        "display_status": "ok",
        "failure_reason": "",
    }
    return evidence, stats
