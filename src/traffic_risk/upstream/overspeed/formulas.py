"""Accepted window aggregation formulas extracted from overspeed module v0."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


from traffic_risk.upstream.overspeed.ipm_speed_utils import (
    MAX_REASONABLE_SPEED_KMH,
    SPEED_LIMIT_KMH_GUI,
    clip01,
    overspeed_level_from_speed_kmh,
    resolve_csv_fps,
)
from traffic_risk.upstream.overspeed.feature_contract import (
    FORBIDDEN_LEAKAGE_COLUMNS,
    JOIN_KEY_FIELDNAMES,
)


WINDOW_EPS_SEC = 1e-6
TRACK_VALID_MIN_RELIABILITY = 0.45

REQUIRED_RAW_COLUMNS = (
    "track_id",
    "x1",
    "y1",
    "x2",
    "y2",
)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def validate_input_columns(df: pd.DataFrame) -> None:
    missing = [field for field in REQUIRED_RAW_COLUMNS if field not in df.columns]
    if missing:
        raise ValueError(f"missing required source columns: {missing}")
    if "timestamp_sec" not in df.columns and "frame" not in df.columns:
        raise ValueError("missing time information: require timestamp_sec or frame")
    forbidden = [field for field in FORBIDDEN_LEAKAGE_COLUMNS if field in df.columns]
    if forbidden:
        raise ValueError(f"forbidden leakage columns found in source csv: {forbidden}")


def normalize_reference_windows(reference_df: pd.DataFrame) -> pd.DataFrame:
    out = reference_df.copy()
    for field_name in JOIN_KEY_FIELDNAMES:
        if field_name not in out.columns:
            raise ValueError(f"reference windows missing required column: {field_name}")
    out["case_key"] = out["case_key"].astype(str).str.strip()
    out["video_id"] = out["video_id"].astype(str).str.strip()
    out["track_id"] = out["track_id"].map(lambda value: str(int(float(value))))
    out["source_type"] = out["source_type"].astype(str).str.strip()
    out["source_id"] = out["source_id"].astype(str).str.strip()
    out["ts_window_idx"] = out["ts_window_idx"].map(lambda value: int(float(value)))
    out["start_sec"] = out["start_sec"].map(lambda value: round(float(value), 6))
    out["end_sec"] = out["end_sec"].map(lambda value: round(float(value), 6))
    return out


def normalized_join_key_values(row_like: Any) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for field_name in JOIN_KEY_FIELDNAMES:
        value = row_like[field_name]
        if field_name in {"track_id", "ts_window_idx"}:
            normalized[field_name] = int(float(value))
        elif field_name in {"start_sec", "end_sec"}:
            normalized[field_name] = round(float(value), 6)
        else:
            normalized[field_name] = str(value).strip()
    return normalized


def extract_video_id(raw_df: pd.DataFrame, csv_path: Path) -> str:
    for column_name in ("source_video_id", "numeric_video_id"):
        if column_name in raw_df.columns:
            value = str(raw_df[column_name].iloc[0]).strip()
            if value and value.lower() != "nan":
                try:
                    return str(int(float(value)))
                except ValueError:
                    return value
    return csv_path.stem


def extract_case_key(track_df: pd.DataFrame, video_id: str, track_id: int) -> str:
    if "canonical_case_key" in track_df.columns:
        value = str(track_df["canonical_case_key"].iloc[0]).strip()
        if value and value.lower() != "nan":
            return value
    return f"{video_id}/{track_id}"


def extract_fps_value(
    raw_df: pd.DataFrame,
    csv_path: Path,
    video_id: str,
    video_dir: str,
) -> tuple[float, dict[str, Any]]:
    if "fps" in raw_df.columns:
        fps_value = safe_float(raw_df["fps"].iloc[0], 0.0)
        if fps_value > 0:
            return fps_value, {
                "fps": fps_value,
                "fps_source": "csv_column",
                "fps_read_success": True,
                "fallback_used": False,
                "video_path": "",
                "notes": "fps column present in timestamp csv",
            }
    fps_value, metadata = resolve_csv_fps(
        csv_path,
        video_dir=video_dir,
        video_id=video_id,
        fps_from_video=True,
    )
    metadata["fps_source"] = "video_metadata_or_default"
    return fps_value, metadata


def build_reference_case_index(reference_df: pd.DataFrame) -> dict[str, dict[str, pd.DataFrame]]:
    by_video: dict[str, dict[str, pd.DataFrame]] = {}
    for video_id, video_group in reference_df.groupby("video_id", sort=False):
        by_video[str(video_id)] = {
            str(case_key): case_group.sort_values(["ts_window_idx", "end_sec"]).reset_index(drop=True)
            for case_key, case_group in video_group.groupby("case_key", sort=False)
        }
    return by_video


def make_failure_row(window_row: pd.Series, failure_reason: str) -> dict[str, Any]:
    return {
        **normalized_join_key_values(window_row),
        "speed_kmh": 0.0,
        "speed_limit_kmh": SPEED_LIMIT_KMH_GUI,
        "overspeed_ratio": 0.0,
        "overspeed_confirmed": 0,
        "overspeed_level": 0,
        "overspeed_reliability": 0.0,
        "overspeed_failure_reason": failure_reason,
        "timestamp_delta_valid": 0,
        "ipm_source_valid": 0,
        "track_motion_valid": 0,
    }


def make_failure_sidecar_row(
    window_row: pd.Series,
    *,
    source_csv: str,
    timestamp_column_used: str,
    fps_value: float,
    failure_reason: str,
) -> dict[str, Any]:
    return {
        **normalized_join_key_values(window_row),
        "source_csv": source_csv,
        "timestamp_column_used": timestamp_column_used,
        "fps": float(fps_value),
        "window_point_count": 0,
        "window_valid_speed_count": 0,
        "window_valid_pair_count": 0,
        "window_valid_speed_ratio": 0.0,
        "window_valid_pair_ratio": 0.0,
        "window_ipm_valid_ratio": 0.0,
        "window_extreme_speed_count": 0,
        "timestamp_delta_sec_p50": 0.0,
        "timestamp_delta_sec_p95": 0.0,
        "ground_distance_m_p95": 0.0,
        "ipm_x_last": 0.0,
        "ipm_y_last": 0.0,
        "bbox_bottom_center_x_last": 0.0,
        "bbox_bottom_center_y_last": 0.0,
        "speed_raw_kmh_p95": 0.0,
        "speed_smoothed_kmh_p95": 0.0,
        "track_display_status": "failed",
        "track_failure_reason": failure_reason,
    }


def compute_window_reliability(
    *,
    valid_speed_ratio: float,
    valid_pair_ratio: float,
    ipm_valid_ratio: float,
    track_valid_speed_ratio: float,
    timestamp_column_used: str,
    extreme_speed_count: int,
) -> float:
    fallback_penalty = 0.10 if timestamp_column_used != "timestamp_sec" else 0.0
    extreme_penalty = 0.30 if extreme_speed_count > 0 else 0.0
    reliability = (
        (0.45 * clip01(valid_speed_ratio))
        + (0.20 * clip01(valid_pair_ratio))
        + (0.20 * clip01(ipm_valid_ratio))
        + (0.15 * clip01(track_valid_speed_ratio))
        - fallback_penalty
        - extreme_penalty
    )
    return clip01(reliability)


def aggregate_window(
    window_row: pd.Series,
    evidence_df: pd.DataFrame,
    track_stats: dict[str, Any],
    *,
    source_csv: str,
    timestamp_column_used: str,
    fps_value: float,
    ipm_loaded: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if str(track_stats.get("display_status", "failed")) != "ok":
        failure_reason = str(track_stats.get("failure_reason", "overspeed module failed"))
        return (
            make_failure_row(window_row, failure_reason),
            make_failure_sidecar_row(
                window_row,
                source_csv=source_csv,
                timestamp_column_used=timestamp_column_used,
                fps_value=fps_value,
                failure_reason=failure_reason,
            ),
        )

    ts_values = pd.to_numeric(evidence_df["timestamp_sec"], errors="coerce")
    mask = (ts_values >= float(window_row["start_sec"]) - WINDOW_EPS_SEC) & (
        ts_values <= float(window_row["end_sec"]) + WINDOW_EPS_SEC
    )
    window_df = evidence_df.loc[mask].copy().reset_index(drop=True)
    if window_df.empty:
        failure_reason = "window_has_no_observed_points"
        return (
            make_failure_row(window_row, failure_reason),
            make_failure_sidecar_row(
                window_row,
                source_csv=source_csv,
                timestamp_column_used=timestamp_column_used,
                fps_value=fps_value,
                failure_reason=failure_reason,
            ),
        )

    speed_values = pd.to_numeric(window_df["speed_kmh"], errors="coerce").to_numpy(dtype=np.float64)
    raw_speed_values = pd.to_numeric(window_df["raw_speed_kmh"], errors="coerce").to_numpy(dtype=np.float64)
    delta_values = pd.to_numeric(window_df["delta_t_sec"], errors="coerce").to_numpy(dtype=np.float64)
    ground_distance_values = pd.to_numeric(window_df["ground_distance_m"], errors="coerce").to_numpy(dtype=np.float64)
    ipm_x_values = pd.to_numeric(window_df["ground_x_m"], errors="coerce").to_numpy(dtype=np.float64)
    ipm_y_values = pd.to_numeric(window_df["ground_y_m"], errors="coerce").to_numpy(dtype=np.float64)
    center_x_values = pd.to_numeric(window_df["center_x"], errors="coerce").to_numpy(dtype=np.float64)
    foot_y_values = pd.to_numeric(window_df["foot_y"], errors="coerce").to_numpy(dtype=np.float64)

    valid_speed_mask = np.isfinite(speed_values)
    valid_speed_values = speed_values[valid_speed_mask]
    valid_pair_mask = np.isfinite(delta_values)
    positive_delta_mask = np.isfinite(delta_values) & (delta_values > 1e-8)
    ipm_valid_mask = np.isfinite(ipm_x_values) & np.isfinite(ipm_y_values)
    extreme_speed_count = int((np.isfinite(speed_values) & (speed_values > MAX_REASONABLE_SPEED_KMH)).sum())

    window_point_count = int(window_df.shape[0])
    valid_speed_count = int(valid_speed_mask.sum())
    valid_pair_count = int(positive_delta_mask.sum())
    valid_speed_ratio = float(valid_speed_count / max(window_point_count, 1))
    valid_pair_ratio = float(valid_pair_count / max(max(window_point_count - 1, 1), 1))
    ipm_valid_ratio = float(np.mean(ipm_valid_mask)) if window_point_count else 0.0

    if valid_speed_count == 0:
        failure_reason = "window_has_no_valid_speed_samples"
        return (
            make_failure_row(window_row, failure_reason),
            make_failure_sidecar_row(
                window_row,
                source_csv=source_csv,
                timestamp_column_used=timestamp_column_used,
                fps_value=fps_value,
                failure_reason=failure_reason,
            ),
        )

    representative_speed = float(np.quantile(valid_speed_values, 0.95))
    overspeed_ratio = float(np.mean(valid_speed_values >= SPEED_LIMIT_KMH_GUI))
    track_motion_valid = int(extreme_speed_count == 0)
    ipm_source_valid = int(ipm_loaded and bool(ipm_valid_mask.any()))
    timestamp_delta_valid = int(bool(positive_delta_mask.any()))
    failure_reason = ""

    if representative_speed > MAX_REASONABLE_SPEED_KMH or not math.isfinite(representative_speed):
        representative_speed = 0.0
        overspeed_ratio = 0.0
        track_motion_valid = 0
        failure_reason = f"extreme_speed_kmh_gt_{MAX_REASONABLE_SPEED_KMH:.0f}"

    reliability = compute_window_reliability(
        valid_speed_ratio=valid_speed_ratio,
        valid_pair_ratio=valid_pair_ratio,
        ipm_valid_ratio=ipm_valid_ratio,
        track_valid_speed_ratio=safe_float(track_stats.get("valid_speed_ratio"), 0.0),
        timestamp_column_used=timestamp_column_used,
        extreme_speed_count=extreme_speed_count,
    )
    if not timestamp_delta_valid and not failure_reason:
        failure_reason = "timestamp_delta_invalid_in_window"
    if not ipm_source_valid and not failure_reason:
        failure_reason = "ipm_invalid_in_window"
    if not track_motion_valid and not failure_reason:
        failure_reason = "track_motion_invalid"
    if failure_reason:
        reliability = 0.0

    level = overspeed_level_from_speed_kmh(representative_speed) if not failure_reason else 0
    confirmed = int(
        level > 0
        and overspeed_ratio >= 0.15
        and reliability >= TRACK_VALID_MIN_RELIABILITY
        and timestamp_delta_valid == 1
        and ipm_source_valid == 1
        and track_motion_valid == 1
    )

    module_row = {
        **normalized_join_key_values(window_row),
        "speed_kmh": representative_speed,
        "speed_limit_kmh": SPEED_LIMIT_KMH_GUI,
        "overspeed_ratio": overspeed_ratio,
        "overspeed_confirmed": confirmed,
        "overspeed_level": level,
        "overspeed_reliability": reliability,
        "overspeed_failure_reason": failure_reason,
        "timestamp_delta_valid": timestamp_delta_valid,
        "ipm_source_valid": ipm_source_valid,
        "track_motion_valid": track_motion_valid,
    }
    sidecar_row = {
        **normalized_join_key_values(window_row),
        "source_csv": source_csv,
        "timestamp_column_used": timestamp_column_used,
        "fps": float(fps_value),
        "window_point_count": window_point_count,
        "window_valid_speed_count": valid_speed_count,
        "window_valid_pair_count": valid_pair_count,
        "window_valid_speed_ratio": valid_speed_ratio,
        "window_valid_pair_ratio": valid_pair_ratio,
        "window_ipm_valid_ratio": ipm_valid_ratio,
        "window_extreme_speed_count": extreme_speed_count,
        "timestamp_delta_sec_p50": (
            float(np.nanmedian(delta_values)) if np.isfinite(delta_values).any() else 0.0
        ),
        "timestamp_delta_sec_p95": (
            float(np.nanquantile(delta_values[np.isfinite(delta_values)], 0.95))
            if np.isfinite(delta_values).any() else 0.0
        ),
        "ground_distance_m_p95": (
            float(np.nanquantile(ground_distance_values[np.isfinite(ground_distance_values)], 0.95))
            if np.isfinite(ground_distance_values).any() else 0.0
        ),
        "ipm_x_last": (
            float(ipm_x_values[np.flatnonzero(np.isfinite(ipm_x_values))[-1]])
            if np.isfinite(ipm_x_values).any() else 0.0
        ),
        "ipm_y_last": (
            float(ipm_y_values[np.flatnonzero(np.isfinite(ipm_y_values))[-1]])
            if np.isfinite(ipm_y_values).any() else 0.0
        ),
        "bbox_bottom_center_x_last": (
            float(center_x_values[np.flatnonzero(np.isfinite(center_x_values))[-1]])
            if np.isfinite(center_x_values).any() else 0.0
        ),
        "bbox_bottom_center_y_last": (
            float(foot_y_values[np.flatnonzero(np.isfinite(foot_y_values))[-1]])
            if np.isfinite(foot_y_values).any() else 0.0
        ),
        "speed_raw_kmh_p95": (
            float(np.nanquantile(raw_speed_values[np.isfinite(raw_speed_values)], 0.95))
            if np.isfinite(raw_speed_values).any() else 0.0
        ),
        "speed_smoothed_kmh_p95": representative_speed,
        "track_display_status": str(track_stats.get("display_status", "failed")),
        "track_failure_reason": str(track_stats.get("failure_reason", failure_reason)),
    }
    return module_row, sidecar_row
