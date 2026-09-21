"""Accepted red-light window formulas extracted from module v0."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from traffic_risk.upstream.redlight.feature_contract import (
    FORBIDDEN_LEAKAGE_COLUMNS,
    JOIN_KEY_FIELDNAMES,
)
from traffic_risk.upstream.redlight.traffic_light import (
    compute_redlight_score,
    frame_is_red,
    summarize_traffic_light_for_frames,
)

WINDOW_EPS_SEC = 1e-6
RED_SIGNAL_CONFIRM_THR = 0.5

REQUIRED_RAW_COLUMNS = (
    "track_id",
    "x1",
    "y1",
    "x2",
    "y2",
)

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def validate_input_columns(df: pd.DataFrame) -> None:
    missing = [field for field in REQUIRED_RAW_COLUMNS if field not in df.columns]
    if missing:
        raise ValueError(f"missing required source columns: {missing}")
    if "timestamp_sec" not in df.columns and "frame" not in df.columns:
        raise ValueError("missing time information: require timestamp_sec or frame")
    if "tl_state" not in df.columns and "tl_prob_red" not in df.columns:
        raise ValueError("missing traffic-light columns: require tl_state or tl_prob_red")
    forbidden = [field for field in FORBIDDEN_LEAKAGE_COLUMNS if field in df.columns]
    if forbidden:
        raise ValueError(f"forbidden leakage columns found in source csv: {forbidden}")


def ensure_timestamp_sec(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    out = raw_df.copy()
    if "timestamp_sec" in out.columns:
        out["timestamp_sec"] = pd.to_numeric(out["timestamp_sec"], errors="coerce")
        return out, "timestamp_sec"
    if "frame" not in out.columns:
        raise ValueError("timestamp_sec missing and frame fallback unavailable")
    fps_value = safe_float(out["fps"].iloc[0], 0.0) if "fps" in out.columns and not out.empty else 0.0
    if fps_value <= 0.0:
        raise ValueError("timestamp_sec missing and fps unavailable for frame fallback")
    out["timestamp_sec"] = pd.to_numeric(out["frame"], errors="coerce") / fps_value
    return out, "frame_div_fps"


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


def extract_fps_value(raw_df: pd.DataFrame) -> float:
    if "fps" in raw_df.columns and not raw_df.empty:
        fps_value = safe_float(raw_df["fps"].iloc[0], 0.0)
        if fps_value > 0.0:
            return fps_value
    return 0.0


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
        "redlight_confirmed": 0,
        "redlight_score": 0.0,
        "redlight_signal_valid": 0,
        "redlight_failure_reason": failure_reason,
        "tl_state_window": "unknown",
        "tl_red_ratio": 0.0,
        "tl_prob_red_mean": 0.0,
        "tl_prob_red_max": 0.0,
        "stopline_crossed_confirmed": 0,
        "forward_motion_after_stopline": 0,
        "redlight_event_candidate": 0,
    }


def make_failure_sidecar_row(
    window_row: pd.Series,
    *,
    source_csv: str,
    timestamp_column_used: str,
    fps_value: float,
    stopline_geometry_source: str,
    failure_reason: str,
) -> dict[str, Any]:
    return {
        **normalized_join_key_values(window_row),
        "source_csv": source_csv,
        "timestamp_column_used": timestamp_column_used,
        "fps": float(fps_value),
        "window_point_count": 0,
        "timestamp_valid": 0,
        "tracking_geometry_valid": 0,
        "tl_state_sequence": "",
        "tl_valid_state_count": 0,
        "tl_known_state_count": 0,
        "tl_frame_count": 0,
        "line_side_value_start": 0.0,
        "line_side_value_end": 0.0,
        "line_side_value_min": 0.0,
        "line_side_value_max": 0.0,
        "signed_distance_px_start": 0.0,
        "signed_distance_px_end": 0.0,
        "signed_distance_px_min": 0.0,
        "signed_distance_px_max": 0.0,
        "stopline_id_start": "",
        "stopline_id_end": "",
        "dominant_stopline_id": "",
        "crossing_stopline_id": "",
        "stopline_id_unique_count": 0,
        "stopline_id_switch_count": 0,
        "stopline_multi_candidate_flag": 0,
        "stopline_identity_confidence": 0.0,
        "crossing_event_in_window": 0,
        "crossing_frame": -1,
        "crossing_timestamp_sec": 0.0,
        "crossing_red_signal": 0,
        "motion_after_crossing_px": 0.0,
        "motion_after_crossing_world": 0.0,
        "bbox_bottom_center_x": 0.0,
        "bbox_bottom_center_y": 0.0,
        "stopline_geometry_source": stopline_geometry_source,
        "stopline_geometry_valid": 0,
        "primary_evidence_summary": failure_reason,
    }


def summarize_stopline_identity(window_df: pd.DataFrame, history_df: pd.DataFrame, crossing_frame: int) -> dict[str, Any]:
    default_summary = {
        "stopline_id_start": "",
        "stopline_id_end": "",
        "dominant_stopline_id": "",
        "crossing_stopline_id": "",
        "stopline_id_unique_count": 0,
        "stopline_id_switch_count": 0,
        "stopline_multi_candidate_flag": 0,
        "stopline_identity_confidence": 0.0,
    }
    if "nearest_stopline_id" not in window_df.columns:
        return default_summary

    window_ids = window_df["nearest_stopline_id"].fillna("").astype(str).str.strip()
    non_empty_window_ids = window_ids[window_ids.ne("")]
    if non_empty_window_ids.empty:
        return default_summary

    unique_ids = list(dict.fromkeys(non_empty_window_ids.tolist()))
    unique_count = len(unique_ids)
    switch_count = 0
    previous_id = ""
    for stopline_id in window_ids.tolist():
        if not stopline_id:
            continue
        if previous_id and stopline_id != previous_id:
            switch_count += 1
        previous_id = stopline_id

    counts = non_empty_window_ids.value_counts()
    dominant_stopline_id = str(counts.index[0]) if not counts.empty else ""
    dominant_share = float(counts.iloc[0]) / float(non_empty_window_ids.shape[0]) if not counts.empty else 0.0

    crossing_stopline_id = ""
    if crossing_frame >= 0 and "frame" in history_df.columns:
        crossing_ids = (
            history_df.loc[
                pd.to_numeric(history_df["frame"], errors="coerce").fillna(-1).astype(int).eq(int(crossing_frame)),
                "nearest_stopline_id",
            ]
            .fillna("")
            .astype(str)
            .str.strip()
        )
        crossing_ids = crossing_ids[crossing_ids.ne("")]
        if not crossing_ids.empty:
            crossing_stopline_id = str(crossing_ids.iloc[0])

    single_candidate_score = 1.0 if unique_count <= 1 else 1.0 / float(unique_count)
    alignment_score = single_candidate_score if dominant_stopline_id else 0.0
    if crossing_stopline_id and dominant_stopline_id:
        alignment_score = 1.0 if crossing_stopline_id == dominant_stopline_id else 0.0

    identity_confidence = float(np.clip(0.45 * dominant_share + 0.35 * single_candidate_score + 0.20 * alignment_score, 0.0, 1.0))
    if switch_count > 0:
        identity_confidence *= 1.0 / (1.0 + 0.5 * switch_count)
    identity_confidence = float(np.clip(identity_confidence, 0.0, 1.0))

    return {
        "stopline_id_start": str(non_empty_window_ids.iloc[0]),
        "stopline_id_end": str(non_empty_window_ids.iloc[-1]),
        "dominant_stopline_id": dominant_stopline_id,
        "crossing_stopline_id": crossing_stopline_id,
        "stopline_id_unique_count": int(unique_count),
        "stopline_id_switch_count": int(switch_count),
        "stopline_multi_candidate_flag": int(unique_count > 1),
        "stopline_identity_confidence": identity_confidence,
    }


def aggregate_window(
    window_row: pd.Series,
    evidence_df: pd.DataFrame,
    track_stats: dict[str, Any],
    traffic_light_table: pd.DataFrame,
    *,
    source_csv: str,
    timestamp_column_used: str,
    fps_value: float,
    stopline_geometry_source: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    stopline_geometry_valid = int(track_stats.get("geometry_valid", 0))
    if evidence_df.empty:
        failure_reason = "track_has_no_points"
        return (
            make_failure_row(window_row, failure_reason),
            make_failure_sidecar_row(
                window_row,
                source_csv=source_csv,
                timestamp_column_used=timestamp_column_used,
                fps_value=fps_value,
                stopline_geometry_source=stopline_geometry_source,
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
                stopline_geometry_source=stopline_geometry_source,
                failure_reason=failure_reason,
            ),
        )

    tl_summary = summarize_traffic_light_for_frames(window_df["frame"], traffic_light_table)
    timestamp_valid = int(pd.to_numeric(window_df["timestamp_sec"], errors="coerce").notna().any())
    tracking_geometry_valid = int(pd.to_numeric(window_df["bbox_valid"], errors="coerce").fillna(0).astype(int).eq(1).any())
    red_signal_score = safe_float(tl_summary["red_signal_score"], 0.0)
    red_signal_confirmed = int(
        safe_float(tl_summary["tl_red_ratio"], 0.0) >= RED_SIGNAL_CONFIRM_THR
        or safe_float(tl_summary["tl_prob_red_mean"], 0.0) >= RED_SIGNAL_CONFIRM_THR
    )

    history_mask = ts_values <= float(window_row["end_sec"]) + WINDOW_EPS_SEC
    history_df = evidence_df.loc[history_mask].copy().reset_index(drop=True)
    stopline_crossed_confirmed = int(pd.to_numeric(history_df["crossed_stop_line"], errors="coerce").fillna(0).astype(int).max() > 0)
    forward_motion_after_stopline = int(pd.to_numeric(window_df["forward_motion_after_stopline"], errors="coerce").fillna(0).astype(int).max() > 0)
    crossing_event_in_window = int(pd.to_numeric(window_df["cross_event"], errors="coerce").fillna(0).astype(int).max() > 0)
    crossing_history_df = history_df.loc[pd.to_numeric(history_df["cross_event"], errors="coerce").fillna(0).astype(int) > 0]
    crossing_frame = -1
    crossing_timestamp_sec = 0.0
    crossing_red_signal = 0
    if not crossing_history_df.empty:
        first_crossing = crossing_history_df.iloc[0]
        crossing_frame = int(first_crossing["frame"])
        crossing_timestamp_sec = safe_float(first_crossing["timestamp_sec"], 0.0)
        crossing_red_signal = int(frame_is_red(crossing_frame, traffic_light_table))

    stopline_identity = summarize_stopline_identity(window_df, history_df, crossing_frame)

    signal_valid = int(
        safe_float(tl_summary["traffic_light_signal_valid"], 0.0) == 1.0
        and stopline_geometry_valid == 1
        and tracking_geometry_valid == 1
        and timestamp_valid == 1
    )
    crossing_temporally_red = int((crossing_event_in_window == 1 and red_signal_confirmed == 1) or crossing_red_signal == 1)
    redlight_event_candidate = int(signal_valid == 1 and red_signal_confirmed == 1 and stopline_crossed_confirmed == 1)
    redlight_confirmed = int(
        signal_valid == 1
        and red_signal_confirmed == 1
        and stopline_crossed_confirmed == 1
        and forward_motion_after_stopline == 1
        and crossing_temporally_red == 1
    )
    redlight_score = compute_redlight_score(
        red_signal_score=red_signal_score,
        stopline_cross_score=float(stopline_crossed_confirmed),
        forward_motion_score=float(forward_motion_after_stopline),
        signal_valid_score=float(signal_valid),
    )

    if stopline_geometry_valid == 0:
        failure_reason = "stop_line_geometry_missing_or_invalid"
    elif safe_float(tl_summary["traffic_light_signal_valid"], 0.0) != 1.0:
        failure_reason = "traffic_light_signal_missing_or_unknown"
    elif timestamp_valid == 0:
        failure_reason = "timestamp_invalid_in_window"
    elif tracking_geometry_valid == 0:
        failure_reason = "tracking_geometry_invalid"
    elif red_signal_confirmed == 0:
        failure_reason = "traffic_light_not_red"
    elif stopline_crossed_confirmed == 0:
        failure_reason = "stopline_not_crossed"
    elif crossing_temporally_red == 0:
        failure_reason = "crossing_not_during_red"
    elif forward_motion_after_stopline == 0:
        failure_reason = "no_forward_motion_after_stopline"
    else:
        failure_reason = ""

    line_values = pd.to_numeric(window_df["line_side_value"], errors="coerce").to_numpy(dtype=np.float64)
    signed_values = pd.to_numeric(window_df["signed_distance_px"], errors="coerce").to_numpy(dtype=np.float64)
    motion_values = pd.to_numeric(window_df["motion_after_crossing_px"], errors="coerce").to_numpy(dtype=np.float64)
    bottom_x_values = pd.to_numeric(window_df["bbox_bottom_center_x"], errors="coerce").to_numpy(dtype=np.float64)
    bottom_y_values = pd.to_numeric(window_df["bbox_bottom_center_y"], errors="coerce").to_numpy(dtype=np.float64)
    primary_evidence_summary = (
        f"tl={tl_summary['tl_state_window']} red_ratio={safe_float(tl_summary['tl_red_ratio']):.3f}; "
        f"crossed={stopline_crossed_confirmed}; forward={forward_motion_after_stopline}; "
        f"stopline={stopline_identity['crossing_stopline_id'] or stopline_identity['dominant_stopline_id'] or 'none'}; "
        f"stopline_conf={stopline_identity['stopline_identity_confidence']:.3f}; "
        f"cross_frame={crossing_frame}; reason={failure_reason or 'confirmed'}"
    )

    module_row = {
        **normalized_join_key_values(window_row),
        "redlight_confirmed": redlight_confirmed,
        "redlight_score": redlight_score,
        "redlight_signal_valid": signal_valid,
        "redlight_failure_reason": failure_reason,
        "tl_state_window": tl_summary["tl_state_window"],
        "tl_red_ratio": safe_float(tl_summary["tl_red_ratio"], 0.0),
        "tl_prob_red_mean": safe_float(tl_summary["tl_prob_red_mean"], 0.0),
        "tl_prob_red_max": safe_float(tl_summary["tl_prob_red_max"], 0.0),
        "stopline_crossed_confirmed": stopline_crossed_confirmed,
        "forward_motion_after_stopline": forward_motion_after_stopline,
        "redlight_event_candidate": redlight_event_candidate,
    }
    sidecar_row = {
        **normalized_join_key_values(window_row),
        "source_csv": source_csv,
        "timestamp_column_used": timestamp_column_used,
        "fps": float(fps_value),
        "window_point_count": int(window_df.shape[0]),
        "timestamp_valid": timestamp_valid,
        "tracking_geometry_valid": tracking_geometry_valid,
        "tl_state_sequence": tl_summary["tl_state_sequence"],
        "tl_valid_state_count": int(tl_summary["tl_valid_state_count"]),
        "tl_known_state_count": int(tl_summary["tl_known_state_count"]),
        "tl_frame_count": int(tl_summary["tl_frame_count"]),
        "line_side_value_start": float(line_values[0]) if len(line_values) else 0.0,
        "line_side_value_end": float(line_values[-1]) if len(line_values) else 0.0,
        "line_side_value_min": float(np.nanmin(line_values)) if len(line_values) else 0.0,
        "line_side_value_max": float(np.nanmax(line_values)) if len(line_values) else 0.0,
        "signed_distance_px_start": float(signed_values[0]) if len(signed_values) else 0.0,
        "signed_distance_px_end": float(signed_values[-1]) if len(signed_values) else 0.0,
        "signed_distance_px_min": float(np.nanmin(signed_values)) if len(signed_values) else 0.0,
        "signed_distance_px_max": float(np.nanmax(signed_values)) if len(signed_values) else 0.0,
        "stopline_id_start": stopline_identity["stopline_id_start"],
        "stopline_id_end": stopline_identity["stopline_id_end"],
        "dominant_stopline_id": stopline_identity["dominant_stopline_id"],
        "crossing_stopline_id": stopline_identity["crossing_stopline_id"],
        "stopline_id_unique_count": stopline_identity["stopline_id_unique_count"],
        "stopline_id_switch_count": stopline_identity["stopline_id_switch_count"],
        "stopline_multi_candidate_flag": stopline_identity["stopline_multi_candidate_flag"],
        "stopline_identity_confidence": stopline_identity["stopline_identity_confidence"],
        "crossing_event_in_window": crossing_event_in_window,
        "crossing_frame": crossing_frame,
        "crossing_timestamp_sec": crossing_timestamp_sec,
        "crossing_red_signal": crossing_red_signal,
        "motion_after_crossing_px": float(np.nanmax(motion_values)) if len(motion_values) else 0.0,
        "motion_after_crossing_world": float(np.nanmax(motion_values)) if len(motion_values) else 0.0,
        "bbox_bottom_center_x": float(bottom_x_values[np.flatnonzero(np.isfinite(bottom_x_values))[-1]]) if np.isfinite(bottom_x_values).any() else 0.0,
        "bbox_bottom_center_y": float(bottom_y_values[np.flatnonzero(np.isfinite(bottom_y_values))[-1]]) if np.isfinite(bottom_y_values).any() else 0.0,
        "stopline_geometry_source": stopline_geometry_source,
        "stopline_geometry_valid": stopline_geometry_valid,
        "primary_evidence_summary": primary_evidence_summary,
    }
    return module_row, sidecar_row


