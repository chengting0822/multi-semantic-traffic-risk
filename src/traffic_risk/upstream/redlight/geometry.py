"""Accepted stop-line geometry and crossing formulas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FORWARD_MOTION_AFTER_STOPLINE_PX = 3.0


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def load_stop_lines(stopline_json_path: str | Path) -> tuple[list[dict[str, Any]], str]:
    path = Path(stopline_json_path)
    if not path.is_file():
        return [], f"stop-line json not found: {path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [], f"stop-line json parse failed: {exc}"
    raw_stop_lines = payload.get("stop_lines", [])
    stop_lines: list[dict[str, Any]] = []
    for index, item in enumerate(raw_stop_lines):
        points = item.get("points", []) if isinstance(item, dict) else []
        if len(points) < 2:
            continue
        p0 = points[0]
        p1 = points[-1]
        if len(p0) < 2 or len(p1) < 2:
            continue
        x1, y1 = safe_float(p0[0]), safe_float(p0[1])
        x2, y2 = safe_float(p1[0]), safe_float(p1[1])
        denom = float(np.hypot(y2 - y1, x2 - x1))
        if denom < 1e-9:
            continue
        stop_lines.append(
            {
                "id": str(item.get("id", f"stop_line_{index + 1}")) if isinstance(item, dict) else f"stop_line_{index + 1}",
                "points": [[x1, y1], [x2, y2]],
            }
        )
    if not stop_lines:
        return [], f"stop-line json has no usable stop_lines: {path}"
    return stop_lines, ""


def compute_nearest_stopline_signed_distance(
    foot_x: np.ndarray,
    foot_y: np.ndarray,
    stop_lines: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    n_points = len(foot_x)
    if n_points == 0 or not stop_lines:
        return np.zeros(n_points, dtype=np.float64), np.full(n_points, "", dtype=object)

    all_distances = np.zeros((n_points, len(stop_lines)), dtype=np.float64)
    line_ids: list[str] = []
    for line_index, stop_line in enumerate(stop_lines):
        points = stop_line["points"]
        x1, y1 = float(points[0][0]), float(points[0][1])
        x2, y2 = float(points[-1][0]), float(points[-1][1])
        a = y2 - y1
        b = -(x2 - x1)
        c = x2 * y1 - y2 * x1
        denom = float(np.hypot(a, b))
        if denom < 1e-9:
            all_distances[:, line_index] = np.nan
        else:
            all_distances[:, line_index] = (a * foot_x + b * foot_y + c) / denom
        line_ids.append(str(stop_line.get("id", f"stop_line_{line_index + 1}")))

    abs_distances = np.abs(all_distances)
    abs_distances[~np.isfinite(abs_distances)] = np.inf
    nearest_idx = abs_distances.argmin(axis=1)
    signed_distance = all_distances[np.arange(n_points), nearest_idx]
    nearest_line_ids = np.asarray([line_ids[index] for index in nearest_idx], dtype=object)
    return signed_distance, nearest_line_ids


def compute_track_stopline_profile(
    track_df: pd.DataFrame,
    stop_lines: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    sorted_df = track_df.sort_values(["timestamp_sec", "frame" if "frame" in track_df.columns else "timestamp_sec"]).reset_index(drop=True)
    n_points = int(sorted_df.shape[0])
    if n_points == 0:
        return pd.DataFrame(), {"geometry_valid": bool(stop_lines), "failure_reason": "track_has_no_points"}

    center_x = ((pd.to_numeric(sorted_df["x1"], errors="coerce") + pd.to_numeric(sorted_df["x2"], errors="coerce")) / 2.0).to_numpy(dtype=np.float64)
    foot_y = pd.to_numeric(sorted_df["y2"], errors="coerce").to_numpy(dtype=np.float64)
    center_y = ((pd.to_numeric(sorted_df["y1"], errors="coerce") + pd.to_numeric(sorted_df["y2"], errors="coerce")) / 2.0).to_numpy(dtype=np.float64)
    timestamp_sec = pd.to_numeric(sorted_df["timestamp_sec"], errors="coerce").to_numpy(dtype=np.float64)
    frame_values = pd.to_numeric(sorted_df["frame"], errors="coerce").fillna(-1).astype(int).to_numpy() if "frame" in sorted_df.columns else np.arange(n_points)

    bbox_valid = np.isfinite(center_x) & np.isfinite(foot_y) & np.isfinite(center_y)
    geometry_valid = bool(stop_lines) and bool(bbox_valid.any())
    signed_distance, nearest_line_ids = compute_nearest_stopline_signed_distance(center_x, foot_y, stop_lines)
    signed_distance[~np.isfinite(signed_distance)] = 0.0
    line_side_value = np.sign(signed_distance).astype(np.float64)

    cross_event = np.zeros(n_points, dtype=np.int8)
    crossed_stop_line = np.zeros(n_points, dtype=np.int8)
    first_cross_index: int | None = None
    post_cross_sign = 0.0
    if geometry_valid:
        ever_crossed = False
        for index in range(1, n_points):
            crossed_now = (
                line_side_value[index] != 0.0
                and line_side_value[index - 1] != 0.0
                and line_side_value[index] != line_side_value[index - 1]
            )
            if crossed_now:
                cross_event[index] = 1
                ever_crossed = True
                if first_cross_index is None:
                    first_cross_index = index
                    post_cross_sign = line_side_value[index]
            if ever_crossed:
                crossed_stop_line[index] = 1

    motion_after_crossing_px = np.zeros(n_points, dtype=np.float64)
    forward_motion_after_stopline = np.zeros(n_points, dtype=np.int8)
    crossing_frame = np.full(n_points, -1, dtype=np.int64)
    crossing_timestamp_sec = np.zeros(n_points, dtype=np.float64)
    if first_cross_index is not None:
        cross_x = center_x[first_cross_index]
        cross_y = foot_y[first_cross_index]
        first_cross_frame = int(frame_values[first_cross_index])
        first_cross_time = float(timestamp_sec[first_cross_index]) if np.isfinite(timestamp_sec[first_cross_index]) else 0.0
        for index in range(first_cross_index, n_points):
            same_crossed_side = post_cross_sign == 0.0 or line_side_value[index] == post_cross_sign
            # Gate motion on same-side only: when the bbox returns to the original
            # side of the stop line (e.g. due to occlusion-induced bbox shrinkage
            # that then recovers), the resulting foot_y displacement must not be
            # counted as forward motion past the stop line.
            if same_crossed_side and bbox_valid[index]:
                motion = float(np.hypot(center_x[index] - cross_x, foot_y[index] - cross_y))
            else:
                motion = 0.0
            motion_after_crossing_px[index] = motion
            crossing_frame[index] = first_cross_frame
            crossing_timestamp_sec[index] = first_cross_time
            if same_crossed_side and motion >= FORWARD_MOTION_AFTER_STOPLINE_PX:
                forward_motion_after_stopline[index] = 1

    evidence = pd.DataFrame(
        {
            "frame": frame_values,
            "timestamp_sec": timestamp_sec,
            "bbox_bottom_center_x": center_x,
            "bbox_bottom_center_y": foot_y,
            "bbox_center_y": center_y,
            "signed_distance_px": signed_distance,
            "line_side_value": line_side_value,
            "nearest_stopline_id": nearest_line_ids,
            "cross_event": cross_event,
            "crossed_stop_line": crossed_stop_line,
            "crossing_frame": crossing_frame,
            "crossing_timestamp_sec": crossing_timestamp_sec,
            "motion_after_crossing_px": motion_after_crossing_px,
            "motion_after_crossing_world": motion_after_crossing_px,
            "forward_motion_after_stopline": forward_motion_after_stopline,
            "bbox_valid": bbox_valid.astype(np.int8),
        }
    )
    stats = {
        "geometry_valid": int(geometry_valid),
        "bbox_valid_ratio": float(np.mean(bbox_valid)) if n_points else 0.0,
        "first_crossing_frame": int(frame_values[first_cross_index]) if first_cross_index is not None else -1,
        "first_crossing_timestamp_sec": float(timestamp_sec[first_cross_index]) if first_cross_index is not None and np.isfinite(timestamp_sec[first_cross_index]) else 0.0,
        "failure_reason": "" if geometry_valid else "stop_line_geometry_or_bbox_invalid",
    }
    return evidence, stats
