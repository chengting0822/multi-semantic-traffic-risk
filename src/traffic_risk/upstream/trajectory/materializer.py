"""Streaming extraction of the accepted Stage2A.17-R fixed-20 features."""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .contracts import FEATURE_COLUMNS, OFFICIAL_40D_FEATURES, SAMPLE_COUNT, STRIDE_SEC, WINDOW_SEC
from traffic_risk.upstream.car_tracks import scan_car_track_ids


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return default if math.isnan(number) else number
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _point_in_polygon(x: float, y: float, polygon: Sequence[Sequence[float]]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i, point in enumerate(polygon):
        xi, yi = float(point[0]), float(point[1])
        xj, yj = float(polygon[j][0]), float(polygon[j][1])
        if (yi > y) != (yj > y):
            crossing = (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
            if x < crossing:
                inside = not inside
        j = i
    return inside


def load_lane_map(path: Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    lanes: list[dict[str, Any]] = []
    for order, item in enumerate(raw.get("lanes", [])):
        polygon = [[float(p[0]), float(p[1])] for p in item.get("polygon", []) if len(p) >= 2]
        if len(polygon) < 3:
            continue
        vectors: list[tuple[float, float]] = []
        segments: list[tuple[float, float, float, float, float, float]] = []
        for path_points in item.get("direction_paths", []):
            for left, right in zip(path_points, path_points[1:]):
                x0, y0 = float(left[0]), float(left[1])
                x1, y1 = float(right[0]), float(right[1])
                dx, dy = x1 - x0, y1 - y0
                norm = math.hypot(dx, dy)
                if norm > 1e-8:
                    vectors.append((dx / norm, dy / norm))
                    segments.append((x0, y0, x1, y1, dx / norm, dy / norm))
        if not vectors:
            direction = item.get("direction", [])
            if len(direction) >= 2:
                dx, dy = float(direction[0]), float(direction[1])
                norm = math.hypot(dx, dy)
                if norm > 1e-8:
                    vectors.append((dx / norm, dy / norm))
        if not vectors:
            continue
        xs, ys = [p[0] for p in polygon], [p[1] for p in polygon]
        lanes.append({
            "lane_id": str(item.get("lane_id", f"lane_{order + 1}")),
            "polygon": polygon,
            "direction_vectors": vectors,
            "direction_segments": segments,
            "direction": vectors[0],
            "bbox": (min(xs), min(ys), max(xs), max(ys)),
            "priority": int(item.get("priority", order + 1)),
            "order": order,
        })
    lanes.sort(key=lambda lane: lane["priority"])
    return {"type": "manual_lane_map", "lanes": lanes}


def _lanes_at_point(lane_map: Mapping[str, Any], x: float, y: float) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for lane in lane_map.get("lanes", []):
        x0, y0, x1, y1 = lane["bbox"]
        if x0 <= x <= x1 and y0 <= y <= y1 and _point_in_polygon(x, y, lane["polygon"]):
            matches.append(lane)
    return matches


def _segment_distance_sq(x: float, y: float, segment: Sequence[float]) -> float:
    x0, y0, x1, y1 = segment[:4]
    dx, dy = x1 - x0, y1 - y0
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:
        return (x - x0) ** 2 + (y - y0) ** 2
    t = float(np.clip(((x - x0) * dx + (y - y0) * dy) / length_sq, 0.0, 1.0))
    return (x - (x0 + t * dx)) ** 2 + (y - (y0 + t * dy)) ** 2


def _best_lane_at_point(
    lane_map: Mapping[str, Any], x: float, y: float, heading: tuple[float, float] | None,
) -> tuple[dict[str, Any] | None, float | None]:
    candidates = _lanes_at_point(lane_map, x, y)
    if not candidates:
        return None, None
    scored: list[tuple[float, dict[str, Any]]] = []
    if heading is not None:
        for lane in candidates:
            score = max(heading[0] * dx + heading[1] * dy for dx, dy in lane["direction_vectors"])
            scored.append((float(score), lane))
        scored.sort(key=lambda item: item[0], reverse=True)
        top = scored[0][0]
        ties = [(score, lane) for score, lane in scored if top - score <= 0.05]
        if len(ties) == 1:
            return ties[0][1], ties[0][0]
        candidates = [lane for _, lane in ties]
    candidates.sort(key=lambda lane: (
        min((_segment_distance_sq(x, y, segment) for segment in lane["direction_segments"]), default=float("inf")),
        lane["priority"], lane["order"],
    ))
    lane = candidates[0]
    score = None if heading is None else max(heading[0] * dx + heading[1] * dy for dx, dy in lane["direction_vectors"])
    return lane, None if score is None else float(score)


def _window_stats(window: np.ndarray) -> np.ndarray:
    stats: list[float] = []
    for index, column in enumerate(FEATURE_COLUMNS):
        values = window[:, index]
        stats.extend([float(np.mean(values)), float(np.std(values))])
        if column in {"vx", "vy", "speed", "ax", "ay", "dw", "dh", "heading_change"}:
            stats.append(float(np.max(np.abs(values))))
    cx, cy = FEATURE_COLUMNS.index("cx"), FEATURE_COLUMNS.index("cy")
    displacement = np.sqrt(
        (window[-1, cx] - window[0, cx]) ** 2
        + (window[-1, cy] - window[0, cy]) ** 2
    )
    stats.append(displacement)
    stats.append(float(np.min(window[:, FEATURE_COLUMNS.index("conf")])))
    return np.asarray(stats, dtype=np.float32)


def _lane_stats(window: np.ndarray, lane_map: Mapping[str, Any]) -> np.ndarray:
    cx = window[:, FEATURE_COLUMNS.index("cx")]
    cy = window[:, FEATURE_COLUMNS.index("cy")]
    vx = float(np.mean(window[:, FEATURE_COLUMNS.index("vx")]))
    vy = float(np.mean(window[:, FEATURE_COLUMNS.index("vy")]))
    norm = math.hypot(vx, vy)
    heading = (vx / norm, vy / norm) if norm > 0.3 else None
    _, score = _best_lane_at_point(lane_map, float(np.mean(cx)), float(np.mean(cy)), heading)
    # The accepted model fitted log(expected bbox area) = a * center_y + b.
    # These two frozen coefficients are the complete perspective artifact.
    expected_area = np.clip(np.exp(0.007516392488086847 * cy + 3.4141061411993343), 100.0, 200000.0)
    width = window[:, FEATURE_COLUMNS.index("w")]
    height = window[:, FEATURE_COLUMNS.index("h")]
    perspective_ratio = float(np.mean((width * height) / expected_area))
    return np.asarray([0.0 if score is None else score, perspective_ratio], dtype=np.float32)


def _normalize(dx: float, dy: float) -> tuple[float, float] | None:
    norm = math.hypot(dx, dy)
    return None if norm <= 1e-8 else (dx / norm, dy / norm)


def _max_consecutive(flags: Iterable[bool]) -> int:
    current = best = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def _lane_family(lane_id: str) -> str:
    parts = str(lane_id or "").split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else str(lane_id or "")


def _flow_features(raw_rows: Sequence[tuple[int, float, float, float, float, float]], lane_map: Mapping[str, Any]) -> dict[str, float]:
    neutral = {
        "flow_cos": 1.0, "bidir_max_cos": 1.0, "reversed_flag": 0.0,
        "flow_conflict_score": 0.0, "negative_flow_margin": 0.0,
        "opposite_lane_occupancy_ratio": 0.0, "opposite_lane_run_length": 0.0,
        "opposite_lane_after_crossing_flag": 0.0, "opposite_lane_after_crossing_run_length": 0.0,
        "selected_lane_known": 0.0, "lane_family_switch_count": 0.0,
        "semantic_support_flag": 0.0, "occupied_lane_flow_cos": 1.0,
    }
    centers = [((x1 + x2) / 2.0, (y1 + y2) / 2.0) for _, x1, y1, x2, y2, _ in raw_rows]
    if len(centers) < 2:
        return neutral
    counts: dict[str, int] = {}
    for x, y in centers:
        lanes = _lanes_at_point(lane_map, x, y)
        if lanes:
            lane_id = str(lanes[0]["lane_id"])
            counts[lane_id] = counts.get(lane_id, 0) + 1
    primary = max(counts, key=counts.get) if counts else None
    best_values: list[float] = []
    occupied_values: list[float] = []
    weights: list[float] = []
    occupied_weights: list[float] = []
    step_lane_ids: list[str] = []
    lane_sets: list[list[dict[str, Any]]] = []
    for left, right in zip(centers, centers[1:]):
        dx, dy = right[0] - left[0], right[1] - left[1]
        speed = math.hypot(dx, dy)
        heading = _normalize(dx, dy) if speed >= 0.5 else None
        lanes = _lanes_at_point(lane_map, (left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0)
        lane_sets.append(lanes)
        if lanes:
            step_lane_ids.append(str(lanes[0]["lane_id"]))
        if heading is not None and lanes:
            best_values.append(max(heading[0] * vx + heading[1] * vy for lane in lanes for vx, vy in lane["direction_vectors"]))
            weights.append(max(speed, 1.0))
            primary_lane = next((lane for lane in lanes if lane["lane_id"] == primary), None)
            if primary_lane is not None:
                vx, vy = primary_lane["direction_vectors"][0]
                occupied_values.append(heading[0] * vx + heading[1] * vy)
            else:
                occupied_values.append(min(heading[0] * vx + heading[1] * vy for lane in lanes for vx, vy in lane["direction_vectors"]))
            occupied_weights.append(max(speed, 1.0))
    flow_cos = float(np.average(best_values, weights=weights)) if best_values else 1.0
    bidir = float(np.average(np.maximum(best_values, -np.asarray(best_values)), weights=weights)) if best_values else 1.0
    occupied_cos = float(np.average(occupied_values, weights=occupied_weights)) if occupied_values else flow_cos
    reversed_flag = float(occupied_cos < 0.3)
    first_lane = next((lanes[0] for lanes in lane_sets if lanes), None)
    opposite_flags: list[bool] = []
    after_flags: list[bool] = []
    if first_lane is not None:
        base_dir = first_lane["direction"]
        for lanes in lane_sets:
            cosines = [base_dir[0] * lane["direction"][0] + base_dir[1] * lane["direction"][1] for lane in lanes]
            same = any(value > 0.5 for value in cosines)
            opposite = any(value < 0.0 for value in cosines)
            opposite_flags.append(opposite)
            after_flags.append(opposite and not same)
    families = [_lane_family(lane_id) for lane_id in step_lane_ids]
    switches = sum(left != right for left, right in zip(families, families[1:]) if left and right)
    after_run = _max_consecutive(after_flags)
    return {
        "flow_cos": flow_cos, "bidir_max_cos": bidir, "reversed_flag": reversed_flag,
        "flow_conflict_score": max(0.0, 1.0 - occupied_cos), "negative_flow_margin": max(0.0, -occupied_cos),
        "opposite_lane_occupancy_ratio": sum(opposite_flags) / len(opposite_flags) if opposite_flags else 0.0,
        "opposite_lane_run_length": float(_max_consecutive(opposite_flags)),
        "opposite_lane_after_crossing_flag": float(any(after_flags)),
        "opposite_lane_after_crossing_run_length": float(after_run),
        "selected_lane_known": float(bool(step_lane_ids)), "lane_family_switch_count": float(switches),
        "semantic_support_flag": float(reversed_flag > 0.5 and after_run > 0),
        "occupied_lane_flow_cos": occupied_cos,
    }


def _segment_projection(point: tuple[float, float], start: Sequence[float], end: Sequence[float]) -> dict[str, Any]:
    px, py = point
    x1, y1 = float(start[0]), float(start[1])
    x2, y2 = float(end[0]), float(end[1])
    dx, dy = x2 - x1, y2 - y1
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-9:
        return {"distance": math.hypot(px - x1, py - y1), "cross": 0.0, "segment_index": 0}
    t = float(np.clip(((px - x1) * dx + (py - y1) * dy) / length_sq, 0.0, 1.0))
    qx, qy = x1 + t * dx, y1 + t * dy
    return {"distance": math.hypot(px - qx, py - qy), "cross": dx * (py - qy) - dy * (px - qx)}


def _nearest_line(point: tuple[float, float], line: Mapping[str, Any]) -> dict[str, Any]:
    points = line.get("points", [])
    best: dict[str, Any] | None = None
    for index, (start, end) in enumerate(zip(points, points[1:])):
        candidate = {**_segment_projection(point, start, end), "segment_index": index}
        if best is None or candidate["distance"] < best["distance"]:
            best = candidate
    if best is None:
        return {"distance": float("inf"), "side_sign": 0, "segment_index": -1}
    cross = float(best["cross"])
    best["side_sign"] = 1 if cross > 1e-6 else (-1 if cross < -1e-6 else 0)
    return best


def _side_switches(signs: Sequence[int], stable_frames: int = 2) -> dict[str, Any]:
    switches: list[dict[str, int]] = []
    smoothed: list[int] = []
    stable = pending = pending_count = last_nonzero = 0
    pending_start = 0
    for index, raw_sign in enumerate(signs):
        sign = 1 if raw_sign > 0 else (-1 if raw_sign < 0 else 0)
        effective = last_nonzero if sign == 0 and last_nonzero else sign
        if sign:
            last_nonzero = sign
        if stable == 0 and effective:
            stable = effective
            smoothed.append(stable)
            continue
        if effective == 0 or effective == stable:
            pending = pending_count = 0
            pending_start = index
            smoothed.append(stable)
            continue
        if pending != effective:
            pending, pending_count, pending_start = effective, 1, index
        else:
            pending_count += 1
        if pending_count >= stable_frames:
            switches.append({"switch_index": pending_start, "from_side": stable, "to_side": pending})
            stable, pending, pending_count = pending, 0, 0
        smoothed.append(stable)
    if len(smoothed) < len(signs):
        smoothed.extend([stable] * (len(signs) - len(smoothed)))
    return {"switch_events": switches, "smoothed_side_signs": smoothed}


def _double_yellow_features(
    raw_rows: Sequence[tuple[int, float, float, float, float, float]], geometry: Mapping[str, Any],
) -> dict[str, float]:
    base = {
        "window_distance_to_double_yellow_min": 0.0,
        "window_double_yellow_cross_count": 0.0,
        "window_double_yellow_crossed_once": 0.0,
        "window_double_yellow_side_switch_count": 0.0,
        "window_after_crossing_opposite_lane_occupancy_ratio": 0.0,
        "window_after_crossing_opposite_lane_run_length": 0.0,
    }
    lines = [line for line in geometry.get("lines", []) if line.get("is_active", True)]
    if not raw_rows or not lines:
        return base
    samples: list[dict[str, Any]] = []
    for frame, x1, _y1, x2, y2, _conf in raw_rows:
        point = ((x1 + x2) / 2.0, y2)
        candidates = [{**_nearest_line(point, line), "line_id": line.get("line_id", "")} for line in lines]
        best = min(candidates, key=lambda item: item["distance"])
        samples.append({"frame": frame, **best})
    switches = _side_switches([int(sample["side_sign"]) for sample in samples], 2)
    distance_gate = float(geometry.get("crossing_distance_px", 20.0))
    gated_signs = [int(s["side_sign"]) if float(s["distance"]) <= distance_gate else 0 for s in samples]
    crossing_result = _side_switches(gated_signs, 2)
    crossings = 0
    crossing_indices: list[int] = []
    for switch in crossing_result["switch_events"]:
        index = int(switch["switch_index"])
        left, right = max(0, index - 1), min(len(samples), index + 3)
        candidates = [i for i in range(left, right) if float(samples[i]["distance"]) <= distance_gate]
        if candidates:
            crossings += 1
            crossing_indices.append(min(candidates, key=lambda i: (float(samples[i]["distance"]), abs(i - index))))
    after_mask = [False] * len(samples)
    nonzero = next((sign for sign in gated_signs if sign), 0)
    if crossing_indices and nonzero:
        start = min(crossing_indices)
        smoothed = crossing_result["smoothed_side_signs"]
        for index in range(start, len(samples)):
            after_mask[index] = smoothed[index] != 0 and smoothed[index] != nonzero
    ratio = 0.0
    if any(after_mask):
        start = after_mask.index(True)
        ratio = sum(after_mask[start:]) / len(after_mask[start:])
    return {
        "window_distance_to_double_yellow_min": min(float(sample["distance"]) for sample in samples),
        "window_double_yellow_cross_count": float(crossings),
        "window_double_yellow_crossed_once": float(crossings > 0),
        "window_double_yellow_side_switch_count": float(len(switches["switch_events"])),
        "window_after_crossing_opposite_lane_occupancy_ratio": float(ratio),
        "window_after_crossing_opposite_lane_run_length": float(_max_consecutive(after_mask)),
    }


def _artifact_features(raw_rows: Sequence[tuple[int, float, float, float, float, float]]) -> dict[str, float]:
    if not raw_rows:
        return {"artifact_frame_gap_max": 0.0, "artifact_bbox_area_cv": 0.0,
                "artifact_max_center_step": 0.0, "artifact_conf_min": 1.0, "tracking_quality_bad": 0.0}
    frames = np.asarray([row[0] for row in raw_rows], dtype=float)
    centers = np.asarray([[(row[1] + row[3]) / 2.0, (row[2] + row[4]) / 2.0] for row in raw_rows])
    areas = np.asarray([max(row[3] - row[1], 1.0) * max(row[4] - row[2], 1.0) for row in raw_rows])
    confs = np.asarray([row[5] for row in raw_rows])
    frame_gap = float(np.max(np.diff(frames))) if len(frames) >= 2 else 1.0
    steps = np.hypot(*np.diff(centers, axis=0).T) if len(centers) >= 2 else np.asarray([])
    center_step = float(np.max(steps)) if len(steps) else 0.0
    area_cv = float(np.std(areas) / (np.mean(areas) + 1e-6))
    conf_min = float(np.min(confs))
    return {
        "artifact_frame_gap_max": frame_gap, "artifact_bbox_area_cv": area_cv,
        "artifact_max_center_step": center_step, "artifact_conf_min": conf_min,
        "tracking_quality_bad": float(frame_gap > 10.0 or area_cv > 1.8 or center_step > 45.0 or conf_min < 0.18),
    }


def _interp(source_times: np.ndarray, values: np.ndarray, sample_times: np.ndarray) -> np.ndarray:
    finite = np.isfinite(source_times) & np.isfinite(values)
    if finite.sum() == 0:
        return np.full(len(sample_times), np.nan)
    if finite.sum() == 1:
        return np.full(len(sample_times), float(values[finite][0]))
    return np.interp(sample_times, source_times[finite], values[finite])


def _resample(window: pd.DataFrame, start: float, end: float, video_id: str, track_id: int, window_id: int, fps: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = window.sort_values(["timestamp_sec", "frame"], kind="mergesort").reset_index(drop=True)
    sample_times = np.linspace(start, end, SAMPLE_COUNT, endpoint=False)
    times = frame["timestamp_sec"].to_numpy(dtype=float)
    nearest = np.clip(np.searchsorted(times, sample_times, side="left"), 0, len(times) - 1)
    previous = np.clip(nearest - 1, 0, len(times) - 1)
    nearest = np.where(np.abs(sample_times - times[previous]) <= np.abs(sample_times - times[nearest]), previous, nearest)
    columns = {name: pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float) for name in ["x1", "y1", "x2", "y2", "conf"]}
    interpolated = {name: _interp(times, values, sample_times) for name, values in columns.items()}
    widths = columns["x2"] - columns["x1"]
    heights = columns["y2"] - columns["y1"]
    interpolated["center_x"] = _interp(times, (columns["x1"] + columns["x2"]) * 0.5, sample_times)
    interpolated["center_y"] = _interp(times, (columns["y1"] + columns["y2"]) * 0.5, sample_times)
    interpolated["width"] = _interp(times, widths, sample_times)
    interpolated["height"] = _interp(times, heights, sample_times)
    interpolated["area"] = _interp(times, widths * heights, sample_times)
    valid = (sample_times >= float(np.min(times)) - 1e-9) & (sample_times <= float(np.max(times)) + 1e-9)
    valid &= np.logical_and.reduce([np.isfinite(interpolated[name]) for name in ["x1", "y1", "x2", "y2"]])
    rows: list[dict[str, Any]] = []
    for index, timestamp in enumerate(sample_times):
        x1, y1, x2, y2, conf = (float(interpolated[name][index]) for name in ["x1", "y1", "x2", "y2", "conf"])
        rows.append({
            "case_key": f"{video_id}/{track_id}", "video_id": video_id, "track_id": track_id,
            "window_id": window_id, "sample_index": index, "sample_timestamp_sec": float(timestamp),
            "nearest_frame": int(frame.iloc[int(nearest[index])]["frame"]),
            "interpolated_x1": x1, "interpolated_y1": y1, "interpolated_x2": x2, "interpolated_y2": y2,
            "interpolated_center_x": float(interpolated["center_x"][index]),
            "interpolated_center_y": float(interpolated["center_y"][index]),
            "interpolated_width": float(interpolated["width"][index]),
            "interpolated_height": float(interpolated["height"][index]),
            "interpolated_area": float(interpolated["area"][index]), "interpolated_confidence": conf,
            "valid_sample": bool(valid[index]), "fps": fps,
        })
    return pd.DataFrame(rows), {"valid_sample_count": int(valid.sum()), "missing_sample_ratio": float(1.0 - valid.mean())}


def _feature_matrix(samples: pd.DataFrame) -> tuple[np.ndarray, list[tuple[int, float, float, float, float, float]]]:
    feature_rows: list[np.ndarray] = []
    raw_rows: list[tuple[int, float, float, float, float, float]] = []
    previous: tuple[float, float, float, float] | None = None
    previous_velocity = (0.0, 0.0)
    previous_heading = 0.0
    for row in samples.sort_values("sample_index").itertuples(index=False):
        x1, y1, x2, y2 = row.interpolated_x1, row.interpolated_y1, row.interpolated_x2, row.interpolated_y2
        cx, cy, width, height = (x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1
        if previous is None:
            vx = vy = ax = ay = dw = dh = heading_change = heading = 0.0
        else:
            vx, vy = cx - previous[0], cy - previous[1]
            ax, ay = vx - previous_velocity[0], vy - previous_velocity[1]
            dw, dh = width - previous[2], height - previous[3]
            heading = math.atan2(vy, vx)
            heading_change = (heading - previous_heading + math.pi) % (2.0 * math.pi) - math.pi
        values = {
            "cx": cx, "cy": cy, "w": width, "h": height, "vx": vx, "vy": vy,
            "speed": math.hypot(vx, vy), "ax": ax, "ay": ay, "dw": dw, "dh": dh,
            "heading_change": heading_change, "aspect_ratio": width / height if abs(height) > 1e-6 else 0.0,
            "conf": float(row.interpolated_confidence),
        }
        feature_rows.append(np.asarray([values[column] for column in FEATURE_COLUMNS], dtype=np.float32))
        raw_rows.append((int(row.nearest_frame), float(x1), float(y1), float(x2), float(y2), float(row.interpolated_confidence)))
        previous = (cx, cy, width, height)
        previous_velocity, previous_heading = (vx, vy), heading
    return np.stack(feature_rows), raw_rows


@dataclass
class TrackState:
    track_id: int
    next_start_sec: float | None = None
    emitted_count: int = 0
    last_timestamp_sec: float = 0.0
    rows: deque[dict[str, Any]] = field(default_factory=deque)

    def append(self, row: Mapping[str, Any]) -> list[tuple[int, float, float, pd.DataFrame]]:
        timestamp = float(row["timestamp_sec"])
        emitted: list[tuple[int, float, float, pd.DataFrame]] = []
        if self.rows and timestamp > self.last_timestamp_sec + WINDOW_SEC:
            emitted.extend(self.tail())
            self.rows.clear()
            self.next_start_sec = timestamp
        elif not self.rows and self.next_start_sec is not None and timestamp > self.next_start_sec + WINDOW_SEC:
            self.next_start_sec = timestamp
        self.last_timestamp_sec = timestamp
        if self.next_start_sec is None:
            self.next_start_sec = timestamp
        self.rows.append(dict(row))
        while self.next_start_sec is not None and timestamp >= self.next_start_sec + WINDOW_SEC - 1e-9:
            current = self._current()
            if current is not None:
                emitted.append(current)
            self.next_start_sec += STRIDE_SEC
            self._drop()
        return emitted

    def tail(self) -> list[tuple[int, float, float, pd.DataFrame]]:
        emitted: list[tuple[int, float, float, pd.DataFrame]] = []
        while self.next_start_sec is not None and self.next_start_sec <= self.last_timestamp_sec + 1e-9:
            current = self._current()
            if current is not None:
                emitted.append(current)
            self.next_start_sec += STRIDE_SEC
            self._drop()
        return emitted

    def _current(self) -> tuple[int, float, float, pd.DataFrame] | None:
        assert self.next_start_sec is not None
        start, end = float(self.next_start_sec), float(self.next_start_sec + WINDOW_SEC)
        rows = [row for row in self.rows if start - 1e-9 <= float(row["timestamp_sec"]) < end - 1e-9]
        if not rows:
            return None
        window_id = self.emitted_count
        self.emitted_count += 1
        return window_id, start, end, pd.DataFrame(rows)

    def _drop(self) -> None:
        assert self.next_start_sec is not None
        while self.rows and float(self.rows[0]["timestamp_sec"]) < self.next_start_sec - 1e-9:
            self.rows.popleft()


def _materialize_window(
    *, video_id: str, track_id: int, window_id: int, start: float, end: float, window: pd.DataFrame,
    fps: float, lane_map: Mapping[str, Any], geometry: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    samples, quality = _resample(window, start, end, video_id, track_id, window_id, fps)
    matrix, raw_rows = _feature_matrix(samples)
    official_values = {name: float(value) for name, value in zip(OFFICIAL_40D_FEATURES, np.concatenate([_window_stats(matrix), _lane_stats(matrix, lane_map)]), strict=True)}
    flow = _flow_features(raw_rows, lane_map)
    double_yellow = _double_yellow_features(raw_rows, geometry)
    artifact = _artifact_features(raw_rows)
    row = {
        "case_key": f"{video_id}/{track_id}", "source_type": "anomaly_csv", "source_id": video_id,
        "video_id": video_id, "track_id": str(track_id), "ts_window_idx": int(window_id),
        "start_sec": float(start), "end_sec": float(end), "window_sec": WINDOW_SEC,
        "stride_sec": STRIDE_SEC, "sample_count": SAMPLE_COUNT, "fps": float(fps),
        "detection_count": int(len(window)), **official_values, **flow, **double_yellow, **artifact, **quality,
    }
    samples["source_type"] = "anomaly_csv"
    samples["source_id"] = video_id
    samples["ts_window_idx"] = int(window_id)
    return row, samples


def _round_csv_float(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        return float(f"{float(value):.10g}") if math.isfinite(float(value)) else np.nan
    return value


def materialize_fixed20(
    *, timestamp_csv: Path, lane_map_json: Path, double_yellow_json: Path, chunksize: int = 250_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read a timestamp CSV in chunks and emit formal fixed-20 feature/sample rows."""

    paths = [Path(timestamp_csv), Path(lane_map_json), Path(double_yellow_json)]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    lane_map = load_lane_map(paths[1])
    geometry = json.loads(paths[2].read_text(encoding="utf-8"))
    accepted_ids = scan_car_track_ids(paths[0], chunksize=chunksize)
    if not accepted_ids:
        raise ValueError("timestamp CSV contains no car-majority track (cls=2)")
    states: dict[int, TrackState] = {}
    feature_rows: list[dict[str, Any]] = []
    sample_frames: list[pd.DataFrame] = []
    video_id = ""
    fps = 0.0
    required = {"frame", "track_id", "x1", "y1", "x2", "y2", "conf", "cls", "timestamp_sec"}
    for chunk in pd.read_csv(paths[0], chunksize=chunksize, low_memory=False):
        missing = sorted(required - set(chunk.columns))
        if missing:
            raise ValueError(f"timestamp CSV is missing columns: {missing}")
        if not video_id:
            id_column = "source_video_id" if "source_video_id" in chunk.columns else "video_id"
            video_id = str(chunk[id_column].dropna().iloc[0]) if id_column in chunk and chunk[id_column].notna().any() else paths[0].stem
            fps = float(pd.to_numeric(chunk.get("fps"), errors="coerce").dropna().iloc[0]) if "fps" in chunk and chunk["fps"].notna().any() else 30.0
        chunk["track_id"] = pd.to_numeric(chunk["track_id"], errors="coerce").fillna(-1).astype(int)
        # Keep every detection of an accepted car track.  Per-frame cls may
        # momentarily flip even though ByteTrack kept the same physical ID.
        chunk = chunk.loc[chunk["track_id"].isin(accepted_ids)]
        for raw in chunk.to_dict("records"):
            track_id = int(raw["track_id"])
            state = states.setdefault(track_id, TrackState(track_id))
            for window_id, start, end, window in state.append(raw):
                feature, samples = _materialize_window(
                    video_id=video_id, track_id=track_id, window_id=window_id, start=start, end=end,
                    window=window, fps=fps, lane_map=lane_map, geometry=geometry,
                )
                feature_rows.append(feature)
                sample_frames.append(samples)
    for track_id, state in states.items():
        for window_id, start, end, window in state.tail():
            feature, samples = _materialize_window(
                video_id=video_id, track_id=track_id, window_id=window_id, start=start, end=end,
                window=window, fps=fps, lane_map=lane_map, geometry=geometry,
            )
            feature_rows.append(feature)
            sample_frames.append(samples)
    features = pd.DataFrame(feature_rows).map(_round_csv_float).sort_values(["track_id", "ts_window_idx"], kind="mergesort").reset_index(drop=True)
    samples = pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame()
    if not samples.empty:
        samples = samples.map(_round_csv_float)
    return features, samples


__all__ = ["load_lane_map", "materialize_fixed20"]
