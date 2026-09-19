#!/usr/bin/env python3
"""Clean red-light-zone movement sidecar generation.

This module distills the accepted upstream generation step behind
``redlight_zone_movement_sidecar_v1``:

    ID window table + frame-level YOLO / traffic-light CSV + stop-line geometry
    -> red-light in-zone movement semantic columns

It only generates semantic sidecar columns.  It does not promote or demote any
risk class; downstream policies such as v42, v52, v58, v63 and v68b consume the
generated columns.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


KEY_COLUMNS = ["case_key", "video_id", "track_id", "ts_window_idx"]


@dataclass(frozen=True)
class CleanRedlightZoneMovementSidecarColumns:
    movement_signal: str = "v92_redlight_zone_movement_signal_v1"
    movement_score: str = "v92_redlight_zone_movement_score_v1"
    red_frame_count: str = "v92_redlight_zone_red_frame_count"
    red_duration_sec: str = "v92_redlight_zone_red_duration_sec"
    forward_progress_px: str = "v92_redlight_zone_forward_progress_px"
    bottom_y_delta_px: str = "v92_redlight_zone_bottom_y_delta_px"
    between_ratio: str = "v92_redlight_zone_between_ratio"
    red_ratio: str = "v92_redlight_zone_red_ratio"
    prior_green_between: str = "v92_redlight_zone_prior_green_between"
    first_seen_between_red: str = "v92_redlight_zone_first_seen_between_red"
    first_frame: str = "v92_redlight_zone_first_frame"
    first_time_sec: str = "v92_redlight_zone_first_time_sec"
    mean_conf: str = "v92_redlight_zone_mean_conf"


COLS = CleanRedlightZoneMovementSidecarColumns()


HISTORICAL_REDLIGHT_ZONE_SIDECAR_COLUMNS = {
    COLS.movement_signal: "redlight_zone_movement_signal_v1",
    COLS.movement_score: "redlight_zone_movement_score_v1",
    COLS.red_frame_count: "redlight_zone_red_frame_count",
    COLS.red_duration_sec: "redlight_zone_red_duration_sec",
    COLS.forward_progress_px: "redlight_zone_forward_progress_px",
    COLS.bottom_y_delta_px: "redlight_zone_bottom_y_delta_px",
    COLS.between_ratio: "redlight_zone_between_ratio",
    COLS.red_ratio: "redlight_zone_red_ratio",
    COLS.prior_green_between: "redlight_zone_prior_green_between",
    COLS.first_seen_between_red: "redlight_zone_first_seen_between_red",
    COLS.first_frame: "redlight_zone_first_frame",
    COLS.first_time_sec: "redlight_zone_first_time_sec",
    COLS.mean_conf: "redlight_zone_mean_conf",
}


def norm_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        num = float(text)
    except ValueError:
        return text
    if num.is_integer():
        return str(int(num))
    return text


def normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "case_key" in out.columns:
        out["case_key"] = out["case_key"].astype(str).str.strip()
    for col in ["video_id", "track_id"]:
        if col in out.columns:
            out[col] = out[col].map(norm_id)
    if "ts_window_idx" in out.columns:
        out["ts_window_idx"] = pd.to_numeric(out["ts_window_idx"], errors="coerce").fillna(-1).astype(int)
    return out


def load_stoplines(path: Path) -> dict[str, list[list[float]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, list[list[float]]] = {}
    for item in payload.get("stop_lines", []):
        points = item.get("points", [])
        if len(points) >= 2:
            out[str(item.get("id", ""))] = [
                [float(points[0][0]), float(points[0][1])],
                [float(points[-1][0]), float(points[-1][1])],
            ]
    required = {"stop_line_1", "stop_line_2"}
    missing = required - set(out)
    if missing:
        raise RuntimeError(f"missing required stop lines: {sorted(missing)}")
    return out


def signed_distance(x: np.ndarray, y: np.ndarray, points: list[list[float]]) -> np.ndarray:
    (x1, y1), (x2, y2) = points
    a = y2 - y1
    b = -(x2 - x1)
    c = x2 * y1 - y2 * x1
    denom = math.hypot(a, b)
    if denom < 1e-9:
        return np.zeros_like(x, dtype=np.float64)
    return (a * x + b * y + c) / denom


def prepare_track(raw_track: pd.DataFrame, stoplines: dict[str, list[list[float]]]) -> pd.DataFrame:
    t = raw_track.sort_values(["timestamp_sec", "frame"], kind="mergesort").copy()
    t["bbox_bottom_center_x"] = (
        pd.to_numeric(t["x1"], errors="coerce") + pd.to_numeric(t["x2"], errors="coerce")
    ) / 2.0
    t["bbox_bottom_center_y"] = pd.to_numeric(t["y2"], errors="coerce")
    x = t["bbox_bottom_center_x"].to_numpy(dtype=np.float64)
    y = t["bbox_bottom_center_y"].to_numpy(dtype=np.float64)
    t["zone_dist_stop_line_1"] = signed_distance(x, y, stoplines["stop_line_1"])
    t["zone_dist_stop_line_2"] = signed_distance(x, y, stoplines["stop_line_2"])

    # In the current camera setup, stop_line_2 is upstream and stop_line_1 is
    # downstream for the signal-controlled direction.  A vehicle is in the
    # red-light zone after passing stop_line_2 but before passing stop_line_1.
    t["zone_between_stoplines"] = (t["zone_dist_stop_line_2"] < -1.0) & (
        t["zone_dist_stop_line_1"] > 1.0
    )
    tl_state = t.get("tl_state", pd.Series("", index=t.index)).fillna("").astype(str).str.lower()
    red_prob = pd.to_numeric(t.get("tl_prob_red", 0.0), errors="coerce").fillna(0.0)
    green_prob = pd.to_numeric(t.get("tl_prob_green", 0.0), errors="coerce").fillna(0.0)
    t["zone_is_red"] = tl_state.eq("red") | (red_prob >= 0.80)
    t["zone_is_green"] = tl_state.eq("green") | (green_prob >= 0.55)
    t["zone_is_red_between"] = t["zone_is_red"] & t["zone_between_stoplines"]
    t["zone_is_green_between"] = t["zone_is_green"] & t["zone_between_stoplines"]
    t["timestamp_sec"] = pd.to_numeric(t["timestamp_sec"], errors="coerce")
    t["conf"] = pd.to_numeric(t.get("conf", 0.0), errors="coerce").fillna(0.0)
    return t


def summarize_window(track: pd.DataFrame, row: pd.Series) -> dict[str, Any]:
    start = float(row["start_sec"])
    end = float(row["end_sec"])
    eps = 1e-6
    w = track[(track["timestamp_sec"] >= start - eps) & (track["timestamp_sec"] <= end + eps)].copy()
    before = track[track["timestamp_sec"] < start - eps]
    first = track.iloc[0]
    if w.empty:
        return {
            COLS.movement_signal: 0,
            COLS.movement_score: 0.0,
            COLS.red_frame_count: 0,
            COLS.red_duration_sec: 0.0,
            COLS.forward_progress_px: 0.0,
            COLS.bottom_y_delta_px: 0.0,
            COLS.between_ratio: 0.0,
            COLS.red_ratio: 0.0,
            COLS.prior_green_between: int(bool(before["zone_is_green_between"].any())),
            COLS.first_seen_between_red: int(bool(first["zone_is_red_between"])),
            COLS.first_frame: int(first["frame"]),
            COLS.first_time_sec: float(first["timestamp_sec"]),
            COLS.mean_conf: 0.0,
        }

    red_zone = w[w["zone_is_red_between"]].copy()
    red_count = int(red_zone.shape[0])
    red_duration = (
        float(red_zone["timestamp_sec"].max() - red_zone["timestamp_sec"].min()) if red_count >= 2 else 0.0
    )
    between_ratio = float(w["zone_between_stoplines"].mean()) if len(w) else 0.0
    red_ratio = float(w["zone_is_red"].mean()) if len(w) else 0.0
    prior_green_between = bool(before["zone_is_green_between"].any())
    first_seen_between_red = bool(first["zone_is_red_between"])
    first_frame = int(first["frame"])
    first_time = float(first["timestamp_sec"])
    mean_conf = float(red_zone["conf"].mean()) if red_count else 0.0
    if red_count >= 2:
        # Positive progress means moving toward the downstream stop_line_1.
        progress = float(red_zone["zone_dist_stop_line_1"].iloc[0] - red_zone["zone_dist_stop_line_1"].iloc[-1])
        bottom_delta = float(red_zone["bbox_bottom_center_y"].iloc[-1] - red_zone["bbox_bottom_center_y"].iloc[0])
    else:
        progress = 0.0
        bottom_delta = 0.0

    movement_strength = min(1.0, max(progress, bottom_delta, 0.0) / 12.0)
    duration_strength = min(1.0, red_duration / 0.45)
    count_strength = min(1.0, red_count / 12.0)
    confidence_strength = min(1.0, mean_conf / 0.55) if mean_conf > 0 else 0.0
    legal_guard = prior_green_between
    signal = (
        red_count >= 8
        and red_duration >= 0.20
        and red_ratio >= 0.55
        and between_ratio >= 0.60
        and max(progress, bottom_delta) >= 6.0
        and confidence_strength >= 0.50
        and not legal_guard
    )
    score = (
        0.30 * movement_strength
        + 0.25 * duration_strength
        + 0.20 * count_strength
        + 0.15 * between_ratio
        + 0.10 * confidence_strength
    )
    if legal_guard:
        score *= 0.25
    return {
        COLS.movement_signal: int(signal),
        COLS.movement_score: float(min(1.0, score)),
        COLS.red_frame_count: red_count,
        COLS.red_duration_sec: red_duration,
        COLS.forward_progress_px: progress,
        COLS.bottom_y_delta_px: bottom_delta,
        COLS.between_ratio: between_ratio,
        COLS.red_ratio: red_ratio,
        COLS.prior_green_between: int(prior_green_between),
        COLS.first_seen_between_red: int(first_seen_between_red),
        COLS.first_frame: first_frame,
        COLS.first_time_sec: first_time,
        COLS.mean_conf: mean_conf,
    }


def _load_raw_video(raw_path: Path, video_id: str) -> pd.DataFrame:
    raw = pd.read_csv(raw_path, low_memory=False)
    if "timestamp_sec" not in raw.columns:
        fps = float(raw.get("fps", pd.Series([30.0])).iloc[0]) if "fps" in raw.columns else 30.0
        raw["timestamp_sec"] = pd.to_numeric(raw["frame"], errors="coerce") / fps
    raw["video_id"] = raw.get("numeric_video_id", video_id)
    raw["video_id"] = raw["video_id"].map(norm_id)
    raw["track_id"] = raw["track_id"].map(norm_id)
    return raw


def _window_reference(windows: pd.DataFrame) -> pd.DataFrame:
    required = ["case_key", "video_id", "track_id", "ts_window_idx", "start_sec", "end_sec"]
    missing = [c for c in required if c not in windows.columns]
    if missing:
        raise KeyError(f"window reference missing required columns: {missing}")
    keep = required + (["source_csv"] if "source_csv" in windows.columns else [])
    ref = normalize_keys(windows[keep].drop_duplicates().copy())
    ref["start_sec"] = pd.to_numeric(ref["start_sec"], errors="coerce")
    ref["end_sec"] = pd.to_numeric(ref["end_sec"], errors="coerce")
    return ref


def build_clean_redlight_zone_movement_sidecar(
    windows: pd.DataFrame,
    *,
    source_dir: Path,
    stopline_json: Path,
) -> pd.DataFrame:
    """Rebuild the accepted red-light-zone movement sidecar."""

    ref = _window_reference(windows)
    stoplines = load_stoplines(stopline_json)
    outputs: list[pd.DataFrame] = []
    raw_cache: dict[str, pd.DataFrame] = {}

    for video_id, video_windows in ref.groupby("video_id", sort=True):
        source_csv = ""
        if "source_csv" in video_windows.columns:
            source_csv_values = video_windows["source_csv"].dropna().astype(str)
            source_csv = str(source_csv_values.iloc[0]) if len(source_csv_values) else ""
        raw_path = Path(source_csv) if source_csv else source_dir / f"{video_id}.csv"
        if not raw_path.is_file():
            raw_path = source_dir / f"{video_id}.csv"
        if not raw_path.is_file():
            raise FileNotFoundError(f"source csv not found for video {video_id}: {raw_path}")
        raw_key = str(raw_path)
        if raw_key not in raw_cache:
            raw_cache[raw_key] = _load_raw_video(raw_path, str(video_id))
        raw = raw_cache[raw_key]

        for track_id, track_windows in video_windows.groupby("track_id", sort=True):
            track_raw = raw[raw["track_id"].eq(str(track_id))].copy()
            if track_raw.empty:
                continue
            track = prepare_track(track_raw, stoplines)
            rows: list[dict[str, Any]] = []
            for _, row in track_windows.sort_values("ts_window_idx").iterrows():
                payload = {
                    "case_key": str(row["case_key"]),
                    "video_id": norm_id(row["video_id"]),
                    "track_id": norm_id(row["track_id"]),
                    "ts_window_idx": int(row["ts_window_idx"]),
                }
                payload.update(summarize_window(track, row))
                rows.append(payload)
            outputs.append(pd.DataFrame(rows))

    if not outputs:
        return pd.DataFrame(columns=KEY_COLUMNS + list(HISTORICAL_REDLIGHT_ZONE_SIDECAR_COLUMNS.keys()))
    return pd.concat(outputs, ignore_index=True)


def add_clean_redlight_zone_movement_sidecar(
    frame: pd.DataFrame,
    *,
    windows_path: Path,
    source_dir: Path,
    stopline_json: Path,
) -> pd.DataFrame:
    out = normalize_keys(frame)
    if not windows_path.exists():
        raise FileNotFoundError(windows_path)
    windows = pd.read_csv(
        windows_path,
        usecols=[
            "case_key",
            "video_id",
            "track_id",
            "ts_window_idx",
            "start_sec",
            "end_sec",
            "source_csv",
        ],
        low_memory=False,
    )
    sidecar = normalize_keys(
        build_clean_redlight_zone_movement_sidecar(
            windows,
            source_dir=source_dir,
            stopline_json=stopline_json,
        )
    )
    if sidecar.duplicated(KEY_COLUMNS).any():
        dup = sidecar.loc[sidecar.duplicated(KEY_COLUMNS, keep=False), KEY_COLUMNS].head()
        raise ValueError(f"duplicated redlight-zone sidecar keys: {dup.to_dict('records')}")
    return out.merge(sidecar, on=KEY_COLUMNS, how="left", validate="many_to_one")


def overwrite_historical_redlight_zone_movement_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for clean_col, historical_col in HISTORICAL_REDLIGHT_ZONE_SIDECAR_COLUMNS.items():
        if clean_col in out.columns:
            out[historical_col] = out[clean_col]
    return out
