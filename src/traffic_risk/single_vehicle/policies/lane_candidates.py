#!/usr/bin/env python3
"""Clean v28 frame-level lane candidate cache generation.

This module distills the accepted v28 upstream cache step:

    timestamp source CSV + manual lane map + requested video/track set
    -> frame-level lane candidate parquet files

It intentionally does not compute any window-level risk signal.  The generated
parquet cache is consumed by the clean v28 / v41 / v48 sidecar generators.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


FRAME_CACHE_COLUMNS = [
    "frame",
    "track_id",
    "timestamp_sec",
    "x1",
    "y1",
    "x2",
    "y2",
    "conf",
    "cx",
    "cy",
    "bcx",
    "bcy",
    "area",
    "cand_top1",
    "cand_top1_score",
    "cand_top2",
    "cand_top2_score",
    "cand_top3",
    "cand_top3_score",
    "cand_count",
    "cand_scores_json",
]


@dataclass(frozen=True)
class Lane:
    lane_id: str
    polygon: tuple[tuple[float, float], ...]
    bbox: tuple[float, float, float, float]
    direction: tuple[float, float]
    order: int
    priority: float


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False, **kwargs)


def norm_id(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return text


def normalize_ids(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in ["case_key", "video_id", "track_id"]:
        if col in out.columns:
            out[col] = out[col].map(norm_id)
    if "ts_window_idx" in out.columns:
        out["ts_window_idx"] = pd.to_numeric(out["ts_window_idx"], errors="coerce").fillna(-1).astype(int)
    return out


def point_in_polygon(x: float, y: float, poly: Sequence[Sequence[float]]) -> bool:
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def is_turn_lane(lane_id: str) -> bool:
    text = str(lane_id or "").lower()
    return "turn" in text or text.endswith("_left") or text.endswith("_right")


def load_lanes(path: Path) -> list[Lane]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    lanes: list[Lane] = []
    for order, lane in enumerate(raw.get("lanes", [])):
        poly = tuple((float(p[0]), float(p[1])) for p in lane.get("polygon", []))
        if len(poly) < 3:
            continue
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        lane_id = str(lane.get("lane_id", f"lane_{order + 1}"))
        direction = lane.get("direction") or [0.0, 0.0]
        # Historical v28 intentionally used only a tiny turn-lane bias.  Do not
        # read or amplify lane-map priority here; motion context is handled by
        # downstream window sidecars.
        priority = 1.05 if is_turn_lane(lane_id) else 1.0
        lanes.append(
            Lane(
                lane_id=lane_id,
                polygon=poly,
                bbox=(min(xs), min(ys), max(xs), max(ys)),
                direction=(float(direction[0]), float(direction[1])),
                order=order,
                priority=priority,
            )
        )
    return lanes


def candidate_lanes_for_point(x: float, y: float, lanes: Sequence[Lane]) -> list[Lane]:
    if not np.isfinite(x) or not np.isfinite(y):
        return []
    hits: list[Lane] = []
    for lane in lanes:
        x0, y0, x1, y1 = lane.bbox
        if x < x0 or x > x1 or y < y0 or y > y1:
            continue
        if point_in_polygon(float(x), float(y), lane.polygon):
            hits.append(lane)
    return hits


def required_raw_columns() -> list[str]:
    return [
        "frame",
        "track_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "conf",
        "timestamp_sec",
    ]


def raw_with_geometry(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.copy()
    out["track_id"] = out["track_id"].map(norm_id)
    out["timestamp_sec"] = pd.to_numeric(out["timestamp_sec"], errors="coerce")
    for c in ["x1", "y1", "x2", "y2", "conf"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["timestamp_sec", "x1", "y1", "x2", "y2"]).copy()
    out["cx"] = (out["x1"] + out["x2"]) / 2.0
    out["cy"] = (out["y1"] + out["y2"]) / 2.0
    out["bcx"] = out["cx"]
    out["bcy"] = out["y2"]
    out["area"] = (out["x2"] - out["x1"]).clip(lower=0.0) * (out["y2"] - out["y1"]).clip(lower=0.0)
    return out


def frame_candidate_scores(row, lanes: Sequence[Lane]) -> dict[str, float]:
    frame_scores: defaultdict[str, float] = defaultdict(float)
    bottom_hits = candidate_lanes_for_point(float(row.bcx), float(row.bcy), lanes)
    center_hits = candidate_lanes_for_point(float(row.cx), float(row.cy), lanes)
    conf = max(0.05, min(float(row.conf) if np.isfinite(row.conf) else 1.0, 1.0))
    for lane in bottom_hits:
        frame_scores[lane.lane_id] += 1.0 * conf * lane.priority
    for lane in center_hits:
        frame_scores[lane.lane_id] += 0.45 * conf * lane.priority
    return dict(frame_scores)


def build_clean_v28_frame_lane_candidate_cache_for_video(
    *,
    video_id: str,
    video_windows: pd.DataFrame,
    lanes: Sequence[Lane],
    timestamp_dir: Path,
) -> pd.DataFrame:
    csv_path = timestamp_dir / f"{video_id}.csv"
    if not csv_path.exists():
        return pd.DataFrame(columns=FRAME_CACHE_COLUMNS)

    raw = read_csv(csv_path, usecols=lambda c: c in set(required_raw_columns()))
    raw = raw_with_geometry(raw)
    wanted_tracks = set(video_windows["track_id"].astype(str).unique().tolist())
    raw = raw[raw["track_id"].astype(str).isin(wanted_tracks)].copy()

    rows: list[dict[str, object]] = []
    for rr in raw.itertuples(index=False):
        scores = frame_candidate_scores(rr, lanes)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        rows.append(
            {
                "frame": int(rr.frame) if np.isfinite(rr.frame) else -1,
                "track_id": str(rr.track_id),
                "timestamp_sec": float(rr.timestamp_sec),
                "x1": float(rr.x1),
                "y1": float(rr.y1),
                "x2": float(rr.x2),
                "y2": float(rr.y2),
                "conf": float(rr.conf) if np.isfinite(rr.conf) else 1.0,
                "cx": float(rr.cx),
                "cy": float(rr.cy),
                "bcx": float(rr.bcx),
                "bcy": float(rr.bcy),
                "area": float(rr.area),
                "cand_top1": ranked[0][0] if ranked else "",
                "cand_top1_score": float(ranked[0][1]) if ranked else 0.0,
                "cand_top2": ranked[1][0] if len(ranked) >= 2 else "",
                "cand_top2_score": float(ranked[1][1]) if len(ranked) >= 2 else 0.0,
                "cand_top3": ranked[2][0] if len(ranked) >= 3 else "",
                "cand_top3_score": float(ranked[2][1]) if len(ranked) >= 3 else 0.0,
                "cand_count": int(len(ranked)),
                "cand_scores_json": json.dumps(scores, ensure_ascii=True, sort_keys=True),
            }
        )
    if not rows:
        return pd.DataFrame(columns=FRAME_CACHE_COLUMNS)
    return pd.DataFrame(rows, columns=FRAME_CACHE_COLUMNS)


def build_clean_v28_frame_lane_candidate_cache(
    windows: pd.DataFrame,
    *,
    lane_map: Path,
    timestamp_dir: Path,
    cache_dir: Path,
    overwrite: bool = False,
) -> dict[str, int]:
    """Build clean v28 frame candidate cache files.

    Returns a small summary with built/skipped/missing/empty counts.  Empty
    videos are intentionally not written, matching the historical builder.
    """

    lanes = load_lanes(lane_map)
    cache_dir.mkdir(parents=True, exist_ok=True)
    windows = normalize_ids(windows[["video_id", "track_id"]].drop_duplicates().copy())

    summary = {"built": 0, "skipped_existing": 0, "missing_timestamp_csv": 0, "empty": 0}
    for video_id, video_windows in windows.groupby("video_id", sort=True):
        out_path = cache_dir / f"{video_id}.parquet"
        if out_path.exists() and not overwrite:
            summary["skipped_existing"] += 1
            continue
        csv_path = timestamp_dir / f"{video_id}.csv"
        if not csv_path.exists():
            if out_path.exists() and overwrite:
                out_path.unlink()
            summary["missing_timestamp_csv"] += 1
            continue
        df = build_clean_v28_frame_lane_candidate_cache_for_video(
            video_id=str(video_id),
            video_windows=video_windows,
            lanes=lanes,
            timestamp_dir=timestamp_dir,
        )
        if df.empty:
            if out_path.exists() and overwrite:
                out_path.unlink()
            summary["empty"] += 1
            continue
        df.to_parquet(out_path, index=False)
        summary["built"] += 1
    return summary
