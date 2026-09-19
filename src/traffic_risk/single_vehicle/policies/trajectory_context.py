#!/usr/bin/env python3
"""Clean v24f trajectory-context sidecar generation.

This module distills the accepted v24f upstream sidecar generation:

    v22 single-prediction windows
    + C4O feature source
    + trajectory v8 sidecar
    + timestamp detection CSV
    + manual lane map
    -> v24f trajectory context sidecar columns

It does not train a model, change labels, or apply a risk rule.  The output is
historical-compatible with
``trajectory_context_sidecar_v24f.csv`` and is consumed by clean v37 trajectory
semantic sidecar generation.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


KEY_COLUMNS = ["case_key", "video_id", "track_id", "ts_window_idx"]

V24F_SIDECAR_COLUMNS = [
    "split",
    "case_key",
    "video_id",
    "track_id",
    "ts_window_idx",
    "start_sec",
    "end_sec",
    "pred_v22_candidate",
    "prob_class0",
    "prob_class1",
    "prob_class2",
    "y_schemaC",
    "redlight_event_score",
    "overspeed_event_score",
    "s33_current_local_semantic_support",
    "s33_prefix_state_confirmed",
    "trajectory_subtype_primary",
    "trajectory_subtype_peak",
    "lane_family_switch_count",
    "opposite_lane_run_length",
    "opposite_lane_occupancy_ratio",
    "reversed_flag",
    "artifact_bbox_area_cv",
    "artifact_max_center_step",
    "osc_h20_score",
    "osc_h30_score",
    "osc_h20_lateral_sign_change_count",
    "osc_h30_lateral_sign_change_count",
    "osc_h20_two_sided_ratio",
    "osc_h30_two_sided_ratio",
    "v24f_raw_count",
    "v24f_lane_reason",
    "v24f_lane_id_raw",
    "v24f_lane_score_top",
    "v24f_lane_score_second",
    "v24f_lane_confidence_raw",
    "v24f_lane_valid_ratio",
    "v24f_lane_ambiguous_ratio",
    "v24f_bbox_area_cv_raw",
    "v24f_bbox_w_cv_raw",
    "v24f_bbox_h_cv_raw",
    "v24f_bbox_area_log_jump_max",
    "v24f_conf_min_raw",
    "v24f_conf_mean_raw",
    "v24f_center_step_max_raw",
    "v24f_center_displacement_raw",
    "v24f_osc_score_max",
    "v24f_osc_sign_change_max",
    "v24f_osc_two_sided_ratio_max",
    "v24f_occlusion_jitter_current",
    "v24f_lane_id_smooth",
    "v24f_lane_sequence_smooth_6w",
    "v24f_transition_type_6w",
    "v24f_lane_change_count_6w",
    "v24f_lane_unique_count_6w",
    "v24f_lane_id_backtrack_6w",
    "v24f_straight_family_backtrack_6w",
    "v24f_contains_turn_lane_6w",
    "v24f_turn_route_like",
    "v24f_single_lane_change_like",
    "v24f_normal_turn_like",
    "v24f_occlusion_jitter_count_6w",
    "v24f_true_weaving_current",
    "v24f_true_weaving_memory",
    "v24f_context_quality",
]

V24F_TEXT_COLUMNS = {
    "split",
    "case_key",
    "video_id",
    "track_id",
    "trajectory_subtype_primary",
    "trajectory_subtype_peak",
    "v24f_lane_reason",
    "v24f_lane_id_raw",
    "v24f_lane_id_smooth",
    "v24f_lane_sequence_smooth_6w",
    "v24f_transition_type_6w",
}


@dataclass(frozen=True)
class Lane:
    lane_id: str
    polygon: tuple[tuple[float, float], ...]
    bbox: tuple[float, float, float, float]
    direction: tuple[float, float]
    order: int
    priority: int


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
    if "source_id" in out.columns:
        out["source_id"] = out["source_id"].map(norm_id)
    if "ts_window_idx" in out.columns:
        out["ts_window_idx"] = pd.to_numeric(out["ts_window_idx"], errors="coerce").fillna(-1).astype(int)
    return out


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False, **kwargs)


def s(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)
    return pd.Series(default, index=df.index, dtype=float)


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
        # v24f historical sidecar used a stronger turn-lane priority than
        # later v28 motion-aware sidecars.  Keep this value for exact replay.
        priority = 2 if is_turn_lane(lane_id) else 1
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


def frame_lane_scores(row, lanes: Sequence[Lane]) -> dict[str, float]:
    conf = float(row.conf) if np.isfinite(row.conf) else 1.0
    conf = max(0.05, min(conf, 1.0))
    scores: defaultdict[str, float] = defaultdict(float)
    for lane in candidate_lanes_for_point(float(row.bcx), float(row.bcy), lanes):
        scores[lane.lane_id] += 1.0 * conf * lane.priority
    for lane in candidate_lanes_for_point(float(row.cx), float(row.cy), lanes):
        scores[lane.lane_id] += 0.55 * conf * lane.priority
    return dict(scores)


def lane_group(lane_id: str) -> str:
    text = str(lane_id or "")
    if not text:
        return ""
    if text.startswith("downbound"):
        return "downbound"
    if text.startswith("upbound"):
        return "upbound"
    if text.startswith("new_lane"):
        return "new_lane"
    return text.split("_")[0]


def compress_sequence(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if not text:
            continue
        if not out or out[-1] != text:
            out.append(text)
    return out


def has_backtrack(seq: Sequence[str]) -> bool:
    if len(seq) < 3:
        return False
    for i in range(len(seq) - 2):
        if seq[i] == seq[i + 2] and seq[i] != seq[i + 1]:
            return True
    return False


def lane_family_sequence(seq: Sequence[str], *, drop_turn: bool = False) -> list[str]:
    values: list[str] = []
    for lane_id in seq:
        lane_id = str(lane_id or "")
        if not lane_id:
            continue
        if drop_turn and is_turn_lane(lane_id):
            continue
        values.append(lane_group(lane_id))
    return compress_sequence(values)


def coeff_var(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return 0.0
    mean = float(np.mean(values))
    if abs(mean) <= 1e-6:
        return 0.0
    return float(np.std(values) / abs(mean))


def max_abs_log_jump(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 1e-6)]
    if values.size <= 1:
        return 0.0
    return float(np.max(np.abs(np.diff(np.log(values)))))


def load_predictions(single_prediction_dir: Path) -> pd.DataFrame:
    frames = []
    for split in ["train", "val", "test"]:
        path = single_prediction_dir / f"v22_frozen_scene_{split}_window_predictions.csv"
        part = normalize_ids(read_csv(path))
        part["split"] = split
        keep = [
            "split",
            "case_key",
            "video_id",
            "track_id",
            "ts_window_idx",
            "start_sec",
            "end_sec",
            "y_true",
            "y_schemaC",
            "pred_v22_candidate",
            "pred_final",
            "prob_class0",
            "prob_class1",
            "prob_class2",
        ]
        frames.append(part[[c for c in keep if c in part.columns]].copy())
    out = pd.concat(frames, ignore_index=True)
    if "pred_v22_candidate" not in out.columns and "pred_final" in out.columns:
        out["pred_v22_candidate"] = out["pred_final"]
    out["pred_v22_candidate"] = pd.to_numeric(out["pred_v22_candidate"], errors="coerce").fillna(0).astype(int)
    if "y_schemaC" in out.columns:
        out["y_schemaC"] = pd.to_numeric(out["y_schemaC"], errors="coerce").fillna(0).astype(int)
    return out


def empty_lane_row(row, reason: str) -> dict[str, object]:
    return {
        "case_key": str(row.case_key),
        "video_id": str(row.video_id),
        "track_id": str(row.track_id),
        "ts_window_idx": int(row.ts_window_idx),
        "v24f_raw_count": 0,
        "v24f_lane_reason": reason,
        "v24f_lane_id_raw": "",
        "v24f_lane_score_top": 0.0,
        "v24f_lane_score_second": 0.0,
        "v24f_lane_confidence_raw": 0.0,
        "v24f_lane_valid_ratio": 0.0,
        "v24f_lane_ambiguous_ratio": 0.0,
        "v24f_bbox_area_cv_raw": 0.0,
        "v24f_bbox_w_cv_raw": 0.0,
        "v24f_bbox_h_cv_raw": 0.0,
        "v24f_bbox_area_log_jump_max": 0.0,
        "v24f_conf_min_raw": 1.0,
        "v24f_conf_mean_raw": 1.0,
        "v24f_center_step_max_raw": 0.0,
        "v24f_center_displacement_raw": 0.0,
    }


def compute_lane_window_row(row, part: pd.DataFrame, lanes: Sequence[Lane]) -> dict[str, object]:
    out = empty_lane_row(row, "ok")
    out["v24f_raw_count"] = int(len(part))
    if part.empty:
        out["v24f_lane_reason"] = "empty_window_rows"
        return out

    score_sum: defaultdict[str, float] = defaultdict(float)
    valid_frames = 0
    ambiguous_frames = 0
    for raw_row in part.itertuples(index=False):
        scores = frame_lane_scores(raw_row, lanes)
        if scores:
            valid_frames += 1
            if len(scores) >= 2:
                ambiguous_frames += 1
            for lane_id, score in scores.items():
                score_sum[lane_id] += score

    ranked = sorted(score_sum.items(), key=lambda kv: (-kv[1], kv[0]))
    top_lane = ranked[0][0] if ranked else ""
    top_score = float(ranked[0][1]) if ranked else 0.0
    second_score = float(ranked[1][1]) if len(ranked) >= 2 else 0.0
    total_score = float(sum(score_sum.values()))
    raw_conf = max(0.0, min(1.0, (top_score - second_score) / total_score)) if total_score > 0 else 0.0
    valid_ratio = valid_frames / max(len(part), 1)
    ambiguous_ratio = ambiguous_frames / max(len(part), 1)

    areas = part["area"].to_numpy(dtype=float)
    widths = part["w"].to_numpy(dtype=float)
    heights = part["h"].to_numpy(dtype=float)
    conf = pd.to_numeric(part["conf"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    cx = part["cx"].to_numpy(dtype=float)
    cy = part["cy"].to_numpy(dtype=float)
    if len(cx) >= 2:
        steps = np.hypot(np.diff(cx), np.diff(cy))
        center_step_max = float(np.nanmax(steps)) if len(steps) else 0.0
        center_disp = float(math.hypot(float(cx[-1] - cx[0]), float(cy[-1] - cy[0])))
    else:
        center_step_max = 0.0
        center_disp = 0.0

    out.update(
        {
            "v24f_lane_reason": "ok",
            "v24f_lane_id_raw": top_lane,
            "v24f_lane_score_top": top_score,
            "v24f_lane_score_second": second_score,
            "v24f_lane_confidence_raw": raw_conf,
            "v24f_lane_valid_ratio": float(valid_ratio),
            "v24f_lane_ambiguous_ratio": float(ambiguous_ratio),
            "v24f_bbox_area_cv_raw": coeff_var(areas),
            "v24f_bbox_w_cv_raw": coeff_var(widths),
            "v24f_bbox_h_cv_raw": coeff_var(heights),
            "v24f_bbox_area_log_jump_max": max_abs_log_jump(areas),
            "v24f_conf_min_raw": float(np.nanmin(conf)) if len(conf) else 1.0,
            "v24f_conf_mean_raw": float(np.nanmean(conf)) if len(conf) else 1.0,
            "v24f_center_step_max_raw": center_step_max,
            "v24f_center_displacement_raw": center_disp,
        }
    )
    return out


def build_window_lane_sidecar(windows: pd.DataFrame, timestamp_dir: Path, lanes: Sequence[Lane]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    base_cols = KEY_COLUMNS + ["start_sec", "end_sec"]
    windows = normalize_ids(windows[base_cols].copy())
    windows["start_sec"] = pd.to_numeric(windows["start_sec"], errors="coerce")
    windows["end_sec"] = pd.to_numeric(windows["end_sec"], errors="coerce")
    for video_id, video_windows in windows.groupby("video_id", sort=True):
        csv_path = timestamp_dir / f"{video_id}.csv"
        if not csv_path.exists():
            for row in video_windows.itertuples(index=False):
                rows.append(empty_lane_row(row, "missing_timestamp_csv"))
            continue
        raw = read_csv(csv_path)
        raw["track_id"] = raw["track_id"].map(norm_id)
        raw["timestamp_sec"] = pd.to_numeric(raw["timestamp_sec"], errors="coerce")
        for c in ["x1", "y1", "x2", "y2", "conf"]:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
        wanted_tracks = set(video_windows["track_id"].astype(str).unique().tolist())
        raw = raw[raw["track_id"].astype(str).isin(wanted_tracks)].dropna(
            subset=["timestamp_sec", "x1", "y1", "x2", "y2"]
        ).copy()
        if raw.empty:
            for row in video_windows.itertuples(index=False):
                rows.append(empty_lane_row(row, "missing_track_rows"))
            continue

        raw["cx"] = (raw["x1"] + raw["x2"]) / 2.0
        raw["cy"] = (raw["y1"] + raw["y2"]) / 2.0
        raw["bcx"] = raw["cx"]
        raw["bcy"] = raw["y2"]
        raw["area"] = (raw["x2"] - raw["x1"]).clip(lower=0.0) * (raw["y2"] - raw["y1"]).clip(lower=0.0)
        raw["w"] = (raw["x2"] - raw["x1"]).clip(lower=0.0)
        raw["h"] = (raw["y2"] - raw["y1"]).clip(lower=0.0)

        for track_id, track_windows in video_windows.groupby("track_id", sort=True):
            tr = raw[raw["track_id"].astype(str).eq(str(track_id))].sort_values("timestamp_sec", kind="mergesort")
            if tr.empty:
                for row in track_windows.itertuples(index=False):
                    rows.append(empty_lane_row(row, "missing_track_rows"))
                continue
            times = tr["timestamp_sec"].to_numpy(dtype=float)
            for row in track_windows.sort_values("ts_window_idx").itertuples(index=False):
                start = float(row.start_sec)
                end = float(row.end_sec)
                left = int(np.searchsorted(times, start, side="left"))
                right = int(np.searchsorted(times, end, side="right"))
                rows.append(compute_lane_window_row(row, tr.iloc[left:right], lanes))
    return pd.DataFrame(rows)


def causal_smooth_lane_ids(part: pd.DataFrame) -> list[str]:
    raw = part["v24f_lane_id_raw"].astype(str).tolist()
    conf = s(part, "v24f_lane_confidence_raw").tolist()
    valid = s(part, "v24f_lane_valid_ratio").tolist()
    smoothed: list[str] = []
    active = ""
    pending = ""
    pending_count = 0
    for lane_id, c, v in zip(raw, conf, valid):
        usable = bool(lane_id) and v >= 0.30
        if not usable:
            smoothed.append(active)
            pending = ""
            pending_count = 0
            continue
        if not active:
            active = lane_id
            smoothed.append(active)
            continue
        if lane_id == active:
            pending = ""
            pending_count = 0
            smoothed.append(active)
            continue
        if c >= 0.62 and v >= 0.45:
            active = lane_id
            pending = ""
            pending_count = 0
            smoothed.append(active)
            continue
        if lane_id == pending:
            pending_count += 1
        else:
            pending = lane_id
            pending_count = 1
        if pending_count >= 2 and c >= 0.35 and v >= 0.35:
            active = lane_id
            pending = ""
            pending_count = 0
        smoothed.append(active)
    return smoothed


def is_turn_route_like(seq: Sequence[str], *, osc_strong: bool, occ_count: int) -> bool:
    if not seq or not any(is_turn_lane(x) for x in seq):
        return False
    if occ_count > 1:
        return False
    straight_families = lane_family_sequence(seq, drop_turn=True)
    straight_backtrack = has_backtrack(straight_families)
    if not straight_backtrack:
        return True
    if len(seq) <= 4 and not osc_strong:
        return True
    return False


def add_context_features(df: pd.DataFrame, recent_window: int = 6) -> pd.DataFrame:
    out = df.copy()
    out["v24f_osc_score_max"] = np.maximum(s(out, "osc_h20_score"), s(out, "osc_h30_score"))
    out["v24f_osc_sign_change_max"] = np.maximum(
        s(out, "osc_h20_lateral_sign_change_count"), s(out, "osc_h30_lateral_sign_change_count")
    )
    out["v24f_osc_two_sided_ratio_max"] = np.maximum(
        s(out, "osc_h20_two_sided_ratio"), s(out, "osc_h30_two_sided_ratio")
    )
    out["v24f_occlusion_jitter_current"] = (
        (s(out, "v24f_bbox_area_cv_raw") >= 0.45)
        | (s(out, "v24f_bbox_area_log_jump_max") >= 0.55)
        | ((s(out, "artifact_bbox_area_cv") >= 0.42) & (s(out, "v24f_center_step_max_raw") >= 10.0))
        | ((s(out, "v24f_conf_min_raw") < 0.35) & (s(out, "v24f_bbox_area_cv_raw") >= 0.25))
        | ((s(out, "v24f_raw_count") <= 2) & (s(out, "v24f_lane_valid_ratio") < 0.40))
        | ((s(out, "v24f_lane_ambiguous_ratio") >= 0.70) & (s(out, "v24f_lane_confidence_raw") < 0.20))
    ).astype(int)

    for col in ["v24f_lane_id_smooth", "v24f_lane_sequence_smooth_6w", "v24f_transition_type_6w"]:
        out[col] = ""
    for col in [
        "v24f_lane_change_count_6w",
        "v24f_lane_unique_count_6w",
        "v24f_lane_id_backtrack_6w",
        "v24f_straight_family_backtrack_6w",
        "v24f_contains_turn_lane_6w",
        "v24f_turn_route_like",
        "v24f_single_lane_change_like",
        "v24f_normal_turn_like",
        "v24f_occlusion_jitter_count_6w",
        "v24f_true_weaving_current",
        "v24f_true_weaving_memory",
        "v24f_context_quality",
    ]:
        out[col] = 0.0

    for _, idx in out.groupby(["split", "case_key"], sort=False).groups.items():
        part = out.loc[list(idx)].sort_values("ts_window_idx", kind="mergesort")
        ordered_idx = part.index.tolist()
        smooth = causal_smooth_lane_ids(part)
        current_occlusion = part["v24f_occlusion_jitter_current"].astype(int).tolist()

        transition_types: list[str] = []
        seq_texts: list[str] = []
        change_values: list[int] = []
        unique_values: list[int] = []
        lane_id_backtrack_values: list[int] = []
        straight_backtrack_values: list[int] = []
        turn_values: list[int] = []
        turn_route_values: list[int] = []
        single_change_values: list[int] = []
        normal_turn_values: list[int] = []
        occ_values: list[int] = []
        quality_values: list[float] = []
        true_current_values: list[int] = []

        for pos in range(len(part)):
            left = max(0, pos - recent_window + 1)
            seq = compress_sequence(smooth[left : pos + 1])
            changes = max(len(seq) - 1, 0)
            unique_count = len(set(seq))
            occ_count = int(sum(current_occlusion[left : pos + 1]))
            turn_seen = int(any(is_turn_lane(x) for x in seq))
            lane_id_backtrack = int(has_backtrack(seq))
            straight_families = lane_family_sequence(seq, drop_turn=True)
            straight_backtrack = int(has_backtrack(straight_families))
            osc_strong = bool(
                (float(part.iloc[pos].get("v24f_osc_score_max", 0.0)) >= 0.55)
                and (float(part.iloc[pos].get("v24f_osc_sign_change_max", 0.0)) >= 1.0)
                and (float(part.iloc[pos].get("v24f_osc_two_sided_ratio_max", 0.0)) >= 0.30)
            )
            turn_route = is_turn_route_like(seq, osc_strong=osc_strong, occ_count=occ_count)
            single_change = bool(changes <= 1 and not straight_backtrack and not turn_seen and not osc_strong and occ_count <= 1)
            normal_turn = bool((turn_seen or turn_route) and not (straight_backtrack and osc_strong) and changes <= 2 and occ_count <= 1)
            true_weaving = bool(
                not normal_turn
                and not turn_route
                and occ_count <= 2
                and (
                    (lane_id_backtrack and (osc_strong or changes >= 2))
                    or (straight_backtrack and (osc_strong or changes >= 2))
                    or (changes >= 2 and osc_strong)
                    or (changes >= 3 and turn_seen <= 0)
                )
            )
            if true_weaving:
                transition_type = "true_weaving"
            elif normal_turn:
                transition_type = "normal_turn"
            elif single_change:
                transition_type = "single_lane_change"
            elif occ_count >= 2:
                transition_type = "occlusion_artifact"
            else:
                transition_type = "stable_or_uncertain"

            seq_texts.append(">".join(seq))
            transition_types.append(transition_type)
            change_values.append(changes)
            unique_values.append(unique_count)
            lane_id_backtrack_values.append(lane_id_backtrack)
            straight_backtrack_values.append(straight_backtrack)
            turn_values.append(turn_seen)
            turn_route_values.append(int(turn_route))
            single_change_values.append(int(single_change))
            normal_turn_values.append(int(normal_turn))
            occ_values.append(occ_count)
            quality_values.append(max(0.0, min(1.0, 1.0 - occ_count / 3.0)))
            true_current_values.append(int(true_weaving))

        true_memory = (
            pd.Series(true_current_values, index=ordered_idx)
            .rolling(window=recent_window, min_periods=1)
            .max()
            .astype(int)
            .tolist()
        )
        out.loc[ordered_idx, "v24f_lane_id_smooth"] = smooth
        out.loc[ordered_idx, "v24f_lane_sequence_smooth_6w"] = seq_texts
        out.loc[ordered_idx, "v24f_transition_type_6w"] = transition_types
        out.loc[ordered_idx, "v24f_lane_change_count_6w"] = change_values
        out.loc[ordered_idx, "v24f_lane_unique_count_6w"] = unique_values
        out.loc[ordered_idx, "v24f_lane_id_backtrack_6w"] = lane_id_backtrack_values
        out.loc[ordered_idx, "v24f_straight_family_backtrack_6w"] = straight_backtrack_values
        out.loc[ordered_idx, "v24f_contains_turn_lane_6w"] = turn_values
        out.loc[ordered_idx, "v24f_turn_route_like"] = turn_route_values
        out.loc[ordered_idx, "v24f_single_lane_change_like"] = single_change_values
        out.loc[ordered_idx, "v24f_normal_turn_like"] = normal_turn_values
        out.loc[ordered_idx, "v24f_occlusion_jitter_count_6w"] = occ_values
        out.loc[ordered_idx, "v24f_true_weaving_current"] = true_current_values
        out.loc[ordered_idx, "v24f_true_weaving_memory"] = true_memory
        out.loc[ordered_idx, "v24f_context_quality"] = quality_values
    return out


def build_base(
    *,
    c4o_feature_csv: Path,
    trajectory_sidecar_csv: Path,
    single_prediction_dir: Path,
) -> pd.DataFrame:
    preds = load_predictions(single_prediction_dir)
    labels = normalize_ids(read_csv(c4o_feature_csv))
    label_keep = KEY_COLUMNS + [
        "split",
        "y_schemaC",
        "redlight_event_score",
        "overspeed_event_score",
        "speed_kmh_p95",
        "s33_current_local_semantic_support",
        "s33_prefix_state_confirmed",
    ]
    labels = labels[[c for c in label_keep if c in labels.columns]].copy()
    sidecar = normalize_ids(read_csv(trajectory_sidecar_csv))
    side_keep = KEY_COLUMNS + [
        "start_sec",
        "end_sec",
        "trajectory_subtype_primary",
        "trajectory_subtype_peak",
        "lane_family_switch_count",
        "opposite_lane_run_length",
        "opposite_lane_occupancy_ratio",
        "reversed_flag",
        "artifact_bbox_area_cv",
        "artifact_max_center_step",
        "osc_h20_score",
        "osc_h30_score",
        "osc_h20_lateral_sign_change_count",
        "osc_h30_lateral_sign_change_count",
        "osc_h20_two_sided_ratio",
        "osc_h30_two_sided_ratio",
    ]
    sidecar = sidecar[[c for c in side_keep if c in sidecar.columns]].copy()

    base = preds.merge(labels.drop(columns=["split"], errors="ignore"), on=KEY_COLUMNS, how="left", suffixes=("", "_label"))
    if "y_schemaC_label" in base.columns:
        base["y_schemaC"] = base["y_schemaC_label"].fillna(base["y_schemaC"])
        base = base.drop(columns=["y_schemaC_label"])
    base = base.merge(sidecar, on=KEY_COLUMNS, how="left", suffixes=("", "_traj"))
    if "start_sec_traj" in base.columns:
        base["start_sec"] = base["start_sec"].fillna(base["start_sec_traj"])
        base["end_sec"] = base["end_sec"].fillna(base["end_sec_traj"])
        base = base.drop(columns=["start_sec_traj", "end_sec_traj"], errors="ignore")
    return base


def build_clean_v24f_trajectory_context_sidecar(
    *,
    c4o_feature_csv: Path,
    trajectory_sidecar_csv: Path,
    single_prediction_dir: Path,
    timestamp_dir: Path,
    lane_map: Path,
) -> pd.DataFrame:
    base = build_base(
        c4o_feature_csv=c4o_feature_csv,
        trajectory_sidecar_csv=trajectory_sidecar_csv,
        single_prediction_dir=single_prediction_dir,
    )
    lanes = load_lanes(lane_map)
    lane_sidecar = build_window_lane_sidecar(base[KEY_COLUMNS + ["start_sec", "end_sec"]], timestamp_dir, lanes)
    merged = base.merge(lane_sidecar, on=KEY_COLUMNS, how="left")
    merged = add_context_features(
        merged.sort_values(["split", "case_key", "ts_window_idx"], kind="mergesort").reset_index(drop=True)
    )
    for col in V24F_SIDECAR_COLUMNS:
        if col not in merged.columns:
            merged[col] = "" if col in V24F_TEXT_COLUMNS else 0.0
    return merged[V24F_SIDECAR_COLUMNS].copy()
