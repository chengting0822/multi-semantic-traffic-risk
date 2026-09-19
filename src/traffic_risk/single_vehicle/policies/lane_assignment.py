#!/usr/bin/env python3
"""Clean v28 motion-aware lane sidecar generation.

This module distills the accepted v28 upstream generation step:

    C4O window source + v28 frame lane candidate cache + manual lane map
    -> v28 motion-aware lane sidecar columns

It does not apply any risk rule.  Downstream v41 / v47 / v48 consume these
generated aliases.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


KEY_COLUMNS = ["case_key", "video_id", "track_id", "ts_window_idx"]


@dataclass(frozen=True)
class CleanV28LaneSidecarColumns:
    raw_count: str = "v92_v28_raw_count"
    lane_reason: str = "v92_v28_lane_reason"
    motion_dx: str = "v92_v28_motion_dx"
    motion_dy: str = "v92_v28_motion_dy"
    motion_mag_px: str = "v92_v28_motion_mag_px"
    lane_id_raw: str = "v92_v28_lane_id_raw"
    lane_id_second: str = "v92_v28_lane_id_second"
    lane_score_top: str = "v92_v28_lane_score_top"
    lane_score_second: str = "v92_v28_lane_score_second"
    lane_confidence_raw: str = "v92_v28_lane_confidence_raw"
    lane_valid_ratio: str = "v92_v28_lane_valid_ratio"
    lane_ambiguous_ratio: str = "v92_v28_lane_ambiguous_ratio"
    lane_motion_cos_top: str = "v92_v28_lane_motion_cos_top"
    lane_motion_cos_second: str = "v92_v28_lane_motion_cos_second"
    lane_opposite_penalty_top: str = "v92_v28_lane_opposite_penalty_top"
    lane_ambiguous: str = "v92_v28_lane_ambiguous"
    bbox_area_cv_raw: str = "v92_v28_bbox_area_cv_raw"
    bbox_area_log_jump_max: str = "v92_v28_bbox_area_log_jump_max"
    conf_min_raw: str = "v92_v28_conf_min_raw"
    conf_mean_raw: str = "v92_v28_conf_mean_raw"
    lane_id_smooth: str = "v92_v28_lane_id_smooth"
    lane_sequence_smooth_6w: str = "v92_v28_lane_sequence_smooth_6w"
    transition_type_6w: str = "v92_v28_transition_type_6w"
    lane_change_count_6w: str = "v92_v28_lane_change_count_6w"
    lane_unique_count_6w: str = "v92_v28_lane_unique_count_6w"
    lane_id_backtrack_6w: str = "v92_v28_lane_id_backtrack_6w"
    straight_family_backtrack_6w: str = "v92_v28_straight_family_backtrack_6w"
    connector_expected_route_6w: str = "v92_v28_connector_expected_route_6w"
    connector_adjacent_route_6w: str = "v92_v28_connector_adjacent_route_6w"
    route_deviation_context: str = "v92_v28_route_deviation_context"
    true_weaving_current: str = "v92_v28_true_weaving_current"
    true_weaving_memory: str = "v92_v28_true_weaving_memory"
    context_quality: str = "v92_v28_context_quality"


COLS = CleanV28LaneSidecarColumns()


HISTORICAL_V28_SIDECAR_COLUMNS = {
    COLS.raw_count: "v28_raw_count",
    COLS.lane_reason: "v28_lane_reason",
    COLS.motion_dx: "v28_motion_dx",
    COLS.motion_dy: "v28_motion_dy",
    COLS.motion_mag_px: "v28_motion_mag_px",
    COLS.lane_id_raw: "v28_lane_id_raw",
    COLS.lane_id_second: "v28_lane_id_second",
    COLS.lane_score_top: "v28_lane_score_top",
    COLS.lane_score_second: "v28_lane_score_second",
    COLS.lane_confidence_raw: "v28_lane_confidence_raw",
    COLS.lane_valid_ratio: "v28_lane_valid_ratio",
    COLS.lane_ambiguous_ratio: "v28_lane_ambiguous_ratio",
    COLS.lane_motion_cos_top: "v28_lane_motion_cos_top",
    COLS.lane_motion_cos_second: "v28_lane_motion_cos_second",
    COLS.lane_opposite_penalty_top: "v28_lane_opposite_penalty_top",
    COLS.lane_ambiguous: "v28_lane_ambiguous",
    COLS.bbox_area_cv_raw: "v28_bbox_area_cv_raw",
    COLS.bbox_area_log_jump_max: "v28_bbox_area_log_jump_max",
    COLS.conf_min_raw: "v28_conf_min_raw",
    COLS.conf_mean_raw: "v28_conf_mean_raw",
    COLS.lane_id_smooth: "v28_lane_id_smooth",
    COLS.lane_sequence_smooth_6w: "v28_lane_sequence_smooth_6w",
    COLS.transition_type_6w: "v28_transition_type_6w",
    COLS.lane_change_count_6w: "v28_lane_change_count_6w",
    COLS.lane_unique_count_6w: "v28_lane_unique_count_6w",
    COLS.lane_id_backtrack_6w: "v28_lane_id_backtrack_6w",
    COLS.straight_family_backtrack_6w: "v28_straight_family_backtrack_6w",
    COLS.connector_expected_route_6w: "v28_connector_expected_route_6w",
    COLS.connector_adjacent_route_6w: "v28_connector_adjacent_route_6w",
    COLS.route_deviation_context: "v28_route_deviation_context",
    COLS.true_weaving_current: "v28_true_weaving_current",
    COLS.true_weaving_memory: "v28_true_weaving_memory",
    COLS.context_quality: "v28_context_quality",
}

TEXT_CLEAN_COLUMNS = {
    COLS.lane_reason,
    COLS.lane_id_raw,
    COLS.lane_id_second,
    COLS.lane_id_smooth,
    COLS.lane_sequence_smooth_6w,
    COLS.transition_type_6w,
}


def norm_id(value) -> str:
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
        out["case_key"] = out["case_key"].map(norm_id)
    for col in ["video_id", "track_id"]:
        if col in out.columns:
            out[col] = out[col].map(norm_id)
    if "ts_window_idx" in out.columns:
        out["ts_window_idx"] = pd.to_numeric(out["ts_window_idx"], errors="coerce").fillna(-1).astype(int)
    return out


def num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default).astype(float)


def load_lane_dirs(path: Path) -> dict[str, tuple[float, float]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(lane["lane_id"]): (float(lane["direction"][0]), float(lane["direction"][1]))
        for lane in raw.get("lanes", [])
        if "lane_id" in lane and "direction" in lane
    }


def cosine(a: tuple[float, float], b: tuple[float, float]) -> float:
    ax, ay = a
    bx, by = b
    am = math.hypot(ax, ay)
    bm = math.hypot(bx, by)
    if am < 1e-6 or bm < 1e-6:
        return 0.0
    return float((ax * bx + ay * by) / (am * bm))


def direction_multiplier(cos_value: float, motion_mag: float) -> float:
    if motion_mag < 4.0:
        return 1.0
    if cos_value <= -0.65:
        return 0.08
    if cos_value <= -0.35:
        return 0.18
    if cos_value <= -0.10:
        return 0.45
    if cos_value >= 0.70:
        return 1.25
    if cos_value >= 0.35:
        return 1.12
    return 1.0


def robust_motion(part: pd.DataFrame) -> tuple[float, float, float]:
    if len(part) <= 1:
        return 0.0, 0.0, 0.0
    n = max(1, int(round(len(part) * 0.30)))
    head = part.iloc[:n]
    tail = part.iloc[-n:]
    dx = float(np.nanmedian(tail["bcx"]) - np.nanmedian(head["bcx"]))
    dy = float(np.nanmedian(tail["bcy"]) - np.nanmedian(head["bcy"]))
    mag = float(math.hypot(dx, dy))
    if mag < 1e-6:
        return 0.0, 0.0, 0.0
    return float(dx / mag), float(dy / mag), mag


def coeff_var(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    mean = float(np.mean(arr))
    if abs(mean) < 1e-6:
        return 0.0
    return float(np.std(arr) / abs(mean))


def max_abs_log_jump(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 1e-6)]
    if arr.size <= 1:
        return 0.0
    return float(np.max(np.abs(np.diff(np.log(arr)))))


def empty_row(row, reason: str) -> dict[str, object]:
    return {
        "case_key": str(row.case_key),
        "video_id": str(row.video_id),
        "track_id": str(row.track_id),
        "ts_window_idx": int(row.ts_window_idx),
        COLS.raw_count: 0,
        COLS.lane_reason: reason,
        COLS.motion_dx: 0.0,
        COLS.motion_dy: 0.0,
        COLS.motion_mag_px: 0.0,
        COLS.lane_id_raw: "",
        COLS.lane_id_second: "",
        COLS.lane_score_top: 0.0,
        COLS.lane_score_second: 0.0,
        COLS.lane_confidence_raw: 0.0,
        COLS.lane_valid_ratio: 0.0,
        COLS.lane_ambiguous_ratio: 0.0,
        COLS.lane_motion_cos_top: 0.0,
        COLS.lane_motion_cos_second: 0.0,
        COLS.lane_opposite_penalty_top: 0,
        COLS.lane_ambiguous: 1,
        COLS.bbox_area_cv_raw: 0.0,
        COLS.bbox_area_log_jump_max: 0.0,
        COLS.conf_min_raw: 1.0,
        COLS.conf_mean_raw: 1.0,
    }


def compute_window_row_from_cache(row, part: pd.DataFrame, lane_dirs: dict[str, tuple[float, float]]) -> dict[str, object]:
    out = empty_row(row, "ok")
    out[COLS.raw_count] = int(len(part))
    if part.empty:
        out[COLS.lane_reason] = "empty_window_rows"
        return out

    mdx, mdy, mmag = robust_motion(part)
    motion = (mdx, mdy)
    adjusted_scores: dict[str, float] = {}
    lane_cos: dict[str, float] = {}
    valid_frames = 0
    ambiguous_frames = 0

    for rr in part.itertuples(index=False):
        frame_scores: dict[str, float] = {}
        cand_json = str(getattr(rr, "cand_scores_json", "") or "")
        if cand_json:
            try:
                frame_scores = {str(k): float(v) for k, v in json.loads(cand_json).items() if float(v) > 0}
            except json.JSONDecodeError:
                frame_scores = {}
        if not frame_scores:
            for lane_id_col, score_col in [
                ("cand_top1", "cand_top1_score"),
                ("cand_top2", "cand_top2_score"),
                ("cand_top3", "cand_top3_score"),
            ]:
                lane_id = str(getattr(rr, lane_id_col, "") or "")
                score = float(getattr(rr, score_col, 0.0) or 0.0)
                if lane_id and score > 0:
                    frame_scores[lane_id] = score
        if not frame_scores:
            continue
        valid_frames += 1
        if len(frame_scores) >= 2:
            ambiguous_frames += 1
        for lane_id, score in frame_scores.items():
            c = cosine(motion, lane_dirs.get(lane_id, (0.0, 0.0)))
            lane_cos[lane_id] = c
            adjusted_scores[lane_id] = adjusted_scores.get(lane_id, 0.0) + score * direction_multiplier(c, mmag)

    ranked = sorted(adjusted_scores.items(), key=lambda kv: (-kv[1], kv[0]))
    top_lane = ranked[0][0] if ranked else ""
    second_lane = ranked[1][0] if len(ranked) >= 2 else ""
    top_score = float(ranked[0][1]) if ranked else 0.0
    second_score = float(ranked[1][1]) if len(ranked) >= 2 else 0.0
    total = float(sum(adjusted_scores.values()))
    raw_conf = max(0.0, min(1.0, (top_score - second_score) / total)) if total > 0 else 0.0
    top_cos = float(lane_cos.get(top_lane, 0.0))
    second_cos = float(lane_cos.get(second_lane, 0.0))
    valid_ratio = valid_frames / max(len(part), 1)
    ambiguous_ratio = ambiguous_frames / max(len(part), 1)
    ambiguous = int(
        (raw_conf < 0.18)
        or (valid_ratio < 0.35)
        or ((second_score > 0) and (second_score / max(top_score, 1e-6) >= 0.82))
    )
    opposite_penalty = int(mmag >= 4.0 and top_cos <= -0.35)
    areas = pd.to_numeric(part["area"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    conf = pd.to_numeric(part["conf"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    out.update(
        {
            COLS.motion_dx: mdx,
            COLS.motion_dy: mdy,
            COLS.motion_mag_px: mmag,
            COLS.lane_id_raw: top_lane,
            COLS.lane_id_second: second_lane,
            COLS.lane_score_top: top_score,
            COLS.lane_score_second: second_score,
            COLS.lane_confidence_raw: raw_conf,
            COLS.lane_valid_ratio: float(valid_ratio),
            COLS.lane_ambiguous_ratio: float(ambiguous_ratio),
            COLS.lane_motion_cos_top: top_cos,
            COLS.lane_motion_cos_second: second_cos,
            COLS.lane_opposite_penalty_top: opposite_penalty,
            COLS.lane_ambiguous: ambiguous,
            COLS.bbox_area_cv_raw: coeff_var(areas),
            COLS.bbox_area_log_jump_max: max_abs_log_jump(areas),
            COLS.conf_min_raw: float(np.nanmin(conf)) if len(conf) else 1.0,
            COLS.conf_mean_raw: float(np.nanmean(conf)) if len(conf) else 1.0,
        }
    )
    return out


def build_window_sidecar_from_cache(windows: pd.DataFrame, *, lane_map: Path, frame_cache_dir: Path) -> pd.DataFrame:
    lane_dirs = load_lane_dirs(lane_map)
    rows: list[dict[str, object]] = []
    windows = normalize_keys(windows[KEY_COLUMNS + ["start_sec", "end_sec"]].drop_duplicates().copy())
    windows["start_sec"] = pd.to_numeric(windows["start_sec"], errors="coerce")
    windows["end_sec"] = pd.to_numeric(windows["end_sec"], errors="coerce")
    for video_id, video_windows in windows.groupby("video_id", sort=True):
        cache_path = frame_cache_dir / f"{video_id}.parquet"
        if not cache_path.exists():
            for row in video_windows.itertuples(index=False):
                rows.append(empty_row(row, "missing_frame_cache"))
            continue
        raw = pd.read_parquet(cache_path)
        raw["track_id"] = raw["track_id"].map(norm_id)
        wanted_tracks = set(video_windows["track_id"].astype(str).unique().tolist())
        raw = raw[raw["track_id"].astype(str).isin(wanted_tracks)].copy()
        if raw.empty:
            for row in video_windows.itertuples(index=False):
                rows.append(empty_row(row, "missing_track_rows"))
            continue
        for track_id, track_windows in video_windows.groupby("track_id", sort=True):
            tr = raw[raw["track_id"].astype(str).eq(str(track_id))].sort_values("timestamp_sec", kind="mergesort")
            if tr.empty:
                for row in track_windows.itertuples(index=False):
                    rows.append(empty_row(row, "missing_track_rows"))
                continue
            times = tr["timestamp_sec"].to_numpy(dtype=float)
            for row in track_windows.sort_values("ts_window_idx").itertuples(index=False):
                left = int(np.searchsorted(times, float(row.start_sec), side="left"))
                right = int(np.searchsorted(times, float(row.end_sec), side="right"))
                rows.append(compute_window_row_from_cache(row, tr.iloc[left:right], lane_dirs))
    return pd.DataFrame(rows)


def compress_sequence(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if not text:
            continue
        if not out or out[-1] != text:
            out.append(text)
    return out


def has_backtrack(seq: list[str]) -> bool:
    return any(seq[i] == seq[i + 2] and seq[i] != seq[i + 1] for i in range(len(seq) - 2))


def lane_group(lane_id: str) -> str:
    lane_id = str(lane_id or "")
    if lane_id.startswith("downbound"):
        return "downbound"
    if lane_id.startswith("upbound"):
        return "upbound"
    if lane_id.startswith("new_lane"):
        return "new_lane"
    return lane_id.split("_")[0] if lane_id else ""


def is_turn_lane(lane_id: str) -> bool:
    text = str(lane_id or "").lower()
    return "turn" in text or text.endswith("_left") or text.endswith("_right")


def lane_family_sequence(seq: list[str], *, drop_turn: bool = False) -> list[str]:
    values: list[str] = []
    for lane_id in seq:
        if not lane_id:
            continue
        if drop_turn and is_turn_lane(lane_id):
            continue
        values.append(lane_group(lane_id))
    return compress_sequence(values)


def causal_smooth(part: pd.DataFrame) -> list[str]:
    raw = part[COLS.lane_id_raw].astype(str).tolist()
    conf = num(part, COLS.lane_confidence_raw).tolist()
    valid = num(part, COLS.lane_valid_ratio).tolist()
    ambiguous = num(part, COLS.lane_ambiguous).tolist()
    opposite = num(part, COLS.lane_opposite_penalty_top).tolist()
    motion_mag = num(part, COLS.motion_mag_px).tolist()
    smoothed: list[str] = []
    active = ""
    pending = ""
    pending_count = 0
    for lane_id, c, v, amb, opp, mag in zip(raw, conf, valid, ambiguous, opposite, motion_mag):
        usable = bool(lane_id) and v >= 0.35 and opp < 0.5
        high_quality = usable and amb < 0.5 and c >= 0.28
        if not usable:
            smoothed.append(active)
            pending = ""
            pending_count = 0
            continue
        if not active:
            active = lane_id if high_quality or c >= 0.45 else ""
            smoothed.append(active)
            continue
        if lane_id == active:
            pending = ""
            pending_count = 0
            smoothed.append(active)
            continue
        if c >= 0.62 and v >= 0.50 and amb < 0.5:
            active = lane_id
            pending = ""
            pending_count = 0
            smoothed.append(active)
            continue
        if high_quality and mag >= 5.0:
            if lane_id == pending:
                pending_count += 1
            else:
                pending = lane_id
                pending_count = 1
            if pending_count >= 2:
                active = lane_id
                pending = ""
                pending_count = 0
        else:
            pending = ""
            pending_count = 0
        smoothed.append(active)
    return smoothed


def is_expected_connector_route(seq: list[str]) -> bool:
    text = ">".join(seq)
    return (
        "new_lane_8>downbound_2" in text
        or "downbound_left_turn>downbound_2" in text
        or "new_lane_8>downbound_left_turn>downbound_2" in text
    )


def is_adjacent_connector_route(seq: list[str]) -> bool:
    text = ">".join(seq)
    return (
        "new_lane_8>downbound_1" in text
        or "downbound_left_turn>downbound_1" in text
        or "new_lane_8>downbound_left_turn>downbound_1" in text
    )


def add_sequence_context(frame: pd.DataFrame, recent_window: int = 6) -> pd.DataFrame:
    out = frame.copy()
    for col in [COLS.lane_id_smooth, COLS.lane_sequence_smooth_6w, COLS.transition_type_6w]:
        out[col] = ""
    for col in [
        COLS.lane_change_count_6w,
        COLS.lane_unique_count_6w,
        COLS.lane_id_backtrack_6w,
        COLS.straight_family_backtrack_6w,
        COLS.connector_expected_route_6w,
        COLS.connector_adjacent_route_6w,
        COLS.route_deviation_context,
        COLS.true_weaving_current,
        COLS.true_weaving_memory,
        COLS.context_quality,
    ]:
        out[col] = 0.0

    for _, idx in out.groupby(["case_key", "video_id", "track_id"], sort=False).groups.items():
        part = out.loc[list(idx)].sort_values("ts_window_idx", kind="mergesort")
        ordered = part.index.tolist()
        smooth = causal_smooth(part)
        out.loc[ordered, COLS.lane_id_smooth] = smooth
        occ = (
            (num(part, COLS.bbox_area_cv_raw) >= 0.45)
            | (num(part, COLS.bbox_area_log_jump_max) >= 0.55)
            | (num(part, COLS.lane_ambiguous) >= 0.5)
        ).astype(int).tolist()

        seq_texts: list[str] = []
        transition_types: list[str] = []
        true_current: list[int] = []
        change_values: list[int] = []
        unique_values: list[int] = []
        backtrack_values: list[int] = []
        family_backtrack_values: list[int] = []
        expected_values: list[int] = []
        adjacent_values: list[int] = []
        route_dev_values: list[int] = []
        quality_values: list[float] = []

        for pos in range(len(part)):
            left = max(0, pos - recent_window + 1)
            seq = compress_sequence(smooth[left : pos + 1])
            changes = max(len(seq) - 1, 0)
            unique = len(set(seq))
            lane_backtrack = int(has_backtrack(seq))
            families = lane_family_sequence(seq, drop_turn=True)
            family_backtrack = int(has_backtrack(families))
            expected = int(is_expected_connector_route(seq))
            adjacent = int(is_adjacent_connector_route(seq))
            occ_count = int(sum(occ[left : pos + 1]))
            true_weaving = bool(
                occ_count <= 2
                and (
                    (lane_backtrack and changes >= 2)
                    or (family_backtrack and changes >= 2)
                    or (changes >= 3 and not expected and not adjacent)
                )
            )
            route_deviation = bool(adjacent and not true_weaving and not lane_backtrack and changes <= 2)
            if true_weaving:
                tr = "true_weaving"
            elif route_deviation:
                tr = "connector_adjacent_route"
            elif expected:
                tr = "connector_expected_route"
            elif changes <= 1:
                tr = "single_lane_change_or_stable"
            elif occ_count >= 2:
                tr = "ambiguous_or_occlusion"
            else:
                tr = "stable_or_uncertain"

            seq_texts.append(">".join(seq))
            transition_types.append(tr)
            true_current.append(int(true_weaving))
            change_values.append(changes)
            unique_values.append(unique)
            backtrack_values.append(lane_backtrack)
            family_backtrack_values.append(family_backtrack)
            expected_values.append(expected)
            adjacent_values.append(adjacent)
            route_dev_values.append(int(route_deviation))
            quality_values.append(max(0.0, min(1.0, 1.0 - occ_count / 3.0)))

        true_memory = (
            pd.Series(true_current, index=ordered)
            .rolling(window=recent_window, min_periods=1)
            .max()
            .astype(int)
            .tolist()
        )
        out.loc[ordered, COLS.lane_sequence_smooth_6w] = seq_texts
        out.loc[ordered, COLS.transition_type_6w] = transition_types
        out.loc[ordered, COLS.true_weaving_current] = true_current
        out.loc[ordered, COLS.true_weaving_memory] = true_memory
        out.loc[ordered, COLS.lane_change_count_6w] = change_values
        out.loc[ordered, COLS.lane_unique_count_6w] = unique_values
        out.loc[ordered, COLS.lane_id_backtrack_6w] = backtrack_values
        out.loc[ordered, COLS.straight_family_backtrack_6w] = family_backtrack_values
        out.loc[ordered, COLS.connector_expected_route_6w] = expected_values
        out.loc[ordered, COLS.connector_adjacent_route_6w] = adjacent_values
        out.loc[ordered, COLS.route_deviation_context] = route_dev_values
        out.loc[ordered, COLS.context_quality] = quality_values
    return out


def build_clean_v28_lane_sidecar(windows: pd.DataFrame, *, lane_map: Path, frame_cache_dir: Path) -> pd.DataFrame:
    sidecar = build_window_sidecar_from_cache(windows, lane_map=lane_map, frame_cache_dir=frame_cache_dir)
    sidecar = add_sequence_context(sidecar)
    keep = KEY_COLUMNS + list(HISTORICAL_V28_SIDECAR_COLUMNS.keys())
    return sidecar[keep]


def add_clean_v28_lane_sidecar(
    frame: pd.DataFrame,
    *,
    window_source_path: Path,
    lane_map: Path,
    frame_cache_dir: Path,
) -> pd.DataFrame:
    out = normalize_keys(frame)
    if not window_source_path.exists():
        raise FileNotFoundError(window_source_path)
    if not lane_map.exists():
        raise FileNotFoundError(lane_map)
    if not frame_cache_dir.exists():
        raise FileNotFoundError(frame_cache_dir)
    windows = pd.read_csv(window_source_path, low_memory=False)
    sidecar = normalize_keys(build_clean_v28_lane_sidecar(windows, lane_map=lane_map, frame_cache_dir=frame_cache_dir))
    if sidecar.duplicated(KEY_COLUMNS).any():
        dup = sidecar.loc[sidecar.duplicated(KEY_COLUMNS, keep=False), KEY_COLUMNS].head()
        raise ValueError(f"duplicated v28 sidecar keys: {dup.to_dict('records')}")
    return out.merge(sidecar, on=KEY_COLUMNS, how="left", validate="many_to_one")


def add_historical_v28_aliases_for_diff(frame: pd.DataFrame, *, historical_v28_sidecar_path: Path) -> pd.DataFrame:
    out = normalize_keys(frame)
    if not historical_v28_sidecar_path.exists():
        raise FileNotFoundError(historical_v28_sidecar_path)
    missing_aliases = [hist for hist in HISTORICAL_V28_SIDECAR_COLUMNS.values() if hist not in out.columns]
    if not missing_aliases:
        return out
    sidecar = normalize_keys(
        pd.read_csv(historical_v28_sidecar_path, usecols=KEY_COLUMNS + missing_aliases, low_memory=False)
    )
    return out.merge(sidecar, on=KEY_COLUMNS, how="left", validate="many_to_one")


def overwrite_historical_v28_lane_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for clean_col, historical_col in HISTORICAL_V28_SIDECAR_COLUMNS.items():
        if clean_col in out.columns:
            out[historical_col] = out[clean_col]
    return out
