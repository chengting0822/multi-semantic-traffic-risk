#!/usr/bin/env python3
"""Clean v41 lane-family sidecar generation.

This module distills only the accepted v41 sidecar generation step:

    frame-level v28 lane candidate cache + manual lane directions
    -> v41 lane-family / reverse-motion sidecar columns

It does not apply any prediction rule.  The downstream risk2 floor / guard
logic lives in clean_v41_lane_family_policy_v92.py.
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
class CleanV41SidecarColumns:
    raw_count: str = "v92_v41_raw_count"
    motion_dx: str = "v92_v41_motion_dx"
    motion_dy: str = "v92_v41_motion_dy"
    motion_mag_px: str = "v92_v41_motion_mag_px"
    family_top: str = "v92_v41_family_top"
    family_second: str = "v92_v41_family_second"
    family_top_score: str = "v92_v41_family_top_score"
    family_second_score: str = "v92_v41_family_second_score"
    family_top_ratio: str = "v92_v41_family_top_ratio"
    family_confidence: str = "v92_v41_family_confidence"
    family_motion_cos_top: str = "v92_v41_family_motion_cos_top"
    family_hit_ratio: str = "v92_v41_family_hit_ratio"
    bbox_area_cv: str = "v92_v41_bbox_area_cv"
    turn_lane_score_ratio: str = "v92_v41_turn_lane_score_ratio"
    top_lane_is_turn: str = "v92_v41_top_lane_is_turn"
    family_wrongway_preconfirm: str = "v92_v41_family_wrongway_preconfirm"
    family_wrongway_preconfirm_strict: str = "v92_v41_family_wrongway_preconfirm_strict"
    family_wrongway_tail_preconfirm: str = "v92_v41_family_wrongway_tail_preconfirm"
    tail_raw_count: str = "v92_v41_tail_raw_count"
    tail_motion_mag_px: str = "v92_v41_tail_motion_mag_px"
    tail_family_top: str = "v92_v41_tail_family_top"
    tail_family_second: str = "v92_v41_tail_family_second"
    tail_family_top_ratio: str = "v92_v41_tail_family_top_ratio"
    tail_family_confidence: str = "v92_v41_tail_family_confidence"
    tail_family_motion_cos_top: str = "v92_v41_tail_family_motion_cos_top"
    tail_family_hit_ratio: str = "v92_v41_tail_family_hit_ratio"
    tail_bbox_area_cv: str = "v92_v41_tail_bbox_area_cv"
    tail_lane_rank_top4: str = "v92_v41_tail_lane_rank_top4"
    lane_rank_top4: str = "v92_v41_lane_rank_top4"


COLS = CleanV41SidecarColumns()


HISTORICAL_V41_SIDECAR_COLUMNS = {
    COLS.raw_count: "v41_raw_count",
    COLS.motion_dx: "v41_motion_dx",
    COLS.motion_dy: "v41_motion_dy",
    COLS.motion_mag_px: "v41_motion_mag_px",
    COLS.family_top: "v41_family_top",
    COLS.family_second: "v41_family_second",
    COLS.family_top_score: "v41_family_top_score",
    COLS.family_second_score: "v41_family_second_score",
    COLS.family_top_ratio: "v41_family_top_ratio",
    COLS.family_confidence: "v41_family_confidence",
    COLS.family_motion_cos_top: "v41_family_motion_cos_top",
    COLS.family_hit_ratio: "v41_family_hit_ratio",
    COLS.bbox_area_cv: "v41_bbox_area_cv",
    COLS.turn_lane_score_ratio: "v41_turn_lane_score_ratio",
    COLS.top_lane_is_turn: "v41_top_lane_is_turn",
    COLS.family_wrongway_preconfirm: "v41_family_wrongway_preconfirm",
    COLS.family_wrongway_preconfirm_strict: "v41_family_wrongway_preconfirm_strict",
    COLS.family_wrongway_tail_preconfirm: "v41_family_wrongway_tail_preconfirm",
    COLS.tail_raw_count: "v41_tail_raw_count",
    COLS.tail_motion_mag_px: "v41_tail_motion_mag_px",
    COLS.tail_family_top: "v41_tail_family_top",
    COLS.tail_family_second: "v41_tail_family_second",
    COLS.tail_family_top_ratio: "v41_tail_family_top_ratio",
    COLS.tail_family_confidence: "v41_tail_family_confidence",
    COLS.tail_family_motion_cos_top: "v41_tail_family_motion_cos_top",
    COLS.tail_family_hit_ratio: "v41_tail_family_hit_ratio",
    COLS.tail_bbox_area_cv: "v41_tail_bbox_area_cv",
    COLS.tail_lane_rank_top4: "v41_tail_lane_rank_top4",
    COLS.lane_rank_top4: "v41_lane_rank_top4",
}

TEXT_CLEAN_COLUMNS = {
    COLS.family_top,
    COLS.family_second,
    COLS.tail_family_top,
    COLS.tail_family_second,
    COLS.tail_lane_rank_top4,
    COLS.lane_rank_top4,
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
        out["case_key"] = out["case_key"].astype(str).str.strip()
    for col in ["video_id", "track_id"]:
        if col in out.columns:
            out[col] = out[col].map(norm_id)
    if "ts_window_idx" in out.columns:
        out["ts_window_idx"] = pd.to_numeric(out["ts_window_idx"], errors="coerce").fillna(-1).astype(int)
    return out


def lane_family(lane_id: str) -> str:
    lane_id = str(lane_id or "")
    if lane_id.startswith("upbound"):
        return "upbound"
    if lane_id.startswith("downbound"):
        return "downbound"
    if lane_id.startswith("new_lane"):
        return "connector"
    return ""


def is_turn_lane(lane_id: str) -> bool:
    text = str(lane_id or "").lower()
    return "turn" in text or text.endswith("_left") or text.endswith("_right")


def cosine(a: tuple[float, float], b: tuple[float, float]) -> float:
    ax, ay = a
    bx, by = b
    am = math.hypot(ax, ay)
    bm = math.hypot(bx, by)
    if am < 1e-6 or bm < 1e-6:
        return 0.0
    return float((ax * bx + ay * by) / (am * bm))


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


def load_lane_dirs(path: Path) -> dict[str, tuple[float, float]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(lane["lane_id"]): (float(lane["direction"][0]), float(lane["direction"][1]))
        for lane in raw.get("lanes", [])
        if "lane_id" in lane and "direction" in lane
    }


def summarize_lane_family(part: pd.DataFrame, lane_dirs: dict[str, tuple[float, float]]) -> dict[str, object]:
    mdx, mdy, mag = robust_motion(part)
    motion = (mdx, mdy)

    lane_scores: dict[str, float] = {}
    family_scores: dict[str, float] = {"upbound": 0.0, "downbound": 0.0, "connector": 0.0}
    family_weighted_cos_sum: dict[str, float] = {"upbound": 0.0, "downbound": 0.0, "connector": 0.0}
    frame_family_hits: dict[str, int] = {"upbound": 0, "downbound": 0, "connector": 0}

    for rr in part.itertuples(index=False):
        try:
            scores = json.loads(str(getattr(rr, "cand_scores_json", "") or "{}"))
        except json.JSONDecodeError:
            scores = {}
        frame_hit_families: set[str] = set()
        for lane_id, score_raw in scores.items():
            lane_id = str(lane_id)
            score = float(score_raw)
            fam = lane_family(lane_id)
            if not fam or score <= 0:
                continue
            lane_scores[lane_id] = lane_scores.get(lane_id, 0.0) + score
            family_scores[fam] = family_scores.get(fam, 0.0) + score
            c = cosine(motion, lane_dirs.get(lane_id, (0.0, 0.0)))
            family_weighted_cos_sum[fam] = family_weighted_cos_sum.get(fam, 0.0) + score * c
            frame_hit_families.add(fam)
        for fam in frame_hit_families:
            frame_family_hits[fam] = frame_family_hits.get(fam, 0) + 1

    total_family = float(sum(family_scores.values()))
    ranked_family = sorted(family_scores.items(), key=lambda kv: (-kv[1], kv[0]))
    top_family = ranked_family[0][0] if ranked_family and ranked_family[0][1] > 0 else ""
    second_family = ranked_family[1][0] if len(ranked_family) >= 2 and ranked_family[1][1] > 0 else ""
    top_score = float(ranked_family[0][1]) if ranked_family else 0.0
    second_score = float(ranked_family[1][1]) if len(ranked_family) >= 2 else 0.0
    top_ratio = top_score / total_family if total_family > 0 else 0.0
    family_conf = (top_score - second_score) / total_family if total_family > 0 else 0.0
    top_cos = family_weighted_cos_sum.get(top_family, 0.0) / top_score if top_score > 0 else 0.0
    hit_ratio = frame_family_hits.get(top_family, 0) / max(len(part), 1) if top_family else 0.0

    area_cv = coeff_var(part["area"].to_numpy(dtype=float)) if "area" in part.columns and len(part) else 0.0

    lane_rank = sorted(lane_scores.items(), key=lambda kv: (-kv[1], kv[0]))[:4]
    total_lane_score = float(sum(lane_scores.values()))
    turn_lane_score = float(sum(score for lane_id, score in lane_scores.items() if is_turn_lane(lane_id)))
    turn_lane_ratio = turn_lane_score / total_lane_score if total_lane_score > 0 else 0.0
    top_lane = lane_rank[0][0] if lane_rank else ""
    return {
        "raw_count": int(len(part)),
        "motion_dx": mdx,
        "motion_dy": mdy,
        "motion_mag_px": mag,
        "family_top": top_family,
        "family_second": second_family,
        "family_top_score": top_score,
        "family_second_score": second_score,
        "family_top_ratio": top_ratio,
        "family_confidence": family_conf,
        "family_motion_cos_top": top_cos,
        "family_hit_ratio": hit_ratio,
        "bbox_area_cv": area_cv,
        "turn_lane_score_ratio": turn_lane_ratio,
        "top_lane_is_turn": int(is_turn_lane(top_lane)),
        "lane_rank_top4": ";".join(f"{k}:{v:.1f}" for k, v in lane_rank),
    }


def summarize_window(row, part: pd.DataFrame, lane_dirs: dict[str, tuple[float, float]]) -> dict[str, object]:
    full = summarize_lane_family(part, lane_dirs)
    tail = part.iloc[len(part) // 2 :] if len(part) else part
    tail_summary = summarize_lane_family(tail, lane_dirs)

    preconfirm = int(
        full["family_top"] in {"upbound", "downbound"}
        and full["raw_count"] >= 6
        and full["motion_mag_px"] >= 12.0
        and full["family_top_ratio"] >= 0.42
        and full["family_confidence"] >= 0.10
        and full["family_hit_ratio"] >= 0.45
        and full["family_motion_cos_top"] <= -0.38
        and full["bbox_area_cv"] <= 0.75
    )
    strict_preconfirm = int(
        full["family_top"] in {"upbound", "downbound"}
        and full["raw_count"] >= 6
        and full["motion_mag_px"] >= 16.0
        and full["family_top_ratio"] >= 0.60
        and full["family_confidence"] >= 0.28
        and full["family_hit_ratio"] >= 0.45
        and full["family_motion_cos_top"] <= -0.55
        and full["bbox_area_cv"] <= 0.65
    )
    tail_preconfirm = int(
        tail_summary["family_top"] in {"upbound", "downbound"}
        and tail_summary["raw_count"] >= 6
        and tail_summary["motion_mag_px"] >= 14.0
        and tail_summary["family_top_ratio"] >= 0.68
        and tail_summary["family_confidence"] >= 0.35
        and tail_summary["family_hit_ratio"] >= 0.60
        and tail_summary["family_motion_cos_top"] <= -0.45
        and tail_summary["bbox_area_cv"] <= 0.65
    )

    return {
        "case_key": str(row.case_key),
        "video_id": norm_id(row.video_id),
        "track_id": norm_id(row.track_id),
        "ts_window_idx": int(row.ts_window_idx),
        COLS.raw_count: full["raw_count"],
        COLS.motion_dx: full["motion_dx"],
        COLS.motion_dy: full["motion_dy"],
        COLS.motion_mag_px: full["motion_mag_px"],
        COLS.family_top: full["family_top"],
        COLS.family_second: full["family_second"],
        COLS.family_top_score: full["family_top_score"],
        COLS.family_second_score: full["family_second_score"],
        COLS.family_top_ratio: full["family_top_ratio"],
        COLS.family_confidence: full["family_confidence"],
        COLS.family_motion_cos_top: full["family_motion_cos_top"],
        COLS.family_hit_ratio: full["family_hit_ratio"],
        COLS.bbox_area_cv: full["bbox_area_cv"],
        COLS.turn_lane_score_ratio: full["turn_lane_score_ratio"],
        COLS.top_lane_is_turn: full["top_lane_is_turn"],
        COLS.family_wrongway_preconfirm: preconfirm,
        COLS.family_wrongway_preconfirm_strict: strict_preconfirm,
        COLS.family_wrongway_tail_preconfirm: tail_preconfirm,
        COLS.tail_raw_count: tail_summary["raw_count"],
        COLS.tail_motion_mag_px: tail_summary["motion_mag_px"],
        COLS.tail_family_top: tail_summary["family_top"],
        COLS.tail_family_second: tail_summary["family_second"],
        COLS.tail_family_top_ratio: tail_summary["family_top_ratio"],
        COLS.tail_family_confidence: tail_summary["family_confidence"],
        COLS.tail_family_motion_cos_top: tail_summary["family_motion_cos_top"],
        COLS.tail_family_hit_ratio: tail_summary["family_hit_ratio"],
        COLS.tail_bbox_area_cv: tail_summary["bbox_area_cv"],
        COLS.tail_lane_rank_top4: tail_summary["lane_rank_top4"],
        COLS.lane_rank_top4: full["lane_rank_top4"],
    }


def build_clean_v41_lane_family_sidecar(
    windows: pd.DataFrame,
    *,
    cache_dir: Path,
    lane_map: Path,
) -> pd.DataFrame:
    lane_dirs = load_lane_dirs(lane_map)
    start_col = "start_sec_feature" if "start_sec_feature" in windows.columns else "start_sec"
    end_col = "end_sec_feature" if "end_sec_feature" in windows.columns else "end_sec"
    base = normalize_keys(windows[KEY_COLUMNS + [start_col, end_col]].drop_duplicates().copy())
    base = base.rename(columns={start_col: "start_sec", end_col: "end_sec"})
    base["start_sec"] = pd.to_numeric(base["start_sec"], errors="coerce")
    base["end_sec"] = pd.to_numeric(base["end_sec"], errors="coerce")

    rows: list[dict[str, object]] = []
    for video_id, video_windows in base.groupby("video_id", sort=True):
        cache_path = cache_dir / f"{video_id}.parquet"
        if not cache_path.exists():
            for row in video_windows.itertuples(index=False):
                rows.append(
                    {
                        "case_key": str(row.case_key),
                        "video_id": norm_id(row.video_id),
                        "track_id": norm_id(row.track_id),
                        "ts_window_idx": int(row.ts_window_idx),
                        COLS.raw_count: 0,
                        COLS.family_wrongway_preconfirm: 0,
                        COLS.family_wrongway_preconfirm_strict: 0,
                        COLS.family_wrongway_tail_preconfirm: 0,
                        COLS.lane_rank_top4: "missing_cache",
                    }
                )
            continue
        raw = pd.read_parquet(cache_path)
        raw["track_id"] = raw["track_id"].map(norm_id)
        for col in ["timestamp_sec", "bcx", "bcy", "area"]:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
        for track_id, track_windows in video_windows.groupby("track_id", sort=True):
            tr = raw[raw["track_id"].astype(str).eq(str(track_id))].sort_values(
                "timestamp_sec", kind="mergesort"
            )
            times = tr["timestamp_sec"].to_numpy(dtype=float)
            for row in track_windows.sort_values("ts_window_idx").itertuples(index=False):
                if tr.empty:
                    part = tr
                else:
                    left = int(np.searchsorted(times, float(row.start_sec), side="left"))
                    right = int(np.searchsorted(times, float(row.end_sec), side="right"))
                    part = tr.iloc[left:right]
                rows.append(summarize_window(row, part, lane_dirs))
    return pd.DataFrame(rows)


def add_clean_v41_lane_family_sidecar(
    frame: pd.DataFrame,
    *,
    cache_dir: Path,
    lane_map: Path,
) -> pd.DataFrame:
    out = normalize_keys(frame)
    sidecar = build_clean_v41_lane_family_sidecar(out, cache_dir=cache_dir, lane_map=lane_map)
    sidecar = normalize_keys(sidecar)
    if sidecar.duplicated(KEY_COLUMNS).any():
        dup = sidecar.loc[sidecar.duplicated(KEY_COLUMNS, keep=False), KEY_COLUMNS].head()
        raise ValueError(f"duplicated v41 sidecar keys: {dup.to_dict('records')}")
    return out.merge(sidecar, on=KEY_COLUMNS, how="left", validate="many_to_one")


def overwrite_historical_v41_sidecar_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for clean_col, historical_col in HISTORICAL_V41_SIDECAR_COLUMNS.items():
        if clean_col in out.columns:
            out[historical_col] = out[clean_col]
    return out
