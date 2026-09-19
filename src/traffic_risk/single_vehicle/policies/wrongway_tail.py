#!/usr/bin/env python3
"""Clean v87/v88 wrongway tail-onset semantics.

This module extracts the accepted v87/v88 branches:

- v87: weak but coherent current-window wrongway tail pre-confirm -> risk2
  support.
- v88 refined p30: last-5-frame high-confidence micro-tail wrongway onset
  -> risk2 support.

The rejected v88 low-confidence last-8 branch is intentionally excluded.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


GROUP = ["split", "case_key", "video_id", "track_id"]
WINDOW_KEYS = ["case_key", "video_id", "track_id", "ts_window_idx"]
V88_LAST5_COLUMNS = [
    "v88_last5_n",
    "v88_last5_top_family",
    "v88_last5_top_ratio",
    "v88_last5_confidence",
    "v88_last5_cos_top",
    "v88_last5_motion_mag",
    "v88_last5_lane_counts",
]


@dataclass(frozen=True)
class CleanWrongwayTailOnsetColumns:
    current: str = "v92_v87_v88_wrongway_tail_onset_current"
    reason: str = "v92_v87_v88_wrongway_tail_onset_reason"
    v87_support: str = "v87_wrongway_tail_preconfirm_risk2"
    v88_support: str = "v88_refined_p30_micro_tail_wrongway"


COLS = CleanWrongwayTailOnsetColumns()


def norm_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else text


def num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default).astype(float)


def text(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series("", index=frame.index, dtype=str)
    return frame[col].fillna("").astype(str)


def lane_family(lane_id: str) -> str:
    lane = str(lane_id or "")
    if lane.startswith("upbound"):
        return "upbound"
    if lane.startswith("downbound"):
        return "downbound"
    if lane.startswith("new_lane"):
        return "connector"
    return ""


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


def load_lane_dirs(lane_map: Path) -> dict[str, tuple[float, float]]:
    raw = json.loads(lane_map.read_text(encoding="utf-8"))
    dirs: dict[str, tuple[float, float]] = {}
    for lane in raw.get("lanes", []):
        if "lane_id" not in lane or "direction" not in lane:
            continue
        dirs[str(lane["lane_id"])] = (float(lane["direction"][0]), float(lane["direction"][1]))
    return dirs


def summarize_tail(part: pd.DataFrame, lane_dirs: dict[str, tuple[float, float]]) -> dict[str, Any]:
    if part.empty:
        return {
            "n": 0,
            "top_family": "",
            "top_ratio": 0.0,
            "confidence": 0.0,
            "cos_top": 0.0,
            "motion_mag": 0.0,
            "lane_counts": "",
        }

    mdx, mdy, mag = robust_motion(part)
    motion = (mdx, mdy)
    lane_counts = part["cand_top1"].fillna("").astype(str).value_counts().to_dict()
    family_scores = {"upbound": 0.0, "downbound": 0.0, "connector": 0.0}
    family_cos_sum = {"upbound": 0.0, "downbound": 0.0, "connector": 0.0}

    for row in part.itertuples(index=False):
        try:
            scores = json.loads(str(getattr(row, "cand_scores_json", "") or "{}"))
        except json.JSONDecodeError:
            scores = {}
        for lane_id, raw_score in scores.items():
            lane_id = str(lane_id)
            score = float(raw_score)
            fam = lane_family(lane_id)
            if score <= 0 or not fam:
                continue
            family_scores[fam] = family_scores.get(fam, 0.0) + score
            family_cos_sum[fam] = family_cos_sum.get(fam, 0.0) + score * cosine(
                motion, lane_dirs.get(lane_id, (0.0, 0.0))
            )

    ranked = sorted(family_scores.items(), key=lambda kv: (-kv[1], kv[0]))
    total = float(sum(family_scores.values()))
    top_family = ranked[0][0] if ranked and ranked[0][1] > 0 else ""
    top_score = float(ranked[0][1]) if ranked else 0.0
    second_score = float(ranked[1][1]) if len(ranked) >= 2 else 0.0
    return {
        "n": int(len(part)),
        "top_family": top_family,
        "top_ratio": top_score / total if total > 0 else 0.0,
        "confidence": (top_score - second_score) / total if total > 0 else 0.0,
        "cos_top": family_cos_sum.get(top_family, 0.0) / top_score if top_score > 0 else 0.0,
        "motion_mag": mag,
        "lane_counts": ";".join(f"{k}:{v}" for k, v in sorted(lane_counts.items(), key=lambda kv: -kv[1])[:6]),
    }


def load_frame_cache(cache_dir: Path, video_id: str) -> pd.DataFrame:
    path = cache_dir / f"{video_id}.parquet"
    if not path.exists():
        return pd.DataFrame()
    out = pd.read_parquet(path)
    out["track_id"] = out["track_id"].map(norm_id)
    return out.sort_values(["track_id", "timestamp_sec", "frame"], kind="mergesort")


def add_v88_last5_features(frame: pd.DataFrame, *, cache_dir: Path, lane_map: Path) -> pd.DataFrame:
    out = frame.copy()
    for col in ["split", "case_key", "video_id", "track_id"]:
        if col in out.columns:
            out[col] = out[col].map(norm_id)
    if "ts_window_idx" in out.columns:
        out["ts_window_idx"] = pd.to_numeric(out["ts_window_idx"], errors="coerce").fillna(-1).astype(int)
    if all(col in out.columns for col in V88_LAST5_COLUMNS):
        return out

    lane_dirs = load_lane_dirs(lane_map)
    feature_rows: list[dict[str, Any]] = []
    for video_id, idx in out.groupby("video_id", sort=False).groups.items():
        cache = load_frame_cache(cache_dir, str(video_id))
        by_track = {tid: part.reset_index(drop=True) for tid, part in cache.groupby("track_id", sort=False)} if not cache.empty else {}
        for track_id, t_idx in out.loc[idx].groupby("track_id", sort=False).groups.items():
            track_frames = by_track.get(str(track_id), pd.DataFrame())
            if track_frames.empty:
                feature_rows.extend({"_index": i} for i in t_idx)
                continue
            times = pd.to_numeric(track_frames["timestamp_sec"], errors="coerce").to_numpy(dtype=float)
            track_rows = out.loc[list(t_idx)]
            for i, row in track_rows.iterrows():
                start = float(row["start_sec"])
                end = float(row["end_sec"])
                left = int(np.searchsorted(times, start, side="left"))
                right = int(np.searchsorted(times, end, side="left"))
                win_tail = track_frames.iloc[max(left, right - 5):right]
                summary = summarize_tail(win_tail, lane_dirs)
                rec = {"_index": i}
                for key, value in summary.items():
                    rec[f"v88_last5_{key}"] = value
                feature_rows.append(rec)

    feats = pd.DataFrame(feature_rows).set_index("_index")
    for col in feats.columns:
        out[col] = feats[col]
    return out


def build_v88_last5_sidecar(frame: pd.DataFrame, *, cache_dir: Path, lane_map: Path) -> pd.DataFrame:
    keyed = add_v88_last5_features(frame, cache_dir=cache_dir, lane_map=lane_map)
    keep = [c for c in WINDOW_KEYS if c in keyed.columns] + V88_LAST5_COLUMNS
    return keyed[keep].drop_duplicates(subset=[c for c in WINDOW_KEYS if c in keyed.columns], keep="last")


def merge_reasons(updates: list[tuple[pd.Series, str]]) -> pd.Series:
    reasons = pd.Series("", index=updates[0][0].index, dtype=object)
    for mask, reason in updates:
        mask = mask.fillna(False)
        reasons.loc[mask] = np.where(
            reasons.loc[mask].astype(str).eq(""),
            reason,
            reasons.loc[mask].astype(str) + ";" + reason,
        )
    return reasons


def apply_clean_v87_v88_wrongway_tail_onset_policy(
    frame: pd.DataFrame,
    *,
    base_current_col: str,
    frame_cache_dir: Path,
    lane_map: Path,
    out_current_col: str | None = None,
    out_reason_col: str | None = None,
) -> pd.DataFrame:
    """Compute accepted v87/v88 wrongway tail onset supports."""

    if base_current_col not in frame.columns:
        raise KeyError(f"base current column not found: {base_current_col}")

    out = add_v88_last5_features(frame, cache_dir=frame_cache_dir, lane_map=lane_map)
    out_current_col = out_current_col or COLS.current
    out_reason_col = out_reason_col or COLS.reason
    current = num(out, base_current_col).astype(int).clip(0, 2)
    artifact_reason = (
        text(out, "v65_official_reject_reason")
        + ";"
        + text(out, "v44_sidecar_raw_recompute_rejected_reason")
    )
    wrongway_tail_artifact_guard = artifact_reason.str.contains(
        "stationary_pixel_jitter|oscillatory_micro_motion|small_bbox_ipm_instability|short_track_micro_motion|tiny_motion|tiny_box",
        case=False,
        regex=True,
    )

    v87_support = (
        current.eq(0)
        & num(out, "v41_family_wrongway_preconfirm").ge(1)
        & num(out, "v41_family_wrongway_tail_preconfirm").ge(1)
        & num(out, "v41_tail_family_confidence").ge(0.50)
        & num(out, "v41_tail_family_motion_cos_top", default=1.0).le(-0.70)
        & num(out, "v41_tail_motion_mag_px").ge(25.0)
        & num(out, "prob_class1").ge(0.20)
        & ~wrongway_tail_artifact_guard
    )

    v88_support = (
        current.eq(0)
        & num(out, "v88_last5_n").ge(5)
        & out["v88_last5_top_family"].fillna("").isin(["upbound", "downbound"])
        & num(out, "v88_last5_top_ratio").ge(0.58)
        & num(out, "v88_last5_confidence").ge(0.15)
        & num(out, "v88_last5_cos_top", default=1.0).le(-0.90)
        & num(out, "v88_last5_motion_mag").ge(5.0)
        & num(out, "prob_class1").ge(0.30)
        & ~wrongway_tail_artifact_guard
    )

    updated = current.copy()
    updated.loc[v87_support | v88_support] = np.maximum(updated.loc[v87_support | v88_support], 1)

    out[COLS.v87_support] = v87_support.astype(int)
    out[COLS.v88_support] = v88_support.astype(int)
    out[out_reason_col] = merge_reasons(
        [
            (v87_support, "v87_wrongway_tail_preconfirm"),
            (v88_support, "v88_refined_p30_micro_tail_wrongway"),
        ]
    )
    out[out_current_col] = updated.astype(int)
    return out
