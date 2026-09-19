#!/usr/bin/env python3
"""Clean v61 stable ge110 onset policy.

This module extracts only the accepted v61 branch:
stable severe overspeed onset (>=110 km/h) -> risk3.

It intentionally excludes the rejected v61 redlight/trajectory combo,
pure-speed cap, wrongway seam, and midband seam variants.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


GROUP = ["split", "case_key", "video_id", "track_id"]


@dataclass(frozen=True)
class CleanV61Ge110OnsetColumns:
    current: str = "v92_v61_ge110_current"
    reason: str = "v92_v61_ge110_reason"
    stable_ge110_pre_event: str = "v92_v61_stable_ge110_pre_event"
    severe_candidate: str = "v92_v61_severe_candidate"
    prev_severe_candidate: str = "v92_v61_prev_severe_candidate"
    very_strong_first: str = "v92_v61_very_strong_first"
    existing_guarded: str = "v92_v61_existing_guarded_ge110"
    short_duplicate_guard: str = "v92_v61_short_duplicate_bbox_guard"
    hard_artifact_guard: str = "v92_v61_hard_artifact_guard"


COLS = CleanV61Ge110OnsetColumns()


def num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default).astype(float)


def flag(frame: pd.DataFrame, col: str) -> pd.Series:
    return num(frame, col).gt(0)


def text(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series("", index=frame.index, dtype=str)
    return frame[col].fillna("").astype(str)


def first_positive(frame: pd.DataFrame, cols: list[str], default: float = 0.0) -> pd.Series:
    out = pd.Series(default, index=frame.index, dtype=float)
    for col in cols:
        if col not in frame.columns:
            continue
        values = num(frame, col, default=0.0)
        out = out.where(out.gt(0), values)
    return out


def optional_threshold(
    frame: pd.DataFrame,
    col: str,
    threshold: float,
    *,
    default_when_missing: bool,
) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default_when_missing, index=frame.index, dtype=bool)
    values = num(frame, col)
    # Treat all-zero optional legacy quality columns as missing when the newer
    # fold-local v65 speed-maturity columns are available.
    if values.max() <= 0 and any(c in frame.columns for c in ["v65_raw_ge110_ratio", "v65_frame_count"]):
        return pd.Series(default_when_missing, index=frame.index, dtype=bool)
    return values.ge(threshold)


def prev_series(frame: pd.DataFrame, values: pd.Series, fill: float = 0.0) -> pd.Series:
    present_group = [c for c in GROUP if c in frame.columns]
    tmp = frame[present_group + ["ts_window_idx"]].copy()
    tmp["_v"] = values.to_numpy()
    tmp = tmp.sort_values(present_group + ["ts_window_idx"], kind="mergesort")
    shifted = tmp.groupby(present_group, sort=False)["_v"].shift(1).fillna(fill)
    out = pd.Series(fill, index=frame.index, dtype=shifted.dtype)
    out.loc[tmp.index] = shifted.to_numpy()
    return out


def apply_clean_v61_ge110_onset_policy(
    frame: pd.DataFrame,
    *,
    base_current_col: str = "pred_v60_integrated_current",
    out_current_col: str | None = None,
    out_reason_col: str | None = None,
) -> pd.DataFrame:
    """Apply the accepted v61 stable ge110 onset policy.

    Values use SchemaC:
    0 = risk0+1, 1 = risk2, 2 = risk3.
    """

    if base_current_col not in frame.columns:
        raise KeyError(f"base current column not found: {base_current_col}")

    out = frame.copy()
    out_current_col = out_current_col or COLS.current
    out_reason_col = out_reason_col or COLS.reason

    current = num(out, base_current_col).astype(int).clip(0, 2)

    raw = first_positive(
        out,
        [
            "v44_sidecar_computed_speed_kmh_p95",
            "v65_raw_speed_p95",
            "computed_speed_kmh_p95",
        ],
    )
    smooth = first_positive(
        out,
        [
            "v44_sidecar_speed_smoothed_kmh_p95",
            "v65_smooth_speed_p95",
            "speed_smoothed_kmh_p95",
        ],
    )
    bbox_ok = optional_threshold(
        out,
        "v44_sidecar_computed_bbox_height_px_p95",
        30.0,
        default_when_missing=True,
    )
    path_eff_ok = optional_threshold(
        out,
        "v44_sidecar_computed_path_efficiency",
        0.60,
        default_when_missing=True,
    )
    quality_ok = optional_threshold(
        out,
        "v44_v8_overspeed_signal_quality",
        0.60,
        default_when_missing=True,
    )
    reliability_ok = optional_threshold(
        out,
        "v44_v8_overspeed_reliability",
        0.98,
        default_when_missing=True,
    )
    raw_ge110_ratio = num(out, "v65_raw_ge110_ratio")
    raw_ge110_run = num(out, "v65_raw_ge110_max_consecutive_frames")
    smooth_ge110_ratio = num(out, "v65_smooth_ge110_ratio")
    smooth_ge110_run = num(out, "v65_smooth_ge110_max_consecutive_frames")
    valid_speed_count = num(out, "v65_valid_speed_count")
    reject = text(out, "v44_sidecar_raw_recompute_rejected_reason")
    v65_reject = text(out, "v65_official_reject_reason")
    combined_reject = reject + ";" + v65_reject

    short_duplicate = flag(out, "overspeed_short_duplicate_bbox_guard_v60") | flag(
        out, "v60_overspeed_short_duplicate_bbox_guard"
    )
    hard_artifact = combined_reject.str.contains(
        "stationary_pixel_jitter|oscillatory_micro_motion|small_bbox_ipm_instability|short_track_micro_motion|ultra_short|tiny_motion|tiny_box",
        case=False,
        regex=True,
    )
    short_track_artifact = combined_reject.str.contains(
        "short_track_unstable_jump|short_track_micro_motion",
        case=False,
        regex=True,
    )
    first_window = num(out, "ts_window_idx").eq(0)
    severe_candidate = (
        raw.ge(110.0)
        & smooth.ge(100.0)
        & bbox_ok
        & reliability_ok
        & quality_ok
        & path_eff_ok
        & (
            raw_ge110_ratio.ge(0.45)
            | raw_ge110_run.ge(5)
            | valid_speed_count.eq(0)
        )
        & ~short_duplicate
        & ~hard_artifact
        & ~short_track_artifact
    )
    prev_severe_candidate = prev_series(out, severe_candidate.astype(int)).astype(bool)
    very_strong_first = (
        raw.ge(170.0)
        & smooth.ge(130.0)
        & bbox_ok
        & optional_threshold(out, "v44_v8_overspeed_signal_quality", 0.90, default_when_missing=True)
        & reliability_ok
        & (
            smooth_ge110_ratio.ge(0.20)
            | smooth_ge110_run.ge(5)
            | raw_ge110_run.ge(10)
        )
        & ~first_window
        & ~short_duplicate
        & ~hard_artifact
        & ~short_track_artifact
    )
    existing_guarded = flag(out, "overspeed_guarded_ge110_onset_v60") | flag(
        out, "v60_overspeed_guarded_ge110_onset"
    )

    stable_ge110 = severe_candidate & (existing_guarded | prev_severe_candidate | very_strong_first)

    updated = current.copy()
    updated.loc[stable_ge110] = np.maximum(updated.loc[stable_ge110], 2)

    out[COLS.stable_ge110_pre_event] = stable_ge110.astype(int)
    out[COLS.severe_candidate] = severe_candidate.astype(int)
    out[COLS.prev_severe_candidate] = prev_severe_candidate.astype(int)
    out[COLS.very_strong_first] = very_strong_first.astype(int)
    out[COLS.existing_guarded] = existing_guarded.astype(int)
    out[COLS.short_duplicate_guard] = short_duplicate.astype(int)
    out[COLS.hard_artifact_guard] = hard_artifact.astype(int)
    out[out_reason_col] = np.where(stable_ge110, "v61_stable_ge110_pre_event", "")
    out[out_current_col] = updated.astype(int)
    return out
