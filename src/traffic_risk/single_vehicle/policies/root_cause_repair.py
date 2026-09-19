#!/usr/bin/env python3
"""Clean v62 root-cause repair policy.

This module extracts the accepted v62 main candidate into a reusable DataFrame
policy.  It deliberately excludes the v62 redlight diagnostic branch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CleanV62RootCauseRepairColumns:
    current: str = "v92_v62_root_cause_repair_current"
    reason: str = "v92_v62_root_cause_repair_reason"
    midband_risk2: str = "v92_v62_midband_coherent_risk2_support"
    track_entry_extreme_speed_risk3: str = "v92_v62_track_entry_extreme_speed_pressure_risk3_support"
    strict_wrongway_risk2: str = "v92_v62_strict_wrongway_preconfirm_risk2_support"


COLS = CleanV62RootCauseRepairColumns()


def num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default).astype(float)


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
    if values.max() <= 0 and any(c in frame.columns for c in ["v65_raw_ge110_ratio", "v65_frame_count"]):
        return pd.Series(default_when_missing, index=frame.index, dtype=bool)
    return values.ge(threshold)


def merge_reasons(updates: list[tuple[pd.Series, str]]) -> pd.Series:
    if not updates:
        return pd.Series(dtype=object)
    reasons = pd.Series("", index=updates[0][0].index, dtype=object)
    for mask, reason in updates:
        mask = mask.fillna(False)
        reasons.loc[mask] = np.where(
            reasons.loc[mask].astype(str).eq(""),
            reason,
            reasons.loc[mask].astype(str) + ";" + reason,
        )
    return reasons


def apply_clean_v62_root_cause_repair_policy(
    frame: pd.DataFrame,
    *,
    base_current_col: str = "pred_v61_ge110_current",
    out_current_col: str | None = None,
    out_reason_col: str | None = None,
) -> pd.DataFrame:
    """Apply the accepted v62 root-cause repair policy.

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
            "computed_speed_kmh_p95",
            "v65_raw_speed_p95",
            "v44_sidecar_computed_speed_kmh_p95",
        ],
    )
    smooth = first_positive(
        out,
        [
            "speed_smoothed_kmh_p95",
            "v65_smooth_speed_p95",
            "v44_sidecar_speed_smoothed_kmh_p95",
        ],
    )
    reject = text(out, "v44_sidecar_raw_recompute_rejected_reason")
    v65_reject = text(out, "v65_official_reject_reason")
    combined_reject = reject + ";" + v65_reject
    reject_artifact = combined_reject.str.contains(
        "stationary_pixel_jitter|oscillatory_micro_motion|small_bbox_ipm_instability|short_track_micro_motion|ultra_short|tiny_motion|tiny_box",
        case=False,
        regex=True,
    )
    short_track_artifact = combined_reject.str.contains(
        "short_track_unstable_jump|short_track_micro_motion",
        case=False,
        regex=True,
    )

    bbox55_ok = optional_threshold(
        out,
        "v44_sidecar_computed_bbox_height_px_p95",
        55.0,
        default_when_missing=True,
    )
    bbox45_ok = optional_threshold(
        out,
        "v44_sidecar_computed_bbox_height_px_p95",
        45.0,
        default_when_missing=True,
    )
    bbox28_ok = optional_threshold(
        out,
        "v44_sidecar_computed_bbox_height_px_p95",
        28.0,
        default_when_missing=True,
    )
    path95_ok = optional_threshold(
        out,
        "v44_sidecar_computed_path_efficiency",
        0.95,
        default_when_missing=True,
    )
    path98_ok = optional_threshold(
        out,
        "v44_sidecar_computed_path_efficiency",
        0.98,
        default_when_missing=True,
    )
    path94_ok = optional_threshold(
        out,
        "v44_sidecar_computed_path_efficiency",
        0.94,
        default_when_missing=True,
    )
    v65_raw_ge90_ratio = num(out, "v65_raw_ge90_ratio")
    v65_raw_ge110_ratio = num(out, "v65_raw_ge110_ratio")
    v65_smooth_ge110_ratio = num(out, "v65_smooth_ge110_ratio")
    v65_raw_ge110_run = num(out, "v65_raw_ge110_max_consecutive_frames")
    v65_smooth_ge110_run = num(out, "v65_smooth_ge110_max_consecutive_frames")

    midband = (
        current.eq(0)
        & raw.ge(105.0)
        & raw.lt(125.0)
        & smooth.ge(80.0)
        & smooth.lt(90.0)
        & num(out, "overspeed_event_score").ge(0.70)
        & num(out, "r32_overspeed_rescue_signal").ge(0.45)
        & num(out, "r32_overspeed_reliable_signal").ge(0.90)
        & (num(out, "prob_class1") + num(out, "prob_class2")).ge(0.45)
        & bbox55_ok
        & path95_ok
        & v65_raw_ge90_ratio.ge(0.35)
        & ~reject_artifact
        & ~short_track_artifact
    )

    track_entry_reject = combined_reject.str.contains(
        "track_entry_ipm_perspective_instability",
        case=False,
        regex=False,
    )
    first_window = num(out, "ts_window_idx").eq(0)
    extreme_no_history = (
        raw.ge(250.0)
        & (smooth.le(1.0) | smooth.ge(180.0))
        & bbox45_ok
        & path98_ok
        & num(out, "prob_class2").ge(0.10)
        & v65_raw_ge110_ratio.ge(0.85)
        & v65_raw_ge110_run.ge(20)
        & ~short_track_artifact
    )
    strong_raw_and_smooth = (
        raw.ge(175.0)
        & smooth.ge(110.0)
        & (
            num(out, "v44_overspeed_event_score_v8").ge(0.65)
            | num(out, "v92_v44_overspeed_event_score_v8").ge(0.65)
            | v65_raw_ge110_ratio.ge(0.50)
        )
        & bbox28_ok
        & path94_ok
        & num(out, "prob_class2").ge(0.10)
        & (
            v65_smooth_ge110_ratio.ge(0.10)
            | v65_smooth_ge110_run.ge(5)
            | v65_raw_ge110_run.ge(12)
        )
        & ~short_track_artifact
    )
    track_entry_extreme = (
        current.eq(0)
        & first_window
        & (track_entry_reject | raw.ge(250.0) | smooth.ge(110.0))
        & (extreme_no_history | strong_raw_and_smooth)
    )

    strict_wrongway = (
        num(out, "v41_family_wrongway_preconfirm_strict").ge(0.5)
        | (
            num(out, "v41_family_wrongway_preconfirm").ge(0.5)
            & num(out, "v41_family_wrongway_tail_preconfirm").ge(0.5)
        )
    )
    strict_wrongway_risk2 = (
        current.eq(0)
        & num(out, "v38_policy_risk2_support").ge(0.5)
        & strict_wrongway
        & num(out, "prob_class1").ge(0.30)
        & num(out, "redlight_event_score").lt(0.20)
    )

    updated = current.copy()
    updated.loc[midband | strict_wrongway_risk2] = np.maximum(
        updated.loc[midband | strict_wrongway_risk2], 1
    )
    updated.loc[track_entry_extreme] = np.maximum(updated.loc[track_entry_extreme], 2)

    out[COLS.midband_risk2] = midband.astype(int)
    out[COLS.track_entry_extreme_speed_risk3] = track_entry_extreme.astype(int)
    out[COLS.strict_wrongway_risk2] = strict_wrongway_risk2.astype(int)
    out[out_reason_col] = merge_reasons(
        [
            (midband, "v62_midband_coherent_risk2"),
            (strict_wrongway_risk2, "v62_strict_wrongway_preconfirm_risk2"),
            (track_entry_extreme, "v62_track_entry_extreme_speed_pressure_risk3"),
        ]
    )
    out[out_current_col] = updated.astype(int)
    return out
