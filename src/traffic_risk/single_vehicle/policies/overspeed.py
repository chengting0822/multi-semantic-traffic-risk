#!/usr/bin/env python3
"""Clean overspeed policy extracted from the v90 historical chain.

This module is intentionally not a historical experiment runner.  It exposes a
reusable DataFrame policy:

    base current risk -> clean overspeed current risk + semantic reason columns

The module does not apply track cumulative max.  Cumulative max belongs to the
final postprocess layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CleanOverspeedColumns:
    current: str = "v92_clean_overspeed_current"
    reason: str = "v92_clean_overspeed_reason"
    midband_risk2: str = "v92_overspeed_midband_risk2"
    stable_ge110_risk3: str = "v92_overspeed_stable_ge110_risk3"
    raw_sustained_ge110_risk3: str = "v92_overspeed_raw_sustained_ge110_risk3"
    track_entry_pressure_risk3: str = "v92_overspeed_track_entry_pressure_risk3"
    ge90_wrongway_combo_risk3: str = "v92_overspeed_ge90_wrongway_combo_risk3"
    track_entry_unstable_cap: str = "v92_overspeed_track_entry_unstable_cap"
    pure_speed_inconsistent_cap: str = "v92_overspeed_pure_speed_inconsistent_cap"
    short_ge110_cap: str = "v92_overspeed_short_ge110_cap"


COLS = CleanOverspeedColumns()


def num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default).astype(float)


def text(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series("", index=frame.index, dtype=str)
    return frame[col].fillna("").astype(str)


def max_existing(frame: pd.DataFrame, cols: Iterable[str]) -> pd.Series:
    present = [num(frame, col) for col in cols if col in frame.columns]
    if not present:
        return pd.Series(0.0, index=frame.index, dtype=float)
    arr = np.maximum.reduce([s.to_numpy() for s in present])
    return pd.Series(arr, index=frame.index, dtype=float)


def clean_reason(base: pd.Series, updates: list[tuple[pd.Series, str]]) -> pd.Series:
    reasons = pd.Series("", index=base.index, dtype=object)
    for mask, reason in updates:
        mask = mask.fillna(False)
        reasons.loc[mask] = np.where(
            reasons.loc[mask].astype(str).eq(""),
            reason,
            reasons.loc[mask].astype(str) + ";" + reason,
        )
    return reasons


def apply_clean_overspeed_policy(
    frame: pd.DataFrame,
    *,
    base_current_col: str,
    out_current_col: str | None = None,
    out_reason_col: str | None = None,
) -> pd.DataFrame:
    """Apply the clean overspeed policy to a window-level prediction table.

    Parameters
    ----------
    frame:
        Window-level table with v90/v65/v56 speed semantic columns.
    base_current_col:
        Current-risk column to update. Values use SchemaC:
        0 = risk0+1, 1 = risk2, 2 = risk3.
    out_current_col / out_reason_col:
        Optional output column overrides.
    """

    out = frame.copy()
    out_current_col = out_current_col or COLS.current
    out_reason_col = out_reason_col or COLS.reason

    current = num(out, base_current_col).astype(int).clip(0, 2)

    raw_p95 = pd.concat(
        [
            num(out, "v65_raw_speed_p95"),
            num(out, "v70_raw_policy_speed_p95"),
            num(out, "computed_speed_kmh_p95"),
            num(out, "speed_kmh_p95"),
        ],
        axis=1,
    ).max(axis=1)
    smooth_p95 = pd.concat(
        [
            num(out, "v65_smooth_speed_p95"),
            num(out, "speed_smoothed_kmh_p95"),
            num(out, "overspeed_smoothed_speed_kmh_p95_v60"),
        ],
        axis=1,
    ).max(axis=1)
    speed_gap = raw_p95 - smooth_p95

    reject_reason = text(out, "v65_official_reject_reason") + ";" + text(
        out, "overspeed_raw_recompute_rejected_reason_v60"
    )
    hard_artifact = reject_reason.str.contains(
        "stationary|jitter|duplicate|tiny|micro|short_track|small_bbox",
        case=False,
        regex=True,
    )
    track_entry_perspective = (
        reject_reason.str.contains("track_entry_ipm_perspective", case=False, regex=True)
        | num(out, "v44_sidecar_track_entry_ipm_perspective_suspect").ge(1)
    )

    redlight_context = (
        num(out, "redlight_event_score").ge(0.45)
        | num(out, "redlight_zone_movement_score_v1").ge(0.80)
        | num(out, "redlight_v8_confirmed_active_v8").ge(1)
        | num(out, "v69b_redlight_zone_preconfirm_risk2").ge(0.5)
        | num(out, "v58_redlight_preconfirm_r2").ge(0.5)
    )
    wrongway_support = (
        num(out, "v41_family_wrongway_preconfirm").ge(0.5)
        | num(out, "v41_family_wrongway_preconfirm_strict").ge(0.5)
        | num(out, "v51_strict_wrongway").ge(0.5)
        | num(out, "v51_strong_reverse_motion").ge(0.5)
        | num(out, "v76_wrongway_preconfirm_onset_risk2").ge(0.5)
        | num(out, "v87_wrongway_tail_preconfirm_risk2").ge(0.5)
        | num(out, "v88_refined_p30_micro_tail_wrongway").ge(0.5)
    )
    weaving_support = (
        num(out, "trajectory_weaving_clean_support_v65").ge(0.5)
        | num(out, "trajectory_true_weaving_tail_preconfirm_v60").ge(0.5)
        | num(out, "trajectory_weaving_preconfirm_risk2_support_v60").ge(0.5)
        | num(out, "v48_tail_lane_backtrack_weaving_onset").ge(0.5)
    )
    double_yellow_support = (
        num(out, "trajectory_double_yellow_contact_preconfirm_risk2_support_v71").ge(0.5)
        | num(out, "v72_double_yellow_contact_preconfirm_risk2_applied").ge(0.5)
    )
    trajectory_context = (
        wrongway_support
        | weaving_support
        | double_yellow_support
        | num(out, "s33_current_local_semantic_support").ge(0.50)
        | num(out, "s33_prefix_state_confirmed").ge(0.50)
        | num(out, "v38_trajectory_risk2_support").ge(0.50)
    )

    # v70: moderate speed onset supports risk2 only.
    midband_risk2 = (
        current.lt(1)
        & num(out, "overspeed_event_score").ge(0.90)
        & num(out, "r32_overspeed_reliable_signal").ge(0.85)
        & num(out, "r32_overspeed_midband_strength").ge(0.80)
        & num(out, "r32_overspeed_ge2_duration_strength").ge(0.45)
        & raw_p95.ge(115.0)
        & smooth_p95.ge(80.0)
        & (num(out, "prob_class1").ge(0.35) | (num(out, "prob_class1") + num(out, "prob_class2")).ge(0.60))
    )

    # v61/v68: stable high speed is risk3.
    #
    # Important: v61_stable_ge110_pre_event by itself is not clean enough for
    # a formal policy.  In the historical chain it was later constrained by
    # other stages.  Here it must be backed by frame-level maturity.
    smooth_stable_ge110 = (
        num(out, "v65_smooth_ge110_ratio").ge(0.20)
        & num(out, "v65_smooth_ge110_max_consecutive_frames").ge(6)
        & smooth_p95.ge(105.0)
    )
    maturity_clean_ge110 = text(out, "v65_speed_onset_maturity").str.contains(
        "stable_ge110_clean", case=False, regex=False
    )
    stable_ge110_risk3 = (
        current.lt(2)
        & (
            smooth_stable_ge110
            | maturity_clean_ge110
            | (
                num(out, "v68b_speed110_clean_policy_consistent").ge(0.5)
                & smooth_stable_ge110
            )
            | (
                (num(out, "overspeed_extreme_pressure_speed_event_v60").ge(0.5)
                | num(out, "v60_overspeed_extreme_pressure_speed_event").ge(0.5))
                & (smooth_stable_ge110 | maturity_clean_ge110)
            )
        )
        & ~hard_artifact
        & ~track_entry_perspective
    )

    # v74/v83: raw is already sustained enough for warning-oriented risk3.
    quality_ok = (
        num(out, "v65_bbox_area_cv").le(0.60)
        & num(out, "v65_center_jump_px_p95").le(25.0)
        & num(out, "overspeed_short_duplicate_bbox_guard_v60").lt(1)
        & num(out, "v44_sidecar_track_entry_ipm_perspective_suspect").lt(1)
        & ~reject_reason.str.contains(
            "unstable|jitter|duplicate|perspective|below_limit_smoothed|conflicts_with_below_limit",
            case=False,
            regex=True,
        )
    )
    raw_sustained_ge110_risk3 = (
        current.lt(2)
        & (
            (
                num(out, "v65_raw_ge110_ratio").ge(0.45)
                & num(out, "v65_raw_ge110_max_consecutive_frames").ge(10)
                & raw_p95.ge(130.0)
            )
            | (
                num(out, "v65_raw_ge110_ratio").ge(0.30)
                & num(out, "v65_raw_ge110_max_consecutive_frames").ge(5)
                & raw_p95.ge(125.0)
            )
        )
        & quality_ok
    )

    # v61/v62 clean extraction: some accepted pressure samples enter the frame
    # with a track-entry perspective warning, but their raw >=110 evidence is
    # sustained and the speed range is plausible.  Keep this separate from
    # track-entry artifacts such as absurd raw spikes / bbox-center jitter.
    track_entry_pressure_risk3 = (
        current.lt(2)
        & track_entry_perspective
        & num(out, "v65_raw_ge110_ratio").ge(0.50)
        & num(out, "v65_raw_ge110_max_consecutive_frames").ge(5)
        & raw_p95.ge(110.0)
        & raw_p95.le(150.0)
        & smooth_p95.ge(100.0)
        & num(out, "v65_bbox_area_cv").le(0.75)
        & num(out, "v65_center_jump_px_p95").le(30.0)
        & ~text(out, "v65_speed_onset_maturity").str.contains(
            "jitter|bbox_or_center|unstable", case=False, regex=True
        )
        & ~reject_reason.str.contains(
            "jitter|duplicate|tiny|micro|short_track|stationary", case=False, regex=True
        )
    )

    # v77: wrongway + stable >=90 is compound risk3.
    stable_ge90 = (
        (
            num(out, "v65_smooth_ge90_ratio").ge(0.25)
            & num(out, "v65_smooth_ge90_max_consecutive_frames").ge(5)
        )
        | (
            num(out, "v65_raw_ge90_ratio").ge(0.75)
            & num(out, "v65_raw_ge90_max_consecutive_frames").ge(10)
            & raw_p95.ge(100.0)
            & num(out, "v65_center_jump_px_p95").le(15.0)
            & num(out, "v65_bbox_area_cv").le(0.85)
            & (~hard_artifact | track_entry_perspective)
        )
    )
    ge90_wrongway_combo_risk3 = current.eq(1) & wrongway_support & stable_ge90
    strict_ge90_wrongway_onset = (
        current.eq(1)
        & wrongway_support
        & num(out, "v41_family_wrongway_preconfirm_strict").ge(0.5)
        & raw_p95.ge(95.0)
        & num(out, "v65_raw_ge90_ratio").ge(0.30)
        & num(out, "v65_raw_ge90_max_consecutive_frames").ge(2)
        & num(out, "prob_class2").ge(0.35)
        & num(out, "v65_bbox_area_cv").le(0.45)
        & num(out, "v65_center_jump_px_p95").le(8.0)
        & ~hard_artifact
    )
    ge90_wrongway_combo_risk3 = ge90_wrongway_combo_risk3 | strict_ge90_wrongway_onset

    # v78: first-window track-entry/persisted-sidecar disagreement cap.
    track_entry_unstable_cap = (
        current.eq(2)
        & num(out, "ts_window_idx").eq(0)
        & reject_reason.str.contains("unstable_against_persisted_sidecar", case=False, regex=False)
        & num(out, "v65_smooth_ge110_ratio").lt(0.20)
        & num(out, "v65_smooth_ge110_max_consecutive_frames").le(5)
        & ~redlight_context
        & ~trajectory_context
    )

    # v70/v73/v85: pure-speed high-risk must have mature support.
    pure_speed_inconsistent_cap = (
        current.eq(2)
        & ~redlight_context
        & ~trajectory_context
        & raw_p95.ge(110.0)
        & smooth_p95.lt(105.0)
        & num(out, "v65_smooth_ge110_ratio").lt(0.10)
        & num(out, "v65_smooth_ge110_tail_ratio").lt(0.10)
        & speed_gap.ge(20.0)
        & num(out, "r32_overspeed_rescue_signal").ge(0.80)
        & num(out, "prob_class2").lt(0.65)
    )
    short_ge110_cap = (
        current.eq(2)
        & ~redlight_context
        & ~trajectory_context
        & raw_p95.ge(110.0)
        & smooth_p95.lt(110.0)
        & num(out, "v65_raw_ge110_ratio").le(0.30)
        & num(out, "v65_raw_ge110_max_consecutive_frames").le(4)
    )

    updated = current.copy()
    updated.loc[midband_risk2] = np.maximum(updated.loc[midband_risk2], 1)
    updated.loc[
        stable_ge110_risk3
        | raw_sustained_ge110_risk3
        | track_entry_pressure_risk3
        | ge90_wrongway_combo_risk3
    ] = 2
    updated.loc[pure_speed_inconsistent_cap | short_ge110_cap] = np.minimum(
        updated.loc[pure_speed_inconsistent_cap | short_ge110_cap], 1
    )
    updated.loc[track_entry_unstable_cap] = 0

    signals = [
        (midband_risk2, "midband_risk2"),
        (stable_ge110_risk3, "stable_ge110_risk3"),
        (raw_sustained_ge110_risk3, "raw_sustained_ge110_risk3"),
        (track_entry_pressure_risk3, "track_entry_pressure_risk3"),
        (ge90_wrongway_combo_risk3, "ge90_wrongway_combo_risk3"),
        (track_entry_unstable_cap, "track_entry_unstable_cap"),
        (pure_speed_inconsistent_cap, "pure_speed_inconsistent_cap"),
        (short_ge110_cap, "short_ge110_cap"),
    ]

    out[COLS.midband_risk2] = midband_risk2.astype(int)
    out[COLS.stable_ge110_risk3] = stable_ge110_risk3.astype(int)
    out[COLS.raw_sustained_ge110_risk3] = raw_sustained_ge110_risk3.astype(int)
    out[COLS.track_entry_pressure_risk3] = track_entry_pressure_risk3.astype(int)
    out[COLS.ge90_wrongway_combo_risk3] = ge90_wrongway_combo_risk3.astype(int)
    out[COLS.track_entry_unstable_cap] = track_entry_unstable_cap.astype(int)
    out[COLS.pure_speed_inconsistent_cap] = pure_speed_inconsistent_cap.astype(int)
    out[COLS.short_ge110_cap] = short_ge110_cap.astype(int)
    out[out_reason_col] = clean_reason(current, signals)
    out[out_current_col] = updated.astype(int)
    out["v92_overspeed_raw_policy_speed_p95"] = raw_p95
    out["v92_overspeed_smooth_policy_speed_p95"] = smooth_p95
    out["v92_overspeed_raw_smooth_gap"] = speed_gap
    return out
