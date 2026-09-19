#!/usr/bin/env python3
"""Clean trajectory policy extracted from the v90 historical chain.

Trajectory-only evidence supports risk2.  It does not create risk3 by itself.
Composite risk3 belongs to redlight / overspeed / final postprocess modules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CleanTrajectoryColumns:
    current: str = "v92_clean_trajectory_current"
    reason: str = "v92_clean_trajectory_reason"
    wrongway_risk2: str = "v92_trajectory_wrongway_risk2"
    weaving_risk2: str = "v92_trajectory_weaving_risk2"
    double_yellow_risk2: str = "v92_trajectory_double_yellow_risk2"
    general_risk2: str = "v92_trajectory_general_risk2"
    weak_onset_cap: str = "v92_trajectory_weak_onset_cap"
    trajectory_only_risk3_cap: str = "v92_trajectory_only_risk3_cap"


COLS = CleanTrajectoryColumns()


def num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default).astype(float)


def text(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series("", index=frame.index, dtype=str)
    return frame[col].fillna("").astype(str)


def clean_reason(updates: list[tuple[pd.Series, str]]) -> pd.Series:
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


def apply_clean_trajectory_policy(
    frame: pd.DataFrame,
    *,
    base_current_col: str,
    out_current_col: str | None = None,
    out_reason_col: str | None = None,
) -> pd.DataFrame:
    """Apply the clean trajectory policy to a window-level prediction table."""

    out = frame.copy()
    out_current_col = out_current_col or COLS.current
    out_reason_col = out_reason_col or COLS.reason

    current = num(out, base_current_col).astype(int).clip(0, 2)

    redlight_support = (
        num(out, "redlight_event_score").ge(0.50)
        | num(out, "v69b_redlight_zone_preconfirm_risk2").ge(0.5)
        | num(out, "v58_redlight_preconfirm_r2").ge(0.5)
        | num(out, "v92_redlight_zone_preconfirm_risk2").ge(0.5)
    )
    speed_support = (
        num(out, "computed_speed_kmh_p95").ge(90.0)
        | num(out, "v65_raw_ge90_ratio").ge(0.50)
        | num(out, "v70_mid_speed_onset_risk2").ge(0.5)
        | num(out, "v92_overspeed_midband_risk2").ge(0.5)
        | num(out, "v92_overspeed_stable_ge110_risk3").ge(0.5)
        | num(out, "v92_overspeed_raw_sustained_ge110_risk3").ge(0.5)
    )
    strict_wrongway_support = (
        num(out, "v41_family_wrongway_preconfirm_strict").ge(0.5)
        | num(out, "v51_strict_wrongway").ge(0.5)
        | num(out, "v51_strong_reverse_motion").ge(0.5)
    )
    clean_weaving_support = (
        num(out, "trajectory_weaving_clean_support_v65").ge(0.5)
        | num(out, "trajectory_true_weaving_tail_preconfirm_v60").ge(0.5)
        | num(out, "trajectory_weaving_preconfirm_risk2_support_v60").ge(0.5)
        | num(out, "v48_tail_lane_backtrack_weaving_onset").ge(0.5)
    )
    double_yellow_support = (
        num(out, "trajectory_double_yellow_contact_preconfirm_risk2_support_v71").ge(0.5)
        | num(out, "v72_double_yellow_contact_preconfirm_risk2_applied").ge(0.5)
    )

    # v51/v76/v87/v88: wrongway onset supports risk2 only.
    wrongway_risk2 = (
        current.lt(1)
        & (
            num(out, "v51_early_strong_reverse_promote_risk2").ge(0.5)
            | num(out, "v76_wrongway_preconfirm_onset_risk2").ge(0.5)
            | num(out, "v87_wrongway_tail_preconfirm_risk2").ge(0.5)
            | num(out, "v88_refined_p30_micro_tail_wrongway").ge(0.5)
        )
    )

    # v60/v65/v48: weaving / head-swing / tail lane backtrack supports risk2.
    # The broad v48/v65 columns are noisy if used alone, but mature tail
    # semantics such as confirmed tail weaving / head-swing onset were part of
    # the v90 historical behavior.  Re-enable only those mature signals and
    # require enough motion/context so stationary jitter is not promoted.
    mature_weaving_maturity = text(out, "trajectory_weaving_onset_maturity_v65").isin(
        ["confirmed_tail_weaving", "head_swing_onset"]
    )
    clean_weaving_motion = (
        num(out, "trajectory_tail_motion_mag_px_v65").ge(50.0)
        & num(out, "trajectory_context_quality_v65").ge(0.30)
        & ~text(out, "v65_official_reject_reason").str.contains(
            "short_track_micro_motion|tiny_motion|tiny_box", case=False, regex=True
        )
    )
    weaving_risk2 = (
        current.lt(1)
        & (clean_weaving_support | mature_weaving_maturity)
        & clean_weaving_motion
    )

    # v71/v72: double-yellow contact supports risk2 only.
    double_yellow_risk2 = current.lt(1) & double_yellow_support

    # v38 broad trajectory risk2 support can be used, but not when v79 says
    # the onset is weak / normal-turn-like.
    guard_reason = text(out, "trajectory_weaving_onset_guard_reason_v65")
    weak_normal_reason = guard_reason.str.contains(
        "normal_turn_or_connector|weak_lane_change_history", case=False, regex=True
    )
    family_mixed_or_opposite = guard_reason.str.contains("family_mixed_or_opposite", case=False, regex=False)
    lane_ambiguous = num(out, "v28_lane_ambiguous").ge(0.5)
    low_context = num(out, "trajectory_context_quality_v65").le(0.05) | lane_ambiguous
    weak_context = weak_normal_reason | (lane_ambiguous & ~family_mixed_or_opposite)
    not_strong_reverse = num(out, "trajectory_tail_motion_cos_top_v65", default=1.0).gt(-0.75)

    weak_onset_cap = (
        current.eq(1)
        & num(out, "v38_trajectory_risk2_support").ge(0.5)
        & ~redlight_support
        & ~speed_support
        & ~strict_wrongway_support
        & ~clean_weaving_support
        & ~double_yellow_support
        & weak_context
        & low_context
        & not_strong_reverse
    )

    general_risk2 = (
        current.lt(1)
        & num(out, "v38_trajectory_risk2_support").ge(0.90)
        & ~weak_context
    )

    # Some v90 windows were supported by clean lane-family tail semantics
    # rather than the broad v38 score.  This is still trajectory-only risk2.
    tail_wrongway_motion_risk2 = (
        current.lt(1)
        & (
            num(out, "v41_family_wrongway_preconfirm").ge(0.5)
            | num(out, "v41_family_wrongway_tail_preconfirm").ge(0.5)
        )
        & num(out, "trajectory_tail_motion_mag_px_v65").ge(30.0)
        & num(out, "trajectory_tail_motion_cos_top_v65", default=1.0).le(-0.75)
        & ~text(out, "v65_official_reject_reason").str.contains(
            "stationary_pixel_jitter|short_track_micro_motion|tiny_motion|tiny_box",
            case=False,
            regex=True,
        )
    )

    # v38: trajectory-only cannot create risk3 without composite risk3 support.
    #
    # Keep this as an explicit diagnostic flag, but do not alter current risk in
    # the trajectory module.  The cap depends on the complete redlight/speed
    # context, so it belongs in clean_final_postprocess.  Applying it here
    # caused false demotion for redlight-combo windows such as 37/14.
    composite_risk3_support = (
        num(out, "v38_policy_risk3_support").ge(0.5)
        | num(out, "v38_stable_ge110_support").ge(0.5)
        | redlight_support
        | num(out, "overspeed_event_score").ge(0.45)
        | num(out, "v92_overspeed_stable_ge110_risk3").ge(0.5)
        | num(out, "v92_overspeed_raw_sustained_ge110_risk3").ge(0.5)
        | num(out, "v92_redlight_combo_mature_risk3").ge(0.5)
        | num(out, "v92_redlight_speed85_mature_risk3").ge(0.5)
        | num(out, "v92_redlight_weaving_lateral_risk3").ge(0.5)
    )
    trajectory_only_risk3_cap = (
        current.eq(2)
        & num(out, "v38_trajectory_only_context").ge(0.5)
        & ~composite_risk3_support
    )

    updated = current.copy()
    promote_risk2 = (
        wrongway_risk2
        | weaving_risk2
        | double_yellow_risk2
        | general_risk2
        | tail_wrongway_motion_risk2
    )
    updated.loc[promote_risk2] = np.maximum(updated.loc[promote_risk2], 1)
    updated.loc[weak_onset_cap] = 0

    signals = [
        (wrongway_risk2, "wrongway_risk2"),
        (weaving_risk2, "weaving_risk2"),
        (double_yellow_risk2, "double_yellow_risk2"),
        (general_risk2, "general_risk2"),
        (tail_wrongway_motion_risk2, "tail_wrongway_motion_risk2"),
        (weak_onset_cap, "weak_onset_cap"),
        (trajectory_only_risk3_cap, "trajectory_only_risk3_cap"),
    ]

    out[COLS.wrongway_risk2] = wrongway_risk2.astype(int)
    out[COLS.weaving_risk2] = weaving_risk2.astype(int)
    out[COLS.double_yellow_risk2] = double_yellow_risk2.astype(int)
    out[COLS.general_risk2] = general_risk2.astype(int)
    out[COLS.weak_onset_cap] = weak_onset_cap.astype(int)
    out[COLS.trajectory_only_risk3_cap] = trajectory_only_risk3_cap.astype(int)
    out[out_reason_col] = clean_reason(signals)
    out[out_current_col] = updated.astype(int)
    return out
