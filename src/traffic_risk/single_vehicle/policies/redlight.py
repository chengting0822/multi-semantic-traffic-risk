#!/usr/bin/env python3
"""Clean redlight policy extracted from the v90 historical chain.

This module exposes a reusable DataFrame policy:

    base current risk -> clean redlight current risk + semantic reason columns

It does not apply track cumulative max.  Cumulative max belongs to the final
postprocess layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CleanRedlightColumns:
    current: str = "v92_clean_redlight_current"
    reason: str = "v92_clean_redlight_reason"
    zone_preconfirm_risk2: str = "v92_redlight_zone_preconfirm_risk2"
    combo_mature_risk3: str = "v92_redlight_combo_mature_risk3"
    wrongway_preconfirm_combo_risk3: str = "v92_redlight_wrongway_preconfirm_combo_risk3"
    speed85_mature_risk3: str = "v92_redlight_speed85_mature_risk3"
    weaving_lateral_risk3: str = "v92_redlight_weaving_lateral_risk3"
    late_stale_combo_cap: str = "v92_redlight_late_stale_combo_cap"
    weak_trajectory_cap: str = "v92_redlight_weak_trajectory_cap"


COLS = CleanRedlightColumns()


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


def apply_clean_redlight_policy(
    frame: pd.DataFrame,
    *,
    base_current_col: str,
    out_current_col: str | None = None,
    out_reason_col: str | None = None,
) -> pd.DataFrame:
    """Apply the clean redlight policy to a window-level prediction table.

    Values use SchemaC:
    0 = risk0+1, 1 = risk2, 2 = risk3.
    """

    out = frame.copy()
    out_current_col = out_current_col or COLS.current
    out_reason_col = out_reason_col or COLS.reason

    current = num(out, base_current_col).astype(int).clip(0, 2)

    redlight = num(out, "redlight_event_score")
    critical_prob = num(out, "prob_class2")
    computed_speed = num(out, "computed_speed_kmh_p95")
    smooth_speed = num(out, "speed_smoothed_kmh_p95")
    reliable_speed = num(out, "r32_overspeed_reliable_signal")
    overspeed_score = num(out, "overspeed_event_score")

    # Fresh trajectory supports.  These are current-window or tail-window
    # semantics, not broad historical support.
    strict_wrongway_support = (
        num(out, "v41_family_wrongway_preconfirm_strict").ge(0.5)
        | num(out, "v51_strict_wrongway").ge(0.5)
        | num(out, "v51_strong_reverse_motion").ge(0.5)
        | num(out, "v76_wrongway_preconfirm_onset_risk2").ge(0.5)
        | num(out, "v87_wrongway_tail_preconfirm_risk2").ge(0.5)
        | num(out, "v88_refined_p30_micro_tail_wrongway").ge(0.5)
    )
    fresh_wrongway_support = (
        strict_wrongway_support
        | num(out, "v41_family_wrongway_preconfirm").ge(0.5)
        | num(out, "v41_family_wrongway_tail_preconfirm").ge(0.5)
    )
    clean_weaving_support = (
        num(out, "trajectory_weaving_clean_support_v65").ge(0.5)
        | num(out, "trajectory_true_weaving_tail_preconfirm_v60").ge(0.5)
        | num(out, "trajectory_weaving_preconfirm_risk2_support_v60").ge(0.5)
        | num(out, "v48_tail_lane_backtrack_weaving_onset").ge(0.5)
    )
    fresh_speed_risk3_support = (
        computed_speed.ge(110.0)
        | num(out, "v65_raw_ge110_ratio").ge(0.45)
        | num(out, "v65_smooth_ge110_ratio").ge(0.20)
        | num(out, "v68b_speed110_clean_risk3").ge(0.5)
        | num(out, "v77_stable_ge90_wrongway_combo").ge(0.5)
        | num(out, "v92_overspeed_stable_ge110_risk3").ge(0.5)
        | num(out, "v92_overspeed_raw_sustained_ge110_risk3").ge(0.5)
    )

    # v69b: redlight-zone movement supports risk2 only.
    strict_zone_preconfirm = (
        num(out, "redlight_zone_movement_signal_v1").ge(1)
        & num(out, "redlight_zone_movement_score_v1").ge(0.65)
        & num(out, "redlight_zone_red_duration_sec").ge(0.45)
        & num(out, "redlight_zone_between_ratio").ge(0.75)
        & num(out, "redlight_zone_red_ratio").ge(0.80)
        & num(out, "redlight_zone_prior_green_between").lt(0.5)
        & num(out, "redlight_zone_mean_conf").ge(0.50)
    )
    low_progress_zone_preconfirm = (
        num(out, "redlight_zone_movement_score_v1").ge(0.80)
        & num(out, "redlight_zone_red_duration_sec").ge(0.45)
        & num(out, "redlight_zone_forward_progress_px").ge(4.50)
        & num(out, "redlight_zone_between_ratio").ge(0.75)
        & num(out, "redlight_zone_red_ratio").ge(0.80)
        & num(out, "redlight_zone_prior_green_between").lt(0.5)
        & num(out, "redlight_zone_mean_conf").ge(0.50)
    )
    zone_preconfirm_risk2 = current.lt(1) & (strict_zone_preconfirm | low_progress_zone_preconfirm)

    # v63: redlight + mature trajectory/wrongway combo supports risk3.
    already_risk2 = current.ge(1)
    red_confirmed = redlight.ge(0.70)
    policy_combo = num(out, "v38_policy_risk3_support").gt(0) & num(out, "v38_trajectory_risk2_support").ge(0.90)
    mature_zone_combo = (
        num(out, "redlight_zone_movement_score_v1").ge(0.50)
        & num(out, "redlight_zone_between_ratio").ge(0.40)
        & num(out, "redlight_zone_forward_progress_px").le(-6.0)
    )
    strict_wrongway_late_crossing = (
        num(out, "v41_family_wrongway_preconfirm").gt(0)
        & num(out, "redlight_zone_red_ratio").ge(0.99)
        & num(out, "v63_redlight_crossing_pos_in_window").ge(0.75)
        & critical_prob.ge(0.30)
    )
    combo_mature_risk3 = (
        current.lt(2)
        & already_risk2
        & red_confirmed
        & policy_combo
        & critical_prob.ge(0.30)
        & (mature_zone_combo | strict_wrongway_late_crossing)
    )
    wrongway_preconfirm_combo_risk3 = (
        current.lt(2)
        & already_risk2
        & red_confirmed
        & num(out, "v38_redlight_wrongway_risk3_support").ge(0.5)
        & fresh_wrongway_support
        & critical_prob.ge(0.35)
        & ~text(out, "v65_official_reject_reason").str.contains(
            "small_bbox|oscillatory_micro_motion|stationary_pixel_jitter|track_entry_ipm_perspective",
            case=False,
            regex=True,
        )
    )

    # v68b: redlight + mature 85 km/h context supports risk3.
    speed85_mature_risk3 = (
        current.lt(2)
        & redlight.ge(0.60)
        & computed_speed.ge(85.0)
        & computed_speed.le(105.0)
        & reliable_speed.ge(0.90)
        & overspeed_score.ge(0.45)
        & overspeed_score.le(0.55)
        & critical_prob.ge(0.65)
        & smooth_speed.ge(75.0)
    )

    # v68b: redlight + second lane-change / weaving lateral onset supports risk3.
    maturity = text(out, "trajectory_weaving_onset_maturity_v65")
    weaving_lateral_risk3 = (
        current.ge(1)
        & current.lt(2)
        & redlight.ge(0.60)
        & maturity.eq("second_lane_change_intent")
        & num(out, "trajectory_tail_motion_mag_px_v65").ge(30.0)
        & num(out, "trajectory_tail_motion_cos_top_v65").le(0.30)
        & critical_prob.ge(0.30)
        & num(out, "trajectory_weaving_review_flag_v65").eq(1)
    )

    # v80: stale redlight+wrongway memory should not become risk3 if the
    # actual crossing is an immature tail event with no fresh support.
    stale_redlight_wrongway_combo = (
        num(out, "v38_redlight_wrongway_risk3_support").ge(0.5)
        & num(out, "v38_redlight_trajectory_risk3_support").lt(0.5)
    )
    legacy_late_immature_crossing = (
        redlight.ge(0.5)
        & num(out, "redlight_v8_crossing_event_in_window").ge(0.5)
        & num(out, "v63_redlight_crossing_pos_in_window").ge(0.75)
        & num(out, "redlight_v8_motion_after_crossing_px").lt(5.0)
        & num(out, "redlight_zone_between_ratio").lt(0.25)
    )
    zone_late_immature_crossing = (
        redlight.ge(0.5)
        & num(out, "redlight_zone_red_ratio").ge(0.95)
        & num(out, "redlight_zone_between_ratio").lt(0.25)
        & num(out, "redlight_zone_movement_score_v1").lt(0.35)
        & num(out, "redlight_zone_forward_progress_px").gt(-5.0)
    )
    late_immature_crossing = legacy_late_immature_crossing | zone_late_immature_crossing
    late_stale_combo_cap = (
        current.eq(2)
        & stale_redlight_wrongway_combo
        & late_immature_crossing
        & ~fresh_wrongway_support
        & ~clean_weaving_support
        & ~fresh_speed_risk3_support
    )

    # v81: redlight + weak trajectory maturity should wait at risk2 until
    # strict wrongway / clean weaving / strong speed evidence matures.
    guard_reason = text(out, "trajectory_weaving_onset_guard_reason_v65")
    weak_trajectory_maturity = (
        maturity.isin(["weak_lane_change_history", "same_direction_backtrack"])
        | guard_reason.str.contains("normal_turn_or_connector|low_motion", case=False, regex=True)
    )
    legacy_redlight_late_and_zone_immature = (
        redlight.ge(0.95)
        & num(out, "redlight_v8_crossing_event_in_window").ge(0.5)
        & num(out, "v63_redlight_crossing_pos_in_window").ge(0.70)
        & num(out, "redlight_zone_between_ratio").lt(0.30)
    )
    zone_redlight_late_and_zone_immature = (
        redlight.ge(0.95)
        & num(out, "redlight_zone_red_ratio").ge(0.95)
        & num(out, "redlight_zone_between_ratio").lt(0.30)
        & num(out, "redlight_zone_movement_score_v1").lt(0.45)
        & num(out, "redlight_zone_forward_progress_px").le(-3.0)
    )
    redlight_late_and_zone_immature = (
        legacy_redlight_late_and_zone_immature | zone_redlight_late_and_zone_immature
    )
    weak_redlight_artifact_context = (
        computed_speed.ge(100.0)
        | smooth_speed.ge(60.0)
        | text(out, "v65_official_reject_reason").str.contains(
            "small_bbox|oscillatory_micro_motion", case=False, regex=True
        )
    )
    weak_trajectory_cap = (
        current.eq(2)
        & num(out, "v38_redlight_trajectory_risk3_support").ge(0.95)
        & redlight_late_and_zone_immature
        & weak_trajectory_maturity
        & weak_redlight_artifact_context
        & ~strict_wrongway_support
        & ~clean_weaving_support
        & ~fresh_speed_risk3_support
    )

    updated = current.copy()
    updated.loc[zone_preconfirm_risk2] = np.maximum(updated.loc[zone_preconfirm_risk2], 1)
    risk3_promote = (
        combo_mature_risk3
        | wrongway_preconfirm_combo_risk3
        | speed85_mature_risk3
        | weaving_lateral_risk3
    )
    updated.loc[risk3_promote] = 2
    cap = late_stale_combo_cap | weak_trajectory_cap
    updated.loc[cap] = np.minimum(updated.loc[cap], 1)

    signals = [
        (zone_preconfirm_risk2, "zone_preconfirm_risk2"),
        (combo_mature_risk3, "combo_mature_risk3"),
        (wrongway_preconfirm_combo_risk3, "wrongway_preconfirm_combo_risk3"),
        (speed85_mature_risk3, "speed85_mature_risk3"),
        (weaving_lateral_risk3, "weaving_lateral_risk3"),
        (late_stale_combo_cap, "late_stale_combo_cap"),
        (weak_trajectory_cap, "weak_trajectory_cap"),
    ]

    out[COLS.zone_preconfirm_risk2] = zone_preconfirm_risk2.astype(int)
    out[COLS.combo_mature_risk3] = combo_mature_risk3.astype(int)
    out[COLS.wrongway_preconfirm_combo_risk3] = wrongway_preconfirm_combo_risk3.astype(int)
    out[COLS.speed85_mature_risk3] = speed85_mature_risk3.astype(int)
    out[COLS.weaving_lateral_risk3] = weaving_lateral_risk3.astype(int)
    out[COLS.late_stale_combo_cap] = late_stale_combo_cap.astype(int)
    out[COLS.weak_trajectory_cap] = weak_trajectory_cap.astype(int)
    out[out_reason_col] = clean_reason(signals)
    out[out_current_col] = updated.astype(int)
    return out
