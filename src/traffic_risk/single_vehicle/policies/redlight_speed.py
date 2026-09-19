#!/usr/bin/env python3
"""Clean v52 redlight / speed onset policy.

This module distills the accepted v52b -> v52c -> v52d chain:

1. redlight-zone movement pre-confirmation can promote to risk2;
2. redlight + stable speed85 contextual support can promote to risk3;
3. early redlight + midband-speed risk3 is capped to risk2 when speed /
   trajectory support is not mature;
4. a pure overspeed onset artifact can be capped from risk2 to normal.

The v52d historical script also rebuilt labels for evaluation.  That label
refresh is intentionally excluded here; this module only implements prediction
logic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CleanV52dRedlightSpeedColumns:
    v52b_current: str = "v92_v52b_redlight_zone_speed85_no_cap_current"
    v52c_current: str = "v92_v52c_redlight_midband_onset_cap_current"
    current: str = "v92_v52d_pure_overspeed_onset_guard_current"
    reason: str = "v92_v52d_redlight_speed_reason"
    redlight_zone_preconfirm_risk2: str = "v92_v52_redlight_zone_preconfirm_risk2"
    redlight85_promote_risk3: str = "v92_v52_redlight85_promote_risk3"
    redlight_speed85_cap_diagnostic: str = "v92_v52_redlight_speed85_cap_diagnostic"
    redlight_midband_onset_cap: str = "v92_v52c_redlight_midband_onset_cap"
    pure_overspeed_onset_guard: str = "v92_v52d_pure_overspeed_onset_guard"
    pure_speed_risk3_review_candidate: str = "v92_v52d_pure_speed_risk3_review_candidate"


COLS = CleanV52dRedlightSpeedColumns()


def num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default).astype(float)


def flag(frame: pd.DataFrame, col: str) -> pd.Series:
    return num(frame, col).gt(0)


def apply_clean_v52d_redlight_speed_policy(
    frame: pd.DataFrame,
    *,
    base_current_col: str = "pred_v51_early_strong_reverse_onset",
    out_current_col: str | None = None,
    out_reason_col: str | None = None,
) -> pd.DataFrame:
    """Apply the accepted v52b/v52c/v52d prediction chain.

    Values use SchemaC:
    0 = risk0+1, 1 = risk2, 2 = risk3.
    """

    if base_current_col not in frame.columns:
        raise KeyError(f"base current column not found: {base_current_col}")

    out = frame.copy()
    out_current_col = out_current_col or COLS.current
    out_reason_col = out_reason_col or COLS.reason

    base = num(out, base_current_col).astype(int).clip(0, 2)

    redlight_zone_preconfirm = (
        base.eq(0)
        & num(out, "redlight_zone_movement_signal_v1").ge(1)
        & num(out, "redlight_zone_movement_score_v1").ge(0.85)
        & num(out, "redlight_zone_red_ratio").ge(0.90)
        & num(out, "redlight_zone_between_ratio").ge(0.90)
        & num(out, "redlight_zone_prior_green_between", 1).le(0)
        & num(out, "redlight_zone_forward_progress_px").ge(6.0)
        & num(out, "redlight_zone_mean_conf").ge(0.75)
    )

    v52b = base.copy()
    v52b.loc[redlight_zone_preconfirm] = np.maximum(
        v52b.loc[redlight_zone_preconfirm], 1
    )

    redlight85_promote = (
        v52b.le(1)
        & num(out, "redlight_event_score").ge(0.60)
        & flag(out, "v46_redlight85_promote_plus_prob")
        & num(out, "speed_smoothed_kmh_p95").ge(85.0)
        & num(out, "prob_class2").ge(0.80)
    )
    v52b.loc[redlight85_promote] = 2

    redlight_speed85_cap_diagnostic = (
        v52b.eq(2)
        & num(out, "redlight_event_score").ge(0.60)
        & num(out, "speed_smoothed_kmh_p95").lt(85.0)
        & num(out, "computed_speed_kmh_p95").lt(110.0)
        & num(out, "v38_redlight_wrongway_risk3_support").lt(0.5)
        & num(out, "v38_redlight_trajectory_risk3_support").lt(0.5)
        & num(out, "v46_redlight85_promote_plus_prob").lt(0.5)
    )

    redlight_midband_onset_cap = (
        v52b.eq(2)
        & num(out, "redlight_event_score").ge(0.60)
        & num(out, "overspeed_event_score").ge(0.25)
        & num(out, "speed_smoothed_kmh_p95").lt(85.0)
        & num(out, "v44_v8_speed_kmh_max").lt(110.0)
        & num(out, "v38_redlight_wrongway_risk3_support").lt(0.5)
        & num(out, "v38_redlight_trajectory_risk3_support").lt(0.5)
        & num(out, "v28_true_weaving_memory").lt(0.5)
        & num(out, "v50_tail_history_same_direction_backtrack_6w").lt(0.5)
        & num(out, "v50c_rule_redlight_tail_history_backtrack_promote").lt(0.5)
    )
    v52c = v52b.copy()
    v52c.loc[redlight_midband_onset_cap] = 1

    raw_speed = num(out, "computed_speed_kmh_p95")
    smooth_speed = num(out, "speed_smoothed_kmh_p95")
    raw_smooth_gap = raw_speed - smooth_speed

    no_redlight_or_trajectory = (
        num(out, "redlight_event_score").lt(0.1)
        & num(out, "redlight_zone_movement_signal_v1").lt(0.5)
        & num(out, "v38_trajectory_risk2_support").lt(0.1)
        & num(out, "v28_true_weaving_memory").lt(0.5)
        & num(out, "v50_tail_history_backtrack_6w").lt(0.5)
        & num(out, "v50_tail_history_same_direction_backtrack_6w").lt(0.5)
    )

    pure_overspeed_onset_guard = (
        v52c.eq(1)
        & no_redlight_or_trajectory
        & num(out, "overspeed_event_score").ge(0.90)
        & smooth_speed.lt(90.0)
        & raw_smooth_gap.ge(45.0)
        & num(out, "v44_v8_speed_kmh_max").ge(130.0)
        & num(out, "prob_class2").lt(0.20)
    )
    v52d = v52c.copy()
    v52d.loc[pure_overspeed_onset_guard] = 0

    pure_speed_risk3_review_candidate = (
        v52c.eq(2)
        & no_redlight_or_trajectory
        & smooth_speed.lt(110.0)
        & raw_smooth_gap.ge(25.0)
        & num(out, "v44_v8_speed_kmh_max").ge(120.0)
        & num(out, "prob_class2").lt(0.65)
    )

    reason = np.full(len(out), "", dtype=object)
    reason = np.where(redlight_zone_preconfirm, "v52_redlight_zone_preconfirm_r2", reason)
    reason = np.where(redlight85_promote, "v52_redlight85_promote_r3", reason)
    reason = np.where(redlight_midband_onset_cap, "v52c_redlight_midband_onset_cap", reason)
    reason = np.where(pure_overspeed_onset_guard, "v52d_pure_overspeed_onset_guard", reason)

    out[COLS.v52b_current] = v52b.astype(int)
    out[COLS.v52c_current] = v52c.astype(int)
    out[COLS.redlight_zone_preconfirm_risk2] = redlight_zone_preconfirm.astype(int)
    out[COLS.redlight85_promote_risk3] = redlight85_promote.astype(int)
    out[COLS.redlight_speed85_cap_diagnostic] = redlight_speed85_cap_diagnostic.astype(int)
    out[COLS.redlight_midband_onset_cap] = redlight_midband_onset_cap.astype(int)
    out[COLS.pure_overspeed_onset_guard] = pure_overspeed_onset_guard.astype(int)
    out[COLS.pure_speed_risk3_review_candidate] = pure_speed_risk3_review_candidate.astype(int)
    out[out_reason_col] = reason
    out[out_current_col] = v52d.astype(int)
    return out
