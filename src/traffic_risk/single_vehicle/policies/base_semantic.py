#!/usr/bin/env python3
"""Base semantic policy extracted from the v63 -> v68b historical chain.

This module turns the v68b semantic policy layer into a reusable DataFrame
policy.  It does not apply track cumulative max.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CleanBaseSemanticColumns:
    current: str = "v92_base_semantic_current"
    reason: str = "v92_base_semantic_reason"
    speed_policy_consistent: str = "v92_base_speed110_clean_policy_consistent"
    speed110_risk3: str = "v92_base_speed110_clean_risk3"
    redlight_speed85_risk3: str = "v92_base_redlight_speed85_risk3"
    redlight_weaving_lateral_risk3: str = "v92_base_redlight_weaving_lateral_risk3"
    any_risk3_support: str = "v92_base_any_current_risk3_support"


COLS = CleanBaseSemanticColumns()


def num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default).astype(float)


def text(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series("", index=frame.index, dtype=str)
    return frame[col].fillna("").astype(str)


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


def apply_clean_base_semantic_policy(
    frame: pd.DataFrame,
    *,
    base_current_col: str = "pred_v63_redlight_combo_onset_current",
    out_current_col: str | None = None,
    out_reason_col: str | None = None,
) -> pd.DataFrame:
    """Apply the v63 -> v68b base semantic policy.

    Values use SchemaC:
    0 = risk0+1, 1 = risk2, 2 = risk3.
    """

    if base_current_col not in frame.columns:
        raise KeyError(f"base current column not found: {base_current_col}")

    out = frame.copy()
    out_current_col = out_current_col or COLS.current
    out_reason_col = out_reason_col or COLS.reason

    current = num(out, base_current_col).astype(int).clip(0, 2)
    redlight = num(out, "redlight_event_score")
    critical = num(out, "prob_class2")
    computed_speed = num(out, "computed_speed_kmh_p95")
    smooth_speed = num(out, "speed_smoothed_kmh_p95")
    reliable_speed = num(out, "r32_overspeed_reliable_signal")
    overspeed_score = num(out, "overspeed_event_score")

    speed_maturity = text(out, "v65_speed_onset_maturity")
    weaving_maturity = text(out, "trajectory_weaving_onset_maturity_v65")
    tail_motion = num(out, "trajectory_tail_motion_mag_px_v65")
    tail_cos = num(out, "trajectory_tail_motion_cos_top_v65")
    weaving_review = num(out, "trajectory_weaving_review_flag_v65")

    stable_ge110_clean = speed_maturity.eq("stable_ge110_clean")
    speed_policy_consistent = smooth_speed.ge(100.0) | (
        reliable_speed.ge(0.85)
        & overspeed_score.ge(0.45)
        & computed_speed.ge(110.0)
    )
    speed110_risk3 = stable_ge110_clean & speed_policy_consistent

    redlight_speed85_risk3 = (
        redlight.ge(0.60)
        & computed_speed.ge(85.0)
        & computed_speed.le(105.0)
        & reliable_speed.ge(0.90)
        & overspeed_score.ge(0.45)
        & overspeed_score.le(0.55)
        & critical.ge(0.65)
        & smooth_speed.ge(75.0)
    )

    redlight_weaving_lateral_risk3 = (
        redlight.ge(0.60)
        & weaving_maturity.eq("second_lane_change_intent")
        & current.ge(1)
        & tail_motion.ge(30.0)
        & tail_cos.le(0.30)
        & critical.ge(0.30)
        & weaving_review.eq(1)
    )

    any_risk3 = speed110_risk3 | redlight_speed85_risk3 | redlight_weaving_lateral_risk3
    updated = current.copy()
    updated.loc[any_risk3] = np.maximum(updated.loc[any_risk3], 2)

    out[COLS.speed_policy_consistent] = speed_policy_consistent.astype(int)
    out[COLS.speed110_risk3] = speed110_risk3.astype(int)
    out[COLS.redlight_speed85_risk3] = redlight_speed85_risk3.astype(int)
    out[COLS.redlight_weaving_lateral_risk3] = redlight_weaving_lateral_risk3.astype(int)
    out[COLS.any_risk3_support] = any_risk3.astype(int)
    out[out_reason_col] = merge_reasons(
        [
            (speed110_risk3, "stable_ge110_clean"),
            (redlight_speed85_risk3, "redlight_speed85_mature"),
            (redlight_weaving_lateral_risk3, "redlight_weaving_lateral_onset"),
        ]
    )
    out[out_current_col] = updated.astype(int)
    return out
