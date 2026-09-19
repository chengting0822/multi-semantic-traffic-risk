#!/usr/bin/env python3
"""Clean v63 redlight-combo maturity policy.

This module extracts the accepted v63 redlight combo onset logic into a
reusable DataFrame policy.  It starts from v62 current risk and does not apply
track cumulative max.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CleanRedlightComboMaturityColumns:
    current: str = "v92_redlight_combo_maturity_current"
    reason: str = "v92_redlight_combo_maturity_reason"
    support: str = "v92_redlight_combo_maturity_support"
    mature_zone_branch: str = "v92_redlight_combo_mature_zone_branch"
    late_crossing_branch: str = "v92_redlight_combo_late_crossing_branch"


COLS = CleanRedlightComboMaturityColumns()


def num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default).astype(float)


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


def apply_clean_redlight_combo_maturity_policy(
    frame: pd.DataFrame,
    *,
    base_current_col: str = "pred_v62_root_cause_repair_current",
    out_current_col: str | None = None,
    out_reason_col: str | None = None,
) -> pd.DataFrame:
    """Apply v63 redlight combo maturity policy.

    Values use SchemaC:
    0 = risk0+1, 1 = risk2, 2 = risk3.
    """

    if base_current_col not in frame.columns:
        raise KeyError(f"base current column not found: {base_current_col}")

    out = frame.copy()
    out_current_col = out_current_col or COLS.current
    out_reason_col = out_reason_col or COLS.reason

    current = num(out, base_current_col).astype(int).clip(0, 2)
    already_risk2 = current.ge(1)
    not_risk3_yet = current.lt(2)
    red_confirmed = num(out, "redlight_event_score").ge(0.70)
    policy_combo = (
        num(out, "v38_policy_risk3_support").ge(0.5)
        & num(out, "v38_trajectory_risk2_support").ge(0.90)
    )
    critical_model = num(out, "prob_class2").ge(0.30)

    mature_zone_combo = (
        num(out, "redlight_zone_movement_score_v1").ge(0.50)
        & num(out, "redlight_zone_between_ratio").ge(0.40)
        & num(out, "redlight_zone_forward_progress_px").le(-6.0)
    )
    strict_wrongway_late_crossing = (
        num(out, "v41_family_wrongway_preconfirm").ge(0.5)
        & num(out, "redlight_zone_red_ratio").ge(0.99)
        & num(out, "v63_redlight_crossing_pos_in_window").ge(0.75)
        & critical_model
    )

    support = (
        already_risk2
        & not_risk3_yet
        & red_confirmed
        & policy_combo
        & critical_model
        & (mature_zone_combo | strict_wrongway_late_crossing)
    )

    mature_zone_branch = support & mature_zone_combo
    late_crossing_branch = support & strict_wrongway_late_crossing

    updated = current.copy()
    updated.loc[support] = np.maximum(updated.loc[support], 2)

    out[COLS.support] = support.astype(int)
    out[COLS.mature_zone_branch] = mature_zone_branch.astype(int)
    out[COLS.late_crossing_branch] = late_crossing_branch.astype(int)
    out[out_reason_col] = merge_reasons(
        [
            (mature_zone_branch, "redlight_combo_mature_zone"),
            (late_crossing_branch, "redlight_combo_late_crossing"),
        ]
    )
    out[out_current_col] = updated.astype(int)
    return out
