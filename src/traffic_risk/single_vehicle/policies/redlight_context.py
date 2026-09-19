#!/usr/bin/env python3
"""Clean v46 redlight + stable 85 km/h contextual policy.

This module extracts only the prediction policy from the historical v46 script.
The historical script also rebuilt labels from GUI annotations, but that is
evaluation-only and intentionally not included here.

Formal chain:
    pred_v45_redlight_speed_ge90_consistency
    -> v92_v46_redlight85_causal_consecutive_current

The historical plus-prob variant is reproduced for audit equivalence, but it is
not used as the downstream source because v47 consumed the causal-consecutive
column.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


GROUP = ["split", "case_key", "video_id", "track_id"]


@dataclass(frozen=True)
class CleanV46Redlight85ContextColumns:
    current: str = "v92_v46_redlight85_causal_consecutive_current"
    plus_prob_current: str = "v92_v46_redlight85_context_plus_prob_current"
    reason: str = "v92_v46_redlight85_context_reason"
    plus_prob_reason: str = "v92_v46_redlight85_plus_prob_reason"
    redlight85_context: str = "v92_v46_redlight85_context"
    stable_speed_ge85: str = "v92_v46_stable_speed_ge85"
    stable_speed_ge90: str = "v92_v46_stable_speed_ge90"
    causal_consecutive: str = "v92_v46_causal_consecutive_redlight85"
    explicit_wrongway_combo: str = "v92_v46_explicit_wrongway_combo"
    explicit_trajectory_combo: str = "v92_v46_explicit_trajectory_combo"
    trajectory_or_wrongway_support: str = "v92_v46_trajectory_or_wrongway_support"
    high_critical_prob: str = "v92_v46_high_critical_prob"
    promote_consecutive: str = "v92_v46_redlight85_promote_consecutive"
    promote_plus_prob: str = "v92_v46_redlight85_promote_plus_prob"
    consecutive_changed: str = "v92_v46_consecutive_changed"
    plus_prob_changed: str = "v92_v46_plus_prob_changed"


COLS = CleanV46Redlight85ContextColumns()


def num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default).astype(float)


def max_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    parts = [num(frame, col, np.nan) for col in columns]
    if not parts:
        return pd.Series(0.0, index=frame.index, dtype=float)
    return pd.concat(parts, axis=1).max(axis=1).fillna(0.0)


def apply_clean_v46_redlight85_context_policy(
    frame: pd.DataFrame,
    *,
    base_current_col: str = "pred_v45_redlight_speed_ge90_consistency",
    out_current_col: str | None = None,
    out_plus_prob_current_col: str | None = None,
) -> pd.DataFrame:
    """Apply the accepted v46 causal redlight85 context promotion.

    Values use SchemaC:
    0 = risk0+1, 1 = risk2, 2 = risk3.
    """

    if base_current_col not in frame.columns:
        raise KeyError(f"base current column not found: {base_current_col}")

    out = frame.copy().sort_values(GROUP + ["ts_window_idx"], kind="mergesort").reset_index(drop=True)
    out_current_col = out_current_col or COLS.current
    out_plus_prob_current_col = out_plus_prob_current_col or COLS.plus_prob_current

    base = num(out, base_current_col).astype(int).clip(0, 2)

    # Historical v46 intentionally used stable/policy smoothed speed, not raw
    # v8 speed, because this rule is about mature redlight + 85 km/h context.
    smooth = max_numeric(
        out,
        [
            "speed_smoothed_kmh_p95",
            "v44_sidecar_speed_smoothed_kmh_p95",
        ],
    )
    stable_ge85 = smooth.ge(85.0)
    stable_ge90 = smooth.ge(90.0)
    redlight85_context = num(out, "redlight_event_score").ge(0.45) & stable_ge85

    explicit_wrongway_combo = (
        num(out, "v38_redlight_wrongway_risk3_support").ge(0.5)
        | num(out, "v31_redlight_wrongway_combo_ready").ge(0.5)
        | num(out, "v41_family_wrongway_preconfirm_strict").ge(0.5)
        | num(out, "v41f_lite_connector_strong_reverse_applied").ge(0.5)
    )
    explicit_trajectory_combo = (
        num(out, "v38_redlight_trajectory_risk3_support").ge(0.5)
        | num(out, "v31_redlight_trajectory_combo_score").ge(0.5)
        | num(out, "v30_true_weaving_confirmed").ge(0.5)
        | num(out, "v30_route_deviation_confirmed").ge(0.5)
        | num(out, "s33_current_local_semantic_support").ge(0.80)
    )
    trajectory_or_wrongway = explicit_wrongway_combo | explicit_trajectory_combo

    same_track = out[GROUP].eq(out[GROUP].shift(1)).all(axis=1)
    prev_window = num(out, "ts_window_idx").eq(num(out, "ts_window_idx").shift(1) + 1)
    prev_redlight85 = redlight85_context.shift(1, fill_value=False)
    causal_consecutive = redlight85_context & same_track & prev_window & prev_redlight85
    high_critical = num(out, "critical_prob").ge(0.80)

    current_risk2 = base.eq(1)
    promote_consecutive = current_risk2 & redlight85_context & (causal_consecutive | trajectory_or_wrongway)
    promote_plus_prob = current_risk2 & redlight85_context & (
        causal_consecutive | trajectory_or_wrongway | high_critical
    )

    current = base.copy()
    current.loc[promote_consecutive] = np.maximum(current.loc[promote_consecutive], 2)
    plus_current = base.copy()
    plus_current.loc[promote_plus_prob] = np.maximum(plus_current.loc[promote_plus_prob], 2)

    out[COLS.redlight85_context] = redlight85_context.astype(int)
    out[COLS.stable_speed_ge85] = stable_ge85.astype(int)
    out[COLS.stable_speed_ge90] = stable_ge90.astype(int)
    out[COLS.causal_consecutive] = causal_consecutive.astype(int)
    out[COLS.explicit_wrongway_combo] = explicit_wrongway_combo.astype(int)
    out[COLS.explicit_trajectory_combo] = explicit_trajectory_combo.astype(int)
    out[COLS.trajectory_or_wrongway_support] = trajectory_or_wrongway.astype(int)
    out[COLS.high_critical_prob] = high_critical.astype(int)
    out[COLS.promote_consecutive] = promote_consecutive.astype(int)
    out[COLS.promote_plus_prob] = promote_plus_prob.astype(int)
    out[COLS.consecutive_changed] = current.ne(base).astype(int)
    out[COLS.plus_prob_changed] = plus_current.ne(base).astype(int)
    out[COLS.reason] = np.where(promote_consecutive, "v46_redlight85_causal_consecutive_promote_risk3", "")
    out[COLS.plus_prob_reason] = np.where(promote_plus_prob, "v46_redlight85_plus_prob_promote_risk3", "")
    out[out_current_col] = current.astype(int)
    out[out_plus_prob_current_col] = plus_current.astype(int)
    return out
