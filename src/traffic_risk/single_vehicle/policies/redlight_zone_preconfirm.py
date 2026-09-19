#!/usr/bin/env python3
"""Clean v42 red-light-zone preconfirm policy.

This module distills the accepted inference logic from historical v42.
It intentionally excludes label rebuilding and review-only diagnostics.

Formal chain:
    pred_v41f_lite_onset_v1e_slow_wrongway
    -> v92_v42_redlight_zone_preconfirm_current

The rule is deliberately narrow:
    - starts from an existing risk2 current;
    - requires red-light-zone movement evidence from the upstream sidecar;
    - requires trajectory / wrongway memory;
    - requires non-trivial risk3 probability;
    - never promotes normal directly to risk.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


GROUP = ["split", "case_key", "video_id", "track_id"]


@dataclass(frozen=True)
class CleanV42RedlightZonePreconfirmColumns:
    current: str = "v92_v42_redlight_zone_preconfirm_current"
    reason: str = "v92_v42_redlight_zone_preconfirm_reason"
    zone_preconfirm: str = "v92_v42_redlight_zone_preconfirm"
    trajectory_memory: str = "v92_v42_redlight_trajectory_memory"
    promoted_current: str = "v92_v42_promoted_current"
    changed: str = "v92_v42_changed"


COLS = CleanV42RedlightZonePreconfirmColumns()


def num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default).astype(float)


def _sort(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.copy().sort_values(GROUP + ["ts_window_idx"], kind="mergesort").reset_index(drop=True)


def apply_clean_v42_redlight_zone_preconfirm_policy(
    frame: pd.DataFrame,
    *,
    base_current_col: str = "pred_v41f_lite_onset_v1e_slow_wrongway",
    out_current_col: str | None = None,
) -> pd.DataFrame:
    """Apply the accepted v42 red-light-zone preconfirm rule."""

    if base_current_col not in frame.columns:
        raise KeyError(f"base current column not found: {base_current_col}")

    out = _sort(frame)
    out_current_col = out_current_col or COLS.current
    base = num(out, base_current_col).astype(int).clip(0, 2)
    out[out_current_col] = base

    zone_preconfirm = (
        num(out, "redlight_zone_movement_score_v1").ge(0.65)
        & num(out, "redlight_zone_red_duration_sec").ge(0.45)
        & num(out, "redlight_zone_between_ratio").ge(0.75)
        & num(out, "redlight_zone_red_ratio").ge(0.80)
        & num(out, "redlight_zone_prior_green_between").lt(0.5)
        & num(out, "redlight_zone_mean_conf").ge(0.55)
    )
    trajectory_memory = (
        num(out, "s33_prefix_state_confirmed").ge(1.0)
        & (
            num(out, "v41_family_wrongway_preconfirm_strict").ge(1.0)
            | num(out, "v41_family_wrongway_tail_preconfirm").ge(1.0)
            | num(out, "s33_current_local_semantic_support").ge(0.35)
        )
    )
    critical_prob = num(out, "prob_class2")
    promote = base.eq(1) & zone_preconfirm & trajectory_memory & critical_prob.ge(0.15)

    out[COLS.zone_preconfirm] = zone_preconfirm.astype(int)
    out[COLS.trajectory_memory] = trajectory_memory.astype(int)
    out[COLS.promoted_current] = promote.astype(int)

    result = base.copy()
    for _, idx in out.groupby(GROUP, sort=False).groups.items():
        idx = list(idx)
        vals = result.loc[idx].to_numpy(dtype=int)
        local_promote = promote.loc[idx].to_numpy(dtype=bool)
        vals[local_promote] = 2
        vals = np.maximum.accumulate(vals)
        result.loc[idx] = vals

    out[out_current_col] = result.astype(int)
    out[COLS.changed] = out[out_current_col].astype(int).ne(base).astype(int)
    out[COLS.reason] = np.where(promote, "v42_redlight_zone_preconfirm_promote_risk3", "")

    # Historical alias for downstream compatibility if a clean-table input does
    # not already carry the historical column.
    if "pred_v42_redlight_zone_preconfirm" not in out.columns:
        out["pred_v42_redlight_zone_preconfirm"] = out[out_current_col]
    return out
