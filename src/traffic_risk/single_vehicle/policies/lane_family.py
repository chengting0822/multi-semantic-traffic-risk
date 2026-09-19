#!/usr/bin/env python3
"""Clean v41 lane-family wrongway / tail-half onset policy.

This module distills the accepted inference chain:

    pred_v37e_v38_policy_cap
    -> pred_v41_probe
    -> pred_v41b_strict_probe
    -> pred_v41c_guarded_probe
    -> pred_v41d_turn_guarded_probe
    -> pred_v41e_directional_turn_guarded_probe
    -> pred_v41f_tail_half_guarded_probe

It assumes v41/v28 sidecar columns already exist in the input table.  It does
not rebuild frame-level lane candidates and does not touch labels or metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CleanV41LaneFamilyColumns:
    probe_current: str = "v92_v41_probe_current"
    strict_current: str = "v92_v41b_strict_probe_current"
    guarded_current: str = "v92_v41c_guarded_probe_current"
    turn_guarded_current: str = "v92_v41d_turn_guarded_probe_current"
    directional_turn_guarded_current: str = "v92_v41e_directional_turn_guarded_probe_current"
    current: str = "v92_v41f_tail_half_guarded_probe_current"
    reason: str = "v92_v41_lane_family_reason"

    probe_applied: str = "v92_v41_probe_applied"
    strict_applied: str = "v92_v41b_strict_probe_applied"
    occlusion_guard: str = "v92_v41c_occlusion_guard"
    legal_outer_turn_guard: str = "v92_v41c_legal_outer_turn_guard"
    guarded_applied: str = "v92_v41c_guarded_probe_applied"
    turn_lane_dominant_guard: str = "v92_v41d_turn_lane_dominant_guard"
    turn_guarded_applied: str = "v92_v41d_turn_guarded_probe_applied"
    downbound_turn_dominant_guard: str = "v92_v41e_downbound_turn_dominant_guard"
    directional_turn_guarded_applied: str = "v92_v41e_directional_turn_guarded_probe_applied"
    tail_boundary_switch: str = "v92_v41f_tail_boundary_switch"
    weak_semantic_support: str = "v92_v41f_weak_semantic_support"
    tail_half_applied: str = "v92_v41f_tail_half_guarded_probe_applied"


COLS = CleanV41LaneFamilyColumns()


def num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default).astype(float)


def text(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series("", index=frame.index, dtype=str)
    return frame[col].fillna("").astype(str)


def apply_clean_v41_lane_family_policy(
    frame: pd.DataFrame,
    *,
    base_current_col: str = "pred_v37e_v38_policy_cap",
    out_current_col: str | None = None,
) -> pd.DataFrame:
    """Apply the accepted v41 lane-family policy through v41f."""

    if base_current_col not in frame.columns:
        raise KeyError(f"base current column not found: {base_current_col}")

    out_current_col = out_current_col or COLS.current
    out = frame.copy()
    base = num(out, base_current_col).astype(int).clip(0, 2)

    risk2_support = (
        num(out, "r46_risk2_policy").ge(1.0)
        | num(out, "prob_class1").ge(0.28)
        | num(out, "s33_current_local_semantic_support").ge(0.10)
    )
    weak_semantic_support = (
        num(out, "r46_risk2_policy").ge(1.0)
        | num(out, "prob_class1").ge(0.20)
        | num(out, "s33_current_local_semantic_support").ge(0.08)
    )

    probe_apply = base.eq(0) & num(out, "v41_family_wrongway_preconfirm").ge(1.0) & risk2_support
    strict_apply = base.eq(0) & num(out, "v41_family_wrongway_preconfirm_strict").ge(1.0) & risk2_support

    out[COLS.probe_current] = base.copy()
    out.loc[probe_apply, COLS.probe_current] = 1
    out[COLS.strict_current] = base.copy()
    out.loc[strict_apply, COLS.strict_current] = 1
    out[COLS.probe_applied] = probe_apply.astype(int)
    out[COLS.strict_applied] = strict_apply.astype(int)

    seq = text(out, "v28_lane_sequence_smooth_6w")
    trans = text(out, "v28_transition_type_6w")
    area_jump = num(out, "v28_bbox_area_log_jump_max")
    area_cv_v28 = num(out, "v28_bbox_area_cv_raw")
    context_quality = num(out, "v28_context_quality")
    occlusion_guard = area_jump.ge(0.70) | (area_cv_v28.ge(0.30) & context_quality.le(0.05))

    turn_lane_ratio = num(out, "v41_turn_lane_score_ratio")
    top_lane_is_turn = num(out, "v41_top_lane_is_turn")
    legal_outer_turn_guard = (
        seq.str.contains("upbound_3>downbound_1", regex=False)
        | seq.str.contains("upbound_3>downbound_2", regex=False)
        | seq.str.contains("new_lane_8>downbound_1", regex=False)
        | seq.str.contains("new_lane_8>downbound_2", regex=False)
        | trans.eq("connector_adjacent_route")
        | trans.eq("connector_expected_route")
    )
    turn_lane_dominant_guard = top_lane_is_turn.ge(1.0) | turn_lane_ratio.ge(0.45)
    family_top = text(out, "v41_family_top")
    downbound_turn_dominant_guard = family_top.eq("downbound") & turn_lane_dominant_guard

    guarded_apply = strict_apply & ~occlusion_guard & ~legal_outer_turn_guard
    turn_guarded_apply = guarded_apply & ~turn_lane_dominant_guard
    directional_turn_guarded_apply = guarded_apply & ~downbound_turn_dominant_guard

    out[COLS.occlusion_guard] = occlusion_guard.astype(int)
    out[COLS.legal_outer_turn_guard] = legal_outer_turn_guard.astype(int)
    out[COLS.turn_lane_dominant_guard] = turn_lane_dominant_guard.astype(int)
    out[COLS.downbound_turn_dominant_guard] = downbound_turn_dominant_guard.astype(int)
    out[COLS.guarded_current] = base.copy()
    out.loc[guarded_apply, COLS.guarded_current] = 1
    out[COLS.turn_guarded_current] = base.copy()
    out.loc[turn_guarded_apply, COLS.turn_guarded_current] = 1
    out[COLS.directional_turn_guarded_current] = base.copy()
    out.loc[directional_turn_guarded_apply, COLS.directional_turn_guarded_current] = 1
    out[COLS.guarded_applied] = guarded_apply.astype(int)
    out[COLS.turn_guarded_applied] = turn_guarded_apply.astype(int)
    out[COLS.directional_turn_guarded_applied] = directional_turn_guarded_apply.astype(int)

    tail_family_top = text(out, "v41_tail_family_top")
    tail_family_second = text(out, "v41_tail_family_second")
    family_second = text(out, "v41_family_second")
    full_confidence = num(out, "v41_family_confidence")
    full_top_ratio = num(out, "v41_family_top_ratio")
    tail_boundary_switch = (
        tail_family_top.ne("")
        & tail_family_top.ne(family_top)
        & (
            tail_family_top.eq(family_second)
            | tail_family_second.eq(family_top)
            | full_confidence.le(0.18)
            | full_top_ratio.le(0.56)
        )
    )

    tail_half_apply = (
        out[COLS.directional_turn_guarded_current].astype(int).eq(0)
        & num(out, "v41_family_wrongway_tail_preconfirm").ge(1.0)
        & tail_boundary_switch
        & weak_semantic_support
        & ~occlusion_guard
        & ~legal_outer_turn_guard
        & ~downbound_turn_dominant_guard
    )

    out[COLS.tail_boundary_switch] = tail_boundary_switch.astype(int)
    out[COLS.weak_semantic_support] = weak_semantic_support.astype(int)
    out[out_current_col] = out[COLS.directional_turn_guarded_current].astype(int)
    out.loc[tail_half_apply, out_current_col] = 1
    out[COLS.tail_half_applied] = tail_half_apply.astype(int)
    out[COLS.reason] = np.select(
        [
            tail_half_apply,
            directional_turn_guarded_apply,
            turn_guarded_apply,
            guarded_apply,
            strict_apply,
            probe_apply,
        ],
        [
            "v41f_tail_half_wrongway_risk2",
            "v41e_directional_turn_guarded_wrongway_risk2",
            "v41d_turn_guarded_wrongway_risk2",
            "v41c_guarded_wrongway_risk2",
            "v41b_strict_wrongway_risk2",
            "v41_probe_wrongway_risk2",
        ],
        default="",
    )

    aliases = {
        "pred_v41_probe": COLS.probe_current,
        "pred_v41b_strict_probe": COLS.strict_current,
        "pred_v41c_guarded_probe": COLS.guarded_current,
        "pred_v41d_turn_guarded_probe": COLS.turn_guarded_current,
        "pred_v41e_directional_turn_guarded_probe": COLS.directional_turn_guarded_current,
        "pred_v41f_tail_half_guarded_probe": out_current_col,
    }
    for dst, src in aliases.items():
        if dst not in out.columns:
            out[dst] = out[src]
    return out
