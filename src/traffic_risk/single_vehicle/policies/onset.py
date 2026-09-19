#!/usr/bin/env python3
"""Clean v41f-lite connector reverse / risk3 cap / slow wrongway policy.

This module distills the accepted inference chain:

    pred_v41f_tail_half_guarded_probe
    -> pred_v41f_lite_connector_strong_reverse
    -> pred_v41f_lite_connector_onset_strong_reverse
    -> pred_v41f_lite_onset_risk3_consistency_cap_v1d_combo_anchor
    -> pred_v41f_lite_onset_v1e_slow_wrongway

It intentionally excludes metrics, labels, case review tables, and rejected
diagnostic variants.  The policy remains post-GRU / semantic-postprocess only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


GROUP = ["split", "case_key", "video_id", "track_id"]


@dataclass(frozen=True)
class CleanV41fLiteOnsetColumns:
    connector_current: str = "v92_v41f_lite_connector_strong_reverse_current"
    onset_current: str = "v92_v41f_lite_connector_onset_strong_reverse_current"
    cap_v1_current: str = "v92_v41f_lite_onset_risk3_consistency_cap_v1_current"
    cap_v1b_current: str = "v92_v41f_lite_onset_risk3_consistency_cap_v1b_strict_current"
    cap_v1c_current: str = "v92_v41f_lite_onset_risk3_consistency_cap_v1c_anchor_current"
    cap_v1d_current: str = "v92_v41f_lite_onset_risk3_consistency_cap_v1d_combo_anchor_current"
    current: str = "v92_v41f_lite_onset_v1e_slow_wrongway_current"
    reason: str = "v92_v41f_lite_onset_reason"

    connector_adjacent_context: str = "v92_v41f_lite_connector_adjacent_context"
    strong_reverse_current: str = "v92_v41f_lite_strong_reverse_current"
    strong_reverse_run2: str = "v92_v41f_lite_strong_reverse_run2"
    onset_strong_reverse: str = "v92_v41f_lite_onset_strong_reverse"
    connector_applied: str = "v92_v41f_lite_applied_current"
    onset_applied: str = "v92_v41f_lite_onset_applied_current"
    connector_changed: str = "v92_v41f_lite_changed"
    onset_changed: str = "v92_v41f_lite_onset_changed"

    cap_v1_strong_semantic: str = "v92_risk3_cap_v1_strong_risk3_semantic"
    cap_v1_raw_speed_spike: str = "v92_risk3_cap_v1_raw_speed_spike"
    cap_v1_moderate_speed_only: str = "v92_risk3_cap_v1_moderate_speed_only"
    cap_v1_trajectory_only: str = "v92_risk3_cap_v1_trajectory_only"
    cap_v1_gru_only: str = "v92_risk3_cap_v1_gru_only"
    cap_v1_applied: str = "v92_risk3_cap_v1_applied"
    cap_v1_changed: str = "v92_risk3_cap_v1_changed"

    cap_v1b_lower_confidence: str = "v92_risk3_cap_v1b_lower_confidence_risk3"
    cap_v1b_low_smooth: str = "v92_risk3_cap_v1b_low_smooth_speed"
    cap_v1b_not_extreme_raw: str = "v92_risk3_cap_v1b_not_extreme_raw"
    cap_v1b_no_redlight: str = "v92_risk3_cap_v1b_no_redlight"
    cap_v1b_no_strict_wrongway: str = "v92_risk3_cap_v1b_no_strict_wrongway"
    cap_v1b_no_explicit_combo: str = "v92_risk3_cap_v1b_no_explicit_risk3_combo"
    cap_v1b_base_condition: str = "v92_risk3_cap_v1b_base_condition"
    cap_v1b_applied: str = "v92_risk3_cap_v1b_applied"
    cap_v1b_changed: str = "v92_risk3_cap_v1b_changed"

    cap_v1c_anchor_current: str = "v92_risk3_cap_v1c_anchor_current"
    cap_v1c_anchor_seen: str = "v92_risk3_cap_v1c_anchor_seen"
    cap_v1c_applied: str = "v92_risk3_cap_v1c_applied"
    cap_v1c_changed: str = "v92_risk3_cap_v1c_changed"

    cap_v1d_combo_anchor_current: str = "v92_risk3_cap_v1d_combo_anchor_current"
    cap_v1d_combo_anchor_seen: str = "v92_risk3_cap_v1d_combo_anchor_seen"
    cap_v1d_applied: str = "v92_risk3_cap_v1d_applied"
    cap_v1d_changed: str = "v92_risk3_cap_v1d_changed"

    v1e_slow_wrongway_onset: str = "v92_v41f_v1e_slow_wrongway_onset"
    v1e_changed: str = "v92_v41f_v1e_changed"


COLS = CleanV41fLiteOnsetColumns()


def num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default).astype(float)


def text(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series("", index=frame.index, dtype=str)
    return frame[col].fillna("").astype(str)


def _sort(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.copy().sort_values(GROUP + ["ts_window_idx"], kind="mergesort").reset_index(drop=True)


def _cummax_by_track(out: pd.DataFrame, source_col: str) -> pd.Series:
    result = num(out, source_col).astype(int).clip(0, 2)
    for _, idx in out.groupby(GROUP, sort=False).groups.items():
        idx = list(idx)
        vals = result.loc[idx].to_numpy(dtype=int)
        result.loc[idx] = np.maximum.accumulate(vals)
    return result.astype(int)


def _apply_connector_strong_reverse(
    frame: pd.DataFrame,
    *,
    base_current_col: str,
    connector_current_col: str,
    onset_current_col: str,
) -> pd.DataFrame:
    out = _sort(frame)
    base = num(out, base_current_col).astype(int).clip(0, 2)
    out[connector_current_col] = base
    out[onset_current_col] = base

    lane_seq = text(out, "v28_lane_sequence_smooth_6w")
    transition = text(out, "v28_transition_type_6w")
    connector_context = (
        transition.eq("connector_adjacent_route")
        | lane_seq.str.contains("new_lane_8>downbound_1", regex=False)
        | lane_seq.str.contains("upbound_3>downbound_1", regex=False)
    )
    strong_reverse_current = (
        text(out, "v41_family_top").isin(["upbound", "downbound"])
        & num(out, "v41_family_top_ratio").ge(0.90)
        & num(out, "v41_family_confidence").ge(0.90)
        & num(out, "v41_family_hit_ratio").ge(0.90)
        & num(out, "v41_family_motion_cos_top", default=1.0).le(-0.90)
        & num(out, "v41_bbox_area_cv", default=9.0).le(0.25)
        & num(out, "v41_motion_mag_px").ge(7.0)
    )

    out[COLS.connector_adjacent_context] = connector_context.astype(int)
    out[COLS.strong_reverse_current] = strong_reverse_current.astype(int)
    out[COLS.strong_reverse_run2] = 0
    out[COLS.onset_strong_reverse] = 0
    out[COLS.connector_applied] = 0
    out[COLS.onset_applied] = 0

    for _, idx in out.groupby(GROUP, sort=False).groups.items():
        idx = list(idx)
        connector = out.loc[idx, COLS.connector_adjacent_context].astype(bool).to_numpy()
        strong = out.loc[idx, COLS.strong_reverse_current].astype(bool).to_numpy()
        current = connector & strong
        prev = np.concatenate([[False], current[:-1]])
        run2 = current & prev
        prev_connector = np.concatenate([[False], connector[:-1]])
        very_strong = (
            current
            & prev_connector
            & (num(out.loc[idx], "v41_family_motion_cos_top", default=1.0).to_numpy() <= -0.94)
        )

        out.loc[idx, COLS.strong_reverse_run2] = run2.astype(int)
        out.loc[idx, COLS.onset_strong_reverse] = very_strong.astype(int)

        vals = out.loc[idx, connector_current_col].to_numpy(dtype=int)
        apply = run2 & (vals < 1)
        out.loc[idx, COLS.connector_applied] = apply.astype(int)
        vals[apply] = np.maximum(vals[apply], 1)
        vals = np.maximum.accumulate(vals)
        out.loc[idx, connector_current_col] = vals

        onset_vals = out.loc[idx, onset_current_col].to_numpy(dtype=int)
        onset_apply = very_strong & (onset_vals < 1)
        out.loc[idx, COLS.onset_applied] = onset_apply.astype(int)
        onset_vals[onset_apply] = np.maximum(onset_vals[onset_apply], 1)
        onset_vals = np.maximum.accumulate(onset_vals)
        out.loc[idx, onset_current_col] = onset_vals

    out[COLS.connector_changed] = out[connector_current_col].astype(int).ne(base).astype(int)
    out[COLS.onset_changed] = out[onset_current_col].astype(int).ne(base).astype(int)
    return out


def _apply_risk3_consistency_cap(
    frame: pd.DataFrame,
    *,
    base_current_col: str,
    cap_v1_col: str,
    cap_v1b_col: str,
    cap_v1c_col: str,
    cap_v1d_col: str,
) -> pd.DataFrame:
    out = _sort(frame)
    base = num(out, base_current_col).astype(int).clip(0, 2)
    out[cap_v1_col] = base
    out[cap_v1b_col] = base
    out[cap_v1c_col] = base
    out[cap_v1d_col] = base

    redlight = num(out, "redlight_event_score")
    overspeed = num(out, "overspeed_event_score")
    computed_p95 = num(out, "computed_speed_kmh_p95")
    smooth_p95 = num(out, "speed_smoothed_kmh_p95")
    raw_smooth_gap = computed_p95 - smooth_p95
    prob2 = num(out, "prob_class2")

    stable_ge110 = num(out, "v38_stable_ge110_support").ge(0.5) | smooth_p95.ge(110.0)
    redlight_wrongway = num(out, "v38_redlight_wrongway_risk3_support").ge(0.5)
    redlight_traj = (
        num(out, "v38_redlight_trajectory_risk3_support").ge(0.65)
        & redlight.ge(0.45)
        & (
            num(out, "s33_prefix_state_confirmed").ge(0.5)
            | num(out, "s33_current_local_semantic_support").ge(0.75)
            | num(out, "v41_family_wrongway_preconfirm_strict").ge(0.5)
        )
    )
    confirmed_wrongway = num(out, "v41_family_wrongway_preconfirm_strict").ge(0.5) & num(
        out, "s33_prefix_state_confirmed"
    ).ge(0.5)
    strong_risk3_semantic = stable_ge110 | redlight_wrongway | redlight_traj | confirmed_wrongway

    raw_speed_spike = computed_p95.ge(110.0) & smooth_p95.lt(100.0) & raw_smooth_gap.ge(25.0)
    moderate_speed_only = (
        computed_p95.ge(90.0)
        & smooth_p95.lt(110.0)
        & redlight.lt(0.45)
        & num(out, "v41_family_wrongway_preconfirm_strict").lt(0.5)
    )
    trajectory_only = num(out, "v38_trajectory_only_context").ge(0.5) & redlight.lt(0.45) & overspeed.lt(0.75)
    gru_only = (
        num(out, "r46_risk3_policy").lt(0.5)
        & num(out, "v38_policy_risk3_support").lt(0.5)
        & prob2.ge(0.40)
        & redlight.lt(0.45)
    )

    cap_context = raw_speed_spike | moderate_speed_only | trajectory_only | gru_only
    cap = base.ge(2) & ~strong_risk3_semantic & cap_context

    out[COLS.cap_v1_strong_semantic] = strong_risk3_semantic.astype(int)
    out[COLS.cap_v1_raw_speed_spike] = raw_speed_spike.astype(int)
    out[COLS.cap_v1_moderate_speed_only] = moderate_speed_only.astype(int)
    out[COLS.cap_v1_trajectory_only] = trajectory_only.astype(int)
    out[COLS.cap_v1_gru_only] = gru_only.astype(int)
    out[COLS.cap_v1_applied] = cap.astype(int)
    out.loc[cap, cap_v1_col] = 1
    out[COLS.cap_v1_changed] = out[cap_v1_col].astype(int).ne(base).astype(int)

    no_redlight = redlight.lt(0.45)
    no_strict_wrongway = num(out, "v41_family_wrongway_preconfirm_strict").lt(0.5)
    low_smooth_speed = smooth_p95.lt(90.0)
    not_extreme_raw = computed_p95.lt(160.0)
    lower_confidence_risk3 = prob2.lt(0.85)
    no_explicit_risk3_combo = (
        num(out, "v38_redlight_wrongway_risk3_support").lt(0.5)
        & num(out, "v38_redlight_trajectory_risk3_support").lt(0.5)
        & num(out, "v38_stable_ge110_support").lt(0.5)
    )
    cap_strict = (
        base.ge(2)
        & lower_confidence_risk3
        & low_smooth_speed
        & not_extreme_raw
        & no_redlight
        & no_strict_wrongway
        & no_explicit_risk3_combo
    )

    out[COLS.cap_v1b_lower_confidence] = lower_confidence_risk3.astype(int)
    out[COLS.cap_v1b_low_smooth] = low_smooth_speed.astype(int)
    out[COLS.cap_v1b_not_extreme_raw] = not_extreme_raw.astype(int)
    out[COLS.cap_v1b_no_redlight] = no_redlight.astype(int)
    out[COLS.cap_v1b_no_strict_wrongway] = no_strict_wrongway.astype(int)
    out[COLS.cap_v1b_no_explicit_combo] = no_explicit_risk3_combo.astype(int)
    out[COLS.cap_v1b_base_condition] = cap_strict.astype(int)
    out[COLS.cap_v1b_applied] = cap_strict.astype(int)
    out.loc[cap_strict, cap_v1b_col] = 1
    out[COLS.cap_v1b_changed] = out[cap_v1b_col].astype(int).ne(base).astype(int)

    anchor_current = (
        (smooth_p95.ge(100.0) & computed_p95.ge(90.0) & prob2.ge(0.40))
        | redlight.ge(0.45)
        | num(out, "v41_family_wrongway_preconfirm_strict").ge(0.5)
        | num(out, "v38_redlight_wrongway_risk3_support").ge(0.5)
        | num(out, "v38_redlight_trajectory_risk3_support").ge(0.5)
    )
    out[COLS.cap_v1c_anchor_current] = anchor_current.astype(int)
    out[COLS.cap_v1c_anchor_seen] = 0
    for _, idx in out.groupby(GROUP, sort=False).groups.items():
        idx = list(idx)
        seen = out.loc[idx, COLS.cap_v1c_anchor_current].astype(bool).cummax().astype(int)
        out.loc[idx, COLS.cap_v1c_anchor_seen] = seen.to_numpy()
    cap_anchor = cap_strict & ~out[COLS.cap_v1c_anchor_seen].astype(bool)
    out[COLS.cap_v1c_applied] = cap_anchor.astype(int)
    out.loc[cap_anchor, cap_v1c_col] = 1
    out[COLS.cap_v1c_changed] = out[cap_v1c_col].astype(int).ne(base).astype(int)

    strict_wrongway = num(out, "v41_family_wrongway_preconfirm_strict").ge(0.5)
    credible_speed_combo = smooth_p95.ge(90.0) & computed_p95.ge(90.0) & overspeed.ge(0.5)
    combo_anchor_current = (
        (smooth_p95.ge(100.0) & computed_p95.ge(90.0) & prob2.ge(0.40))
        | num(out, "v38_redlight_wrongway_risk3_support").ge(0.5)
        | num(out, "v38_redlight_trajectory_risk3_support").ge(0.5)
        | redlight.ge(0.45)
        | (strict_wrongway & credible_speed_combo)
    )
    out[COLS.cap_v1d_combo_anchor_current] = combo_anchor_current.astype(int)
    out[COLS.cap_v1d_combo_anchor_seen] = 0
    for _, idx in out.groupby(GROUP, sort=False).groups.items():
        idx = list(idx)
        seen = out.loc[idx, COLS.cap_v1d_combo_anchor_current].astype(bool).cummax().astype(int)
        out.loc[idx, COLS.cap_v1d_combo_anchor_seen] = seen.to_numpy()
    cap_combo_anchor = cap_strict & ~out[COLS.cap_v1d_combo_anchor_seen].astype(bool)
    out[COLS.cap_v1d_applied] = cap_combo_anchor.astype(int)
    out.loc[cap_combo_anchor, cap_v1d_col] = 1
    out[COLS.cap_v1d_changed] = out[cap_v1d_col].astype(int).ne(base).astype(int)
    return out


def _apply_slow_wrongway_v1e(
    frame: pd.DataFrame,
    *,
    base_current_col: str,
    out_current_col: str,
) -> pd.DataFrame:
    out = _sort(frame)
    base = num(out, base_current_col).astype(int).clip(0, 2)
    out[out_current_col] = base

    family = text(out, "v41_family_top")
    transition = text(out, "v28_transition_type_6w")
    lane_seq = text(out, "v28_lane_sequence_smooth_6w")
    stable_straight_family = family.isin(["upbound", "downbound"])
    not_connector_route = ~(transition.isin(["connector_expected_route", "connector_adjacent_route"]) | lane_seq.str.contains(">", regex=False))
    slow_wrongway = (
        base.eq(0)
        & stable_straight_family
        & not_connector_route
        & num(out, "v41_raw_count").ge(14)
        & num(out, "v41_family_top_ratio").ge(0.98)
        & num(out, "v41_family_confidence").ge(0.98)
        & num(out, "v41_family_hit_ratio").ge(0.95)
        & num(out, "v41_family_motion_cos_top", default=1.0).le(-0.55)
        & num(out, "v41_motion_mag_px").ge(4.0)
        & num(out, "v41_motion_mag_px").lt(12.0)
        & num(out, "v41_bbox_area_cv", default=9.0).ge(0.08)
        & num(out, "v41_bbox_area_cv", default=9.0).le(0.25)
        & num(out, "v28_context_quality").ge(0.8)
        & num(out, "v28_bbox_area_log_jump_max", default=9.0).le(0.25)
    )
    out[COLS.v1e_slow_wrongway_onset] = slow_wrongway.astype(int)

    result = base.copy()
    for _, idx in out.groupby(GROUP, sort=False).groups.items():
        idx = list(idx)
        vals = result.loc[idx].to_numpy(dtype=int)
        onset = slow_wrongway.loc[idx].to_numpy(dtype=bool)
        vals[onset] = np.maximum(vals[onset], 1)
        vals = np.maximum.accumulate(vals)
        result.loc[idx] = vals

    out[out_current_col] = result.astype(int)
    out[COLS.v1e_changed] = out[out_current_col].astype(int).ne(base).astype(int)
    out[COLS.reason] = np.select(
        [
            out[COLS.v1e_slow_wrongway_onset].eq(1),
            out[COLS.cap_v1d_applied].eq(1),
            out[COLS.onset_applied].eq(1),
            out[COLS.connector_applied].eq(1),
        ],
        [
            "v41f_v1e_slow_wrongway_risk2",
            "v41f_risk3_consistency_cap_to_risk2",
            "v41f_connector_onset_strong_reverse_risk2",
            "v41f_connector_strong_reverse_run2_risk2",
        ],
        default="",
    )
    return out


def apply_clean_v41f_lite_onset_policy(
    frame: pd.DataFrame,
    *,
    base_current_col: str = "pred_v41f_tail_half_guarded_probe",
    out_current_col: str | None = None,
) -> pd.DataFrame:
    """Apply the accepted v41f-lite chain through v1e."""

    if base_current_col not in frame.columns:
        raise KeyError(f"base current column not found: {base_current_col}")

    out_current_col = out_current_col or COLS.current
    out = _apply_connector_strong_reverse(
        frame,
        base_current_col=base_current_col,
        connector_current_col=COLS.connector_current,
        onset_current_col=COLS.onset_current,
    )
    out = _apply_risk3_consistency_cap(
        out,
        base_current_col=COLS.onset_current,
        cap_v1_col=COLS.cap_v1_current,
        cap_v1b_col=COLS.cap_v1b_current,
        cap_v1c_col=COLS.cap_v1c_current,
        cap_v1d_col=COLS.cap_v1d_current,
    )
    out = _apply_slow_wrongway_v1e(out, base_current_col=COLS.cap_v1d_current, out_current_col=out_current_col)

    aliases = {
        "pred_v41f_lite_connector_strong_reverse": COLS.connector_current,
        "pred_v41f_lite_connector_onset_strong_reverse": COLS.onset_current,
        "pred_v41f_lite_onset_risk3_consistency_cap_v1": COLS.cap_v1_current,
        "pred_v41f_lite_onset_risk3_consistency_cap_v1b_strict": COLS.cap_v1b_current,
        "pred_v41f_lite_onset_risk3_consistency_cap_v1c_anchor": COLS.cap_v1c_current,
        "pred_v41f_lite_onset_risk3_consistency_cap_v1d_combo_anchor": COLS.cap_v1d_current,
        "pred_v41f_lite_onset_v1e_slow_wrongway": out_current_col,
    }
    for dst, src in aliases.items():
        if dst not in out.columns:
            out[dst] = out[src]
    return out
