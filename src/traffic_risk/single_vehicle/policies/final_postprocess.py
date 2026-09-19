#!/usr/bin/env python3
"""Final postprocess for the v92 clean single-vehicle risk pipeline.

This module combines the extracted clean semantic policies and applies the
last consistency rules before track cumulative max.

It intentionally does not compute cumulative max inside the policy function.
The caller should apply cumulative max once, after all current-window decisions
are complete.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .overspeed import apply_clean_overspeed_policy
from .redlight import apply_clean_redlight_policy
from .trajectory import apply_clean_trajectory_policy


@dataclass(frozen=True)
class CleanFinalColumns:
    current: str = "v92_clean_pipeline_current"
    reason: str = "v92_clean_pipeline_reason"
    trajectory_current: str = "v92_clean_pipeline_after_trajectory"
    redlight_current: str = "v92_clean_pipeline_after_redlight"
    overspeed_current: str = "v92_clean_pipeline_after_overspeed"
    ge90_wrongway_combo_risk3: str = "v92_final_ge90_wrongway_combo_risk3"
    speed_severity_cap: str = "v92_final_speed_severity_cap"
    track_entry_speed_artifact_cap: str = "v92_final_track_entry_speed_artifact_cap"
    trajectory_only_risk3_cap: str = "v92_final_trajectory_only_risk3_cap"
    weak_track_entry_trajectory_cap: str = "v92_final_weak_track_entry_trajectory_cap"


COLS = CleanFinalColumns()


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


def merge_reason_columns(frame: pd.DataFrame, cols: list[str]) -> pd.Series:
    merged = pd.Series("", index=frame.index, dtype=object)
    for col in cols:
        if col not in frame.columns:
            continue
        values = frame[col].fillna("").astype(str)
        mask = values.ne("")
        merged.loc[mask] = np.where(
            merged.loc[mask].astype(str).eq(""),
            values.loc[mask],
            merged.loc[mask].astype(str) + ";" + values.loc[mask],
        )
    return merged


def apply_clean_final_postprocess(
    frame: pd.DataFrame,
    *,
    base_current_col: str = "pred_v68b_semantic_policy_current",
    out_current_col: str | None = None,
    out_reason_col: str | None = None,
) -> pd.DataFrame:
    """Apply the v92 clean pipeline current-risk policy.

    Values use SchemaC:
    0 = risk0+1, 1 = risk2, 2 = risk3.
    """

    out = frame.copy()
    out_current_col = out_current_col or COLS.current
    out_reason_col = out_reason_col or COLS.reason

    if base_current_col not in out.columns:
        raise KeyError(f"base current column not found: {base_current_col}")

    out = apply_clean_trajectory_policy(
        out,
        base_current_col=base_current_col,
        out_current_col=COLS.trajectory_current,
        out_reason_col="v92_clean_pipeline_trajectory_reason",
    )
    out = apply_clean_redlight_policy(
        out,
        base_current_col=COLS.trajectory_current,
        out_current_col=COLS.redlight_current,
        out_reason_col="v92_clean_pipeline_redlight_reason",
    )
    out = apply_clean_overspeed_policy(
        out,
        base_current_col=COLS.redlight_current,
        out_current_col=COLS.overspeed_current,
        out_reason_col="v92_clean_pipeline_overspeed_reason",
    )

    current = num(out, COLS.overspeed_current).astype(int).clip(0, 2)

    redlight_context = (
        num(out, "redlight_event_score").ge(0.50)
        | num(out, "v92_redlight_zone_preconfirm_risk2").ge(0.5)
        | num(out, "v92_redlight_combo_mature_risk3").ge(0.5)
        | num(out, "v92_redlight_speed85_mature_risk3").ge(0.5)
        | num(out, "v92_redlight_weaving_lateral_risk3").ge(0.5)
    )
    wrongway_context = (
        num(out, "v41_family_wrongway_preconfirm").ge(0.5)
        | num(out, "v41_family_wrongway_preconfirm_strict").ge(0.5)
        | num(out, "v41_family_wrongway_tail_preconfirm").ge(0.5)
        | num(out, "v51_strict_wrongway").ge(0.5)
        | num(out, "v51_strong_reverse_motion").ge(0.5)
        | num(out, "v76_wrongway_preconfirm_onset_risk2").ge(0.5)
        | num(out, "v87_wrongway_tail_preconfirm_risk2").ge(0.5)
        | num(out, "v88_refined_p30_micro_tail_wrongway").ge(0.5)
    )
    weaving_context = (
        num(out, "trajectory_weaving_clean_support_v65").ge(0.5)
        | num(out, "trajectory_true_weaving_tail_preconfirm_v60").ge(0.5)
        | num(out, "trajectory_weaving_preconfirm_risk2_support_v60").ge(0.5)
        | num(out, "v48_tail_lane_backtrack_weaving_onset").ge(0.5)
    )
    speed_risk3_context = (
        num(out, "v92_overspeed_stable_ge110_risk3").ge(0.5)
        | num(out, "v92_overspeed_raw_sustained_ge110_risk3").ge(0.5)
        | num(out, "v92_overspeed_ge90_wrongway_combo_risk3").ge(0.5)
        | num(out, "v38_stable_ge110_support").ge(0.5)
        | num(out, "v68b_speed110_clean_risk3").ge(0.5)
    )

    # v77: emerging >=90 + strict wrongway is a compound risk3 signal.
    ge90_wrongway_combo_risk3 = (
        current.eq(1)
        & (
            num(out, "v77_emerging_ge90_strict_wrongway_combo").ge(0.5)
            | num(out, "v77_ge90_wrongway_combo_promote_risk3").ge(0.5)
        )
        & wrongway_context
    )

    # v85: speed-only high risk must be mature.  If the historical speed
    # severity cap already identified the current window, keep the cap here.
    speed_severity_cap = (
        current.eq(2)
        & num(out, "v85_speed_severity_cap").ge(0.5)
        & ~redlight_context
        & ~wrongway_context
        & ~weaving_context
    )

    reject_reason = (
        text(out, "v65_official_reject_reason")
        + ";"
        + text(out, "overspeed_raw_recompute_rejected_reason_v60")
        + ";"
        + text(out, "v44_sidecar_raw_recompute_rejected_reason")
    )
    speed_maturity = text(out, "v65_speed_onset_maturity")
    track_entry_speed_artifact_review = (
        current.eq(2)
        & num(out, "ts_window_idx").eq(0)
        & (
            speed_maturity.str.contains("raw_spike_bbox_or_center_jitter", case=False, regex=False)
            | reject_reason.str.contains("track_entry_ipm_perspective", case=False, regex=True)
            | num(out, "v44_sidecar_track_entry_ipm_perspective_suspect").ge(1)
        )
        & num(out, "prob_class2").lt(0.30)
        & ~wrongway_context
        & ~weaving_context
    )

    composite_risk3_context = (
        redlight_context
        | speed_risk3_context
        | ge90_wrongway_combo_risk3
        | num(out, "v38_redlight_trajectory_risk3_support").ge(0.5)
        | num(out, "v38_redlight_wrongway_risk3_support").ge(0.5)
        | num(out, "v38_overspeed_ge90_trajectory_risk3_support").ge(0.5)
    )
    trajectory_only_risk3_review = (
        current.eq(2)
        & num(out, "v38_trajectory_only_context").ge(0.5)
        & ~composite_risk3_context
    )

    weak_track_entry_trajectory_cap = (
        current.eq(1)
        & ~redlight_context
        & ~speed_risk3_context
        & num(out, "computed_speed_kmh_p95").lt(90.0)
        & num(out, "v65_raw_ge90_ratio").lt(0.10)
        & num(out, "v38_trajectory_risk2_support").gt(0.0)
        & num(out, "v38_trajectory_risk2_support").le(0.35)
        & num(out, "v41_family_wrongway_preconfirm_strict").lt(0.5)
        & num(out, "v51_strict_wrongway").lt(0.5)
        & num(out, "v51_strong_reverse_motion").lt(0.5)
        & num(out, "trajectory_weaving_clean_support_v65").lt(0.5)
        & num(out, "trajectory_true_weaving_tail_preconfirm_v60").lt(0.5)
        & reject_reason.str.contains("track_entry_ipm_perspective", case=False, regex=True)
    )

    updated = current.copy()
    updated.loc[ge90_wrongway_combo_risk3] = 2
    updated.loc[speed_severity_cap] = np.minimum(updated.loc[speed_severity_cap], 1)
    updated.loc[weak_track_entry_trajectory_cap] = 0

    signals = [
        (ge90_wrongway_combo_risk3, "ge90_wrongway_combo_risk3"),
        (speed_severity_cap, "speed_severity_cap"),
        (weak_track_entry_trajectory_cap, "weak_track_entry_trajectory_cap"),
    ]

    out[COLS.ge90_wrongway_combo_risk3] = ge90_wrongway_combo_risk3.astype(int)
    out[COLS.speed_severity_cap] = speed_severity_cap.astype(int)
    # These two remain diagnostic flags.  They are intentionally not applied
    # because a broad final cap demotes accepted high-speed pressure samples
    # and redlight/wrongway combo windows.  They should be resolved upstream,
    # not hidden in final postprocess.
    out[COLS.track_entry_speed_artifact_cap] = track_entry_speed_artifact_review.astype(int)
    out[COLS.trajectory_only_risk3_cap] = trajectory_only_risk3_review.astype(int)
    out[COLS.weak_track_entry_trajectory_cap] = weak_track_entry_trajectory_cap.astype(int)
    out[out_reason_col] = clean_reason(signals)
    out["v92_clean_pipeline_all_reason"] = merge_reason_columns(
        out,
        [
            "v92_clean_pipeline_trajectory_reason",
            "v92_clean_pipeline_redlight_reason",
            "v92_clean_pipeline_overspeed_reason",
            out_reason_col,
        ],
    )
    out[out_current_col] = updated.astype(int)
    return out
