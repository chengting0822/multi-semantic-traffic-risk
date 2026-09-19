#!/usr/bin/env python3
"""Clean v44/v45 overspeed-source and redlight-speed consistency policies.

This module distills the accepted prediction logic from the historical v44
and v45 scripts.  It intentionally excludes label rebuilding and report-only
diagnostics.

Formal chain:
    pred_v42_redlight_zone_preconfirm
    -> v92_v44_overspeed_event_source_refresh_current
    -> v92_v44c_zero_score_smooth_confirmed_refresh_current
    -> v92_v45_redlight_speed_ge90_consistency_current

The broad v44 current is reproduced for equivalence and for the v44c feature
refresh.  The downstream accepted source is v44c, then v45.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


GROUP = ["split", "case_key", "video_id", "track_id"]
R46_SPEED_EVENT_MIN_DURATION_SEC = 1.30


@dataclass(frozen=True)
class CleanV44V45OverspeedRedlightColumns:
    v44_current: str = "v92_v44_overspeed_event_source_refresh_current"
    v44c_current: str = "v92_v44c_zero_score_smooth_confirmed_refresh_current"
    current: str = "v92_v45_redlight_speed_ge90_consistency_current"

    v44_overspeed_event_score: str = "v92_v44_overspeed_event_score_v8"
    v44_r32_midband: str = "v92_v44_r32_overspeed_midband_strength_v8"
    v44_r32_ge2_duration: str = "v92_v44_r32_overspeed_ge2_duration_strength_v8"
    v44_r32_reliable: str = "v92_v44_r32_overspeed_reliable_signal_v8"
    v44_r32_rescue: str = "v92_v44_r32_overspeed_rescue_signal_v8"
    v44_severe_support: str = "v92_v44_overspeed_severe_support_v8"
    v44_speed_ge90_event: str = "v92_v44_speed_ge90_event_v8"
    v44_speed_ge110_event: str = "v92_v44_speed_ge110_event_v8"
    v44_speed_risk2_preconfirm: str = "v92_v44_speed_risk2_preconfirm_v8"
    v44_overspeed_source_gap: str = "v92_v44_overspeed_event_source_gap"
    v44_promote_risk2: str = "v92_v44_promote_risk2_from_v8_overspeed"
    v44_promote_risk3: str = "v92_v44_promote_risk3_from_v8_overspeed"
    v44_changed: str = "v92_v44_changed"

    v44c_repair_scope: str = "v92_v44c_repair_scope"
    v44c_bad_quality_reason: str = "v92_v44c_bad_quality_reason"
    v44c_promote_risk2: str = "v92_v44c_promote_risk2"
    v44c_promote_risk3: str = "v92_v44c_promote_risk3"
    v44c_changed: str = "v92_v44c_changed"

    reason: str = "v92_v45_redlight_speed_ge90_reason"
    redlight_speed_context: str = "v92_v45_redlight_speed_context"
    stable_speed_ge90: str = "v92_v45_stable_speed_ge90"
    explicit_wrongway_combo: str = "v92_v45_explicit_wrongway_combo"
    explicit_trajectory_combo: str = "v92_v45_explicit_trajectory_combo"
    severe_speed_confirmed: str = "v92_v45_severe_speed_confirmed"
    redlight_speed_ge90_cap: str = "v92_v45_redlight_speed_ge90_cap"
    changed: str = "v92_v45_changed"


COLS = CleanV44V45OverspeedRedlightColumns()


def num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default).astype(float)


def max_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    parts = [num(frame, col, np.nan) for col in columns]
    if not parts:
        return pd.Series(0.0, index=frame.index, dtype=float)
    return pd.concat(parts, axis=1).max(axis=1).fillna(0.0)


def text(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series("", index=frame.index, dtype=str)
    return frame[col].fillna("").astype(str)


def clip01(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)


def sigmoid(series: pd.Series, *, center: float, scale: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return clip01(1.0 / (1.0 + np.exp(-(values - center) / max(scale, 1e-6))))


def banded_speed_strength(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    result = pd.Series(0.0, index=values.index, dtype=float)
    risk1 = (values > 70.0) & (values <= 90.0)
    risk2 = (values > 90.0) & (values <= 110.0)
    risk3 = values > 110.0
    result.loc[risk1] = ((values.loc[risk1] - 70.0) / 20.0) * (1.0 / 3.0)
    result.loc[risk2] = (1.0 / 3.0) + ((values.loc[risk2] - 90.0) / 20.0) * (1.0 / 3.0)
    result.loc[risk3] = ((2.0 / 3.0) + ((values.loc[risk3] - 110.0) / 20.0) * (1.0 / 3.0)).clip(
        upper=1.0
    )
    return clip01(result)


def _sort(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.copy().sort_values(GROUP + ["ts_window_idx"], kind="mergesort").reset_index(drop=True)


def _group_cummax_current(
    out: pd.DataFrame,
    current_col: str,
    promote_risk2: pd.Series,
    promote_risk3: pd.Series,
) -> pd.Series:
    current = num(out, current_col).astype(int).clip(0, 2)
    result = current.copy()
    for _, idx in out.groupby(GROUP, sort=False).groups.items():
        idx = list(idx)
        vals = result.loc[idx].to_numpy(dtype=int)
        p2 = promote_risk2.loc[idx].to_numpy(dtype=bool)
        p3 = promote_risk3.loc[idx].to_numpy(dtype=bool)
        vals[p2] = np.maximum(vals[p2], 1)
        vals[p3] = 2
        vals = np.maximum.accumulate(vals)
        result.loc[idx] = vals
    return result.astype(int)


def add_clean_v44_refreshed_overspeed_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Recompute official v8-derived overspeed features used by v44/v44c."""

    out = frame.copy()
    speed = num(out, "v44_v8_speed_kmh_p95")
    ge2 = num(out, "v44_v8_overspeed_level_ge2_duration")
    ge3 = num(out, "v44_v8_overspeed_level_ge3_duration")
    reliability = clip01(num(out, "v44_v8_overspeed_reliability"))
    quality = clip01(num(out, "v44_v8_overspeed_signal_quality"))
    reliable = clip01(reliability * quality)

    strength = banded_speed_strength(speed)
    persistence = sigmoid(ge2, center=0.6667, scale=0.3333)
    support = clip01(0.72 * (0.60 * reliability + 0.40 * quality) + 0.28 * persistence)
    refreshed_event = clip01(strength * np.sqrt(0.08 + 0.92 * support))

    midband = clip01((speed - 90.0) / 20.0)
    duration = sigmoid(ge2, center=1.30, scale=0.40)
    rescue = clip01(midband * duration * reliable)
    severe = clip01((speed - 110.0) / 20.0) * reliable

    speed_ge90_active = (
        speed.ge(90.0)
        | num(out, "v44_v8_speed_band_90_110_flag").ge(0.5)
        | num(out, "v44_v8_speed_band_ge_110_flag").ge(0.5)
    )
    speed_ge110_active = speed.gt(110.0) | num(out, "v44_v8_speed_band_ge_110_flag").ge(0.5)
    speed_ge90_event = speed_ge90_active & ge2.ge(R46_SPEED_EVENT_MIN_DURATION_SEC) & reliable.ge(0.85)
    speed_ge110_event = speed_ge110_active & ge3.ge(R46_SPEED_EVENT_MIN_DURATION_SEC) & reliable.ge(0.85)

    risk2_preconfirm = refreshed_event.ge(0.55) & speed.ge(90.0) & reliable.ge(0.85)

    out[COLS.v44_overspeed_event_score] = refreshed_event
    out[COLS.v44_r32_midband] = midband
    out[COLS.v44_r32_ge2_duration] = duration
    out[COLS.v44_r32_reliable] = reliable
    out[COLS.v44_r32_rescue] = rescue
    out[COLS.v44_severe_support] = severe
    out[COLS.v44_speed_ge90_event] = speed_ge90_event.astype(int)
    out[COLS.v44_speed_ge110_event] = speed_ge110_event.astype(int)
    out[COLS.v44_speed_risk2_preconfirm] = risk2_preconfirm.astype(int)
    out[COLS.v44_overspeed_source_gap] = refreshed_event - num(out, "overspeed_event_score")

    # Historical column names are also populated when absent, because later
    # stages use the historical names as upstream semantic inputs.
    alias_map = {
        "v44_overspeed_event_score_v8": COLS.v44_overspeed_event_score,
        "v44_r32_overspeed_midband_strength_v8": COLS.v44_r32_midband,
        "v44_r32_overspeed_ge2_duration_strength_v8": COLS.v44_r32_ge2_duration,
        "v44_r32_overspeed_reliable_signal_v8": COLS.v44_r32_reliable,
        "v44_r32_overspeed_rescue_signal_v8": COLS.v44_r32_rescue,
        "v44_overspeed_severe_support_v8": COLS.v44_severe_support,
        "v44_speed_ge90_event_v8": COLS.v44_speed_ge90_event,
        "v44_speed_ge110_event_v8": COLS.v44_speed_ge110_event,
        "v44_speed_risk2_preconfirm_v8": COLS.v44_speed_risk2_preconfirm,
        "v44_overspeed_event_source_gap": COLS.v44_overspeed_source_gap,
    }
    for dst, src in alias_map.items():
        if dst not in out.columns:
            out[dst] = out[src]
    return out


def apply_clean_v44_v45_overspeed_redlight_policy(
    frame: pd.DataFrame,
    *,
    base_current_col: str = "pred_v42_redlight_zone_preconfirm",
    out_v44_current_col: str | None = None,
    out_v44c_current_col: str | None = None,
    out_current_col: str | None = None,
) -> pd.DataFrame:
    """Apply clean v44 broad refresh, accepted v44c repair, and v45 cap."""

    if base_current_col not in frame.columns:
        raise KeyError(f"base current column not found: {base_current_col}")

    out = _sort(frame)
    out_v44_current_col = out_v44_current_col or COLS.v44_current
    out_v44c_current_col = out_v44c_current_col or COLS.v44c_current
    out_current_col = out_current_col or COLS.current

    base = num(out, base_current_col).astype(int).clip(0, 2)
    out = add_clean_v44_refreshed_overspeed_features(out)

    # v44 broad refresh.  This stage is reproduced for equivalence, but the
    # accepted downstream source is the narrower v44c repair below.
    out[out_v44_current_col] = base
    v44_risk2_pre = num(out, COLS.v44_speed_risk2_preconfirm).ge(1.0)
    v44_risk3_event = num(out, COLS.v44_speed_ge110_event).ge(1.0)
    v44_promote_risk2 = base.lt(1) & v44_risk2_pre
    v44_promote_risk3 = base.lt(2) & v44_risk3_event
    out[COLS.v44_promote_risk2] = v44_promote_risk2.astype(int)
    out[COLS.v44_promote_risk3] = v44_promote_risk3.astype(int)
    out[out_v44_current_col] = _group_cummax_current(
        out,
        out_v44_current_col,
        v44_promote_risk2,
        v44_promote_risk3,
    )
    out[COLS.v44_changed] = out[out_v44_current_col].astype(int).ne(base).astype(int)

    # v44c accepted repair.  It starts again from the v42 base current and only
    # repairs rows where the old overspeed score was missing while official v8
    # and smoothed speed both support high overspeed.
    out[out_v44c_current_col] = base
    reason = text(out, "v44_sidecar_raw_recompute_rejected_reason")
    bad_quality = reason.str.contains("short_track|oscillatory|stationary|tiny|small_bbox", case=False, regex=True)
    source_gap = num(out, COLS.v44_overspeed_event_score) - num(out, "overspeed_event_score")
    original_missing = num(out, "overspeed_event_score").le(0.05)
    smooth_ge110 = num(out, "v44_sidecar_speed_smoothed_kmh_p95").ge(110.0)
    official_speed_ge90 = num(out, "v44_v8_speed_kmh_p95").ge(90.0)
    official_speed_ge110 = num(out, "v44_v8_speed_kmh_p95").gt(110.0)
    reliable = num(out, COLS.v44_r32_reliable).ge(0.85)
    event_score = num(out, COLS.v44_overspeed_event_score)
    ge3_duration = num(out, "v44_v8_overspeed_level_ge3_duration")

    repair_scope = original_missing & source_gap.ge(0.50) & smooth_ge110 & reliable & ~bad_quality
    v44c_risk2 = repair_scope & official_speed_ge90 & event_score.ge(0.55)
    v44c_risk3 = repair_scope & official_speed_ge110 & ge3_duration.ge(1.30)

    out[COLS.v44c_repair_scope] = repair_scope.astype(int)
    out[COLS.v44c_bad_quality_reason] = bad_quality.astype(int)
    out[COLS.v44c_promote_risk2] = (base.lt(1) & v44c_risk2).astype(int)
    out[COLS.v44c_promote_risk3] = (base.lt(2) & v44c_risk3).astype(int)
    out[out_v44c_current_col] = _group_cummax_current(out, out_v44c_current_col, v44c_risk2, v44c_risk3)
    out[COLS.v44c_changed] = out[out_v44c_current_col].astype(int).ne(base).astype(int)

    # Historical aliases used by downstream clean stages if the input table does
    # not already contain them.
    if "pred_v44_overspeed_event_source_refresh" not in out.columns:
        out["pred_v44_overspeed_event_source_refresh"] = out[out_v44_current_col]
    if "pred_v44c_zero_score_smooth_confirmed_refresh" not in out.columns:
        out["pred_v44c_zero_score_smooth_confirmed_refresh"] = out[out_v44c_current_col]

    # v45 accepted severity cap.
    current = num(out, out_v44c_current_col).astype(int).clip(0, 2)
    redlight = num(out, "redlight_event_score")
    overspeed = num(out, "overspeed_event_score")
    smooth = max_numeric(out, ["speed_smoothed_kmh_p95", "v44_sidecar_speed_smoothed_kmh_p95"])
    raw = max_numeric(
        out,
        [
            "computed_speed_kmh_p95",
            "v44_sidecar_computed_speed_kmh_p95",
            "v44_v8_speed_kmh_max",
        ],
    )
    reliable_any = max_numeric(
        out,
        [
            "r32_overspeed_reliable_signal",
            COLS.v44_r32_reliable,
            "v44_v8_overspeed_reliability",
        ],
    )

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
    severe_speed_confirmed = smooth.ge(110.0) | (raw.ge(130.0) & smooth.ge(100.0))
    stable_ge90 = smooth.ge(90.0)
    redlight_speed_context = redlight.ge(0.45) & overspeed.ge(0.50) & reliable_any.ge(0.85)
    no_other_risk3_semantic = ~(explicit_wrongway_combo | explicit_trajectory_combo | severe_speed_confirmed)
    cap = current.ge(2) & redlight_speed_context & ~stable_ge90 & no_other_risk3_semantic

    capped = current.copy()
    capped.loc[cap] = 1

    out[COLS.redlight_speed_context] = redlight_speed_context.astype(int)
    out[COLS.stable_speed_ge90] = stable_ge90.astype(int)
    out[COLS.explicit_wrongway_combo] = explicit_wrongway_combo.astype(int)
    out[COLS.explicit_trajectory_combo] = explicit_trajectory_combo.astype(int)
    out[COLS.severe_speed_confirmed] = severe_speed_confirmed.astype(int)
    out[COLS.redlight_speed_ge90_cap] = cap.astype(int)
    out[COLS.changed] = capped.ne(current).astype(int)
    out[COLS.reason] = np.where(cap, "v45_redlight_speed_ge90_cap_to_risk2", "")
    out[out_current_col] = capped.astype(int)
    return out
