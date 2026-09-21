#!/usr/bin/env python3
"""Clean v38 policy-severity feature generation.

This module distills the accepted v38 feature contract used by the v90 chain.
It rebuilds the eight `v38_*` semantic support columns from upstream evidence:

    official C4O / R46 source fields
    + v37 clean trajectory semantic sidecar
    -> v38 policy-severity support features

It does not train GRU, touch labels, apply prediction gates, or rebuild final
risk.  The output is only feature columns.  The following v38 retraining
candidate was explicitly rejected historically and is not reproduced here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from traffic_risk.identifiers import normalize_window_keys


KEY_COLUMNS = ["case_key", "video_id", "track_id", "ts_window_idx"]

R46_SPEED_EVENT_MIN_DURATION_SEC = 1.30
R46_REDLIGHT_MIN_DURATION_SEC = 0.3333
S33_PREFIX_CONFIRMED_COLUMN = "s33_prefix_state_confirmed"
S33_LOCAL_SUPPORT_COLUMN = "s33_current_local_semantic_support"


@dataclass(frozen=True)
class CleanV38FeatureColumns:
    trajectory_risk2_support: str = "v92_v38_trajectory_risk2_support"
    policy_risk2_support: str = "v92_v38_policy_risk2_support"
    policy_risk3_support: str = "v92_v38_policy_risk3_support"
    trajectory_only_context: str = "v92_v38_trajectory_only_context"
    stable_ge110_support: str = "v92_v38_stable_ge110_support"
    redlight_trajectory_risk3_support: str = "v92_v38_redlight_trajectory_risk3_support"
    redlight_wrongway_risk3_support: str = "v92_v38_redlight_wrongway_risk3_support"
    overspeed_ge90_trajectory_risk3_support: str = "v92_v38_overspeed_ge90_trajectory_risk3_support"


COLS = CleanV38FeatureColumns()

HISTORICAL_V38_COLUMNS: dict[str, str] = {
    COLS.trajectory_risk2_support: "v38_trajectory_risk2_support",
    COLS.policy_risk2_support: "v38_policy_risk2_support",
    COLS.policy_risk3_support: "v38_policy_risk3_support",
    COLS.trajectory_only_context: "v38_trajectory_only_context",
    COLS.stable_ge110_support: "v38_stable_ge110_support",
    COLS.redlight_trajectory_risk3_support: "v38_redlight_trajectory_risk3_support",
    COLS.redlight_wrongway_risk3_support: "v38_redlight_wrongway_risk3_support",
    COLS.overspeed_ge90_trajectory_risk3_support: "v38_overspeed_ge90_trajectory_risk3_support",
}

V37_REQUIRED_COLUMNS = [
    "v37_redlight_wrongway_combo_ready",
    "v37_current_true_weaving_event",
]

R46_COLUMNS = [
    "v92_r46_speed_ge90_event",
    "v92_r46_speed_90_110_event",
    "v92_r46_speed_ge110_event",
    "v92_r46_wrongway_event",
    "v92_r46_redlight_event",
    "v92_r46_risk3_speed_ge110_policy",
    "v92_r46_risk3_wrongway_redlight_policy",
    "v92_r46_risk3_wrongway_speed_ge90_policy",
    "v92_r46_risk3_redlight_speed_ge90_policy",
    "v92_r46_risk3_policy",
    "v92_r46_risk2_policy",
    "v92_r46_policy_severity_score",
]


def numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype(float)


def clip01(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)


def normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    return normalize_window_keys(frame)


def add_clean_r46_policy_flags(source_frame: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the official R46 policy flags from source-frame fields."""

    out = normalize_keys(source_frame)
    speed_p95 = numeric(out, "speed_kmh_p95")
    overspeed_level = numeric(out, "overspeed_level")
    overspeed_ge2_duration = numeric(out, "overspeed_level_ge2_duration")
    overspeed_ge3_duration = numeric(out, "overspeed_level_ge3_duration")
    overspeed_reliable_signal = clip01(
        clip01(numeric(out, "overspeed_reliability")) * clip01(numeric(out, "overspeed_signal_quality"))
    )

    speed_ge90_active = (
        speed_p95.ge(90.0)
        | numeric(out, "speed_band_90_110_flag").ge(0.5)
        | numeric(out, "speed_band_ge_110_flag").ge(0.5)
        | overspeed_level.ge(2.0)
    )
    speed_ge110_active = (
        speed_p95.gt(110.0)
        | numeric(out, "speed_band_ge_110_flag").ge(0.5)
        | overspeed_level.ge(3.0)
    )
    speed_reliable = overspeed_reliable_signal.ge(0.85)
    speed_ge90_event = speed_ge90_active & overspeed_ge2_duration.ge(R46_SPEED_EVENT_MIN_DURATION_SEC) & speed_reliable
    speed_ge110_event = speed_ge110_active & overspeed_ge3_duration.ge(R46_SPEED_EVENT_MIN_DURATION_SEC) & speed_reliable
    speed_90_110_event = (
        speed_ge90_event
        & ~speed_ge110_event
        & (speed_p95.le(110.0) | numeric(out, "speed_band_90_110_flag").ge(0.5))
    )

    wrongway_raw = numeric(out, "trajectory_wrong_way_flag").ge(0.5) | numeric(
        out, "trajectory_illegal_flow_direction_flag"
    ).ge(0.5)
    wrongway_score = pd.concat(
        [
            clip01(numeric(out, "trajectory_wrong_way_score"))
            * clip01(numeric(out, "trajectory_wrong_way_confidence")),
            clip01(numeric(out, "trajectory_illegal_flow_direction_score"))
            * clip01(numeric(out, "trajectory_illegal_flow_direction_confidence")),
        ],
        axis=1,
    ).max(axis=1)
    wrongway_duration = pd.concat(
        [
            numeric(out, "trajectory_wrong_way_duration"),
            numeric(out, "trajectory_illegal_flow_direction_duration"),
        ],
        axis=1,
    ).max(axis=1)
    s33_wrongway = numeric(out, S33_PREFIX_CONFIRMED_COLUMN).ge(0.5) & numeric(
        out, S33_LOCAL_SUPPORT_COLUMN
    ).ge(0.80)
    wrongway_event = wrongway_raw | s33_wrongway | (wrongway_score.ge(0.45) & wrongway_duration.ge(0.6667))

    redlight_event_peak = pd.concat(
        [
            clip01(numeric(out, "redlight_event_peak_score")),
            clip01(numeric(out, "redlight_policy_strength")),
            clip01(numeric(out, "redlight_confirmed")),
            clip01(numeric(out, "redlight_event_score")),
        ],
        axis=1,
    ).max(axis=1)
    redlight_event = (
        numeric(out, "redlight_confirmed").ge(0.5)
        | numeric(out, "redlight_confirmed_duration").ge(R46_REDLIGHT_MIN_DURATION_SEC)
        | (redlight_event_peak.ge(0.60) & numeric(out, "redlight_signal_valid", 1.0).ge(0.5))
    )

    risk3_speed_ge110 = speed_ge110_event
    risk3_wrongway_redlight = wrongway_event & redlight_event
    risk3_wrongway_speed_ge90 = wrongway_event & speed_ge90_event
    risk3_redlight_speed_ge90 = redlight_event & speed_ge90_event
    risk3_policy = (
        risk3_speed_ge110
        | risk3_wrongway_redlight
        | risk3_wrongway_speed_ge90
        | risk3_redlight_speed_ge90
    )
    risk2_policy = (wrongway_event | redlight_event | speed_90_110_event) & ~risk3_policy

    out["v92_r46_speed_ge90_event"] = speed_ge90_event.astype(float)
    out["v92_r46_speed_90_110_event"] = speed_90_110_event.astype(float)
    out["v92_r46_speed_ge110_event"] = speed_ge110_event.astype(float)
    out["v92_r46_wrongway_event"] = wrongway_event.astype(float)
    out["v92_r46_redlight_event"] = redlight_event.astype(float)
    out["v92_r46_risk3_speed_ge110_policy"] = risk3_speed_ge110.astype(float)
    out["v92_r46_risk3_wrongway_redlight_policy"] = risk3_wrongway_redlight.astype(float)
    out["v92_r46_risk3_wrongway_speed_ge90_policy"] = risk3_wrongway_speed_ge90.astype(float)
    out["v92_r46_risk3_redlight_speed_ge90_policy"] = risk3_redlight_speed_ge90.astype(float)
    out["v92_r46_risk3_policy"] = risk3_policy.astype(float)
    out["v92_r46_risk2_policy"] = risk2_policy.astype(float)
    out["v92_r46_policy_severity_score"] = np.where(risk3_policy, 1.0, np.where(risk2_policy, 0.5, 0.0))
    return out


def merge_v37_sidecar(frame: pd.DataFrame, v37_sidecar: pd.DataFrame | None) -> pd.DataFrame:
    out = normalize_keys(frame)
    if v37_sidecar is None:
        for column in V37_REQUIRED_COLUMNS:
            if column not in out.columns:
                out[column] = 0.0
        return out

    sidecar = normalize_keys(v37_sidecar)
    missing = [column for column in KEY_COLUMNS + V37_REQUIRED_COLUMNS if column not in sidecar.columns]
    if missing:
        raise ValueError(f"v37 sidecar missing columns: {missing}")
    duplicated = int(sidecar.duplicated(KEY_COLUMNS).sum())
    if duplicated:
        raise ValueError(f"v37 sidecar duplicated keys: {duplicated}")

    out = out.drop(columns=[c for c in V37_REQUIRED_COLUMNS if c in out.columns], errors="ignore")
    out = out.merge(sidecar[KEY_COLUMNS + V37_REQUIRED_COLUMNS], on=KEY_COLUMNS, how="left", validate="many_to_one")
    for column in V37_REQUIRED_COLUMNS:
        out[column] = numeric(out, column).clip(0.0, 1.0)
    return out


def merge_r46_source(frame: pd.DataFrame, source_frame: pd.DataFrame | None) -> pd.DataFrame:
    out = normalize_keys(frame)
    if source_frame is None:
        for column in R46_COLUMNS:
            if column not in out.columns:
                out[column] = 0.0
        return out

    source = add_clean_r46_policy_flags(source_frame)
    missing = [column for column in KEY_COLUMNS + R46_COLUMNS if column not in source.columns]
    if missing:
        raise ValueError(f"r46 source missing columns after rebuild: {missing}")
    duplicated = int(source.duplicated(KEY_COLUMNS).sum())
    if duplicated:
        raise ValueError(f"r46 source duplicated keys: {duplicated}")

    out = out.drop(columns=[c for c in R46_COLUMNS if c in out.columns], errors="ignore")
    out = out.merge(source[KEY_COLUMNS + R46_COLUMNS], on=KEY_COLUMNS, how="left", validate="many_to_one")
    for column in R46_COLUMNS:
        out[column] = numeric(out, column).clip(0.0, 1.0)
    return out


def add_clean_v38_feature_generation(
    frame: pd.DataFrame,
    *,
    source_frame: pd.DataFrame | None = None,
    v37_sidecar: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate clean v38 policy-severity features from upstream evidence."""

    out = merge_r46_source(frame, source_frame)
    out = merge_v37_sidecar(out, v37_sidecar)

    s33_prefix = numeric(out, "s33_prefix_state_confirmed")
    s33_margin = numeric(out, "s33_confirmed_ema_margin").clip(0.0, 1.0)
    s33_current = numeric(out, "s33_current_local_semantic_support").clip(0.0, 1.0)
    v37_weaving = numeric(out, "v37_current_true_weaving_event").clip(0.0, 1.0)
    trajectory_support = pd.concat(
        [
            s33_prefix.ge(0.5).astype(float),
            s33_current,
            s33_margin,
            v37_weaving,
        ],
        axis=1,
    ).max(axis=1).clip(0.0, 1.0)

    redlight_event = numeric(out, "redlight_event_score").clip(0.0, 1.0)
    overspeed_event = numeric(out, "overspeed_event_score").clip(0.0, 1.0)
    r46_risk2 = numeric(out, "v92_r46_risk2_policy").clip(0.0, 1.0)
    r46_risk3 = numeric(out, "v92_r46_risk3_policy").clip(0.0, 1.0)
    r46_speed_ge90 = numeric(out, "v92_r46_speed_ge90_event").clip(0.0, 1.0)
    r46_speed_ge110 = numeric(out, "v92_r46_speed_ge110_event").clip(0.0, 1.0)
    r46_wrong_red = numeric(out, "v92_r46_risk3_wrongway_redlight_policy").clip(0.0, 1.0)
    smoothed_speed = numeric(out, "speed_smoothed_kmh_p95")
    reliable_speed = numeric(out, "r32_overspeed_reliable_signal").clip(0.0, 1.0)

    redlight_trajectory = (redlight_event * trajectory_support).clip(0.0, 1.0)
    redlight_wrongway = pd.concat(
        [numeric(out, "v37_redlight_wrongway_combo_ready").clip(0.0, 1.0), r46_wrong_red],
        axis=1,
    ).max(axis=1)
    overspeed_trajectory = (pd.concat([overspeed_event, r46_speed_ge90], axis=1).max(axis=1) * trajectory_support).clip(
        0.0, 1.0
    )
    stable_ge110 = pd.concat(
        [
            (smoothed_speed.ge(110.0) & reliable_speed.ge(0.70)).astype(float),
            (smoothed_speed.ge(108.0) & r46_speed_ge110.ge(0.5) & reliable_speed.ge(0.80)).astype(float),
        ],
        axis=1,
    ).max(axis=1)

    composite_risk3 = pd.concat(
        [
            r46_risk3,
            stable_ge110,
            redlight_wrongway,
            redlight_trajectory.ge(0.55).astype(float),
            overspeed_trajectory.ge(0.55).astype(float),
        ],
        axis=1,
    ).max(axis=1).clip(0.0, 1.0)
    policy_risk2 = pd.concat([r46_risk2, trajectory_support], axis=1).max(axis=1).clip(0.0, 1.0)
    trajectory_only = (
        trajectory_support.ge(0.5)
        & redlight_event.lt(0.45)
        & overspeed_event.lt(0.45)
        & composite_risk3.lt(0.5)
        & stable_ge110.lt(0.5)
    ).astype(float)

    out[COLS.trajectory_risk2_support] = trajectory_support
    out[COLS.policy_risk2_support] = policy_risk2
    out[COLS.policy_risk3_support] = composite_risk3
    out[COLS.trajectory_only_context] = trajectory_only
    out[COLS.stable_ge110_support] = stable_ge110
    out[COLS.redlight_trajectory_risk3_support] = redlight_trajectory
    out[COLS.redlight_wrongway_risk3_support] = redlight_wrongway
    out[COLS.overspeed_ge90_trajectory_risk3_support] = overspeed_trajectory
    for column in HISTORICAL_V38_COLUMNS:
        out[column] = numeric(out, column).clip(0.0, 1.0)
    return out


def overwrite_historical_v38_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    """Use clean v38 feature columns under the historical names expected downstream."""

    out = frame.copy()
    for clean_col, historical_col in HISTORICAL_V38_COLUMNS.items():
        if clean_col not in out.columns:
            raise KeyError(f"clean v38 column not found: {clean_col}")
        out[historical_col] = numeric(out, clean_col).clip(0.0, 1.0)
    return out
