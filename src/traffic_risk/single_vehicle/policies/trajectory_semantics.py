#!/usr/bin/env python3
"""Clean v37 trajectory semantics sidecar generation.

This module distills the accepted v37 upstream semantic sidecar:

    C4O source windows
    + v28 motion-aware lane sidecar
    + v24f trajectory context sidecar
    + red-light-zone movement sidecar
    -> v37 clean trajectory semantic sidecar columns

It does not train GRU, touch labels, or change any prediction.  The generated
columns are a clean-room equivalent of
``clean_trajectory_semantics_sidecar_v37c.csv``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


KEY_COLUMNS = ["case_key", "video_id", "track_id", "ts_window_idx"]
GROUP_COLUMNS = ["case_key", "video_id", "track_id"]


@dataclass(frozen=True)
class CleanV37TrajectorySidecarColumns:
    redlight_zone_movement_confirmed: str = "v92_v37_redlight_zone_movement_confirmed"
    clean_wrongway_memory: str = "v92_v37_clean_wrongway_memory"
    clean_trajectory_memory_strength: str = "v92_v37_clean_trajectory_memory_strength"
    redlight_wrongway_combo_ready: str = "v92_v37_redlight_wrongway_combo_ready"
    redlight_trajectory_combo_score: str = "v92_v37_redlight_trajectory_combo_score"
    current_clean_wrongway_event: str = "v92_v37_current_clean_wrongway_event"
    current_true_weaving_event: str = "v92_v37_current_true_weaving_event"
    normal_turn_connector_context: str = "v92_v37_normal_turn_connector_context"
    lane_mapping_low_quality: str = "v92_v37_lane_mapping_low_quality"
    route_deviation_ignored: str = "v92_v37_route_deviation_ignored"
    sustained_opposite_context: str = "v92_v37_sustained_opposite_context"
    reversed_blocked_by_turn: str = "v92_v37_reversed_blocked_by_turn"


COLS = CleanV37TrajectorySidecarColumns()


HISTORICAL_V37_SIDECAR_COLUMNS: dict[str, str] = {
    COLS.redlight_zone_movement_confirmed: "v37_redlight_zone_movement_confirmed",
    COLS.clean_wrongway_memory: "v37_clean_wrongway_memory",
    COLS.clean_trajectory_memory_strength: "v37_clean_trajectory_memory_strength",
    COLS.redlight_wrongway_combo_ready: "v37_redlight_wrongway_combo_ready",
    COLS.redlight_trajectory_combo_score: "v37_redlight_trajectory_combo_score",
    COLS.current_clean_wrongway_event: "v37_current_clean_wrongway_event",
    COLS.current_true_weaving_event: "v37_current_true_weaving_event",
    COLS.normal_turn_connector_context: "v37_normal_turn_connector_context",
    COLS.lane_mapping_low_quality: "v37_lane_mapping_low_quality",
    COLS.route_deviation_ignored: "v37_route_deviation_ignored",
    COLS.sustained_opposite_context: "v37_sustained_opposite_context",
    COLS.reversed_blocked_by_turn: "v37_reversed_blocked_by_turn",
}

V37_TRAINING_FEATURE_COLUMNS = [
    COLS.redlight_zone_movement_confirmed,
    COLS.redlight_wrongway_combo_ready,
    COLS.redlight_trajectory_combo_score,
    COLS.current_true_weaving_event,
]


V28_ALIASES: dict[str, tuple[str, str]] = {
    "lane_id_raw": ("v92_v28_lane_id_raw", "v28_lane_id_raw"),
    "lane_id_smooth": ("v92_v28_lane_id_smooth", "v28_lane_id_smooth"),
    "lane_sequence_smooth_6w": ("v92_v28_lane_sequence_smooth_6w", "v28_lane_sequence_smooth_6w"),
    "transition_type_6w": ("v92_v28_transition_type_6w", "v28_transition_type_6w"),
    "lane_change_count_6w": ("v92_v28_lane_change_count_6w", "v28_lane_change_count_6w"),
    "lane_id_backtrack_6w": ("v92_v28_lane_id_backtrack_6w", "v28_lane_id_backtrack_6w"),
    "straight_family_backtrack_6w": (
        "v92_v28_straight_family_backtrack_6w",
        "v28_straight_family_backtrack_6w",
    ),
    "connector_expected_route_6w": (
        "v92_v28_connector_expected_route_6w",
        "v28_connector_expected_route_6w",
    ),
    "connector_adjacent_route_6w": (
        "v92_v28_connector_adjacent_route_6w",
        "v28_connector_adjacent_route_6w",
    ),
    "route_deviation_context": ("v92_v28_route_deviation_context", "v28_route_deviation_context"),
    "true_weaving_current": ("v92_v28_true_weaving_current", "v28_true_weaving_current"),
    "true_weaving_memory": ("v92_v28_true_weaving_memory", "v28_true_weaving_memory"),
    "context_quality": ("v92_v28_context_quality", "v28_context_quality"),
    "lane_ambiguous": ("v92_v28_lane_ambiguous", "v28_lane_ambiguous"),
    "lane_valid_ratio": ("v92_v28_lane_valid_ratio", "v28_lane_valid_ratio"),
    "bbox_area_cv_raw": ("v92_v28_bbox_area_cv_raw", "v28_bbox_area_cv_raw"),
    "bbox_area_log_jump_max": ("v92_v28_bbox_area_log_jump_max", "v28_bbox_area_log_jump_max"),
    "lane_motion_cos_top": ("v92_v28_lane_motion_cos_top", "v28_lane_motion_cos_top"),
    "lane_opposite_penalty_top": ("v92_v28_lane_opposite_penalty_top", "v28_lane_opposite_penalty_top"),
}

V24F_COLUMNS = [
    "opposite_lane_run_length",
    "opposite_lane_occupancy_ratio",
    "reversed_flag",
    "v24f_context_quality",
    "v24f_true_weaving_current",
    "v24f_true_weaving_memory",
    "v24f_normal_turn_like",
    "v24f_turn_route_like",
    "v24f_single_lane_change_like",
    "v24f_occlusion_jitter_current",
    "v24f_occlusion_jitter_count_6w",
    "osc_h20_score",
    "osc_h30_score",
]

REDLIGHT_ZONE_ALIASES: dict[str, tuple[str, str]] = {
    "movement_signal": ("v92_redlight_zone_movement_signal_v1", "redlight_zone_movement_signal_v1"),
    "movement_score": ("v92_redlight_zone_movement_score_v1", "redlight_zone_movement_score_v1"),
    "prior_green_between": ("v92_redlight_zone_prior_green_between", "redlight_zone_prior_green_between"),
}


def norm_id(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return text


def normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "case_key" in out.columns:
        out["case_key"] = out["case_key"].map(norm_id)
    for col in ["video_id", "track_id"]:
        if col in out.columns:
            out[col] = out[col].map(norm_id)
    if "ts_window_idx" in out.columns:
        out["ts_window_idx"] = pd.to_numeric(out["ts_window_idx"], errors="coerce").fillna(-1).astype(int)
    return out

def numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype(float)


def first_existing(frame: pd.DataFrame, aliases: Iterable[str], default: float | str = 0.0) -> pd.Series:
    for column in aliases:
        if column in frame.columns:
            return frame[column]
    return pd.Series(default, index=frame.index)


def rolling_max(series: pd.Series, window: int) -> pd.Series:
    return series.astype(float).rolling(window=window, min_periods=1).max()


def rolling_sum(series: pd.Series, window: int) -> pd.Series:
    return series.astype(float).rolling(window=window, min_periods=1).sum()


def validate_unique(name: str, frame: pd.DataFrame) -> None:
    missing = [c for c in KEY_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{name} missing keys: {missing}")
    duplicated = int(frame.duplicated(KEY_COLUMNS).sum())
    if duplicated:
        sample = frame.loc[frame.duplicated(KEY_COLUMNS, keep=False), KEY_COLUMNS].head(10).to_dict("records")
        raise ValueError(f"{name} has duplicated keys: {duplicated}; sample={sample}")


def _prepare_v28(v28_sidecar: pd.DataFrame) -> pd.DataFrame:
    sidecar = normalize_keys(v28_sidecar)
    validate_unique("v28_sidecar", sidecar)
    out = sidecar[KEY_COLUMNS].copy()
    for canonical, aliases in V28_ALIASES.items():
        out[f"v37in_v28_{canonical}"] = first_existing(sidecar, aliases, default="")
    return out


def _prepare_v24f(v24f_sidecar: pd.DataFrame) -> pd.DataFrame:
    sidecar = normalize_keys(v24f_sidecar)
    validate_unique("v24f_sidecar", sidecar)
    cols = KEY_COLUMNS + [c for c in V24F_COLUMNS if c in sidecar.columns]
    return sidecar[cols].copy()


def _prepare_redlight_zone(redlight_zone_sidecar: pd.DataFrame) -> pd.DataFrame:
    sidecar = normalize_keys(redlight_zone_sidecar)
    validate_unique("redlight_zone_sidecar", sidecar)
    out = sidecar[KEY_COLUMNS].copy()
    for canonical, aliases in REDLIGHT_ZONE_ALIASES.items():
        out[f"v37in_redlight_zone_{canonical}"] = first_existing(sidecar, aliases, default=0.0)
    return out


def build_clean_v37_trajectory_sidecar(
    c4o_source: pd.DataFrame,
    *,
    v28_sidecar: pd.DataFrame,
    v24f_sidecar: pd.DataFrame,
    redlight_zone_sidecar: pd.DataFrame,
) -> pd.DataFrame:
    """Rebuild accepted v37 trajectory semantic sidecar columns."""

    c4o = normalize_keys(c4o_source)
    validate_unique("c4o_source", c4o)
    base_cols = [
        c
        for c in [
            *KEY_COLUMNS,
            "start_sec",
            "end_sec",
            "split",
            "y_schemaC",
            "redlight_event_score",
            "active_family_count",
            "cooccurrence_strength",
        ]
        if c in c4o.columns
    ]
    merged = c4o[base_cols].copy()
    merged = merged.merge(_prepare_v28(v28_sidecar), on=KEY_COLUMNS, how="left", validate="one_to_one")
    merged = merged.merge(_prepare_v24f(v24f_sidecar), on=KEY_COLUMNS, how="left", validate="one_to_one")
    merged = merged.merge(_prepare_redlight_zone(redlight_zone_sidecar), on=KEY_COLUMNS, how="left", validate="one_to_one")

    for col in merged.columns:
        if col in KEY_COLUMNS or col in ["split"]:
            continue
        if col.startswith("v37in_v28_lane_id") or col.startswith("v37in_v28_lane_sequence") or col.endswith(
            "transition_type_6w"
        ):
            continue
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    redlight_event = numeric(merged, "redlight_event_score").clip(0, 1)
    red_zone_score = numeric(merged, "v37in_redlight_zone_movement_score").clip(0, 1)
    red_zone_signal = numeric(merged, "v37in_redlight_zone_movement_signal")
    prior_green = numeric(merged, "v37in_redlight_zone_prior_green_between")
    red_zone_soft = pd.Series(
        np.maximum(red_zone_signal, ((red_zone_score >= 0.70) & (prior_green < 0.5)).astype(float) * red_zone_score),
        index=merged.index,
        dtype=float,
    )
    red_current = pd.Series(np.maximum(redlight_event, red_zone_soft), index=merged.index, dtype=float)

    lane_text = (
        first_existing(merged, ["v37in_v28_lane_id_smooth"], default="")
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    lane_mapped = lane_text.ne("") & ~lane_text.isin(["nan", "none", "unknown"])

    v28_quality = (
        (numeric(merged, "v37in_v28_context_quality", 1.0) >= 0.66)
        & (numeric(merged, "v37in_v28_lane_ambiguous") < 0.5)
        & (numeric(merged, "v37in_v28_lane_valid_ratio", 1.0) >= 0.35)
        & (numeric(merged, "v37in_v28_bbox_area_cv_raw") < 0.45)
        & (numeric(merged, "v37in_v28_bbox_area_log_jump_max") < 0.55)
        & lane_mapped
    )
    v24_quality = (
        (numeric(merged, "v24f_context_quality", 1.0) >= 0.60)
        & (numeric(merged, "v24f_occlusion_jitter_current") < 0.5)
        & (numeric(merged, "v24f_occlusion_jitter_count_6w") <= 2)
    )
    clean_quality = v28_quality & v24_quality

    turn_connector = (
        (numeric(merged, "v37in_v28_connector_expected_route_6w") >= 0.5)
        | (numeric(merged, "v37in_v28_connector_adjacent_route_6w") >= 0.5)
        | (numeric(merged, "v24f_normal_turn_like") >= 0.5)
        | (numeric(merged, "v24f_turn_route_like") >= 0.5)
    )
    normal_turn_connector = turn_connector | (numeric(merged, "v24f_single_lane_change_like") >= 0.5)
    low_quality = ~clean_quality

    opposite_run = numeric(merged, "opposite_lane_run_length")
    opposite_ratio = numeric(merged, "opposite_lane_occupancy_ratio")
    reversed_flag = numeric(merged, "reversed_flag")
    lane_cos = numeric(merged, "v37in_v28_lane_motion_cos_top")
    non_forward_on_turn = lane_cos <= 0.25
    negative_lane_flow = clean_quality & (lane_cos <= -0.55)
    strong_opposite_raw = (opposite_run >= 8) & (opposite_ratio >= 0.50) & clean_quality
    very_strong_opposite_raw = (opposite_run >= 14) & (opposite_ratio >= 0.85) & clean_quality
    strong_opposite = strong_opposite_raw & (~turn_connector | non_forward_on_turn | negative_lane_flow)
    very_strong_opposite = very_strong_opposite_raw & (~turn_connector | non_forward_on_turn | negative_lane_flow)
    reversed_clean_base = (
        (reversed_flag >= 0.5)
        & (numeric(merged, "v24f_context_quality", 1.0) >= 0.66)
        & (numeric(merged, "v24f_occlusion_jitter_current") < 0.5)
        & (numeric(merged, "v24f_occlusion_jitter_count_6w") <= 2)
        & (lane_cos <= -0.25)
    )
    reversed_blocked_by_turn = reversed_clean_base & turn_connector & ~very_strong_opposite
    current_wrongway_base = (
        strong_opposite
        | very_strong_opposite
        | negative_lane_flow
        | (reversed_clean_base & (~turn_connector | non_forward_on_turn))
    )

    v28_weaving = numeric(merged, "v37in_v28_true_weaving_current")
    v24_weaving = numeric(merged, "v24f_true_weaving_current")
    osc_peak = np.maximum(numeric(merged, "osc_h20_score"), numeric(merged, "osc_h30_score")).clip(0, 1)
    lane_backtrack = (
        (numeric(merged, "v37in_v28_lane_id_backtrack_6w") >= 0.5)
        | (numeric(merged, "v37in_v28_straight_family_backtrack_6w") >= 0.5)
    )
    repeated_change = numeric(merged, "v37in_v28_lane_change_count_6w") >= 2
    current_weaving = (
        clean_quality
        & ((v28_weaving >= 0.5) | (v24_weaving >= 0.5) | ((osc_peak >= 0.66) & lane_backtrack & repeated_change))
        & ~((normal_turn_connector) & ~lane_backtrack & (numeric(merged, "v37in_v28_lane_change_count_6w") <= 2))
    )

    route_deviation_ignored = (
        (numeric(merged, "v37in_v28_route_deviation_context") >= 0.5)
        & ~current_weaving
        & ~current_wrongway_base
    )

    out_rows: list[pd.DataFrame] = []
    for _, idx in merged.groupby(GROUP_COLUMNS, sort=False).groups.items():
        ordered = merged.loc[list(idx)].sort_values("ts_window_idx", kind="mergesort").index
        wrong_cur = pd.Series(current_wrongway_base.loc[ordered].astype(float), index=ordered, dtype=float)
        weave_cur = pd.Series(current_weaving.loc[ordered].astype(float), index=ordered, dtype=float)
        red_cur = pd.Series(red_current.loc[ordered], index=ordered, dtype=float)

        wrong_mem = ((rolling_sum(wrong_cur, 18) >= 1) | (rolling_max(wrong_cur, 18) >= 0.5)).astype(float)
        weave_mem = ((rolling_sum(weave_cur, 8) >= 1) | (rolling_max(weave_cur, 8) >= 0.5)).astype(float)
        trajectory_mem = np.maximum(wrong_mem.to_numpy(dtype=float), weave_mem.to_numpy(dtype=float))
        red_zone_confirmed = ((red_zone_soft.loc[ordered] >= 0.70) | (red_zone_signal.loc[ordered] >= 0.5)).astype(float)
        red_wrong_combo = ((red_cur >= 0.55) & (wrong_mem >= 0.5)).astype(float)
        red_traj_combo = np.minimum(1.0, red_cur.to_numpy(dtype=float) * np.maximum(wrong_mem, weave_mem).to_numpy(dtype=float))

        keep = [c for c in [*KEY_COLUMNS, "start_sec", "end_sec", "split", "y_schemaC"] if c in merged.columns]
        local = merged.loc[ordered, keep].copy()
        local[COLS.redlight_zone_movement_confirmed] = red_zone_confirmed.to_numpy(dtype=float)
        local[COLS.clean_wrongway_memory] = wrong_mem.to_numpy(dtype=float)
        local[COLS.clean_trajectory_memory_strength] = trajectory_mem
        local[COLS.redlight_wrongway_combo_ready] = red_wrong_combo.to_numpy(dtype=float)
        local[COLS.redlight_trajectory_combo_score] = red_traj_combo
        local[COLS.current_clean_wrongway_event] = wrong_cur.to_numpy(dtype=float)
        local[COLS.current_true_weaving_event] = weave_cur.to_numpy(dtype=float)
        local[COLS.normal_turn_connector_context] = normal_turn_connector.loc[ordered].astype(float).to_numpy(dtype=float)
        local[COLS.lane_mapping_low_quality] = low_quality.loc[ordered].astype(float).to_numpy(dtype=float)
        local[COLS.route_deviation_ignored] = route_deviation_ignored.loc[ordered].astype(float).to_numpy(dtype=float)
        local[COLS.sustained_opposite_context] = very_strong_opposite.loc[ordered].astype(float).to_numpy(dtype=float)
        local[COLS.reversed_blocked_by_turn] = reversed_blocked_by_turn.loc[ordered].astype(float).to_numpy(dtype=float)
        out_rows.append(local)

    out = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame(columns=KEY_COLUMNS)
    for col in HISTORICAL_V37_SIDECAR_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    return out


def overwrite_historical_v37_trajectory_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for clean_col, historical_col in HISTORICAL_V37_SIDECAR_COLUMNS.items():
        if clean_col in out.columns:
            out[historical_col] = out[clean_col]
    return out
