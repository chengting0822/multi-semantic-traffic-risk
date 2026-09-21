"""Inference-only extraction of the accepted C4O canonical feature formulas.

This module starts from the four *v8* policy tables, not the earlier 15-D v0
fusion table.  It intentionally has no training labels, split files, or
absolute research-directory paths.  The v0-to-v8 builders are a separate
required stage; callers must not pass v0 tables here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from traffic_risk.identifiers import normalize_window_keys


KEYS = ["case_key", "source_type", "source_id", "video_id", "track_id", "ts_window_idx"]
TRACK_KEYS = ["case_key", "video_id", "track_id"]
OUTPUT_FEATURES = [
    "s33_prefix_state_confirmed", "s33_confirmed_ema_margin",
    "s33_current_local_semantic_support", "r32_overspeed_midband_strength",
    "r32_overspeed_ge2_duration_strength", "r32_overspeed_reliable_signal",
    "r32_overspeed_rescue_signal", "overspeed_event_score", "redlight_event_score",
    "active_family_count", "cooccurrence_strength", "elapsed_track_progress",
]
FUSION_SOURCE_COLUMNS = [
    "trajectory_anomaly_score", "trajectory_anomaly_duration_sec_proxy",
    "overspeed_level", "trajectory_confidence_peak",
    "pa_trajectory_score_reliable", "trajectory_score_reliable",
    "pa_trajectory_duration_reliable", "trajectory_duration_reliable",
    "moderate_overspeed_duration", "sustained_overspeed_ratio",
    "active_family_count", "cooccurrence_duration_sec",
    "trajectory_overspeed_overlap_ratio", "trajectory_redlight_overlap_ratio",
    "overspeed_redlight_overlap_ratio", "policy_pair_peak_trajectory_overspeed",
    "policy_pair_peak_trajectory_redlight", "policy_pair_peak_overspeed_redlight",
    "pa_total_duration_proxy",
]
REQUIRED_MODULE_COLUMNS = {
    "fusion_v8": [
        "trajectory_anomaly_score", "trajectory_anomaly_duration_sec_proxy",
        "trajectory_confidence_peak", "pa_trajectory_score_reliable",
        "pa_trajectory_duration_reliable", "trajectory_score_reliable",
        "trajectory_duration_reliable", "moderate_overspeed_duration",
        "sustained_overspeed_ratio", "active_family_count",
        "cooccurrence_duration_sec", "trajectory_overspeed_overlap_ratio",
        "trajectory_redlight_overlap_ratio", "overspeed_redlight_overlap_ratio",
        "policy_pair_peak_trajectory_overspeed", "policy_pair_peak_trajectory_redlight",
        "policy_pair_peak_overspeed_redlight",
    ],
    "trajectory_v8": ["trajectory_subtype_peak", "trajectory_post_peak_persistence"],
    "overspeed_v8": [
        "speed_kmh_p95", "overspeed_reliability", "overspeed_signal_quality",
        "overspeed_level_ge2_duration", "computed_speed_kmh_p95",
        "speed_smoothed_kmh_p95",
    ],
    "redlight_v8": [
        "redlight_policy_strength", "redlight_event_peak_score",
        "redlight_policy_confidence", "redlight_policy_commitment",
        "redlight_signal_valid", "redlight_stopline_crossing_confidence",
        "redlight_crossing_freshness_score", "redlight_after_line_motion_score",
        "redlight_confirmed_duration",
    ],
}


def _number(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype(float)


def _clip01(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").fillna(0.0).clip(0.0, 1.0)


def _sigmoid(values: pd.Series, *, center: float, scale: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    return _clip01(1.0 / (1.0 + np.exp(-(numeric - center) / max(scale, 1e-6))))


def _max(*values: pd.Series) -> pd.Series:
    return pd.concat(values, axis=1).max(axis=1)


def _banded_speed(speed: pd.Series) -> pd.Series:
    result = pd.Series(0.0, index=speed.index, dtype=float)
    one = speed.gt(70) & speed.le(90)
    two = speed.gt(90) & speed.le(110)
    three = speed.gt(110)
    result.loc[one] = ((speed.loc[one] - 70) / 20) / 3
    result.loc[two] = 1 / 3 + ((speed.loc[two] - 90) / 20) / 3
    result.loc[three] = (2 / 3 + ((speed.loc[three] - 110) / 20) / 3).clip(upper=1)
    return _clip01(result)


def _prepare(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    frame = normalize_window_keys(frame.copy())
    missing = [column for column in KEYS + REQUIRED_MODULE_COLUMNS[label] if column not in frame]
    if missing:
        raise ValueError(f"{label} is missing join columns: {missing}")
    if frame.duplicated(KEYS).any():
        raise ValueError(f"{label} has duplicate windows")
    return frame


def _join(base: pd.DataFrame, other: pd.DataFrame, label: str) -> pd.DataFrame:
    payload = [column for column in other if column not in KEYS and column not in base]
    result = base.merge(other[KEYS + payload], on=KEYS, how="left", validate="one_to_one", indicator=True)
    if not result["_merge"].eq("both").all():
        raise ValueError(f"{label} does not cover every fusion-v8 window")
    return result.drop(columns="_merge")


def build_c4o_from_v8(
    *,
    fusion_v8: pd.DataFrame,
    trajectory_v8: pd.DataFrame,
    overspeed_v8: pd.DataFrame,
    redlight_v8: pd.DataFrame,
    s33_prefix: pd.DataFrame,
) -> pd.DataFrame:
    """Build the formal C4O source columns from exact v8 module outputs.

    ``s33_prefix`` must be calculated by the accepted causal S3.3 state
    machine.  It is mandatory because zero-filling this signal silently changes
    GRU and policy predictions.
    """

    fusion = _prepare(fusion_v8, "fusion_v8")
    base_columns = KEYS + ["start_sec", "end_sec"]
    missing_base = [column for column in base_columns if column not in fusion]
    if missing_base:
        raise ValueError(f"fusion_v8 is missing time columns: {missing_base}")
    frame = fusion[base_columns + [column for column in FUSION_SOURCE_COLUMNS if column in fusion]].copy()
    for name, extra in (
        ("trajectory_v8", trajectory_v8),
        ("overspeed_v8", overspeed_v8),
        ("redlight_v8", redlight_v8),
    ):
        frame = _join(frame, _prepare(extra, name), name)
    prefix = normalize_window_keys(s33_prefix.copy())
    prefix_keys = ["case_key", "video_id", "track_id", "ts_window_idx"]
    prefix_columns = [
        "s33_prefix_state_confirmed", "s33_confirmed_ema_margin",
        "s33_current_local_semantic_support", "s33_confirmed_ema_pos",
    ]
    missing = [column for column in prefix_keys + prefix_columns if column not in prefix]
    if missing:
        raise ValueError(f"S3.3 prefix is missing columns: {missing}")
    frame = frame.merge(prefix[prefix_keys + prefix_columns], on=prefix_keys, how="left", validate="one_to_one")
    if frame[prefix_columns].isna().any().any():
        raise ValueError("S3.3 prefix does not cover every fusion-v8 window")
    frame = frame.sort_values(["video_id", "track_id", "ts_window_idx", "start_sec", "end_sec"], kind="mergesort").reset_index(drop=True)

    # Exact accepted canonical 8-D formulas (the first three are policy support;
    # C4O retains the final five and replaces the trajectory three with S3.3).
    duration = (_number(frame, "end_sec") - _number(frame, "start_sec")).clip(lower=0)
    elapsed = duration.groupby([frame["video_id"], frame["track_id"]], sort=False).cumsum()
    frame["elapsed_track_duration_sec"] = elapsed
    frame["elapsed_track_progress"] = _clip01(1 - np.exp(-elapsed / 6.0))
    frame["trajectory_strength"] = _clip01(_max(_number(frame, "trajectory_anomaly_score"), _number(frame, "trajectory_subtype_peak")))
    frame["trajectory_confidence"] = _clip01(_max(
        _number(frame, "trajectory_confidence_peak"), _number(frame, "pa_trajectory_score_reliable"),
        _number(frame, "trajectory_score_reliable"),
    ))
    frame["trajectory_persistence"] = _clip01(_max(
        _sigmoid(_number(frame, "trajectory_anomaly_duration_sec_proxy"), center=0.6667, scale=0.3333),
        _number(frame, "trajectory_post_peak_persistence"), _number(frame, "pa_trajectory_duration_reliable"),
        _number(frame, "trajectory_duration_reliable"),
    ))
    speed = _number(frame, "speed_kmh_p95")
    frame["overspeed_strength"] = _banded_speed(speed)
    frame["overspeed_confidence"] = _clip01(
        0.60 * _number(frame, "overspeed_reliability") + 0.40 * _number(frame, "overspeed_signal_quality")
    )
    frame["overspeed_persistence"] = _clip01(_max(
        _sigmoid(_number(frame, "overspeed_level_ge2_duration"), center=0.6667, scale=0.3333),
        _sigmoid(_number(frame, "moderate_overspeed_duration"), center=0.6667, scale=0.3333),
        _number(frame, "sustained_overspeed_ratio"),
    ))
    overspeed_support = _clip01(0.72 * frame["overspeed_confidence"] + 0.28 * frame["overspeed_persistence"])
    frame["overspeed_event_score"] = _clip01(frame["overspeed_strength"] * np.sqrt(0.08 + 0.92 * overspeed_support))

    crossing_presence = _clip01((_number(frame, "redlight_crossing_onset") > 0).astype(float) * _number(frame, "redlight_stopline_crossing_confidence"))
    crossing_freshness = _clip01(_max(
        _number(frame, "redlight_crossing_freshness_score"),
        crossing_presence * np.exp(-_number(frame, "redlight_crossing_onset") / 0.45),
    ))
    motion_commitment = _clip01(_number(frame, "redlight_after_line_motion_score") * _number(frame, "redlight_stopline_crossing_confidence"))
    redlight_support = _clip01(_max(
        _number(frame, "redlight_policy_strength"), _number(frame, "redlight_event_peak_score"),
        _number(frame, "redlight_confirmed"),
        _sigmoid(_number(frame, "redlight_confirmed_duration"), center=0.3333, scale=0.1667),
    ))
    frame["redlight_strength"] = _clip01(_max(_number(frame, "redlight_policy_strength"), _number(frame, "redlight_event_peak_score")))
    frame["redlight_confidence"] = _clip01(_max(
        _number(frame, "redlight_policy_confidence"),
        redlight_support * (0.65 * _number(frame, "redlight_stopline_crossing_confidence") + 0.35 * _number(frame, "redlight_signal_valid")),
    ))
    frame["redlight_commitment"] = _clip01(_max(
        _number(frame, "redlight_policy_commitment"), crossing_freshness * redlight_support,
        motion_commitment * redlight_support,
    ))
    support = _clip01(0.60 * frame["redlight_confidence"] + 0.40 * frame["redlight_commitment"])
    frame["redlight_event_score"] = _clip01(frame["redlight_strength"] * np.sqrt(0.14 + 0.86 * support))
    frame["active_family_count"] = _number(frame, "active_family_count")
    overlap = _clip01(_max(
        _number(frame, "trajectory_overspeed_overlap_ratio"), _number(frame, "trajectory_redlight_overlap_ratio"),
        _number(frame, "overspeed_redlight_overlap_ratio"),
    ))
    pair = _clip01(_max(
        _number(frame, "policy_pair_peak_trajectory_overspeed"), _number(frame, "policy_pair_peak_trajectory_redlight"),
        _number(frame, "policy_pair_peak_overspeed_redlight"),
    ))
    cooc_duration = _number(frame, "cooccurrence_duration_sec")
    frame["cooccurrence_strength"] = _clip01(_max(
        (cooc_duration > 0).astype(float) * _sigmoid(cooc_duration, center=1.0, scale=0.5), overlap, pair,
    ))

    # C4O's four R32 overspeed inputs are not learnable placeholders: these are
    # the frozen training formulas, including the nonzero sigmoid baseline.
    reliability = _clip01(_number(frame, "overspeed_reliability"))
    quality = _clip01(_number(frame, "overspeed_signal_quality"))
    frame["r32_overspeed_midband_strength"] = _clip01((speed - 90.0) / 20.0)
    frame["r32_overspeed_ge2_duration_strength"] = _sigmoid(_number(frame, "overspeed_level_ge2_duration"), center=1.30, scale=0.40)
    frame["r32_overspeed_reliable_signal"] = _clip01(reliability * quality)
    frame["r32_overspeed_rescue_signal"] = _clip01(
        frame["r32_overspeed_midband_strength"]
        * frame["r32_overspeed_ge2_duration_strength"]
        * frame["r32_overspeed_reliable_signal"]
    )

    # The frozen track postprocessor and later semantic policies consume these
    # R46 side channels even though they are not part of the 16 GRU inputs.
    level = _number(frame, "overspeed_level")
    ge2_duration = _number(frame, "overspeed_level_ge2_duration")
    ge3_duration = _number(frame, "overspeed_level_ge3_duration")
    reliable_speed = frame["r32_overspeed_reliable_signal"].ge(0.85)
    ge90 = (
        speed.ge(90) | _number(frame, "speed_band_90_110_flag").ge(0.5)
        | _number(frame, "speed_band_ge_110_flag").ge(0.5) | level.ge(2)
    ) & ge2_duration.ge(1.30) & reliable_speed
    ge110 = (
        speed.gt(110) | _number(frame, "speed_band_ge_110_flag").ge(0.5) | level.ge(3)
    ) & ge3_duration.ge(1.30) & reliable_speed
    speed_90_110 = ge90 & ~ge110 & (speed.le(110) | _number(frame, "speed_band_90_110_flag").ge(0.5))
    wrongway_raw = _number(frame, "trajectory_wrong_way_flag").ge(0.5) | _number(frame, "trajectory_illegal_flow_direction_flag").ge(0.5)
    wrongway_score = _max(
        _clip01(_number(frame, "trajectory_wrong_way_score")) * _clip01(_number(frame, "trajectory_wrong_way_confidence")),
        _clip01(_number(frame, "trajectory_illegal_flow_direction_score")) * _clip01(_number(frame, "trajectory_illegal_flow_direction_confidence")),
    )
    wrongway_duration = _max(_number(frame, "trajectory_wrong_way_duration"), _number(frame, "trajectory_illegal_flow_direction_duration"))
    s33_wrongway = frame["s33_prefix_state_confirmed"].ge(0.5) & frame["s33_current_local_semantic_support"].ge(0.80)
    wrongway = wrongway_raw | s33_wrongway | (wrongway_score.ge(0.45) & wrongway_duration.ge(0.6667))
    redlight_peak = _max(
        _clip01(_number(frame, "redlight_event_peak_score")), _clip01(_number(frame, "redlight_policy_strength")),
        _clip01(_number(frame, "redlight_confirmed")), _clip01(frame["redlight_event_score"]),
    )
    redlight = (
        _number(frame, "redlight_confirmed").ge(0.5)
        | _number(frame, "redlight_confirmed_duration").ge(0.3333)
        | (redlight_peak.ge(0.60) & _number(frame, "redlight_signal_valid", 1.0).ge(0.5))
    )
    risk3 = ge110 | (wrongway & redlight) | (wrongway & ge90) | (redlight & ge90)
    risk2 = (wrongway | redlight | speed_90_110) & ~risk3
    frame["r46_risk3_policy"] = risk3.astype(float)
    frame["r46_risk2_policy"] = risk2.astype(float)
    frame["r46_policy_severity_score"] = np.where(risk3, 1.0, np.where(risk2, 0.5, 0.0))

    # The formal C4O table omits one-window tracks as tracking fragments.  This
    # rule is applied only after all causal per-track features were calculated.
    window_count = frame.groupby(TRACK_KEYS)["ts_window_idx"].transform("count")
    result = frame.loc[window_count > 1].reset_index(drop=True)
    result["split"] = "inference"
    return result


__all__ = ["OUTPUT_FEATURES", "build_c4o_from_v8"]
