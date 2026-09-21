"""Accepted red-light v8 policy outputs from the v0 module and evidence sidecar."""

from __future__ import annotations

import numpy as np
import pandas as pd

from traffic_risk.upstream.v8_common import KEYS, TIMES, clip01, cumulative_duration, join, maximum, number, sigmoid

OUTPUT_COLUMNS = [
    "redlight_confirmed", "redlight_score", "redlight_signal_valid",
    "redlight_confirmed_duration", "redlight_crossing_onset",
    "redlight_forward_motion_after_line", "redlight_after_line_motion_score",
    "redlight_stopline_identity_confidence", "redlight_stopline_crossing_confidence",
    "redlight_crossing_freshness_score", "redlight_policy_strength",
    "redlight_policy_confidence", "redlight_policy_commitment",
    "redlight_event_peak_score", "redlight_geometry_valid_flag",
]


def build_redlight_v8(module: pd.DataFrame, sidecar: pd.DataFrame) -> pd.DataFrame:
    frame = join(module, sidecar, label="redlight_sidecar")
    required = ["redlight_confirmed", "redlight_score", "crossing_event_in_window", "crossing_red_signal", "stopline_geometry_valid", "tl_red_ratio", "tl_prob_red_mean", "tl_prob_red_max", "stopline_identity_confidence", "stopline_multi_candidate_flag"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"redlight v8 lacks evidence: {missing}")
    confirmed = clip01(number(frame, "redlight_confirmed"))
    score = clip01(number(frame, "redlight_score"))
    crossing = clip01(number(frame, "crossing_event_in_window"))
    crossing_red = clip01(number(frame, "crossing_red_signal"))
    geometry = clip01(number(frame, "stopline_geometry_valid"))
    states = frame.get("tl_state_window", pd.Series("", index=frame.index)).fillna("").astype(str)
    known = (~states.str.strip().eq("")).astype(float)
    red_ratio = clip01(number(frame, "tl_red_ratio"))
    red_mean = clip01(number(frame, "tl_prob_red_mean"))
    red_max = clip01(number(frame, "tl_prob_red_max"))
    side_delta = clip01(sigmoid((number(frame, "line_side_value_end") - number(frame, "line_side_value_start")).abs(), 0.6, 0.20))
    distance_delta = clip01(sigmoid((number(frame, "signed_distance_px_end") - number(frame, "signed_distance_px_start")).abs(), 20.0, 8.0))
    signal = clip01(0.30 * known + 0.30 * (red_mean - 0.5).abs() * 2 + 0.20 * (red_max - 0.5).abs() * 2 + 0.20 * (red_ratio - 0.5).abs() * 2)
    motion_px = number(frame, "motion_after_crossing_px").clip(lower=0)
    motion_world = number(frame, "motion_after_crossing_world").clip(lower=0)
    motion = clip01(0.25 * clip01(sigmoid(motion_world, 0.8, 0.30)) + 0.75 * clip01(sigmoid(motion_px, 8.0, 3.0)))
    identity = clip01(number(frame, "stopline_identity_confidence"))
    identity_weight = clip01(0.70 * identity + 0.30 * (1 - clip01(number(frame, "stopline_multi_candidate_flag"))))
    raw_crossing = clip01(0.18 * crossing + 0.28 * crossing_red + (crossing >= 1).astype(float) * (0.14 * geometry + 0.10 * side_delta + 0.10 * distance_delta + 0.20 * identity))
    confidence = clip01(raw_crossing * (0.55 + 0.45 * identity_weight))
    onset = (number(frame, "crossing_timestamp_sec") - number(frame, "start_sec")).clip(lower=0).where(crossing >= 1, 0.0)
    freshness = clip01(crossing * identity_weight * np.exp(-onset / 0.45))
    near_transition = (red_ratio >= 0.6) & (red_ratio < 1) & (crossing >= 1) & (crossing_red == 0) & (motion >= 0.25)
    active = ((crossing_red >= 1) & (confidence >= 0.55) & (motion >= 0.35)) | near_transition | (confirmed >= 1)
    duration = cumulative_duration(frame, active.astype(float))
    persistence = (duration > 0).astype(float) * clip01(sigmoid(duration, 0.3333, 0.1667))
    support = clip01(maximum(confirmed, crossing_red * (motion >= 0.35).astype(float), persistence))
    strength = clip01(maximum(score * support, confidence * motion * support, confirmed * identity_weight))
    policy_confidence = clip01(support * (0.55 * confidence + 0.25 * signal + 0.20 * identity_weight))
    commitment = clip01(maximum(score * (0.50 + 0.50 * identity_weight) * support, freshness * support, confidence * motion * support))
    frame["redlight_confirmed"] = confirmed
    frame["redlight_score"] = score
    frame["redlight_signal_valid"] = signal
    frame["redlight_confirmed_duration"] = duration
    frame["redlight_crossing_onset"] = onset
    frame["redlight_forward_motion_after_line"] = motion_world.where(motion_world > 0, motion_px / 10)
    frame["redlight_after_line_motion_score"] = motion
    frame["redlight_stopline_identity_confidence"] = identity_weight
    frame["redlight_stopline_crossing_confidence"] = confidence
    frame["redlight_crossing_freshness_score"] = freshness
    frame["redlight_policy_strength"] = strength
    frame["redlight_policy_confidence"] = policy_confidence
    frame["redlight_policy_commitment"] = commitment
    frame["redlight_event_peak_score"] = maximum(strength, confirmed)
    frame["redlight_geometry_valid_flag"] = geometry
    return frame[KEYS + TIMES + OUTPUT_COLUMNS].copy()


__all__ = ["build_redlight_v8", "OUTPUT_COLUMNS"]
