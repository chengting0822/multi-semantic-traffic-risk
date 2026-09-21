"""Accepted policy-grounded v8 fusion fields needed by C4O."""
from __future__ import annotations
import numpy as np
import pandas as pd
from traffic_risk.upstream.v8_common import (KEYS, TIMES, clip_nonnegative, clip01, cumulative_duration_by_track, join, number as numeric_series, safe_divide)
SHORT_OVERSPEED_BURST_SEC = 1.5

def binary(series: pd.Series, threshold: float = 0.5) -> pd.Series:
    return (series.astype(float) >= threshold).astype(float)

def window_duration(frame: pd.DataFrame) -> pd.Series:
    if "window_sec" in frame:
        candidate = pd.to_numeric(frame["window_sec"], errors="coerce")
        if candidate.notna().any():
            return clip_nonnegative(candidate.fillna(0.0))
    return clip_nonnegative(numeric_series(frame, "end_sec") - numeric_series(frame, "start_sec"))

def compute_family_relations(
    frame: pd.DataFrame,
    trajectory_active: pd.Series,
    overspeed_active: pd.Series,
    redlight_active: pd.Series,
    trajectory_level: pd.Series,
    overspeed_level: pd.Series,
    redlight_level: pd.Series,
) -> dict[str, pd.Series]:
    result = {
        "active_family_count": pd.Series(0.0, index=frame.index, dtype=float),
        "family_pair_overspeed_redlight": pd.Series(0.0, index=frame.index, dtype=float),
        "family_pair_overspeed_trajectory": pd.Series(0.0, index=frame.index, dtype=float),
        "family_pair_redlight_trajectory": pd.Series(0.0, index=frame.index, dtype=float),
        "family_all_three": pd.Series(0.0, index=frame.index, dtype=float),
        "trajectory_overspeed_overlap_ratio": pd.Series(0.0, index=frame.index, dtype=float),
        "trajectory_redlight_overlap_ratio": pd.Series(0.0, index=frame.index, dtype=float),
        "overspeed_redlight_overlap_ratio": pd.Series(0.0, index=frame.index, dtype=float),
        "all_three_overlap_ratio": pd.Series(0.0, index=frame.index, dtype=float),
        "cooccurrence_duration_sec": pd.Series(0.0, index=frame.index, dtype=float),
        "peak_window_family_count": pd.Series(0.0, index=frame.index, dtype=float),
        "multi_family_burst_count": pd.Series(0.0, index=frame.index, dtype=float),
        "trajectory_before_redlight_count": pd.Series(0.0, index=frame.index, dtype=float),
        "trajectory_before_overspeed_count": pd.Series(0.0, index=frame.index, dtype=float),
        "overspeed_before_trajectory_count": pd.Series(0.0, index=frame.index, dtype=float),
        "redlight_then_trajectory_persistence": pd.Series(0.0, index=frame.index, dtype=float),
        "redlight_overspeed_overlap_ratio": pd.Series(0.0, index=frame.index, dtype=float),
        "redlight_trajectory_overlap_ratio": pd.Series(0.0, index=frame.index, dtype=float),
        "max_module_level": pd.Series(0.0, index=frame.index, dtype=float),
    }

    ordered = frame.sort_values(["video_id", "track_id", "ts_window_idx", "start_sec", "end_sec"], kind="mergesort")
    duration = clip_nonnegative(numeric_series(frame, "end_sec") - numeric_series(frame, "start_sec"))
    for (_, _), index in ordered.groupby(["video_id", "track_id"], sort=False).groups.items():
        idx = pd.Index(index)
        t = trajectory_active.loc[idx].to_numpy(dtype=float)
        o = overspeed_active.loc[idx].to_numpy(dtype=float)
        r = redlight_active.loc[idx].to_numpy(dtype=float)
        d = duration.loc[idx].to_numpy(dtype=float)
        family_count = t + o + r
        pair_or = ((o >= 1.0) & (r >= 1.0)).astype(float)
        pair_ot = ((o >= 1.0) & (t >= 1.0)).astype(float)
        pair_rt = ((r >= 1.0) & (t >= 1.0)).astype(float)
        all_three = ((t >= 1.0) & (o >= 1.0) & (r >= 1.0)).astype(float)
        steps = np.arange(1, len(idx) + 1, dtype=float)
        prev_multi = np.concatenate([[0.0], family_count[:-1]])
        prior_traj = np.concatenate([[0.0], np.cumsum(t)[:-1]])
        prior_over = np.concatenate([[0.0], np.cumsum(o)[:-1]])
        prior_red = np.concatenate([[0.0], np.cumsum(r)[:-1]])

        result["active_family_count"].loc[idx] = family_count
        result["family_pair_overspeed_redlight"].loc[idx] = pair_or
        result["family_pair_overspeed_trajectory"].loc[idx] = pair_ot
        result["family_pair_redlight_trajectory"].loc[idx] = pair_rt
        result["family_all_three"].loc[idx] = all_three
        result["trajectory_overspeed_overlap_ratio"].loc[idx] = np.cumsum(pair_ot) / steps
        result["trajectory_redlight_overlap_ratio"].loc[idx] = np.cumsum(pair_rt) / steps
        result["overspeed_redlight_overlap_ratio"].loc[idx] = np.cumsum(pair_or) / steps
        result["all_three_overlap_ratio"].loc[idx] = np.cumsum(all_three) / steps
        # 改為 K=3 滑動視窗加總（取最近 3 個時間窗口的共現時長），可隨共現消失而下降
        _cooc_per_win = d * (family_count >= 2.0)
        _k = 3
        _rolling = np.array([
            _cooc_per_win[max(0, i - _k + 1):i + 1].sum()
            for i in range(len(_cooc_per_win))
        ])
        result["cooccurrence_duration_sec"].loc[idx] = _rolling
        result["peak_window_family_count"].loc[idx] = np.maximum.accumulate(family_count)
        result["multi_family_burst_count"].loc[idx] = np.cumsum((family_count >= 2.0) & (prev_multi < 2.0))
        result["trajectory_before_redlight_count"].loc[idx] = np.cumsum((r >= 1.0) & (prior_traj > 0.0))
        result["trajectory_before_overspeed_count"].loc[idx] = np.cumsum((o >= 1.0) & (prior_traj > 0.0))
        result["overspeed_before_trajectory_count"].loc[idx] = np.cumsum((t >= 1.0) & (prior_over > 0.0))
        result["redlight_then_trajectory_persistence"].loc[idx] = np.cumsum(d * ((t >= 1.0) & (prior_red > 0.0)))
        result["redlight_overspeed_overlap_ratio"].loc[idx] = np.cumsum(pair_or) / steps
        result["redlight_trajectory_overlap_ratio"].loc[idx] = np.cumsum(pair_rt) / steps
        result["max_module_level"].loc[idx] = np.maximum.reduce(
            [
                trajectory_level.loc[idx].to_numpy(dtype=float),
                overspeed_level.loc[idx].to_numpy(dtype=float),
                redlight_level.loc[idx].to_numpy(dtype=float),
            ]
        )
    return result



def compute_overspeed_temporal_shape_features(
    frame: pd.DataFrame,
    speed_reference: pd.Series,
    speed_peak: pd.Series,
    speed_mean: pd.Series,
    overspeed_any_active: pd.Series,
    moderate_overspeed_active: pd.Series,
    severe_overspeed_active: pd.Series,
) -> dict[str, pd.Series]:
    ordered = frame.sort_values(["video_id", "track_id", "ts_window_idx", "start_sec", "end_sec"], kind="mergesort")
    duration = window_duration(frame).clip(lower=1e-6)
    speed_rise_slope = pd.Series(0.0, index=frame.index, dtype=float)
    speed_decay_slope = pd.Series(0.0, index=frame.index, dtype=float)
    short_overspeed_burst_count = pd.Series(0.0, index=frame.index, dtype=float)

    for (_, _), index in ordered.groupby(["video_id", "track_id"], sort=False).groups.items():
        idx = pd.Index(index)
        reference_values = speed_reference.loc[idx].to_numpy(dtype=float)
        duration_values = duration.loc[idx].to_numpy(dtype=float)
        previous_reference = np.concatenate([[reference_values[0]], reference_values[:-1]])
        delta = reference_values - previous_reference
        speed_rise_slope.loc[idx] = np.maximum(delta, 0.0) / np.maximum(duration_values, 1e-6)
        speed_decay_slope.loc[idx] = np.maximum(-delta, 0.0) / np.maximum(duration_values, 1e-6)

        active_values = overspeed_any_active.loc[idx].to_numpy(dtype=float)
        burst_count = 0.0
        burst_duration = 0.0
        in_burst = False
        burst_counts = np.zeros(len(idx), dtype=float)
        for position, active_value in enumerate(active_values):
            if active_value >= 1.0:
                burst_duration += duration_values[position]
                in_burst = True
            elif in_burst:
                if burst_duration <= SHORT_OVERSPEED_BURST_SEC:
                    burst_count += 1.0
                in_burst = False
                burst_duration = 0.0
            burst_counts[position] = burst_count
        short_overspeed_burst_count.loc[idx] = burst_counts

    moderate_overspeed_duration = cumulative_duration_by_track(frame, moderate_overspeed_active)
    severe_overspeed_duration = cumulative_duration_by_track(frame, severe_overspeed_active)
    overspeed_active_duration = cumulative_duration_by_track(frame, overspeed_any_active)
    elapsed_track_duration = cumulative_duration_by_track(frame, pd.Series(1.0, index=frame.index, dtype=float))
    ratio_denominator = moderate_overspeed_duration.where(moderate_overspeed_duration > 0.0, duration)

    return {
        "speed_rise_slope": speed_rise_slope,
        "speed_decay_slope": speed_decay_slope,
        "peak_to_mean_speed_gap": clip_nonnegative(speed_peak - speed_mean),
        "moderate_overspeed_duration": moderate_overspeed_duration,
        "severe_overspeed_duration": severe_overspeed_duration,
        "severe_to_moderate_duration_ratio": safe_divide(severe_overspeed_duration, ratio_denominator.clip(lower=1e-6), default=0.0),
        "short_overspeed_burst_count": short_overspeed_burst_count,
        "sustained_overspeed_ratio": clip01(safe_divide(overspeed_active_duration, elapsed_track_duration.clip(lower=1e-6), default=0.0)),
    }


def compute_family_role_codes(
    trajectory_strength: pd.Series,
    overspeed_strength: pd.Series,
    redlight_strength: pd.Series,
) -> dict[str, pd.Series]:
    family_strength = np.column_stack(
        [
            trajectory_strength.to_numpy(dtype=float),
            overspeed_strength.to_numpy(dtype=float),
            redlight_strength.to_numpy(dtype=float),
        ]
    )
    family_codes = np.array([1.0, 2.0, 3.0], dtype=float)
    dominant_family = np.zeros(family_strength.shape[0], dtype=float)
    secondary_family = np.zeros(family_strength.shape[0], dtype=float)

    for row_index, row in enumerate(family_strength):
        order = np.argsort(-row, kind="mergesort")
        if row[order[0]] > 0.0:
            dominant_family[row_index] = family_codes[order[0]]
        if row[order[1]] > 0.0:
            secondary_family[row_index] = family_codes[order[1]]

    return {
        "dominant_family": pd.Series(dominant_family, index=trajectory_strength.index, dtype=float),
        "secondary_family": pd.Series(secondary_family, index=trajectory_strength.index, dtype=float),
    }




def build_fusion_v8(base_frame: pd.DataFrame, trajectory_v8: pd.DataFrame, overspeed_v8: pd.DataFrame, redlight_v8: pd.DataFrame) -> pd.DataFrame:
    """Compute original minimal-C4O v8 fusion formulas on aligned windows."""
    frame = join(base_frame, trajectory_v8, label="trajectory_v8")
    frame = join(frame, overspeed_v8, label="overspeed_v8")
    frame = join(frame, redlight_v8, label="redlight_v8")
    frame = frame.sort_values(["video_id", "track_id", "ts_window_idx", "start_sec", "end_sec"], kind="mergesort").reset_index(drop=True)
    trajectory_peak = pd.concat(
        [
            numeric_series(frame, "trajectory_wrong_way_score"),
            numeric_series(frame, "trajectory_weaving_score"),
            numeric_series(frame, "trajectory_double_yellow_crossing_score"),
            numeric_series(frame, "trajectory_opposite_lane_driving_score"),
            numeric_series(frame, "trajectory_illegal_flow_direction_score"),
        ],
        axis=1,
    ).max(axis=1)
    trajectory_confidence_peak = pd.concat(
        [
            numeric_series(frame, "trajectory_wrong_way_confidence"),
            numeric_series(frame, "trajectory_weaving_confidence"),
            numeric_series(frame, "trajectory_double_yellow_crossing_confidence"),
            numeric_series(frame, "trajectory_opposite_lane_driving_confidence"),
            numeric_series(frame, "trajectory_illegal_flow_direction_confidence"),
        ],
        axis=1,
    ).max(axis=1)
    trajectory_other_peak = pd.concat(
        [
            numeric_series(frame, "trajectory_wrong_way_score"),
            numeric_series(frame, "trajectory_double_yellow_crossing_score"),
            numeric_series(frame, "trajectory_opposite_lane_driving_score"),
            numeric_series(frame, "trajectory_illegal_flow_direction_score"),
        ],
        axis=1,
    ).max(axis=1)

    overspeed_policy_peak = clip01(
        pd.concat(
            [
                safe_divide(clip_nonnegative(numeric_series(frame, "speed_kmh_p95") - 70.0), 60.0, default=0.0),
                numeric_series(frame, "speed_band_90_110_flag"),
                numeric_series(frame, "speed_band_ge_110_flag"),
            ],
            axis=1,
        ).max(axis=1)
    )
    overspeed_margin_kmh = clip_nonnegative(numeric_series(frame, "speed_kmh_p95") - 70.0)
    redlight_policy_peak = pd.concat(
        [
            numeric_series(frame, "redlight_event_peak_score"),
            numeric_series(frame, "redlight_policy_strength"),
        ],
        axis=1,
    ).max(axis=1)
    overspeed_level_from_bands = pd.concat(
        [
            3.0 * numeric_series(frame, "speed_band_ge_110_flag"),
            2.0 * numeric_series(frame, "speed_band_90_110_flag"),
            1.0 * numeric_series(frame, "speed_band_70_90_flag"),
        ],
        axis=1,
    ).max(axis=1)

    trajectory_weaving_soft_support = binary(
        pd.concat(
            [
                binary(numeric_series(frame, "trajectory_subtype_switch_count"), threshold=1.0),
                binary(numeric_series(frame, "trajectory_weaving_confidence"), threshold=0.54),
                binary(numeric_series(frame, "trajectory_post_peak_persistence"), threshold=0.70),
            ],
            axis=1,
        ).max(axis=1)
    )
    trajectory_weaving_soft_active = (
        binary(numeric_series(frame, "trajectory_weaving_score"), threshold=0.55)
        * (numeric_series(frame, "trajectory_weaving_score") >= trajectory_other_peak).astype(float)
        * trajectory_weaving_soft_support
    )
    trajectory_family_active = binary(
        pd.concat(
            [
                binary(numeric_series(frame, "trajectory_wrong_way_flag")),
                binary(numeric_series(frame, "trajectory_weaving_flag")),
                binary(numeric_series(frame, "trajectory_double_yellow_crossing_flag")),
                binary(numeric_series(frame, "trajectory_opposite_lane_driving_flag")),
                binary(numeric_series(frame, "trajectory_illegal_flow_direction_flag")),
                binary(trajectory_other_peak, threshold=0.70),
                trajectory_weaving_soft_active,
                binary(numeric_series(frame, "trajectory_anomaly_level"), threshold=1.0),
            ],
            axis=1,
        ).max(axis=1)
    )
    overspeed_family_active = binary(
        pd.concat(
            [
                binary(overspeed_margin_kmh, threshold=15.0),
                numeric_series(frame, "speed_band_90_110_flag"),
                numeric_series(frame, "speed_band_ge_110_flag"),
            ],
            axis=1,
        ).max(axis=1)
    )
    redlight_family_active = binary(
        pd.concat(
            [
                binary(numeric_series(frame, "redlight_event_peak_score"), threshold=0.50),
                binary(numeric_series(frame, "redlight_confirmed_duration"), threshold=0.01),
                binary(numeric_series(frame, "redlight_confirmed")),
            ],
            axis=1,
        ).max(axis=1)
    )

    trajectory_level = pd.concat(
        [
            3.0 * binary(numeric_series(frame, "trajectory_wrong_way_flag")),
            2.0 * binary(numeric_series(frame, "trajectory_weaving_flag")),
            3.0 * binary(numeric_series(frame, "trajectory_double_yellow_crossing_flag")),
            3.0 * binary(numeric_series(frame, "trajectory_opposite_lane_driving_flag")),
            3.0 * binary(numeric_series(frame, "trajectory_illegal_flow_direction_flag")),
            numeric_series(frame, "trajectory_anomaly_level"),
        ],
        axis=1,
    ).max(axis=1)
    overspeed_level = overspeed_level_from_bands
    redlight_level = pd.concat(
        [
            3.0 * binary(numeric_series(frame, "redlight_event_peak_score"), threshold=0.85),
            2.0 * redlight_family_active,
        ],
        axis=1,
    ).max(axis=1)

    relation_features = compute_family_relations(
        frame,
        trajectory_family_active,
        overspeed_family_active,
        redlight_family_active,
        trajectory_level,
        overspeed_level,
        redlight_level,
    )
    for name, series in relation_features.items():
        frame[name] = series

    severe_overspeed_active = binary(
        pd.concat(
            [
                numeric_series(frame, "speed_band_ge_110_flag"),
                binary(overspeed_level, threshold=3.0),
            ],
            axis=1,
        ).max(axis=1)
    )
    moderate_overspeed_active = (
        binary(
            pd.concat(
                [
                    numeric_series(frame, "speed_band_70_90_flag"),
                    numeric_series(frame, "speed_band_90_110_flag"),
                    binary(overspeed_level, threshold=1.0),
                ],
                axis=1,
            ).max(axis=1)
        )
        * (1.0 - severe_overspeed_active)
    ).clip(lower=0.0, upper=1.0)
    overspeed_temporal_features = compute_overspeed_temporal_shape_features(
        frame,
        speed_reference=numeric_series(frame, "speed_kmh_p95"),
        speed_peak=numeric_series(frame, "speed_kmh_max"),
        speed_mean=numeric_series(frame, "speed_kmh_mean"),
        overspeed_any_active=overspeed_family_active,
        moderate_overspeed_active=moderate_overspeed_active,
        severe_overspeed_active=severe_overspeed_active,
    )
    for name, series in overspeed_temporal_features.items():
        frame[name] = series

    family_role_features = compute_family_role_codes(
        trajectory_strength=pd.concat([trajectory_peak, clip01(safe_divide(trajectory_level, 3.0, default=0.0))], axis=1).max(axis=1),
        overspeed_strength=pd.concat([overspeed_policy_peak, clip01(safe_divide(overspeed_level, 3.0, default=0.0))], axis=1).max(axis=1),
        redlight_strength=pd.concat([redlight_policy_peak, clip01(safe_divide(redlight_level, 3.0, default=0.0))], axis=1).max(axis=1),
    )
    for name, series in family_role_features.items():
        frame[name] = series

    frame["trajectory_policy_peak"] = trajectory_peak
    frame["trajectory_confidence_peak"] = trajectory_confidence_peak
    frame["overspeed_policy_peak"] = overspeed_policy_peak
    frame["redlight_policy_peak"] = redlight_policy_peak
    module_reliability_frame = pd.concat(
        [trajectory_confidence_peak, numeric_series(frame, "overspeed_reliability"), numeric_series(frame, "redlight_signal_valid")],
        axis=1,
    )
    frame["policy_reliability_mean"] = module_reliability_frame.mean(axis=1)
    frame["policy_reliability_min"] = module_reliability_frame.min(axis=1)
    frame["policy_reliability_gap"] = clip_nonnegative(frame["policy_reliability_mean"] - frame["policy_reliability_min"])
    frame["module_reliability_gap"] = clip_nonnegative(module_reliability_frame.max(axis=1) - module_reliability_frame.min(axis=1))
    frame["policy_pair_peak_trajectory_overspeed"] = trajectory_peak * overspeed_policy_peak
    frame["policy_pair_peak_trajectory_redlight"] = trajectory_peak * redlight_policy_peak
    frame["policy_pair_peak_overspeed_redlight"] = overspeed_policy_peak * redlight_policy_peak
    frame["policy_peak_triplet"] = pd.concat([trajectory_peak, overspeed_policy_peak, redlight_policy_peak], axis=1).max(axis=1)
    frame["semantic_current_score"] = pd.concat([trajectory_peak, overspeed_policy_peak, redlight_policy_peak], axis=1).max(axis=1)
    frame["trajectory_family_active"] = trajectory_family_active
    frame["overspeed_family_active"] = overspeed_family_active
    frame["redlight_family_active"] = redlight_family_active
    frame["trajectory_peak"] = trajectory_peak
    frame["trajectory_duration_reliable"] = numeric_series(frame, "trajectory_anomaly_duration_sec_proxy") * numeric_series(frame, "trajectory_anomaly_reliability")
    frame["trajectory_score_reliable"] = numeric_series(frame, "trajectory_anomaly_score") * numeric_series(frame, "trajectory_anomaly_reliability")
    frame["pa_trajectory_duration_reliable"] = frame["trajectory_duration_reliable"]
    frame["pa_trajectory_score_reliable"] = frame["trajectory_score_reliable"]
    frame["pa_total_duration_proxy"] = (
        numeric_series(frame, "trajectory_anomaly_duration_sec_proxy")
        + numeric_series(frame, "overspeed_duration_sec_proxy")
        + numeric_series(frame, "redlight_duration_sec_proxy")
    )


    return frame

__all__ = ["build_fusion_v8"]
