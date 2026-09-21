"""Frozen B5 GMM scoring and causal v0 trajectory semantic features."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import B5_FEATURES, STRIDE_SEC

THRESHOLD = 321.366
EMA_ALPHA = 0.3
MIN_COUNT = 5
MIN_RATIO = 0.1
MIN_CONSECUTIVE = 4
TOP_K_WINDOWS = 3


def _logsumexp(values: np.ndarray, axis: int = 1) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(maximum, axis=axis) + np.log(np.sum(np.exp(values - maximum), axis=axis))


def score_fixed20(features: pd.DataFrame, model_json: Path) -> pd.DataFrame:
    """Apply the frozen StandardScaler and full-covariance GMM without sklearn."""

    payload = json.loads(Path(model_json).read_text(encoding="utf-8"))
    columns = list(payload["feature_columns"])
    if columns != B5_FEATURES:
        raise ValueError("trajectory GMM feature contract does not match B5")
    missing = [column for column in columns if column not in features]
    if missing:
        raise ValueError(f"fixed20 table is missing B5 columns: {missing}")
    model_frame = features[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).copy()
    # The frozen B5 GMM predates the occupied-lane-direction repair.  Its
    # `reversed_flag` input was defined from best-lane flow_cos; v8 still uses
    # the repaired occupied-lane value carried by the unchanged feature table.
    model_frame["reversed_flag"] = (model_frame["flow_cos"] < 0.3).astype(float)
    values = model_frame.to_numpy(dtype=float)
    mean = np.asarray(payload["scaler"]["mean"], dtype=float)
    scale = np.asarray(payload["scaler"]["scale"], dtype=float)
    scaled = (values - mean) / scale
    weights = np.asarray(payload["gmm"]["weights"], dtype=float)
    means = np.asarray(payload["gmm"]["means"], dtype=float)
    precisions = np.asarray(payload["gmm"]["precisions_cholesky"], dtype=float)
    components: list[np.ndarray] = []
    constant = len(columns) * math.log(2.0 * math.pi)
    for component in range(len(weights)):
        transformed = scaled @ precisions[component] - means[component] @ precisions[component]
        log_det = float(np.sum(np.log(np.diag(precisions[component]))))
        components.append(-0.5 * (np.sum(transformed**2, axis=1) + constant) + log_det + math.log(weights[component]))
    output = features.copy()
    output["gmm_raw_score"] = -_logsumexp(np.column_stack(components), axis=1)
    return output


def _clip01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _max_run(values: list[bool]) -> int:
    current = best = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def _semantic_row(row: pd.Series, observed_count: int) -> dict[str, Any]:
    path_bonus = _clip01(
        0.25 * float(row["prefix_mix_lane_aware_triggered"])
        + 0.35 * float(row["prefix_mix_candidate2_triggered"])
        + 0.35 * float(row["prefix_mix_aggressive_lane_rescue_triggered"])
        + 0.25 * float(row["prefix_mix_double_yellow_triggered"])
        + 0.20 * float(row["prefix_mix_score_aware_semantic_triggered"])
    )
    conf_min = _clip01(float(row.get("conf_min", 0.0)))
    bad_ratio = _clip01(float(row["prefix_tracking_quality_bad_window_count"]) / max(1, observed_count))
    severe = bool(row["prefix_severe_tracking_quality"])
    quality_penalty = _clip01(0.45 * bad_ratio + 0.25 * (1.0 - conf_min) + (0.35 if severe else 0.0))
    score = _clip01(
        0.42 * _clip01(max(0.0, float(row["prefix_top_k_mean_gmm_ema"])) / 2000.0)
        + 0.18 * _clip01(float(row["prefix_abnormal_window_ratio"]))
        + 0.15 * _clip01(float(row["prefix_max_consecutive_abnormal_windows"]) / 6.0)
        + 0.10 * _clip01(float(row["prefix_semantic_support_window_count"]) / 3.0)
        + 0.15 * path_bonus
        - 0.20 * quality_penalty
    )
    level = 0 if score < 0.20 else (1 if score < 0.45 else (2 if score < 0.70 else 3))
    if bool(row["prefix_mix_aggressive_lane_rescue_triggered"]) and score >= 0.55:
        level = max(level, 3)
    reliability = _clip01(0.65 * conf_min + 0.35 * (1.0 - bad_ratio) - (0.25 if severe else 0.0))
    return {
        "trajectory_anomaly_score": score,
        "trajectory_anomaly_level": level,
        "trajectory_anomaly_duration_sec_proxy": min(max(float(row["prefix_positive_duration_sec"]), 0.0), 6.0),
        "trajectory_anomaly_reliability": reliability,
        "trajectory_anomaly_signal_valid": int(reliability >= 0.45 and not severe),
    }


def build_causal_semantics(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce the accepted causal-prefix evidence and v0 semantic bridge."""

    prefix_rows: list[dict[str, Any]] = []
    semantic_rows: list[dict[str, Any]] = []
    for _case_key, group in scored.groupby("case_key", sort=False):
        ordered = group.sort_values(["ts_window_idx", "start_sec"], kind="mergesort")
        ema_values: list[float] = []
        above: list[bool] = []
        reversed_flags: list[bool] = []
        semantic_flags: list[bool] = []
        opposite_flags: list[bool] = []
        bad_flags: list[bool] = []
        min_flow = 1.0
        max_speed = 0.0
        max_after_run = 0
        double_yellow_count = 0
        for observed, (_, source) in enumerate(ordered.iterrows(), start=1):
            raw = float(source["gmm_raw_score"])
            ema = raw if not ema_values else EMA_ALPHA * raw + (1.0 - EMA_ALPHA) * ema_values[-1]
            ema_values.append(ema)
            above.append(ema > THRESHOLD)
            # The causal-prefix policy was fitted before the occupied-lane
            # direction repair.  Preserve that historical contract here,
            # while the v8 branch below continues to use the repaired fields
            # from the fixed20 table.
            legacy_reversed = bool(float(source.get("flow_cos", 1.0)) < 0.3)
            legacy_semantic_support = bool(
                legacy_reversed
                and int(float(source.get("opposite_lane_after_crossing_run_length", 0.0))) > 0
            )
            reversed_flags.append(legacy_reversed)
            after = bool(float(source.get("opposite_lane_after_crossing_flag", 0.0)) >= 0.5)
            semantic_flags.append(legacy_semantic_support or (legacy_reversed and after))
            opposite_flags.append(float(source.get("opposite_lane_occupancy_ratio", 0.0)) > 0.0)
            bad_flags.append(bool(float(source.get("tracking_quality_bad", 0.0)) >= 0.5))
            min_flow = min(min_flow, float(source.get("flow_cos", 1.0)))
            max_speed = max(max_speed, float(source.get("speed_mean", 0.0)))
            max_after_run = max(max_after_run, int(float(source.get("opposite_lane_after_crossing_run_length", 0.0))))
            double_yellow_count += int(float(source.get("window_double_yellow_cross_count", 0.0)))
            abnormal_count, abnormal_ratio = sum(above), sum(above) / observed
            abnormal_run = _max_run(above)
            reverse_count, reverse_run = sum(reversed_flags), _max_run(reversed_flags)
            semantic_count, semantic_run = sum(semantic_flags), _max_run(semantic_flags)
            bad_count = sum(bad_flags)
            severe = bad_count >= max(2, math.ceil(observed * 0.35))
            opposite_ratio = sum(opposite_flags) / observed
            lane_aware = semantic_count >= 1 and semantic_run >= 1 and (reverse_run >= 1 or min_flow <= 0.3) and not severe
            candidate2 = opposite_ratio >= 0.42 and max_after_run >= 2 and reverse_count >= 2 and reverse_run >= 2 and min_flow <= 0.3 and not severe
            aggressive = max_after_run >= 6 and opposite_ratio >= 0.42 and max_speed >= 1.0 and min_flow <= 0.7 and float(source.get("speed_mean", 0.0)) >= 0.9 and not severe
            yellow = double_yellow_count > 0 and max_speed >= 1.0 and not severe
            top_k = sorted(ema_values)[-min(TOP_K_WINDOWS, len(ema_values)):]
            medium = max(ema_values) >= THRESHOLD + 5.0
            row = source.to_dict()
            row["reversed_flag"] = int(legacy_reversed)
            row["semantic_support_flag"] = int(legacy_semantic_support)
            row.update({
                "gmm_ema_score": ema, "above_threshold": above[-1],
                "prefix_abnormal_window_count": abnormal_count,
                "prefix_abnormal_window_ratio": abnormal_ratio,
                "prefix_max_consecutive_abnormal_windows": abnormal_run,
                "prefix_positive_duration_sec": abnormal_count * STRIDE_SEC,
                "prefix_top_k_mean_gmm_ema": float(np.mean(top_k)),
                "prefix_semantic_support_window_count": semantic_count,
                "prefix_tracking_quality_bad_window_count": bad_count,
                "prefix_severe_tracking_quality": severe,
                "prefix_mix_lane_aware_triggered": lane_aware,
                "prefix_mix_candidate2_triggered": candidate2,
                "prefix_mix_aggressive_lane_rescue_triggered": aggressive,
                "prefix_mix_double_yellow_triggered": yellow,
                "prefix_mix_score_aware_semantic_triggered": medium and (lane_aware or candidate2 or yellow),
            })
            prefix_rows.append(row)
            semantic = {key: row[key] for key in ["case_key", "source_type", "source_id", "video_id", "track_id", "ts_window_idx", "start_sec", "end_sec"]}
            semantic.update(_semantic_row(pd.Series(row), observed))
            semantic_rows.append(semantic)
    return pd.DataFrame(prefix_rows), pd.DataFrame(semantic_rows)


__all__ = ["build_causal_semantics", "score_fixed20"]
