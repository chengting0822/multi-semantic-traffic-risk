"""Frozen S3.3 prefix replay used by the C4O single-vehicle model.

Extracted from the accepted S2/S3.2/S3.3 inference rules.  This module uses
only window scores and trajectory evidence; no labels or case-ID exceptions.
The research-time double-yellow-only guard reads the complete track, so this
is an offline replay, not a claim of fully streaming causal inference.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from traffic_risk.identifiers import normalize_window_keys


KEYS = ["case_key", "video_id", "track_id", "ts_window_idx"]
THRESHOLD = 0.25
ALPHA = 0.7


def _v(row: dict, name: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(name, default))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _b(row: dict, name: str) -> bool:
    return _v(row, name) != 0.0


def _run(flags: list[bool]) -> int:
    current = maximum = 0
    for flag in flags:
        current = current + 1 if flag else 0
        maximum = max(maximum, current)
    return maximum


def _local(row: dict, count: int) -> float:
    semantic = float(_b(row, "semantic_support_flag"))
    reverse = float(_b(row, "reversed_flag"))
    flow = _v(row, "flow_cos", 1.0)
    occupied = _v(row, "occupied_lane_flow_cos", flow)
    ratio = float(np.clip(_v(row, "opposite_lane_occupancy_ratio"), 0, 1))
    after = float(_b(row, "opposite_lane_after_crossing_flag"))
    area_cv = _v(row, "artifact_bbox_area_cv")
    step = _v(row, "artifact_max_center_step")
    yellow = _b(row, "window_double_yellow_crossed_once")
    clean_reverse = reverse > 0.5 and occupied <= -0.75 and _v(row, "trajectory_anomaly_score") >= 0.30 and area_cv <= 0.18 and step >= 2.5
    direction = (reverse > 0.5 and flow < 0.75) or flow < 0.2 or clean_reverse
    yellow_support = 0.95 if yellow and direction else 0.0
    occupancy = ratio * (0.35 + 0.65 * after) if direction else 0.0
    artifact = area_cv > 0.4 and step < 5.0
    turn_artifact = area_cv > 0.18 and reverse > 0.5 and flow > 0.45 and semantic <= 0 and not yellow
    wrongway = 0.0 if count == 1 or artifact or step < 4.0 else reverse * max(float(np.clip((0.35 - flow) / 1.35, 0, 1)), occupancy)
    if turn_artifact:
        yellow_support = occupancy = wrongway = 0.0
    return float(np.clip(max(semantic, yellow_support, occupancy, wrongway), 0, 1))


def _oscillation(row: dict) -> bool:
    score = _v(row, "trajectory_anomaly_score")
    return bool(
        (_v(row, "osc_h30_score") >= 0.80 and score >= 0.30 and _v(row, "osc_h30_two_sided_amp_px") >= 20
         and _v(row, "osc_h30_area_cv", 1.0) <= 0.55 and _v(row, "osc_h30_conf_min") >= 0.15)
        or (_v(row, "osc_h20_score") >= 0.80 and score >= 0.50 and _v(row, "osc_h20_two_sided_amp_px") >= 20
            and _v(row, "osc_h20_area_cv", 1.0) <= 0.55 and _v(row, "osc_h20_conf_min") >= 0.35)
    )


def _double_yellow_normal_flow(rows: list[dict]) -> bool:
    return bool(
        not any(_b(row, "semantic_support_flag") for row in rows)
        and sum(_b(row, "reversed_flag") for row in rows) <= 1
        and any(_b(row, "window_double_yellow_crossed_once") or _b(row, "window_double_yellow_cross_count") for row in rows)
        and min(_v(row, "flow_cos", 1.0) for row in rows) > 0.45
        and max(_v(row, "opposite_lane_after_crossing_run_length") for row in rows) >= 5
    )


def _features(rows: list[dict], scores: list[float], emas: list[float]) -> dict:
    above = [score > THRESHOLD for score in emas]
    reverse = [_b(row, "reversed_flag") for row in rows]
    opposite = [_v(row, "opposite_lane_occupancy_ratio") > 0 for row in rows]
    after = [_b(row, "opposite_lane_after_crossing_flag") for row in rows]
    semantic = [_b(row, "semantic_support_flag") or (r and a) for row, r, a in zip(rows, reverse, after)]
    bad = sum(_b(row, "tracking_quality_bad") for row in rows)
    yellow_count = sum(int(_v(row, "window_double_yellow_cross_count")) for row in rows)
    n = len(rows)
    return {
        "window_count": n, "above_threshold": above, "ema_scores": emas,
        "abnormal_window_count": sum(above), "abnormal_window_ratio": sum(above) / n,
        "max_consecutive_abnormal_windows": _run(above), "max_gmm_ema": max(emas),
        "min_flow_cos": min(_v(row, "flow_cos", 1.0) for row in rows),
        "wrong_way_run_length": _run(reverse), "opposite_lane_run_length": _run(opposite),
        "opposite_lane_occupancy_ratio": sum(opposite) / n,
        "opposite_lane_after_crossing_run_length": _run(after),
        "semantic_support_window_count": sum(semantic),
        "double_yellow_cross_count": yellow_count,
        "double_yellow_distance_min": min((_v(row, "window_distance_to_double_yellow_min") for row in rows if _v(row, "window_distance_to_double_yellow_min") > 0), default=0.0),
        "tracking_quality_bad_window_count": bad,
        "severe_tracking_quality": bad >= max(2, math.ceil(n * 0.35)),
    }


def _decision(f: dict) -> bool:
    """S2 base, then selected S3.2 A+B-strict+C+D+E and S3.3 B+guard."""
    n, abnormal, ratio, consecutive = f["window_count"], f["abnormal_window_count"], f["abnormal_window_ratio"], f["max_consecutive_abnormal_windows"]
    maximum, flow, wrong_run = f["max_gmm_ema"], f["min_flow_cos"], f["wrong_way_run_length"]
    occ, occ_run, after = f["opposite_lane_occupancy_ratio"], f["opposite_lane_run_length"], f["opposite_lane_after_crossing_run_length"]
    support, yellow, yellow_distance = f["semantic_support_window_count"], f["double_yellow_cross_count"], f["double_yellow_distance_min"]
    bad_ratio = f["tracking_quality_bad_window_count"] / n
    severe = f["severe_tracking_quality"]
    base = abnormal >= 5 and ratio >= 0.10 and consecutive >= 4
    guard_e = maximum >= THRESHOLD and support == 0 and yellow == 0 and wrong_run == 0 and after == 0 and occ_run <= 3 and flow >= 0.80
    recover_a = abnormal >= 5 and ratio >= 0.10 and 3 <= consecutive < 4 and maximum >= THRESHOLD and bad_ratio <= 0.20 and (yellow >= 1 or occ >= 0.25 or support >= 1 or wrong_run >= 1 or after >= 1)
    recover_b = maximum >= THRESHOLD and abnormal >= 4 and occ >= 0.30 and occ_run >= 4 and wrong_run >= 5 and flow <= -0.50 and not severe
    recover_c = maximum >= THRESHOLD and yellow >= 2 and yellow_distance <= 3.0 and occ >= 0.30 and (after >= 1 or wrong_run >= 2) and not severe
    predicted = base or (not guard_e and (recover_a or recover_b or recover_c))
    suppress_d = support == 0 and yellow == 0 and occ == 0 and yellow_distance >= 80.0 and (bad_ratio > 0 or flow > -0.20)
    if predicted and suppress_d:
        predicted = False
    if not predicted:
        guard_s33 = maximum >= THRESHOLD and support == 0 and yellow == 0 and wrong_run == 0 and after <= 0 and occ_run <= 3 and flow >= 0.8
        recover_s33 = (
            maximum >= THRESHOLD * 1.3 and abnormal >= 1 and ratio <= 0.10 and consecutive <= 1
            and occ >= 0.15 and after >= 2 and occ_run >= 2 and yellow_distance <= 26.0
            and flow <= 0.55 and bad_ratio <= 0.0 and not severe
        )
        predicted = not guard_s33 and recover_s33
    return bool(predicted)


def build_s33_prefix(fusion_v0: pd.DataFrame, trajectory_sidecar: pd.DataFrame) -> pd.DataFrame:
    """Return all four C4O-required prefix channels for any car-only window set."""
    fusion = normalize_window_keys(fusion_v0)
    sidecar = normalize_window_keys(trajectory_sidecar)
    for label, frame, required in (
        ("fusion", fusion, KEYS + ["start_sec", "trajectory_anomaly_score"]),
        ("trajectory", sidecar, KEYS + ["flow_cos", "artifact_max_center_step"]),
    ):
        missing = [column for column in required if column not in frame]
        if missing:
            raise ValueError(f"{label} lacks S3.3 fields: {missing}")
        if frame.duplicated(KEYS).any():
            raise ValueError(f"{label} has duplicate S3.3 windows")
    source = fusion[[*KEYS, "start_sec", "trajectory_anomaly_score"]].merge(
        sidecar[[*KEYS, *[column for column in sidecar if column not in KEYS and column not in ("start_sec", "end_sec", "trajectory_anomaly_score")]]],
        on=KEYS, how="left", validate="one_to_one", indicator=True,
    )
    if not source["_merge"].eq("both").all():
        raise ValueError("trajectory sidecar does not cover every fusion window")
    source = source.drop(columns="_merge").sort_values(["case_key", "ts_window_idx", "start_sec"], kind="mergesort")
    output: list[dict] = []
    for (case_key, video_id, track_id), group in source.groupby(KEYS[:3], sort=False):
        rows = group.to_dict("records")
        gate = _double_yellow_normal_flow(rows)
        scores: list[float] = []
        emas: list[float] = []
        recent_semantic: list[bool] = []
        recent_oscillation: list[bool] = []
        for row in rows:
            score = _v(row, "trajectory_anomaly_score")
            scores.append(min(score, THRESHOLD - 1e-6) if gate else score)
            emas.append(scores[-1] if not emas else ALPHA * scores[-1] + (1 - ALPHA) * emas[-1])
            features = _features(rows[:len(scores)], scores, emas)
            local = _local(row, len(rows))
            oscillation = _oscillation(row)
            if gate:
                local = 0.0
                oscillation = False
            recent_semantic.append(local > 0.1)
            recent_oscillation.append(oscillation)
            confirmed = any(recent_semantic[-13:]) or any(recent_oscillation[-3:])
            margin = emas[-1] - THRESHOLD
            output.append({
                "case_key": case_key, "video_id": video_id, "track_id": track_id,
                "ts_window_idx": int(row["ts_window_idx"]),
                "s33_prefix_state_confirmed": float(_decision(features) and confirmed),
                "s33_confirmed_ema_margin": margin * float(confirmed),
                "s33_current_local_semantic_support": max(local, 0.35 * float(oscillation)),
                "s33_confirmed_ema_pos": max(0.0, margin) * float(confirmed),
            })
    return pd.DataFrame(output, columns=[*KEYS, "s33_prefix_state_confirmed", "s33_confirmed_ema_margin", "s33_current_local_semantic_support", "s33_confirmed_ema_pos"])


__all__ = ["build_s33_prefix"]
