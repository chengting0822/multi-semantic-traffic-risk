from __future__ import annotations

import pandas as pd
import pytest

from traffic_risk.upstream.fusion_v8 import build_fusion_v8
from traffic_risk.upstream.redlight.v8 import build_redlight_v8
from traffic_risk.upstream.s33 import build_s33_prefix


def _keys() -> list[dict]:
    return [
        {"case_key": "junction-a/7", "source_type": "anomaly_csv", "source_id": "junction-a",
         "video_id": "junction-a", "track_id": "7", "ts_window_idx": i,
         "start_sec": i * 0.3333, "end_sec": i * 0.3333 + 0.6667}
        for i in range(2)
    ]


def test_s33_prefix_uses_text_video_id_and_local_evidence() -> None:
    rows = _keys()
    fusion = pd.DataFrame([{**row, "trajectory_anomaly_score": score} for row, score in zip(rows, [0.1, 0.5])])
    trajectory = pd.DataFrame([
        {**row, "flow_cos": flow, "artifact_max_center_step": 6.0,
         "semantic_support_flag": semantic, "occupied_lane_flow_cos": flow,
         "opposite_lane_after_crossing_flag": semantic}
        for row, flow, semantic in zip(rows, [1.0, -0.9], [0, 1])
    ])
    result = build_s33_prefix(fusion, trajectory)
    assert result.video_id.tolist() == ["junction-a", "junction-a"]
    assert result.s33_current_local_semantic_support.tolist() == [0.0, 1.0]
    assert result.s33_prefix_state_confirmed.tolist() == [0.0, 0.0]  # fewer than five abnormal windows
    assert result.s33_confirmed_ema_margin.iloc[0] == 0.0
    assert result.s33_confirmed_ema_pos.iloc[1] > 0.0


def test_redlight_v8_uses_crossing_and_causal_duration() -> None:
    rows = _keys()
    module = pd.DataFrame([{**row, "redlight_confirmed": 1, "redlight_score": 0.9} for row in rows])
    sidecar = pd.DataFrame([
        {**row, "crossing_event_in_window": 1, "crossing_red_signal": 1,
         "stopline_geometry_valid": 1, "tl_state_window": "red", "tl_red_ratio": 1,
         "tl_prob_red_mean": 0.9, "tl_prob_red_max": 0.95,
         "stopline_identity_confidence": 1, "stopline_multi_candidate_flag": 0,
         "motion_after_crossing_px": 20, "crossing_timestamp_sec": row["start_sec"] + 0.1}
        for row in rows
    ])
    result = build_redlight_v8(module, sidecar)
    assert result.redlight_confirmed_duration.tolist() == pytest.approx([0.6667, 1.3334])
    assert result.redlight_policy_strength.min() > 0.0
    assert result.redlight_event_peak_score.max() == 1.0


def test_fusion_v8_counts_multiple_active_families() -> None:
    rows = _keys()
    base = pd.DataFrame([{**row, "trajectory_anomaly_score": 0.8,
                          "trajectory_anomaly_level": 2,
                          "trajectory_anomaly_reliability": 0.9,
                          "trajectory_anomaly_duration_sec_proxy": 0.6667} for row in rows])
    trajectory = pd.DataFrame([{**row, "trajectory_wrong_way_flag": 1,
                                "trajectory_wrong_way_score": 0.9,
                                "trajectory_wrong_way_confidence": 0.9} for row in rows])
    overspeed = pd.DataFrame([{**row, "speed_kmh_p95": 100,
                               "speed_kmh_max": 105, "speed_kmh_mean": 95,
                               "speed_band_90_110_flag": 1,
                               "overspeed_reliability": 1} for row in rows])
    redlight = pd.DataFrame([{**row, "redlight_event_peak_score": 0.9,
                              "redlight_confirmed": 1,
                              "redlight_confirmed_duration": 0.6667,
                              "redlight_policy_strength": 0.9,
                              "redlight_signal_valid": 1} for row in rows])
    result = build_fusion_v8(base, trajectory, overspeed, redlight)
    assert result.active_family_count.tolist() == [3.0, 3.0]
    assert result.cooccurrence_duration_sec.iloc[1] > result.cooccurrence_duration_sec.iloc[0]
    assert result.policy_pair_peak_trajectory_overspeed.min() > 0.0
