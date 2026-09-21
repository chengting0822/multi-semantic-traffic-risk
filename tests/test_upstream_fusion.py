import pandas as pd
import pytest

from traffic_risk.upstream.fusion import adapt_overspeed, adapt_redlight, build_fusion_features


def _keys() -> dict[str, list]:
    return {
        "case_key": ["any-video/1"] * 3,
        "source_type": ["anomaly_csv"] * 3,
        "source_id": ["any-video"] * 3,
        "video_id": ["any-video"] * 3,
        "track_id": [1] * 3,
        "ts_window_idx": [0, 1, 2],
        "start_sec": [0.0, 0.3333, 0.6666],
        "end_sec": [0.6667, 1.0, 1.3333],
    }


def test_formal_v0_adapters_and_fusion_are_causal() -> None:
    trajectory = pd.DataFrame({
        **_keys(),
        "trajectory_anomaly_score": [0.0, 0.4, 0.0],
        "trajectory_anomaly_level": [0, 1, 0],
        "trajectory_anomaly_duration_sec_proxy": [0, 0.3333, 0],
        "trajectory_anomaly_reliability": [1, 1, 1],
        "trajectory_anomaly_signal_valid": [1, 1, 1],
    })
    overspeed = pd.DataFrame({
        **_keys(), "speed_kmh": [0, 95, 0], "overspeed_level": [0, 2, 0],
        "overspeed_reliability": [1, 1, 1], "timestamp_delta_valid": [1, 1, 1],
        "ipm_source_valid": [1, 1, 1], "track_motion_valid": [1, 1, 1],
    })
    redlight = pd.DataFrame({
        **_keys(), "redlight_confirmed": [0, 1, 0],
        "redlight_score": [0.0, 0.8, 0.0], "redlight_signal_valid": [1, 1, 1],
    })
    assert adapt_overspeed(overspeed)["overspeed_duration_sec_proxy"].tolist() == [0.0, 0.3333, 0.0]
    assert adapt_redlight(redlight)["redlight_duration_sec_proxy"].tolist() == [0.0, 0.3333, 0.0]
    result = build_fusion_features(trajectory, overspeed, redlight)
    assert result.windows["semantic_active_module_count"].tolist() == [0, 3, 0]
    assert result.windows["semantic_history_ema"].tolist() == pytest.approx([0.0, 0.0, 0.16])
    assert result.windows["semantic_recent_peak_level"].tolist() == [0, 2, 2]


def test_fusion_rejects_missing_branch_window() -> None:
    keys = _keys()
    trajectory = pd.DataFrame({**keys, **{name: [0] * 3 for name in [
        "trajectory_anomaly_score", "trajectory_anomaly_level", "trajectory_anomaly_duration_sec_proxy",
        "trajectory_anomaly_reliability", "trajectory_anomaly_signal_valid",
    ]}})
    overspeed = pd.DataFrame({**keys, **{name: [0] * 3 for name in [
        "overspeed_score", "overspeed_level", "overspeed_duration_sec_proxy", "overspeed_reliability",
    ]}})
    redlight = pd.DataFrame({**keys, **{name: [0] * 3 for name in [
        "redlight_confirmed", "redlight_score", "redlight_duration_sec_proxy", "redlight_signal_valid",
    ]}}).iloc[:2]
    with pytest.raises(ValueError, match="lacks 1 trajectory windows"):
        build_fusion_features(trajectory, overspeed, redlight, overspeed_is_gru=True, redlight_is_gru=True)
