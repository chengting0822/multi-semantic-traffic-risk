import pandas as pd
import pytest

from traffic_risk.upstream.canonical import REQUIRED_MODULE_COLUMNS, build_c4o_from_v8


def _module(label: str) -> pd.DataFrame:
    rows = []
    for index in range(2):
        row = {
            "case_key": "camera-a/7", "source_type": "anomaly_csv", "source_id": "camera-a",
            "video_id": "camera-a", "track_id": "7", "ts_window_idx": index,
            "start_sec": index * 0.3333, "end_sec": index * 0.3333 + 0.6667,
        }
        row.update({column: 0.0 for column in REQUIRED_MODULE_COLUMNS[label]})
        rows.append(row)
    return pd.DataFrame(rows)


def test_c4o_requires_v8_and_prefix_and_preserves_text_video_id() -> None:
    inputs = {name: _module(name) for name in REQUIRED_MODULE_COLUMNS}
    speed = inputs["overspeed_v8"]
    speed["speed_kmh_p95"] = 100.0
    speed["overspeed_reliability"] = 1.0
    speed["overspeed_signal_quality"] = 1.0
    speed["overspeed_level_ge2_duration"] = 1.30
    speed["speed_smoothed_kmh_p95"] = 100.0
    speed["computed_speed_kmh_p95"] = 100.0
    prefix = inputs["fusion_v8"][["case_key", "video_id", "track_id", "ts_window_idx"]].copy()
    for column in (
        "s33_prefix_state_confirmed", "s33_confirmed_ema_margin",
        "s33_current_local_semantic_support", "s33_confirmed_ema_pos",
    ):
        prefix[column] = 0.0

    output = build_c4o_from_v8(**inputs, s33_prefix=prefix)
    assert output["video_id"].tolist() == ["camera-a", "camera-a"]
    assert output["r32_overspeed_midband_strength"].tolist() == [0.5, 0.5]
    assert output["r32_overspeed_ge2_duration_strength"].tolist() == [0.5, 0.5]
    assert output["r32_overspeed_reliable_signal"].tolist() == [1.0, 1.0]
    assert output["r32_overspeed_rescue_signal"].tolist() == [0.25, 0.25]
    assert output["r46_risk2_policy"].tolist() == [1.0, 1.0]

    with pytest.raises(ValueError, match="S3.3 prefix"):
        build_c4o_from_v8(**inputs, s33_prefix=prefix.iloc[:1])
    with pytest.raises(ValueError, match="missing join columns"):
        build_c4o_from_v8(**{**inputs, "fusion_v8": pd.DataFrame()}, s33_prefix=prefix)
    with pytest.raises(ValueError, match="missing join columns"):
        build_c4o_from_v8(**{**inputs, "overspeed_v8": pd.DataFrame()}, s33_prefix=prefix)
