import pandas as pd

from traffic_risk.identifiers import normalize_window_keys


def test_window_keys_preserve_arbitrary_video_ids() -> None:
    frame = pd.DataFrame(
        {
            "case_key": [" morning ", "numeric"],
            "video_id": ["intersection-a", 115.0],
            "track_id": ["vehicle-7", 4.0],
            "ts_window_idx": ["2", 3.0],
        }
    )

    result = normalize_window_keys(frame)

    assert result["video_id"].tolist() == ["intersection-a", "115"]
    assert result["track_id"].tolist() == ["vehicle-7", "4"]
    assert result["ts_window_idx"].tolist() == [2, 3]
    assert result["case_key"].tolist() == ["morning", "numeric"]
