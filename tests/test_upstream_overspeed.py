import json
from pathlib import Path

import pandas as pd

from traffic_risk.upstream.overspeed.runner import build_overspeed_features


def test_overspeed_runner_supports_text_video_id(tmp_path: Path) -> None:
    timestamps = tmp_path / "intersection-a.csv"
    pd.DataFrame(
        {
            "frame": [0, 10, 20],
            "track_id": [1, 1, 1],
            "x1": [0.0, 1.0, 2.0],
            "y1": [0.0, 0.0, 0.0],
            "x2": [2.0, 3.0, 4.0],
            "y2": [1.0, 1.0, 1.0],
            "timestamp_sec": [0.0, 1.0 / 3.0, 2.0 / 3.0],
            "source_video_id": ["intersection-a"] * 3,
            "canonical_case_key": ["intersection-a/1"] * 3,
            "fps": [30.0] * 3,
        }
    ).to_csv(timestamps, index=False)
    references = tmp_path / "windows.csv"
    pd.DataFrame(
        {
            "case_key": ["intersection-a/1"],
            "source_type": ["anomaly_csv"],
            "source_id": ["intersection-a"],
            "video_id": ["intersection-a"],
            "track_id": [1],
            "ts_window_idx": [0],
            "start_sec": [0.0],
            "end_sec": [0.6667],
        }
    ).to_csv(references, index=False)
    ipm = tmp_path / "ipm.json"
    ipm.write_text(json.dumps({"homography_img_to_world": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}))

    result = build_overspeed_features(
        timestamp_csv=timestamps,
        reference_windows_csv=references,
        ipm_json=ipm,
    )

    assert len(result.windows) == 1
    assert result.windows.loc[0, "video_id"] == "intersection-a"
    assert result.windows.loc[0, "case_key"] == "intersection-a/1"
    assert result.windows.loc[0, "track_motion_valid"] == 1
    assert result.sidecar.loc[0, "window_point_count"] == 3
