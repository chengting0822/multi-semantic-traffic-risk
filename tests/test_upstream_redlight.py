import json
from pathlib import Path

import pandas as pd

from traffic_risk.upstream.redlight.runner import build_redlight_features


def test_redlight_runner_supports_text_video_id(tmp_path: Path) -> None:
    timestamps = tmp_path / "intersection-a.csv"
    pd.DataFrame(
        {
            "frame": [0, 10, 20],
            "track_id": [1, 1, 1],
            "x1": [0.0, 0.0, 0.0],
            "y1": [2.0, 4.0, 8.0],
            "x2": [2.0, 2.0, 2.0],
            "y2": [4.0, 6.0, 10.0],
            "timestamp_sec": [0.0, 1.0 / 3.0, 2.0 / 3.0],
            "source_video_id": ["intersection-a"] * 3,
            "canonical_case_key": ["intersection-a/1"] * 3,
            "fps": [30.0] * 3,
            "tl_state": ["red"] * 3,
            "tl_prob_red": [0.99] * 3,
            "tl_prob_green": [0.01] * 3,
            "cls": [2] * 3,
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
    stop_lines = tmp_path / "stop_lines.json"
    stop_lines.write_text(json.dumps({
        "stop_lines": [{"id": "line-1", "points": [[-10, 5], [10, 5]]}]
    }))

    result = build_redlight_features(
        timestamp_csv=timestamps,
        reference_windows_csv=references,
        stop_lines_json=stop_lines,
    )

    assert len(result.windows) == 1
    assert result.windows.loc[0, "video_id"] == "intersection-a"
    assert result.windows.loc[0, "stopline_crossed_confirmed"] == 1
    assert result.windows.loc[0, "forward_motion_after_stopline"] == 1
    assert result.windows.loc[0, "redlight_confirmed"] == 1
