from pathlib import Path

import pandas as pd

from traffic_risk.upstream.tracking import prepare_tracking_csv


def test_prepare_tracks_streams_arbitrary_video_id(tmp_path: Path) -> None:
    detector = tmp_path / "tracks.csv"
    pd.DataFrame(
        {
            "frame": [0, 10, 20, 5, 15],
            "track_id": [1, 1, 1, 2, 2],
            "x1": [0, 1, 2, 3, 4],
            "y1": [0, 1, 2, 3, 4],
            "x2": [10, 11, 12, 13, 14],
            "y2": [10, 11, 12, 13, 14],
            "conf": [0.9] * 5,
            "cls": [2] * 5,
        }
    ).to_csv(detector, index=False)
    video = tmp_path / "intersection-a.mp4"
    video.touch()

    result = prepare_tracking_csv(
        detector_csv=detector,
        video_path=video,
        output_dir=tmp_path / "prepared",
        fps=30.0,
        chunksize=2,
    )

    timestamp = pd.read_csv(result.timestamp_csv)
    windows = pd.read_csv(result.reference_windows_csv)
    assert result.video_id == "intersection-a"
    assert result.detector_rows == 5
    assert result.track_count == 2
    assert timestamp["source_video_id"].unique().tolist() == ["intersection-a"]
    assert timestamp["canonical_case_key"].unique().tolist() == [
        "intersection-a/1", "intersection-a/2"
    ]
    assert windows["video_id"].unique().tolist() == ["intersection-a"]
    assert windows.groupby("track_id").size().to_dict() == {1: 3, 2: 2}
    assert windows.loc[windows["track_id"].eq(1), "start_sec"].tolist() == [0.0, 0.3333, 0.6666]
