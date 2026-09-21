from pathlib import Path
from collections import Counter

import pandas as pd

from traffic_risk.paths import CONFIG_DIR, MODEL_DIR
from traffic_risk.upstream.car_tracks import car_track_ids
from traffic_risk.upstream.tracking import prepare_tracking_csv
from traffic_risk.upstream.trajectory.runner import build_trajectory_features


def test_car_track_requires_majority_but_tolerates_class_jitter() -> None:
    counts = {
        1: Counter({2: 8, 3: 1, 7: 1}),
        2: Counter({3: 8, 2: 1}),
        3: Counter({2: 2, 3: 2}),
        4: Counter({2: 3, 3: 2, 7: 2}),
    }
    assert car_track_ids(counts) == {1}


def test_car_track_keeps_same_id_when_frame_class_flips(tmp_path: Path) -> None:
    car_frames = [0, 5, 10, 15, 20, 25, 30]
    rows = [
        {
            "frame": frame,
            "track_id": 1,
            "x1": 1000 + frame,
            "y1": 700 + frame,
            "x2": 1040 + frame,
            "y2": 740 + frame,
            "conf": 0.95,
            "cls": 3 if frame in {10, 20} else 2,
        }
        for frame in car_frames
    ]
    rows += [
        {"frame": frame, "track_id": 2, "x1": 500, "y1": 500, "x2": 530, "y2": 550, "conf": 0.9, "cls": 3}
        for frame in car_frames
    ]
    rows.append({"frame": 0, "track_id": -9, "x1": 100, "y1": 100, "x2": 120, "y2": 120, "conf": 0.99, "cls": 9})
    detector = tmp_path / "tracks.csv"
    pd.DataFrame(rows).sort_values(["frame", "track_id"]).to_csv(detector, index=False)
    video = tmp_path / "any-video.mp4"
    video.touch()

    prepared = prepare_tracking_csv(
        detector_csv=detector,
        video_path=video,
        output_dir=tmp_path / "prepared",
        fps=30.0,
        chunksize=2,
    )
    references = pd.read_csv(prepared.reference_windows_csv)
    timestamp = pd.read_csv(prepared.timestamp_csv)
    assert prepared.track_count == 1
    assert set(references["track_id"]) == {1}
    assert set(timestamp["track_id"]) == {-9, 1, 2}
    assert timestamp.loc[timestamp["track_id"].eq(1), "cls"].eq(3).sum() == 2

    trajectory = build_trajectory_features(
        timestamp_csv=prepared.timestamp_csv,
        lane_map_json=CONFIG_DIR / "lane_map.json",
        double_yellow_json=CONFIG_DIR / "double_yellow.json",
        model_json=MODEL_DIR / "trajectory_gmm_b5.json",
        chunksize=2,
    )
    assert set(trajectory.fixed20["track_id"].astype(str)) == {"1"}
    assert int(trajectory.fixed20.iloc[0]["detection_count"]) > 2
    assert len(trajectory.windows) == len(trajectory.sidecar)
