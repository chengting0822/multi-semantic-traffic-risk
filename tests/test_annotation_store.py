from pathlib import Path

import pandas as pd

from traffic_risk.annotation.gui import load_case
from traffic_risk.annotation.store import AnnotationDocument, Segment
from traffic_risk.scene.hybrid import refresh_labels_from_scene_windows


def test_annotation_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "14.json"
    document = AnnotationDocument("14", 30.0)
    document.add(Segment(0, 29, 0, 30.0))
    document.add(Segment(30, 59, 2, 30.0), "11")
    document.save(path)

    loaded = AnnotationDocument.load(path, "14", 30.0)
    assert loaded.scene_segments == [Segment(0, 29, 0, 30.0)]
    assert loaded.track_segments["11"] == [Segment(30, 59, 2, 30.0)]


def test_segment_rejects_invalid_values() -> None:
    try:
        Segment(10, 5, 0, 30.0)
    except ValueError as error:
        assert "range" in str(error)
    else:
        raise AssertionError("invalid segment must fail")


def test_annotation_gui_is_limited_to_demo_cases() -> None:
    assert load_case("115")["video"].endswith("115.mp4")
    try:
        load_case("999")
    except ValueError as error:
        assert "case must be" in str(error)
    else:
        raise AssertionError("non-demo case must fail")


def test_hybrid_allows_unlabeled_live_inference() -> None:
    frame = pd.DataFrame(
        {
            "scene_y_original": [-1],
            "scene_y_binary": [-1],
            "scene_y_schemaC": [-1],
        }
    )
    result = refresh_labels_from_scene_windows(frame)
    assert result[["y_4cls", "y_binary", "y_severity"]].iloc[0].tolist() == [-1, -1, -1]
