from __future__ import annotations

from pathlib import Path

import pandas as pd

from traffic_risk.scene.pipeline import SceneRiskPipeline


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples/demo"


def test_four_case_demo_matches_formal_pipeline() -> None:
    actual = SceneRiskPipeline(device="cpu").predict(
        dataset=DEMO / "scene_features.npz",
        scene_windows=pd.read_csv(DEMO / "scene_windows.csv", low_memory=False),
        scene_tokens=pd.read_csv(DEMO / "scene_tokens.csv", low_memory=False),
        interaction_features=pd.read_csv(DEMO / "interaction_features.csv", low_memory=False),
        labeled_only=True,
    )
    expected = pd.read_csv(DEMO / "expected_scene_risk.csv")
    actual["video_id"] = actual["video_id"].astype(str)
    expected["video_id"] = expected["video_id"].astype(str)
    keys = ["row_idx", "video_id", "scene_window_idx"]
    joined = actual[keys + ["risk_level"]].merge(expected, on=keys, validate="one_to_one")
    assert len(joined) == 183
    assert (joined["risk_level"] == joined["expected_risk_level"]).all()
    assert set(joined["video_id"]) == {"14", "76", "96", "115"}
