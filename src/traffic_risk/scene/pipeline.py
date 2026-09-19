"""High-level scene-risk inference with the validation-selected hybrid rule."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from traffic_risk.paths import CONFIG_DIR

from .hybrid import build_hybrid_predictions
from .inference import SceneRiskModel


class SceneRiskPipeline:
    def __init__(
        self,
        *,
        config: Path = CONFIG_DIR / "scene_hybrid.json",
        device: str = "auto",
    ) -> None:
        self.config: dict[str, Any] = json.loads(config.read_text(encoding="utf-8"))
        variants = self.config.get("variants", [])
        if len(variants) != 1:
            raise ValueError("scene hybrid config must contain exactly one selected variant")
        self.variant = str(variants[0])
        self.model = SceneRiskModel(device=device)

    def predict(
        self,
        *,
        dataset: Path,
        scene_windows: pd.DataFrame,
        scene_tokens: pd.DataFrame,
        interaction_features: pd.DataFrame,
        batch_size: int = 256,
        labeled_only: bool = False,
    ) -> pd.DataFrame:
        learned = self.model.predict_npz(dataset, batch_size=batch_size, labeled_only=labeled_only)
        result = build_hybrid_predictions(
            learned,
            scene_windows,
            scene_tokens,
            interaction_features,
            None,
            None,
            self.config,
        )
        selected_column = f"scene_pred_{self.variant}"
        result["risk_level"] = result[selected_column].astype(int)
        return result


__all__ = ["SceneRiskPipeline"]
