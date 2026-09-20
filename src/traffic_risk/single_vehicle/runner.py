"""End-to-end single-vehicle inference from current-run semantic tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .feature_builder import build_semantic_feature_bundle, write_semantic_feature_bundle
from .inference import SingleVehicleRiskModel
from .pipeline import apply_policy_chain


@dataclass(frozen=True)
class SingleVehicleRunResult:
    learned_windows: pd.DataFrame
    final_windows: pd.DataFrame
    frame_cache_summary: dict[str, int]


def run_single_vehicle_pipeline(
    *,
    c4o_features: Path,
    trajectory_features: Path,
    timestamp_dir: Path,
    lane_map: Path,
    stop_lines: Path,
    output_dir: Path,
    device: str = "auto",
    overwrite_cache: bool = False,
) -> SingleVehicleRunResult:
    """Build semantic evidence, run the GRU, then apply the frozen policies."""

    output_dir.mkdir(parents=True, exist_ok=True)
    upstream = pd.read_csv(c4o_features, low_memory=False)
    trajectory = pd.read_csv(trajectory_features, low_memory=False)
    bundle = build_semantic_feature_bundle(
        c4o_features=upstream,
        trajectory_features=trajectory,
        timestamp_dir=timestamp_dir,
        lane_map=lane_map,
        stop_lines=stop_lines,
        cache_dir=output_dir / "cache",
        overwrite_cache=overwrite_cache,
    )
    learned = SingleVehicleRiskModel(device=device).predict(bundle.model_features)
    final = apply_policy_chain(learned, bundle.policy_inputs)

    write_semantic_feature_bundle(bundle, output_dir / "semantic_features")
    learned.to_csv(output_dir / "single_vehicle_learned_windows.csv", index=False)
    final.to_csv(output_dir / "single_vehicle_risk.csv", index=False)
    return SingleVehicleRunResult(learned, final, bundle.frame_cache_summary)


__all__ = ["SingleVehicleRunResult", "run_single_vehicle_pipeline"]
