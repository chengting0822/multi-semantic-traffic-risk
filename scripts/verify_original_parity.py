#!/usr/bin/env python3
"""Compare the compact package with frozen outputs in the original workspace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from traffic_risk.scene.hybrid import build_hybrid_predictions
from traffic_risk.scene.inference import SceneRiskModel
from traffic_risk.single_vehicle.inference import SingleVehicleRiskModel
from traffic_risk.single_vehicle.pipeline import PolicyInputs, apply_policy_chain


SINGLE_REL = Path(
    "單車風險/kfold_single_vehicle_risk/Val-calibrated version/"
    "frozen_candidates/v92_clean_root_cause_modules_20260621"
)
SCENE_REL = Path("場景風險/formal_training_data/scene_single_20260622_v93_clean_refresh")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument(
        "--check",
        choices=("all", "single-model", "single-policy", "scene-model", "scene-hybrid"),
        default="all",
    )
    return parser.parse_args()


def single_model_check(root: Path) -> dict[str, Any]:
    outputs = root / SINGLE_REL / (
        "runs/paper_eval_20260622/original_schema_train_eval_v94/fold_00/"
        "pred_v37e_clean/outputs"
    )
    model = SingleVehicleRiskModel(device="cpu")
    feature_path = next(
        (outputs / "tables").glob(
            "*_features_20260621_v92_clean_pred_v37e_c4o_gru.csv"
        )
    )
    features = pd.read_csv(
        feature_path,
        dtype={"case_key": str, "video_id": str, "track_id": str},
        low_memory=False,
    )
    report: dict[str, Any] = {}
    for split in ("val", "test"):
        split_features = features.loc[features["split"].astype(str) == split].copy()
        actual = model.predict(split_features).rename(columns={"ts_window_idx": "timestep_idx"})
        expected_path = next(
            (outputs / "predictions/V37E_C4O_combo_and_weaving_events").glob(
                f"*_{split}_window_predictions.csv"
            )
        )
        expected = pd.read_csv(expected_path, dtype={"case_key": str, "video_id": str, "track_id": str})
        keys = ["case_key", "video_id", "track_id", "timestep_idx"]
        joined = actual[keys + [
            "prob_class0", "prob_class1", "prob_class2", "pred_gru_raw", "pred_v37e"
        ]].merge(expected, on=keys, suffixes=("_actual", "_expected"), validate="one_to_one")
        actual_probs = joined[[
            "prob_class0_actual", "prob_class1_actual", "prob_class2_actual"
        ]].to_numpy()
        expected_probs = joined[[
            "prob_class0_expected", "prob_class1_expected", "prob_class2_expected"
        ]].to_numpy()
        report[split] = {
            "rows": len(joined),
            "raw_prediction_mismatches": int(
                (joined["pred_gru_raw"] != joined["y_pred_raw_argmax"]).sum()
            ),
            "formal_prediction_mismatches": int(
                (joined["pred_v37e"] != joined["y_pred_argmax"]).sum()
            ),
            "max_probability_absolute_error": float(np.abs(actual_probs - expected_probs).max()),
        }
    return report


def single_policy_check(root: Path) -> dict[str, Any]:
    base = root / SINGLE_REL
    run = base / "runs/paper_eval_20260622/original_schema_train_eval_v94/fold_00"
    cache = base / "cache/clean_v92_fast_precomputed_sidecars"
    windows = pd.read_csv(run / "inputs/clean_v92_pipeline_input.csv", low_memory=False)
    inputs = PolicyInputs.from_csv(
        source_features=cache / "clean_official_source_frame.csv",
        trajectory_features=cache / "clean_v37_trajectory_sidecar.csv",
        lane_features=cache / "clean_v28_lane_sidecar.csv",
        lane_family_features=cache / "clean_v41_lane_family_sidecar.csv",
        redlight_zone_features=cache / "clean_redlight_zone_movement_sidecar.csv",
        tail_lane_features=cache / "clean_v48_tail_lane_sidecar.csv",
        wrongway_tail_features=cache / "clean_v88_last5_sidecar.csv",
        double_yellow_features=root / "異常軌跡模組/outputs/tables/trajectory_double_yellow_contact_preconfirm_v71_sidecar.csv",
    )
    actual = apply_policy_chain(windows, inputs)
    expected = pd.read_csv(run / "clean_v92_pipeline/all_window_predictions_clean_v92_pipeline.csv", low_memory=False)
    keys = ["case_key", "video_id", "track_id", "ts_window_idx"]
    joined = actual[keys + ["v92_clean_pipeline_current", "risk_level"]].merge(
        expected[keys + ["v92_clean_pipeline_current", "v92_clean_pipeline_final"]],
        on=keys,
        suffixes=("_actual", "_expected"),
        validate="one_to_one",
    )
    return {
        "rows": len(joined),
        "current_mismatches": int(
            (joined["v92_clean_pipeline_current_actual"] != joined["v92_clean_pipeline_current_expected"]).sum()
        ),
        "final_mismatches": int((joined["risk_level"] != joined["v92_clean_pipeline_final"]).sum()),
    }


def scene_paths(root: Path) -> tuple[Path, Path, Path]:
    base = root / SCENE_REL
    run = base / "scene_v93_retrain_hybrid_select_v2_learned_high"
    tensor = base / "tensors/scene_dataset_scene_single_20260622_v93_clean_refresh_interaction_features_top16.npz"
    return base, run, tensor


def scene_model_check(root: Path) -> dict[str, Any]:
    _base, run, tensor = scene_paths(root)
    actual = SceneRiskModel(device="cpu").predict_npz(tensor).sort_values("row_idx").reset_index(drop=True)
    expected = pd.read_csv(
        run / "predictions/scene_token_pool_v93_retrain_hybrid_select_v2_learned_high_predictions.csv"
    ).sort_values("row_idx").reset_index(drop=True)
    prediction_columns = ["pred_4cls_raw", "pred_4cls", "pred_binary", "pred_severity"]
    probability_columns = [
        "prob4_0", "prob4_1", "prob4_2", "prob4_3", "prob_binary_positive",
        "prob_severity_0", "prob_severity_1", "prob_severity_2",
    ]
    return {
        "rows": len(actual),
        "class_mismatches": {column: int((actual[column] != expected[column]).sum()) for column in prediction_columns},
        "max_probability_absolute_error": max(
            float(np.abs(actual[column].to_numpy() - expected[column].to_numpy()).max())
            for column in probability_columns
        ),
    }


def scene_hybrid_check(root: Path) -> dict[str, Any]:
    base, run, _tensor = scene_paths(root)
    config = json.loads(
        (run / "configs/scene_hybrid_v93_retrain_hybrid_select_v2_learned_high_cpa_collision_strict_v2.json")
        .read_text(encoding="utf-8")
    )
    learned = pd.read_csv(run / "predictions/scene_token_pool_v93_retrain_hybrid_select_v2_learned_high_predictions.csv")
    actual = build_hybrid_predictions(
        learned,
        pd.read_csv(base / "tables/scene_windows_scene_single_20260622_v93_clean_refresh.csv", low_memory=False),
        pd.read_csv(base / "tables/scene_tokens_scene_single_20260622_v93_clean_refresh.csv", low_memory=False),
        pd.read_csv(base / "tables/scene_interaction_sidecar_v2.csv", low_memory=False),
        None,
        None,
        config,
    ).sort_values("row_idx").reset_index(drop=True)
    expected = pd.read_csv(
        run / "predictions/scene_hybrid_v93_retrain_hybrid_select_v2_learned_high_cpa_collision_strict_v2_predictions.csv",
        low_memory=False,
    ).sort_values("row_idx").reset_index(drop=True)
    selected = "scene_pred_single_floor_severity_gate_strict_pure_overspeed_rescue_learned_interaction_track_sticky_cpa_onset_high_confidence"
    prediction_columns = [column for column in actual if column.startswith("scene_pred_")]
    return {
        "rows": len(actual),
        "columns": len(actual.columns),
        "selected_mismatches": int((actual[selected] != expected[selected]).sum()),
        "all_prediction_mismatches": sum(int((actual[column] != expected[column]).sum()) for column in prediction_columns),
    }


def main() -> int:
    args = parse_args()
    checks = {
        "single-model": single_model_check,
        "single-policy": single_policy_check,
        "scene-model": scene_model_check,
        "scene-hybrid": scene_hybrid_check,
    }
    selected = checks if args.check == "all" else {args.check: checks[args.check]}
    report = {name: function(args.original_root.resolve()) for name, function in selected.items()}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
