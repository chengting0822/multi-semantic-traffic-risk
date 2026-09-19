from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .annotations import label_window, load_scene_segments


KEY_COLUMNS = ["case_key", "video_id", "track_id", "ts_window_idx"]

PREDICTION_COLUMNS = [
    "y_pred_argmax",
    "y_pred_raw_argmax",
    "pred_v22_candidate",
    "prob_class0",
    "prob_class1",
    "prob_class2",
    "high_risk_prob",
    "critical_prob",
]

FEATURE_SIGNAL_COLUMNS = [
    "overspeed_event_score",
    "redlight_event_score",
    "active_family_count",
    "cooccurrence_strength",
    "r46_risk3_policy",
    "r46_risk2_policy",
    "r46_policy_severity_score",
]

DEFAULT_SCENE_STRIDE_SEC = 1.0 / 3.0


def build_scene_dataset(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    features = _load_features(config["feature_table"])
    predictions = _load_predictions(config["prediction_files"])
    merged = _merge_features_predictions(features, predictions)
    scene_windows = _aggregate_scene_windows(merged)

    segments_by_video, annotation_manifest = load_scene_segments(config["annotation_dir"])
    labels = [
        label_window(
            segments_by_video.get(str(row.video_id)),
            float(row.start_sec),
            float(row.end_sec),
        )
        for row in scene_windows.itertuples(index=False)
    ]
    label_df = pd.DataFrame(labels)
    scene_windows = pd.concat([scene_windows.reset_index(drop=True), label_df], axis=1)
    scene_windows = scene_windows.sort_values(["split", "video_id_num", "scene_window_idx"]).reset_index(drop=True)
    scene_windows = scene_windows.drop(columns=["video_id_num"])

    manifest = {
        "feature_table": str(config["feature_table"]),
        "feature_rows": int(len(features)),
        "prediction_rows": int(len(predictions)),
        "merged_rows": int(len(merged)),
        "scene_window_rows": int(len(scene_windows)),
        "scene_window_rows_by_split": _value_counts(scene_windows, "split"),
        "labeled_scene_window_rows": int(scene_windows["scene_label_available"].sum()),
        "unlabeled_scene_window_rows": int((scene_windows["scene_label_available"] == 0).sum()),
        "prediction_available_rows": int(merged["prediction_available"].sum()),
        "scene_windows_with_predictions": int((scene_windows["prediction_available_count"] > 0).sum()),
        "mixed_split_scene_windows": int((scene_windows["split_nunique"] > 1).sum()),
        "annotation_manifest": annotation_manifest,
    }
    return scene_windows, manifest


def _load_features(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    missing = [col for col in KEY_COLUMNS + ["split", "start_sec", "end_sec"] if col not in df.columns]
    if missing:
        raise KeyError(f"Feature table missing columns: {missing}")
    df = _normalize_keys(df)
    for col in FEATURE_SIGNAL_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    return df


def _load_predictions(prediction_files: dict[str, str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split, path in prediction_files.items():
        pred = pd.read_csv(path, low_memory=False)
        pred = _normalize_keys(pred)
        missing = [col for col in KEY_COLUMNS if col not in pred.columns]
        if missing:
            raise KeyError(f"Prediction file {path} missing columns: {missing}")
        keep_cols = [col for col in KEY_COLUMNS + PREDICTION_COLUMNS if col in pred.columns]
        pred = pred[keep_cols].copy()
        for col in PREDICTION_COLUMNS:
            if col not in pred.columns:
                pred[col] = np.nan
        pred["prediction_split_source"] = split
        pred["prediction_available"] = 1
        frames.append(pred)
    if not frames:
        return pd.DataFrame(columns=KEY_COLUMNS + PREDICTION_COLUMNS + ["prediction_split_source", "prediction_available"])
    return pd.concat(frames, ignore_index=True)


def _merge_features_predictions(features: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    merged = features.merge(
        predictions,
        on=KEY_COLUMNS,
        how="left",
        suffixes=("", "_pred"),
        validate="one_to_one",
    )
    merged["prediction_available"] = merged["prediction_available"].fillna(0).astype(int)
    merged["prediction_split_source"] = merged["prediction_split_source"].fillna("")

    for col in PREDICTION_COLUMNS:
        if col not in merged.columns:
            merged[col] = np.nan
    for col in ["prob_class0", "prob_class1", "prob_class2", "high_risk_prob", "critical_prob"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
    for col in ["y_pred_argmax", "y_pred_raw_argmax", "pred_v22_candidate"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(-1).astype(int)
    for col in FEATURE_SIGNAL_COLUMNS:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    merged["pred_schemaC"] = np.where(merged["prediction_available"].eq(1), merged["y_pred_argmax"], -1)
    merged["pred_risk2_flag"] = (merged["pred_schemaC"] == 1).astype(int)
    merged["pred_risk3_flag"] = (merged["pred_schemaC"] == 2).astype(int)
    merged["r46_risk2_flag"] = (merged["r46_risk2_policy"] > 0).astype(int)
    merged["r46_risk3_flag"] = (merged["r46_risk3_policy"] > 0).astype(int)
    merged["prob_support_flag"] = (
        (merged["prediction_available"].eq(1))
        & ((merged["high_risk_prob"] >= 0.35) | (merged["critical_prob"] >= 0.35))
    ).astype(int)
    merged["rule_support_flag"] = (
        (merged["r46_risk2_flag"].eq(1))
        | (merged["r46_risk3_flag"].eq(1))
        | (merged["overspeed_event_score"] >= 0.4)
        | (merged["redlight_event_score"] >= 0.4)
    ).astype(int)
    merged["track_support_flag"] = (
        (merged["pred_schemaC"] > 0)
        | (merged["prob_support_flag"].eq(1))
        | (merged["rule_support_flag"].eq(1))
    ).astype(int)
    merged["strong_risk3_support_flag"] = (
        (merged["pred_schemaC"] == 2)
        | (merged["critical_prob"] >= 0.55)
        | (merged["r46_risk3_flag"].eq(1))
        | (merged["r46_policy_severity_score"] >= 2.0)
    ).astype(int)
    merged["medium_risk2_support_flag"] = (
        (merged["pred_schemaC"] == 1)
        | (merged["high_risk_prob"] >= 0.55)
        | (merged["r46_risk2_flag"].eq(1))
        | (merged["overspeed_event_score"] >= 0.5)
        | (merged["redlight_event_score"] >= 0.5)
        | ((merged["active_family_count"] >= 2) & (merged["cooccurrence_strength"] >= 0.25))
    ).astype(int)
    merged = _add_scene_window_index(merged)
    return merged


def _aggregate_scene_windows(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["video_id", "scene_window_idx"]
    grouped = df.groupby(group_cols, sort=False)
    agg = grouped.agg(
        split=("split", "first"),
        split_nunique=("split", "nunique"),
        track_window_idx_min=("ts_window_idx", "min"),
        track_window_idx_max=("ts_window_idx", "max"),
        start_sec=("start_sec", "min"),
        end_sec=("end_sec", "max"),
        active_track_count=("case_key", "nunique"),
        prediction_available_count=("prediction_available", "sum"),
        max_pred_schemaC=("pred_schemaC", "max"),
        count_pred_risk2=("pred_risk2_flag", "sum"),
        count_pred_risk3=("pred_risk3_flag", "sum"),
        max_prob_class1=("prob_class1", "max"),
        max_prob_class2=("prob_class2", "max"),
        max_high_risk_prob=("high_risk_prob", "max"),
        max_critical_prob=("critical_prob", "max"),
        mean_high_risk_prob=("high_risk_prob", "mean"),
        mean_critical_prob=("critical_prob", "mean"),
        max_overspeed_event_score=("overspeed_event_score", "max"),
        max_redlight_event_score=("redlight_event_score", "max"),
        max_active_family_count=("active_family_count", "max"),
        max_cooccurrence_strength=("cooccurrence_strength", "max"),
        max_r46_risk2_policy=("r46_risk2_policy", "max"),
        max_r46_risk3_policy=("r46_risk3_policy", "max"),
        max_r46_policy_severity_score=("r46_policy_severity_score", "max"),
        count_r46_risk2_policy=("r46_risk2_flag", "sum"),
        count_r46_risk3_policy=("r46_risk3_flag", "sum"),
        track_support_count=("track_support_flag", "sum"),
        rule_support_count=("rule_support_flag", "sum"),
        prob_support_count=("prob_support_flag", "sum"),
        strong_risk3_support_count=("strong_risk3_support_flag", "sum"),
        medium_risk2_support_count=("medium_risk2_support_flag", "sum"),
    ).reset_index(drop=False)

    case_keys = grouped["case_key"].apply(_join_sorted_unique).rename("case_keys")
    track_ids = grouped["track_id"].apply(_join_sorted_unique).rename("track_ids")
    agg = agg.merge(case_keys.reset_index(), on=group_cols, how="left")
    agg = agg.merge(track_ids.reset_index(), on=group_cols, how="left")
    agg["scene_window_idx"] = agg["scene_window_idx"].astype(int)
    agg["video_id_num"] = pd.to_numeric(agg["video_id"], errors="coerce").fillna(-1).astype(int)
    agg["prediction_source"] = np.where(
        agg["prediction_available_count"] > 0,
        "r52_selected_prediction",
        "feature_policy_only",
    )
    return agg


def _add_scene_window_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["start_sec"] = pd.to_numeric(df["start_sec"], errors="raise")
    # R52 ts_window_idx is track-local. Scene windows are video-global 20-frame
    # windows with 10-frame stride, matching the old scene annotation ids.
    df["scene_window_idx"] = np.rint(df["start_sec"] / DEFAULT_SCENE_STRIDE_SEC).astype(int)
    df["scene_stride_sec"] = DEFAULT_SCENE_STRIDE_SEC
    return df


def _normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["case_key", "video_id", "track_id"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    if "ts_window_idx" in df.columns:
        df["ts_window_idx"] = pd.to_numeric(df["ts_window_idx"], errors="raise").astype(int)
    return df


def _join_sorted_unique(values: pd.Series) -> str:
    unique = sorted({str(v) for v in values.dropna()}, key=_natural_key)
    return ",".join(unique)


def _natural_key(value: str) -> tuple[int, str]:
    try:
        return (0, f"{int(value):012d}")
    except ValueError:
        return (1, value)


def _value_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    return {str(k): int(v) for k, v in df[column].value_counts(dropna=False).sort_index().items()}
