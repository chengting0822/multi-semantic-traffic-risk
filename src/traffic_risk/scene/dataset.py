from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .annotations import label_window_by_index, load_scene_segments, scene_schema_c
from .config import path_stat
from .dataset_base import _load_features, _load_predictions, _merge_features_predictions


TOKEN_MODEL_FEATURE_COLUMNS = [
    "prob_class1",
    "prob_class2",
    "s33_prefix_state_confirmed",
    "s33_confirmed_ema_margin",
    "s33_current_local_semantic_support",
    "r32_overspeed_midband_strength",
    "r32_overspeed_ge2_duration_strength",
    "r32_overspeed_reliable_signal",
    "r32_overspeed_rescue_signal",
    "overspeed_event_score",
    "redlight_event_score",
    "active_family_count",
    "cooccurrence_strength",
    "elapsed_track_progress",
]

GLOBAL_MODEL_FEATURE_COLUMNS = [
    "active_track_count",
    "max_prob_class1",
    "max_prob_class2",
    "mean_prob_class1",
    "mean_prob_class2",
    "top2_prob_class1",
    "top2_prob_class2",
    "top3_prob_class1",
    "top3_prob_class2",
    "max_overspeed_event_score",
    "mean_overspeed_event_score",
    "max_redlight_event_score",
    "mean_redlight_event_score",
    "max_active_family_count",
    "mean_active_family_count",
    "max_cooccurrence_strength",
    "mean_cooccurrence_strength",
    "current_event_score",
    "hist_event_score_ema",
    "hist_positive_ratio_8",
    "hist_recent_max_event_score_8",
    "hist_windows_since_positive",
]

SCENE_FLOOR_FEATURE_COLUMNS = [
    "single_vehicle_floor_schemaC",
    "single_vehicle_floor_risk",
    "has_single_vehicle_risk2",
    "has_single_vehicle_risk3",
    "count_single_vehicle_risk2",
    "count_single_vehicle_risk3",
    "strict_policy_floor_risk",
    "has_strict_policy_risk2",
    "has_strict_policy_risk3",
    "count_strict_policy_risk2",
    "count_strict_policy_risk3",
    "support_floor_risk",
    "count_medium_risk2_support",
    "count_strong_risk3_support",
    "default_floor_risk",
]

TOKEN_FLOOR_AUDIT_COLUMNS = [
    "token_single_vehicle_floor_schemaC",
    "token_single_vehicle_floor_risk",
    "token_strict_policy_floor_risk",
    "token_support_floor_risk",
    "token_final_single_vehicle_floor_schemaC",
    "token_final_single_vehicle_floor_risk",
    "token_final_strict_policy_floor_risk",
    "token_final_support_floor_risk",
    "token_active_single_vehicle_floor_schemaC",
    "token_active_single_vehicle_floor_risk",
    "token_active_floor_reason",
]

TOKEN_AUDIT_ONLY_COLUMNS = [
    "prob_class0",
    "high_risk_prob",
    "critical_prob",
    "pred_schemaC_window",
    "single_vehicle_window_final_schemaC",
    "single_vehicle_cumulative_final_schemaC",
    "pred_schemaC",
    "pred_risk2_flag",
    "pred_risk3_flag",
    "r46_risk2_policy",
    "r46_risk3_policy",
    "r46_policy_severity_score",
    "prediction_available",
    "track_support_flag",
    "rule_support_flag",
    "prob_support_flag",
    "strong_risk3_support_flag",
    "medium_risk2_support_flag",
    "token_selection_score",
]

REQUIRED_UPSTREAM_COLUMNS = [
    "s33_prefix_state_confirmed",
    "s33_confirmed_ema_margin",
    "s33_current_local_semantic_support",
    "r32_overspeed_midband_strength",
    "r32_overspeed_ge2_duration_strength",
    "r32_overspeed_reliable_signal",
    "r32_overspeed_rescue_signal",
    "overspeed_event_score",
    "redlight_event_score",
    "active_family_count",
    "cooccurrence_strength",
    "elapsed_track_progress",
]

REMOVED_FROM_V2_LEARNED_INPUT = [
    "prob_class0",
    "high_risk_prob",
    "critical_prob",
    "pred_schemaC",
    "pred_risk2_flag",
    "pred_risk3_flag",
    "r46_risk2_policy",
    "r46_risk3_policy",
    "r46_policy_severity_score",
    "prediction_available",
    "track_support_flag",
    "rule_support_flag",
    "prob_support_flag",
    "strong_risk3_support_flag",
    "medium_risk2_support_flag",
    "token_saliency",
]


def build_scene_dataset_v3(config: dict[str, Any]) -> dict[str, Any]:
    scene_cfg = config.get("scene", {})
    top_k = int(scene_cfg.get("top_k", 16))

    features = _load_features(config["feature_table"])
    _require_columns(features, REQUIRED_UPSTREAM_COLUMNS, "feature_table")
    predictions = _load_predictions(config["prediction_files"])
    merged = _merge_features_predictions(features, predictions)
    merged = _apply_single_vehicle_floor_source_v3(merged, scene_cfg)
    merged = _add_v3_token_columns(merged)
    merged = _apply_scene_token_alignment_v3(merged, scene_cfg)
    merged = _apply_active_single_vehicle_floor_v1(merged, scene_cfg)

    scene_windows = _aggregate_scene_windows_v3(merged)
    scene_windows = _apply_single_vehicle_temporal_floor_overlap_v3(scene_windows, merged, scene_cfg)
    scene_windows = _add_first_last_labels(scene_windows, config["annotation_dir"])
    scene_windows = _apply_boundary_label_policy(scene_windows, scene_cfg)
    annotation_manifest = scene_windows.attrs.get("annotation_manifest", {})
    scene_windows = _add_global_history_v3(scene_windows, scene_cfg)
    scene_windows = scene_windows.sort_values(["split", "video_id_num", "scene_window_idx"]).reset_index(drop=True)
    scene_windows["scene_row_id"] = np.arange(len(scene_windows), dtype=np.int64)
    scene_windows = scene_windows.drop(columns=["video_id_num"])
    scene_windows.attrs["annotation_manifest"] = annotation_manifest

    selected_tokens = _select_tokens_v3(merged, scene_windows, top_k)
    tensor_payload = _build_tensor_payload_v3(scene_windows, selected_tokens, top_k)
    audit = _build_audit_v3(config, scene_windows, selected_tokens, features, predictions, tensor_payload)

    return {
        "scene_windows": scene_windows,
        "scene_tokens": selected_tokens,
        "tensor_payload": tensor_payload,
        "audit": audit,
    }


def write_scene_dataset_v3(result: dict[str, Any], config: dict[str, Any]) -> None:
    outputs = config["outputs"]
    scene_windows_path = Path(outputs["scene_windows_csv"])
    scene_tokens_path = Path(outputs["scene_tokens_csv"])
    tensor_path = Path(outputs["scene_tensor_npz"])
    audit_json_path = Path(outputs["audit_json"])
    audit_md_path = Path(outputs["audit_md"])
    manifest_path = Path(outputs["manifest_json"])

    for path in [scene_windows_path, scene_tokens_path, tensor_path, audit_json_path, audit_md_path, manifest_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    result["scene_windows"].to_csv(scene_windows_path, index=False)
    result["scene_tokens"].to_csv(scene_tokens_path, index=False)
    np.savez_compressed(tensor_path, **result["tensor_payload"])
    audit_json_path.write_text(json.dumps(result["audit"], ensure_ascii=False, indent=2), encoding="utf-8")
    audit_md_path.write_text(render_audit_md_v3(result["audit"]), encoding="utf-8")

    manifest = {
        "schema_version": "scene_risk.dataset_r52_v3.manifest/v1",
        "run_id": config["run_id"],
        "inputs": {
            "feature_table": path_stat(config["feature_table"]),
            "annotation_dir": path_stat(config["annotation_dir"]),
            "prediction_files": {split: path_stat(path) for split, path in config["prediction_files"].items()},
        },
        "outputs": {name: path_stat(path) for name, path in outputs.items()},
        "scene_config": config.get("scene", {}),
        "single_vehicle_floor_source": config.get("scene", {}).get("single_vehicle_floor_source", {"mode": "window_pred_schemaC"}),
        "feature_groups": result["audit"]["feature_groups"],
        "audit_summary": {
            "scene_windows": result["audit"]["scene_window_count"],
            "labeled_scene_windows": result["audit"]["labeled_scene_window_count"],
            "token_rows": result["audit"]["selected_token_count"],
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _apply_single_vehicle_floor_source_v3(df: pd.DataFrame, scene_cfg: dict[str, Any]) -> pd.DataFrame:
    """Choose the scene-facing single-vehicle final-risk source.

    Historical scene datasets used the per-window prediction stored in
    ``pred_schemaC``.  For the final scene floor, however, the intended contract
    is to consume the single-vehicle module's outward final risk.  When that
    final risk is monotonic at the track level, scene tokens should carry the
    cumulative maximum risk observed so far for the same track, not the local
    per-window classifier output.
    """

    cfg = scene_cfg.get("single_vehicle_floor_source", {}) or {}
    mode = str(cfg.get("mode", "window_pred_schemaC")).lower()
    out = df.copy()

    if "pred_schemaC" not in out.columns:
        raise KeyError("merged_features_predictions missing pred_schemaC")

    out["pred_schemaC_window"] = pd.to_numeric(out["pred_schemaC"], errors="coerce").fillna(-1).astype(int)

    requested_source = cfg.get("source_column")
    if requested_source is None:
        if "pred_v22_candidate" in out.columns and pd.to_numeric(out["pred_v22_candidate"], errors="coerce").fillna(-1).ge(0).any():
            requested_source = "pred_v22_candidate"
        else:
            requested_source = "pred_schemaC_window"
    source_col = str(requested_source)
    if source_col not in out.columns:
        raise KeyError(f"single_vehicle_floor_source source_column not found: {source_col}")

    source_schema = pd.to_numeric(out[source_col], errors="coerce").fillna(-1).astype(int)
    source_schema = source_schema.where(source_schema >= 0, out["pred_schemaC_window"])
    source_schema = source_schema.clip(lower=-1, upper=2).astype(int)
    out["single_vehicle_window_final_schemaC"] = source_schema

    cumulative_modes = {
        "track_cumulative_pred_schemac",
        "track_cumulative_final_schemac",
        "track_cumulative_final",
        "cumulative_final",
    }
    if mode in cumulative_modes:
        group_cols = [col for col in ["video_id", "track_id", "case_key"] if col in out.columns]
        if not group_cols:
            raise KeyError("Cannot build track cumulative floor without video_id/track_id/case_key")

        sort_cols = group_cols + [col for col in ["ts_window_idx", "start_sec", "end_sec"] if col in out.columns]
        sorted_out = out.sort_values(sort_cols, kind="mergesort").copy()
        nonnegative = sorted_out["single_vehicle_window_final_schemaC"].clip(lower=0).astype(int)
        sorted_out["single_vehicle_cumulative_final_schemaC"] = nonnegative.groupby(
            [sorted_out[col].astype(str) for col in group_cols],
            sort=False,
        ).cummax()
        out["single_vehicle_cumulative_final_schemaC"] = (
            sorted_out["single_vehicle_cumulative_final_schemaC"].reindex(out.index).fillna(0).astype(int)
        )
        out["pred_schemaC"] = out["single_vehicle_cumulative_final_schemaC"].astype(int)
    elif mode in {"window_pred_schemac", "window_final_schemac", "window_final", "window_pred_schemaC".lower()}:
        out["single_vehicle_cumulative_final_schemaC"] = out["single_vehicle_window_final_schemaC"].clip(lower=0).astype(int)
        out["pred_schemaC"] = out["single_vehicle_window_final_schemaC"].astype(int)
    else:
        raise ValueError(
            "Unsupported single_vehicle_floor_source.mode: "
            f"{mode}. Use window_pred_schemaC or track_cumulative_pred_schemaC."
        )

    out["pred_schemaC"] = pd.to_numeric(out["pred_schemaC"], errors="coerce").fillna(-1).astype(int)
    out["pred_risk2_flag"] = (out["pred_schemaC"] == 1).astype(int)
    out["pred_risk3_flag"] = (out["pred_schemaC"] == 2).astype(int)

    if {"high_risk_prob", "critical_prob"}.issubset(out.columns):
        out["prob_support_flag"] = (
            (pd.to_numeric(out.get("prediction_available", 0), errors="coerce").fillna(0).astype(int).eq(1))
            & (
                (pd.to_numeric(out["high_risk_prob"], errors="coerce").fillna(0.0) >= 0.35)
                | (pd.to_numeric(out["critical_prob"], errors="coerce").fillna(0.0) >= 0.35)
            )
        ).astype(int)

    if {"r46_risk2_flag", "r46_risk3_flag", "overspeed_event_score", "redlight_event_score"}.issubset(out.columns):
        out["rule_support_flag"] = (
            (pd.to_numeric(out["r46_risk2_flag"], errors="coerce").fillna(0).astype(int).eq(1))
            | (pd.to_numeric(out["r46_risk3_flag"], errors="coerce").fillna(0).astype(int).eq(1))
            | (pd.to_numeric(out["overspeed_event_score"], errors="coerce").fillna(0.0) >= 0.4)
            | (pd.to_numeric(out["redlight_event_score"], errors="coerce").fillna(0.0) >= 0.4)
        ).astype(int)

    if {"prob_support_flag", "rule_support_flag"}.issubset(out.columns):
        out["track_support_flag"] = (
            (out["pred_schemaC"] > 0)
            | (pd.to_numeric(out["prob_support_flag"], errors="coerce").fillna(0).astype(int).eq(1))
            | (pd.to_numeric(out["rule_support_flag"], errors="coerce").fillna(0).astype(int).eq(1))
        ).astype(int)

    if {"critical_prob", "r46_risk3_flag", "r46_policy_severity_score"}.issubset(out.columns):
        out["strong_risk3_support_flag"] = (
            (out["pred_schemaC"] == 2)
            | (pd.to_numeric(out["critical_prob"], errors="coerce").fillna(0.0) >= 0.55)
            | (pd.to_numeric(out["r46_risk3_flag"], errors="coerce").fillna(0).astype(int).eq(1))
            | (pd.to_numeric(out["r46_policy_severity_score"], errors="coerce").fillna(0.0) >= 2.0)
        ).astype(int)

    if {
        "high_risk_prob",
        "r46_risk2_flag",
        "overspeed_event_score",
        "redlight_event_score",
        "active_family_count",
        "cooccurrence_strength",
    }.issubset(out.columns):
        out["medium_risk2_support_flag"] = (
            (out["pred_schemaC"] == 1)
            | (pd.to_numeric(out["high_risk_prob"], errors="coerce").fillna(0.0) >= 0.55)
            | (pd.to_numeric(out["r46_risk2_flag"], errors="coerce").fillna(0).astype(int).eq(1))
            | (pd.to_numeric(out["overspeed_event_score"], errors="coerce").fillna(0.0) >= 0.5)
            | (pd.to_numeric(out["redlight_event_score"], errors="coerce").fillna(0.0) >= 0.5)
            | (
                (pd.to_numeric(out["active_family_count"], errors="coerce").fillna(0.0) >= 2)
                & (pd.to_numeric(out["cooccurrence_strength"], errors="coerce").fillna(0.0) >= 0.25)
            )
        ).astype(int)

    return out


def _add_v3_token_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    _require_columns(out, TOKEN_MODEL_FEATURE_COLUMNS, "merged_features_predictions")
    for col in TOKEN_MODEL_FEATURE_COLUMNS + TOKEN_AUDIT_ONLY_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    pred_schema = out["pred_schemaC"].clip(lower=0).astype(int)
    out["token_single_vehicle_floor_schemaC"] = pred_schema
    out["token_single_vehicle_floor_risk"] = pred_schema.map(_schema_c_to_original_floor).astype(int)
    out["token_strict_policy_floor_risk"] = np.select(
        [out["r46_risk3_policy"] > 0, out["r46_risk2_policy"] > 0],
        [3, 2],
        default=0,
    ).astype(int)
    out["token_strict_policy_risk2_flag"] = (out["token_strict_policy_floor_risk"] >= 2).astype(int)
    out["token_strict_policy_risk3_flag"] = (out["token_strict_policy_floor_risk"] >= 3).astype(int)
    out["token_support_floor_risk"] = np.select(
        [out["strong_risk3_support_flag"] > 0, out["medium_risk2_support_flag"] > 0],
        [3, 2],
        default=0,
    ).astype(int)
    out["token_support_risk2_flag"] = (out["token_support_floor_risk"] >= 2).astype(int)
    out["token_support_risk3_flag"] = (out["token_support_floor_risk"] >= 3).astype(int)
    out["token_selection_score"] = (
        2.2 * out["prob_class2"]
        + 1.5 * out["prob_class1"]
        + 1.0 * out["overspeed_event_score"]
        + 1.0 * out["redlight_event_score"]
        + 0.45 * out["active_family_count"].clip(upper=3.0)
        + 0.45 * out["cooccurrence_strength"]
        + 0.35 * (out["token_single_vehicle_floor_risk"] / 3.0)
        + 0.25 * (out["token_strict_policy_floor_risk"] / 3.0)
    )
    out["token_must_keep"] = (
        (out["token_single_vehicle_floor_risk"] > 0)
        | (out["token_strict_policy_floor_risk"] > 0)
        | (out["prob_class1"] >= 0.35)
        | (out["prob_class2"] >= 0.35)
        | (out["overspeed_event_score"] >= 0.25)
        | (out["redlight_event_score"] >= 0.25)
    ).astype(int)
    return out


def _apply_scene_token_alignment_v3(df: pd.DataFrame, scene_cfg: dict[str, Any]) -> pd.DataFrame:
    """Assign single-vehicle ID windows to scene windows.

    Default behavior keeps the historical round(start_sec / stride) assignment.
    The optional causal_overlap_v1 mode expands one ID window to all sufficiently
    overlapping canonical scene windows, but only when the ID window has already
    completed by the scene-window end time.

    The causal_completion_v1 mode is stricter: every ID window is assigned only to
    the earliest canonical scene window whose end time can already observe that
    completed ID window. This keeps streaming semantics clean and prevents the
    same single-vehicle decision from being smeared across several adjacent scene
    windows.
    """
    alignment_cfg = scene_cfg.get("alignment", {}) or {}
    mode = str(alignment_cfg.get("mode", "round_start")).lower()
    if mode not in {"causal_overlap_v1", "causal_overlap", "causal_completion_v1", "causal_completion"}:
        out = df.copy()
        out["alignment_mode"] = "round_start"
        out["round_scene_window_idx"] = pd.to_numeric(out["scene_window_idx"], errors="coerce").fillna(-1).astype(int)
        return out

    required = ["video_id", "start_sec", "end_sec", "scene_window_idx"]
    _require_columns(df, required, "merged_features_predictions")

    stride_sec = float(alignment_cfg.get("scene_stride_sec", scene_cfg.get("scene_stride_sec", 1.0 / 3.0)))
    window_sec = float(alignment_cfg.get("scene_window_sec", scene_cfg.get("scene_window_sec", 0.6667)))
    min_overlap_sec = float(alignment_cfg.get("min_overlap_sec", 0.20))
    min_scene_overlap_ratio = float(alignment_cfg.get("min_scene_overlap_ratio", 0.30))
    min_token_overlap_ratio = float(alignment_cfg.get("min_token_overlap_ratio", 0.30))
    require_completed = bool(alignment_cfg.get("require_completed_single_window", True))
    max_scene_windows_per_token = int(alignment_cfg.get("max_scene_windows_per_token", 4))
    eps = float(alignment_cfg.get("epsilon_sec", 1e-6))

    if stride_sec <= 0 or window_sec <= 0:
        raise ValueError(f"Invalid scene alignment seconds: stride={stride_sec}, window={window_sec}")

    base = df.copy()
    base["start_sec"] = pd.to_numeric(base["start_sec"], errors="coerce")
    base["end_sec"] = pd.to_numeric(base["end_sec"], errors="coerce")
    base["round_scene_window_idx"] = pd.to_numeric(base["scene_window_idx"], errors="coerce").fillna(-1).astype(int)
    base = base[base["start_sec"].notna() & base["end_sec"].notna() & (base["end_sec"] > base["start_sec"])].copy()

    aligned_rows: list[dict[str, Any]] = []
    for row in base.itertuples(index=False):
        token_start = float(row.start_sec)
        token_end = float(row.end_sec)
        token_duration = max(0.0, token_end - token_start)
        if token_duration <= 0:
            continue

        if mode in {"causal_completion_v1", "causal_completion"}:
            first_idx = max(0, int(np.ceil((token_end - window_sec - eps) / stride_sec)))
            last_idx = first_idx
        else:
            first_idx = max(0, int(np.floor((token_start - window_sec) / stride_sec)) - 1)
            last_idx = max(first_idx, int(np.floor(token_end / stride_sec)) + 2)
        matches: list[dict[str, float | int]] = []
        for scene_idx in range(first_idx, last_idx + 1):
            scene_start = float(scene_idx * stride_sec)
            scene_end = float(scene_start + window_sec)
            if require_completed and token_end > scene_end + eps:
                continue
            overlap = max(0.0, min(scene_end, token_end) - max(scene_start, token_start))
            if overlap <= 0:
                continue
            scene_ratio = overlap / window_sec
            token_ratio = overlap / token_duration
            if (
                overlap + eps < min_overlap_sec
                or scene_ratio + eps < min_scene_overlap_ratio
                or token_ratio + eps < min_token_overlap_ratio
            ):
                continue
            matches.append(
                {
                    "scene_window_idx": int(scene_idx),
                    "canonical_scene_start_sec": scene_start,
                    "canonical_scene_end_sec": scene_end,
                    "scene_token_overlap_sec": float(overlap),
                    "scene_token_scene_overlap_ratio": float(scene_ratio),
                    "scene_token_token_overlap_ratio": float(token_ratio),
                }
            )

        if max_scene_windows_per_token > 0 and len(matches) > max_scene_windows_per_token:
            matches = sorted(
                matches,
                key=lambda item: (
                    float(item["scene_token_overlap_sec"]),
                    float(item["scene_token_scene_overlap_ratio"]),
                    -abs(int(item["scene_window_idx"]) - int(row.round_scene_window_idx)),
                ),
                reverse=True,
            )[:max_scene_windows_per_token]
            matches = sorted(matches, key=lambda item: int(item["scene_window_idx"]))

        if not matches:
            continue

        row_dict = row._asdict()
        for match in matches:
            aligned = row_dict.copy()
            aligned.update(match)
            aligned["alignment_mode"] = "causal_completion_v1" if mode in {"causal_completion_v1", "causal_completion"} else "causal_overlap_v1"
            aligned["scene_token_completed_by_scene_end"] = int(token_end <= float(match["canonical_scene_end_sec"]) + eps)
            aligned_rows.append(aligned)

    if not aligned_rows:
        return pd.DataFrame(columns=list(base.columns) + [
            "round_scene_window_idx",
            "canonical_scene_start_sec",
            "canonical_scene_end_sec",
            "scene_token_overlap_sec",
            "scene_token_scene_overlap_ratio",
            "scene_token_token_overlap_ratio",
            "scene_token_completed_by_scene_end",
            "alignment_mode",
        ])

    out = pd.DataFrame(aligned_rows)
    out["scene_window_idx"] = pd.to_numeric(out["scene_window_idx"], errors="coerce").fillna(-1).astype(int)
    out["alignment_mode"] = "causal_completion_v1" if mode in {"causal_completion_v1", "causal_completion"} else "causal_overlap_v1"
    return out


def _original_floor_to_schema_c(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").fillna(0).astype(int)
    return values.map({0: 0, 1: 0, 2: 1, 3: 2}).fillna(0).astype(int)


def _apply_active_single_vehicle_floor_v1(df: pd.DataFrame, scene_cfg: dict[str, Any]) -> pd.DataFrame:
    """Build a scene-facing active floor from the single-vehicle final risk.

    The single-vehicle final risk is intentionally conservative for track-level
    reporting: it may include recurrent memory, cumulative max behavior, and
    late-window residue.  Scene floor needs a narrower interpretation: only
    risks that are still active enough for the current completed scene window
    should force the scene risk upward.

    This function keeps the original final floor columns for audit and, when
    enabled, replaces the scene-facing floor columns with active-floor values.
    The first supported guard is a narrow tail-stale suppressor for risk2 only:
    near the end of a track, if there is no current trajectory support, no
    strong speed/redlight evidence, no multi-family co-occurrence, and no risk3
    support, the risk2 floor is considered stale for scene-floor purposes.
    """

    cfg = scene_cfg.get("single_vehicle_active_floor", {}) or {}
    out = df.copy()

    floor_cols = [
        "token_single_vehicle_floor_schemaC",
        "token_single_vehicle_floor_risk",
        "token_strict_policy_floor_risk",
        "token_support_floor_risk",
        "token_strict_policy_risk2_flag",
        "token_strict_policy_risk3_flag",
        "token_support_risk2_flag",
        "token_support_risk3_flag",
        "pred_risk2_flag",
        "pred_risk3_flag",
    ]
    _require_columns(out, floor_cols, "active_single_vehicle_floor_v1")

    if "token_final_single_vehicle_floor_risk" not in out.columns:
        out["token_final_single_vehicle_floor_schemaC"] = out["token_single_vehicle_floor_schemaC"]
        out["token_final_single_vehicle_floor_risk"] = out["token_single_vehicle_floor_risk"]
        out["token_final_strict_policy_floor_risk"] = out["token_strict_policy_floor_risk"]
        out["token_final_support_floor_risk"] = out["token_support_floor_risk"]
        out["token_final_strict_policy_risk2_flag"] = out["token_strict_policy_risk2_flag"]
        out["token_final_strict_policy_risk3_flag"] = out["token_strict_policy_risk3_flag"]
        out["token_final_support_risk2_flag"] = out["token_support_risk2_flag"]
        out["token_final_support_risk3_flag"] = out["token_support_risk3_flag"]
        out["token_final_pred_risk2_flag"] = out["pred_risk2_flag"]
        out["token_final_pred_risk3_flag"] = out["pred_risk3_flag"]

    enabled = bool(cfg.get("enabled", False))
    replace_scene_floor = bool(cfg.get("replace_scene_floor", enabled))
    active_risk = pd.to_numeric(out["token_final_single_vehicle_floor_risk"], errors="coerce").fillna(0).astype(int)
    active_strict = pd.to_numeric(out["token_final_strict_policy_floor_risk"], errors="coerce").fillna(0).astype(int)
    active_support = pd.to_numeric(out["token_final_support_floor_risk"], errors="coerce").fillna(0).astype(int)
    reason = pd.Series("kept_final_floor", index=out.index, dtype=object)

    if enabled and bool(cfg.get("tail_stale_guard_enabled", True)):
        elapsed = pd.to_numeric(out.get("elapsed_track_progress", 0), errors="coerce").fillna(0.0)
        active_family = pd.to_numeric(out.get("active_family_count", 0), errors="coerce").fillna(0.0)
        cooccur = pd.to_numeric(out.get("cooccurrence_strength", 0), errors="coerce").fillna(0.0)
        current_semantic = pd.to_numeric(
            out.get("s33_current_local_semantic_support", 0), errors="coerce"
        ).fillna(0.0)
        overspeed = pd.to_numeric(out.get("overspeed_event_score", 0), errors="coerce").fillna(0.0)
        redlight = pd.to_numeric(out.get("redlight_event_score", 0), errors="coerce").fillna(0.0)
        critical_prob = pd.to_numeric(out.get("prob_class2", 0), errors="coerce").fillna(0.0)
        strong_risk3 = pd.to_numeric(out.get("strong_risk3_support_flag", 0), errors="coerce").fillna(0.0)
        r46_risk3 = pd.to_numeric(out.get("r46_risk3_policy", 0), errors="coerce").fillna(0.0)
        track_last_idx = (
            pd.to_numeric(out["ts_window_idx"], errors="coerce")
            .groupby([out["video_id"].astype(str), out["track_id"].astype(str)])
            .transform("max")
            .fillna(-1)
        )
        token_idx = pd.to_numeric(out["ts_window_idx"], errors="coerce").fillna(-1)
        last_window_tolerance = int(cfg.get("tail_last_window_tolerance", 0))
        is_track_tail_window = token_idx >= (track_last_idx - last_window_tolerance)

        tail_near_end = elapsed >= float(cfg.get("tail_min_elapsed_track_progress", 0.95))
        weak_current_context = (
            (active_family <= float(cfg.get("tail_max_active_family_count", 1.0)))
            & (cooccur <= float(cfg.get("tail_max_cooccurrence_strength", 0.20)))
            & (current_semantic <= float(cfg.get("tail_max_current_local_semantic_support", 0.0)))
        )
        weak_direct_evidence = (
            (overspeed < float(cfg.get("tail_min_overspeed_event_score_keep", 0.55)))
            & (redlight < float(cfg.get("tail_min_redlight_event_score_keep", 0.75)))
            & (critical_prob < float(cfg.get("tail_min_critical_prob_keep", 0.35)))
        )
        no_risk3_support = (
            (active_strict < 3)
            & (active_support < 3)
            & (strong_risk3 <= 0)
            & (r46_risk3 <= 0)
        )
        stale_mask = (
            (active_risk == 2)
            & is_track_tail_window
            & tail_near_end
            & weak_current_context
            & weak_direct_evidence
            & no_risk3_support
        )

        active_risk = active_risk.mask(stale_mask, 0).astype(int)
        active_strict = active_strict.mask(stale_mask, 0).astype(int)
        active_support = active_support.mask(stale_mask, 0).astype(int)
        reason = reason.mask(stale_mask, "tail_stale_risk2_suppressed_for_scene_floor")

    out["token_active_single_vehicle_floor_risk"] = active_risk.astype(int)
    out["token_active_single_vehicle_floor_schemaC"] = _original_floor_to_schema_c(active_risk)
    out["token_active_strict_policy_floor_risk"] = active_strict.astype(int)
    out["token_active_support_floor_risk"] = active_support.astype(int)
    out["token_active_floor_reason"] = reason

    if enabled and replace_scene_floor:
        out["token_single_vehicle_floor_risk"] = out["token_active_single_vehicle_floor_risk"]
        out["token_single_vehicle_floor_schemaC"] = out["token_active_single_vehicle_floor_schemaC"]
        out["token_strict_policy_floor_risk"] = out["token_active_strict_policy_floor_risk"]
        out["token_support_floor_risk"] = out["token_active_support_floor_risk"]
        out["token_strict_policy_risk2_flag"] = (out["token_strict_policy_floor_risk"] >= 2).astype(int)
        out["token_strict_policy_risk3_flag"] = (out["token_strict_policy_floor_risk"] >= 3).astype(int)
        out["token_support_risk2_flag"] = (out["token_support_floor_risk"] >= 2).astype(int)
        out["token_support_risk3_flag"] = (out["token_support_floor_risk"] >= 3).astype(int)
        out["pred_risk2_flag"] = (out["token_active_single_vehicle_floor_risk"] >= 2).astype(int)
        out["pred_risk3_flag"] = (out["token_active_single_vehicle_floor_risk"] >= 3).astype(int)

    return out


def _aggregate_scene_windows_v3(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["video_id", "scene_window_idx"]
    grouped = df.groupby(group_cols, sort=False)
    canonical_time = {"canonical_scene_start_sec", "canonical_scene_end_sec"}.issubset(df.columns)
    start_agg = ("canonical_scene_start_sec", "first") if canonical_time else ("start_sec", "min")
    end_agg = ("canonical_scene_end_sec", "first") if canonical_time else ("end_sec", "max")
    agg = grouped.agg(
        split=("split", "first"),
        split_nunique=("split", "nunique"),
        track_window_idx_min=("ts_window_idx", "min"),
        track_window_idx_max=("ts_window_idx", "max"),
        start_sec=start_agg,
        end_sec=end_agg,
        token_time_min=("start_sec", "min"),
        token_time_max=("end_sec", "max"),
        active_track_count=("case_key", "nunique"),
        prediction_available_count=("prediction_available", "sum"),
        max_prob_class1=("prob_class1", "max"),
        max_prob_class2=("prob_class2", "max"),
        mean_prob_class1=("prob_class1", "mean"),
        mean_prob_class2=("prob_class2", "mean"),
        max_overspeed_event_score=("overspeed_event_score", "max"),
        mean_overspeed_event_score=("overspeed_event_score", "mean"),
        max_redlight_event_score=("redlight_event_score", "max"),
        mean_redlight_event_score=("redlight_event_score", "mean"),
        max_active_family_count=("active_family_count", "max"),
        mean_active_family_count=("active_family_count", "mean"),
        max_cooccurrence_strength=("cooccurrence_strength", "max"),
        mean_cooccurrence_strength=("cooccurrence_strength", "mean"),
        single_vehicle_floor_schemaC=("token_single_vehicle_floor_schemaC", "max"),
        single_vehicle_floor_risk=("token_single_vehicle_floor_risk", "max"),
        count_single_vehicle_risk2=("pred_risk2_flag", "sum"),
        count_single_vehicle_risk3=("pred_risk3_flag", "sum"),
        count_strict_policy_risk2=("token_strict_policy_risk2_flag", "sum"),
        count_strict_policy_risk3=("token_strict_policy_risk3_flag", "sum"),
        count_medium_risk2_support=("token_support_risk2_flag", "sum"),
        count_strong_risk3_support=("token_support_risk3_flag", "sum"),
        max_pred_schemaC=("pred_schemaC", "max"),
        max_high_risk_prob=("high_risk_prob", "max"),
        max_critical_prob=("critical_prob", "max"),
        track_support_count=("track_support_flag", "sum"),
        rule_support_count=("rule_support_flag", "sum"),
        prob_support_count=("prob_support_flag", "sum"),
    ).reset_index(drop=False)

    agg["top2_prob_class1"] = grouped["prob_class1"].apply(lambda s: _nth_largest(s, 2)).to_numpy(dtype=float)
    agg["top2_prob_class2"] = grouped["prob_class2"].apply(lambda s: _nth_largest(s, 2)).to_numpy(dtype=float)
    agg["top3_prob_class1"] = grouped["prob_class1"].apply(lambda s: _nth_largest(s, 3)).to_numpy(dtype=float)
    agg["top3_prob_class2"] = grouped["prob_class2"].apply(lambda s: _nth_largest(s, 3)).to_numpy(dtype=float)

    agg["has_single_vehicle_risk2"] = (agg["count_single_vehicle_risk2"] > 0).astype(int)
    agg["has_single_vehicle_risk3"] = (agg["count_single_vehicle_risk3"] > 0).astype(int)
    agg["has_strict_policy_risk2"] = (agg["count_strict_policy_risk2"] > 0).astype(int)
    agg["has_strict_policy_risk3"] = (agg["count_strict_policy_risk3"] > 0).astype(int)
    agg["strict_policy_floor_risk"] = np.select(
        [agg["has_strict_policy_risk3"] > 0, agg["has_strict_policy_risk2"] > 0],
        [3, 2],
        default=0,
    ).astype(int)
    agg["support_floor_risk"] = np.select(
        [agg["count_strong_risk3_support"] > 0, agg["count_medium_risk2_support"] > 0],
        [3, 2],
        default=0,
    ).astype(int)
    agg["default_floor_risk"] = np.maximum(agg["single_vehicle_floor_risk"], agg["strict_policy_floor_risk"]).astype(int)

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


def _apply_single_vehicle_temporal_floor_overlap_v3(
    scene_windows: pd.DataFrame,
    merged: pd.DataFrame,
    scene_cfg: dict[str, Any],
) -> pd.DataFrame:
    out = scene_windows.copy()
    floor_cfg = scene_cfg.get("single_vehicle_temporal_floor_overlap", {}) or {}
    enabled = bool(floor_cfg.get("enabled", False))
    out["single_vehicle_temporal_overlap_floor_risk"] = 0
    out["single_vehicle_temporal_overlap_risk2_track_ids"] = ""
    out["single_vehicle_temporal_overlap_risk3_track_ids"] = ""
    out["single_vehicle_temporal_overlap_applied"] = 0
    if not enabled:
        return out

    required = [
        "video_id",
        "case_key",
        "track_id",
        "scene_window_idx",
        "ts_window_idx",
        "start_sec",
        "end_sec",
        "token_single_vehicle_floor_risk",
    ]
    _require_columns(merged, required, "merged_features_predictions")

    min_overlap_sec = float(floor_cfg.get("min_overlap_sec", 0.20))
    min_scene_overlap_ratio = float(floor_cfg.get("min_scene_overlap_ratio", 0.55))
    min_token_overlap_ratio = float(floor_cfg.get("min_token_overlap_ratio", 0.55))
    require_track_first_window = bool(floor_cfg.get("require_track_first_window", True))
    max_track_window_idx = int(floor_cfg.get("max_track_window_idx", 0))
    only_prior_scene_windows = bool(floor_cfg.get("only_prior_scene_windows", True))
    max_scene_window_idx_gap = int(floor_cfg.get("max_scene_window_idx_gap", 1))

    risk_tokens = merged[required].copy()
    risk_tokens["token_single_vehicle_floor_risk"] = pd.to_numeric(
        risk_tokens["token_single_vehicle_floor_risk"], errors="coerce"
    ).fillna(0).astype(int)
    risk_tokens["scene_window_idx"] = pd.to_numeric(risk_tokens["scene_window_idx"], errors="coerce")
    risk_tokens["ts_window_idx"] = pd.to_numeric(risk_tokens["ts_window_idx"], errors="coerce")
    risk_tokens["start_sec"] = pd.to_numeric(risk_tokens["start_sec"], errors="coerce")
    risk_tokens["end_sec"] = pd.to_numeric(risk_tokens["end_sec"], errors="coerce")
    risk_tokens = risk_tokens[
        (risk_tokens["token_single_vehicle_floor_risk"] > 0)
        & risk_tokens["scene_window_idx"].notna()
        & risk_tokens["ts_window_idx"].notna()
        & risk_tokens["start_sec"].notna()
        & risk_tokens["end_sec"].notna()
        & (risk_tokens["end_sec"] > risk_tokens["start_sec"])
    ].copy()
    if require_track_first_window:
        risk_tokens = risk_tokens[risk_tokens["ts_window_idx"].astype(int) <= max_track_window_idx].copy()
    if risk_tokens.empty:
        return out

    original_floor = pd.to_numeric(out["single_vehicle_floor_risk"], errors="coerce").fillna(0).astype(int)
    original_risk2_count = pd.to_numeric(out["count_single_vehicle_risk2"], errors="coerce").fillna(0).astype(int)
    original_risk3_count = pd.to_numeric(out["count_single_vehicle_risk3"], errors="coerce").fillna(0).astype(int)

    rows: list[dict[str, Any]] = []
    scene_parts = {
        str(video_id): group.copy()
        for video_id, group in out.groupby(out["video_id"].astype(str), sort=False)
    }
    for video_id, token_group in risk_tokens.groupby(risk_tokens["video_id"].astype(str), sort=False):
        windows = scene_parts.get(str(video_id))
        if windows is None or windows.empty:
            continue
        win_start = pd.to_numeric(windows["start_sec"], errors="coerce").to_numpy(dtype=float)
        win_end = pd.to_numeric(windows["end_sec"], errors="coerce").to_numpy(dtype=float)
        win_scene_idx = pd.to_numeric(windows["scene_window_idx"], errors="coerce").fillna(-1).to_numpy(dtype=int)
        win_duration = np.maximum(0.0, win_end - win_start)
        win_index = windows.index.to_numpy()
        for token in token_group.itertuples(index=False):
            token_start = float(token.start_sec)
            token_end = float(token.end_sec)
            token_scene_idx = int(token.scene_window_idx)
            token_duration = max(0.0, token_end - token_start)
            if token_duration <= 0:
                continue
            overlap = np.maximum(0.0, np.minimum(win_end, token_end) - np.maximum(win_start, token_start))
            scene_ratio = np.divide(
                overlap,
                win_duration,
                out=np.zeros_like(overlap, dtype=float),
                where=win_duration > 0,
            )
            token_ratio = overlap / token_duration
            matched = (
                (overlap >= min_overlap_sec)
                & (scene_ratio >= min_scene_overlap_ratio)
                & (token_ratio >= min_token_overlap_ratio)
            )
            if only_prior_scene_windows:
                matched &= win_scene_idx < token_scene_idx
            if max_scene_window_idx_gap >= 0:
                matched &= (token_scene_idx - win_scene_idx) <= max_scene_window_idx_gap
            for idx in win_index[matched]:
                rows.append(
                    {
                        "scene_index": int(idx),
                        "risk": int(token.token_single_vehicle_floor_risk),
                        "track_id": str(token.track_id),
                        "case_key": str(token.case_key),
                    }
                )

    if not rows:
        return out

    overlap_frame = pd.DataFrame(rows)
    max_risk = overlap_frame.groupby("scene_index")["risk"].max()
    risk2_ids = overlap_frame[overlap_frame["risk"] >= 2].groupby("scene_index")["track_id"].apply(_join_sorted_unique)
    risk3_ids = overlap_frame[overlap_frame["risk"] >= 3].groupby("scene_index")["track_id"].apply(_join_sorted_unique)
    risk2_counts = overlap_frame[overlap_frame["risk"] >= 2].groupby("scene_index")["track_id"].nunique()
    risk3_counts = overlap_frame[overlap_frame["risk"] >= 3].groupby("scene_index")["track_id"].nunique()

    for idx, risk in max_risk.items():
        idx = int(idx)
        out.at[idx, "single_vehicle_temporal_overlap_floor_risk"] = int(risk)
        if int(risk) > int(original_floor.iloc[idx]):
            out.at[idx, "single_vehicle_temporal_overlap_applied"] = 1
    for idx, value in risk2_ids.items():
        out.at[int(idx), "single_vehicle_temporal_overlap_risk2_track_ids"] = value
    for idx, value in risk3_ids.items():
        out.at[int(idx), "single_vehicle_temporal_overlap_risk3_track_ids"] = value

    overlap_floor = pd.to_numeric(out["single_vehicle_temporal_overlap_floor_risk"], errors="coerce").fillna(0).astype(int)
    out["single_vehicle_floor_risk"] = np.maximum(original_floor.to_numpy(dtype=int), overlap_floor.to_numpy(dtype=int)).astype(int)
    out["count_single_vehicle_risk2"] = np.maximum(
        original_risk2_count.to_numpy(dtype=int),
        pd.Series(risk2_counts, index=risk2_counts.index).reindex(out.index).fillna(0).to_numpy(dtype=int),
    ).astype(int)
    out["count_single_vehicle_risk3"] = np.maximum(
        original_risk3_count.to_numpy(dtype=int),
        pd.Series(risk3_counts, index=risk3_counts.index).reindex(out.index).fillna(0).to_numpy(dtype=int),
    ).astype(int)
    out["has_single_vehicle_risk2"] = (out["count_single_vehicle_risk2"] > 0).astype(int)
    out["has_single_vehicle_risk3"] = (out["count_single_vehicle_risk3"] > 0).astype(int)
    out["default_floor_risk"] = np.maximum(out["single_vehicle_floor_risk"], out["strict_policy_floor_risk"]).astype(int)
    return out


def _add_first_last_labels(scene_windows: pd.DataFrame, annotation_dir: str | Path) -> pd.DataFrame:
    segments_by_video, annotation_manifest = load_scene_segments(annotation_dir)
    labels = [
        label_window_by_index(
            segments_by_video.get(str(row.video_id)),
            int(row.scene_window_idx),
            float(row.start_sec),
            float(row.end_sec),
        )
        for row in scene_windows.itertuples(index=False)
    ]
    details = [
        _window_overlap_details(
            segments_by_video.get(str(row.video_id)),
            float(row.start_sec),
            float(row.end_sec),
        )
        for row in scene_windows.itertuples(index=False)
    ]
    labeled = pd.concat([scene_windows.reset_index(drop=True), pd.DataFrame(labels), pd.DataFrame(details)], axis=1)
    labeled.attrs["annotation_manifest"] = annotation_manifest
    return labeled


def _window_overlap_details(segments: list[Any] | None, start_sec: float, end_sec: float) -> dict[str, Any]:
    duration = max(0.0, float(end_sec) - float(start_sec))
    if not segments:
        return {
            "scene_label_window_duration_sec": duration,
            "scene_label_overlap_ratio": 0.0,
            "scene_label_dominant_risk": -1,
            "scene_label_dominant_overlap_sec": 0.0,
            "scene_label_dominant_overlap_ratio": 0.0,
            "scene_label_max_time_risk": -1,
            "scene_label_positive_overlap_sec": 0.0,
            "scene_label_max_positive_overlap_ratio": 0.0,
        }

    overlaps: list[tuple[float, int, int]] = []
    positive_overlap = 0.0
    max_positive_overlap = 0.0
    for seg in segments:
        overlap = max(0.0, min(float(end_sec), float(seg.end_sec)) - max(float(start_sec), float(seg.start_sec)))
        if overlap <= 0:
            continue
        risk = int(seg.risk)
        overlaps.append((overlap, risk, int(seg.segment_idx)))
        if risk > 0:
            positive_overlap += overlap
            max_positive_overlap = max(max_positive_overlap, overlap)

    if not overlaps:
        return {
            "scene_label_window_duration_sec": duration,
            "scene_label_overlap_ratio": 0.0,
            "scene_label_dominant_risk": -1,
            "scene_label_dominant_overlap_sec": 0.0,
            "scene_label_dominant_overlap_ratio": 0.0,
            "scene_label_max_time_risk": -1,
            "scene_label_positive_overlap_sec": 0.0,
            "scene_label_max_positive_overlap_ratio": 0.0,
        }

    dominant_overlap, dominant_risk, _dominant_idx = max(overlaps, key=lambda item: (item[0], item[1]))
    _max_time_overlap, max_time_risk, _max_time_idx = max(overlaps, key=lambda item: (item[1], item[0]))
    return {
        "scene_label_window_duration_sec": duration,
        "scene_label_overlap_ratio": float(max_positive_overlap / duration) if duration > 0 else 0.0,
        "scene_label_dominant_risk": int(dominant_risk),
        "scene_label_dominant_overlap_sec": float(dominant_overlap),
        "scene_label_dominant_overlap_ratio": float(dominant_overlap / duration) if duration > 0 else 0.0,
        "scene_label_max_time_risk": int(max_time_risk),
        "scene_label_positive_overlap_sec": float(positive_overlap),
        "scene_label_max_positive_overlap_ratio": float(max_positive_overlap / duration) if duration > 0 else 0.0,
    }


def _apply_boundary_label_policy(scene_windows: pd.DataFrame, scene_cfg: dict[str, Any]) -> pd.DataFrame:
    df = scene_windows.copy()
    duration = pd.to_numeric(df["scene_label_window_duration_sec"], errors="coerce").fillna(0.0)
    overlap = pd.to_numeric(df["scene_label_overlap_sec"], errors="coerce").fillna(0.0)
    df["scene_label_overlap_ratio"] = np.where(duration.to_numpy(dtype=float) > 0, overlap.to_numpy(dtype=float) / duration.to_numpy(dtype=float), 0.0)

    raw_cols = ["scene_label_available", "scene_y_original", "scene_y_schemaC", "scene_y_binary"]
    for col in raw_cols:
        df[f"{col}_raw"] = df[col]

    policy = scene_cfg.get("boundary_label_policy", {}) or {}
    enabled = bool(policy.get("enabled", False))
    df["scene_boundary_policy"] = str(policy.get("name", "disabled" if not enabled else "boundary_overlap_ratio"))
    df["scene_boundary_ambiguous"] = 0
    df["scene_boundary_reason"] = ""

    if not enabled:
        return df

    min_overlap_ratio = float(policy.get("hard_min_overlap_ratio", 0.33))
    dominant_guard_max_ratio = float(policy.get("dominant_guard_max_overlap_ratio", min_overlap_ratio))
    require_non_primary = bool(policy.get("require_non_primary_window", True))
    affected_risks = {int(value) for value in policy.get("affected_risks", [1, 2, 3])}
    exclude = bool(policy.get("exclude_from_training_and_primary_eval", True))

    y = pd.to_numeric(df["scene_y_original"], errors="coerce").fillna(-1).astype(int)
    available = pd.to_numeric(df["scene_label_available"], errors="coerce").fillna(0).astype(int) == 1
    primary = pd.to_numeric(df["scene_label_primary_match"], errors="coerce").fillna(0).astype(int) == 1
    ratio = pd.to_numeric(df["scene_label_overlap_ratio"], errors="coerce").fillna(0.0)
    dominant_risk = pd.to_numeric(df["scene_label_dominant_risk"], errors="coerce").fillna(-1).astype(int)

    affected = y.map(lambda value: int(value) in affected_risks).astype(bool)
    primary_ok = ~primary if require_non_primary else pd.Series(True, index=df.index)
    low_overlap = available & affected & (y > 0) & primary_ok & (ratio < min_overlap_ratio)
    dominant_boundary = (
        available
        & affected
        & (y > 0)
        & primary_ok
        & (dominant_risk >= 0)
        & (y > dominant_risk)
        & (ratio < dominant_guard_max_ratio)
    )
    ambiguous = low_overlap | dominant_boundary

    reasons = np.full(len(df), "", dtype=object)
    reasons[low_overlap.to_numpy(dtype=bool)] = "positive_overlap_ratio_below_hard_min"
    dominant_only = dominant_boundary.to_numpy(dtype=bool) & ~low_overlap.to_numpy(dtype=bool)
    reasons[dominant_only] = "higher_risk_boundary_lower_than_dominant_overlap"
    df["scene_boundary_ambiguous"] = ambiguous.astype(int)
    df["scene_boundary_reason"] = reasons

    if exclude:
        df.loc[ambiguous, "scene_label_available"] = 0
        df.loc[ambiguous, "scene_y_original"] = -1
        df.loc[ambiguous, "scene_y_schemaC"] = -1
        df.loc[ambiguous, "scene_y_binary"] = -1
    else:
        downgrade_mask = ambiguous & (dominant_risk >= 0)
        df.loc[downgrade_mask, "scene_y_original"] = dominant_risk[downgrade_mask].astype(int)
        df.loc[downgrade_mask, "scene_y_schemaC"] = dominant_risk[downgrade_mask].map(scene_schema_c).astype(int)
        df.loc[downgrade_mask, "scene_y_binary"] = (dominant_risk[downgrade_mask].astype(int) > 0).astype(int)
    return df


def _add_global_history_v3(scene_windows: pd.DataFrame, scene_cfg: dict[str, Any]) -> pd.DataFrame:
    df = scene_windows.copy()
    decay = float(scene_cfg.get("history_decay", 0.85))
    history_window_count = int(scene_cfg.get("history_window_count", 8))
    df["current_event_score"] = np.maximum.reduce(
        [
            df["max_prob_class1"].to_numpy(dtype=float),
            df["max_prob_class2"].to_numpy(dtype=float),
            df["max_overspeed_event_score"].to_numpy(dtype=float),
            df["max_redlight_event_score"].to_numpy(dtype=float),
            (df["max_active_family_count"].clip(upper=3) / 3.0).to_numpy(dtype=float),
            df["max_cooccurrence_strength"].to_numpy(dtype=float),
        ]
    )

    parts: list[pd.DataFrame] = []
    for _video_id, group in df.sort_values(["video_id_num", "scene_window_idx"]).groupby("video_id", sort=False):
        g = group.copy()
        event_values = g["current_event_score"].to_numpy(dtype=float)
        hist_ema = np.zeros(len(g), dtype=np.float32)
        hist_ratio = np.zeros(len(g), dtype=np.float32)
        hist_recent_max = np.zeros(len(g), dtype=np.float32)
        windows_since = np.full(len(g), 999, dtype=np.float32)
        ema = 0.0
        last_positive_index: int | None = None
        for idx in range(len(g)):
            hist_ema[idx] = ema
            start = max(0, idx - history_window_count)
            prev = event_values[start:idx]
            hist_ratio[idx] = float((prev >= 0.55).mean()) if len(prev) else 0.0
            hist_recent_max[idx] = float(prev.max()) if len(prev) else 0.0
            windows_since[idx] = float(idx - last_positive_index) if last_positive_index is not None else 999.0
            if event_values[idx] >= 0.55:
                last_positive_index = idx
            ema = decay * ema + (1.0 - decay) * float(event_values[idx])
        g["hist_event_score_ema"] = hist_ema
        g["hist_positive_ratio_8"] = hist_ratio
        g["hist_recent_max_event_score_8"] = hist_recent_max
        g["hist_windows_since_positive"] = windows_since
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def _select_tokens_v3(merged: pd.DataFrame, scene_windows: pd.DataFrame, top_k: int) -> pd.DataFrame:
    key_to_row_id = {
        (str(row.video_id), int(row.scene_window_idx)): int(row.scene_row_id)
        for row in scene_windows.itertuples(index=False)
    }
    token_frame = merged.copy()
    token_frame["scene_row_id"] = [
        key_to_row_id[(str(row.video_id), int(row.scene_window_idx))]
        for row in token_frame.itertuples(index=False)
    ]
    token_frame = token_frame.sort_values(
        ["scene_row_id", "token_must_keep", "token_selection_score", "track_id"],
        ascending=[True, False, False, True],
        kind="mergesort",
    )
    selected = token_frame.groupby("scene_row_id", sort=False).head(top_k).copy()
    selected["token_slot"] = selected.groupby("scene_row_id").cumcount().astype(int)
    keep_cols = _unique_columns(
        [
            "scene_row_id",
            "token_slot",
            "case_key",
            "video_id",
            "track_id",
            "scene_window_idx",
            "ts_window_idx",
            "start_sec",
            "end_sec",
            "round_scene_window_idx",
            "canonical_scene_start_sec",
            "canonical_scene_end_sec",
            "scene_token_overlap_sec",
            "scene_token_scene_overlap_ratio",
            "scene_token_token_overlap_ratio",
            "scene_token_completed_by_scene_end",
            "alignment_mode",
            "token_must_keep",
        ]
        + TOKEN_MODEL_FEATURE_COLUMNS
        + TOKEN_FLOOR_AUDIT_COLUMNS
        + TOKEN_AUDIT_ONLY_COLUMNS
    )
    existing_keep_cols = [col for col in keep_cols if col in selected.columns]
    return selected[existing_keep_cols].sort_values(["scene_row_id", "token_slot"]).reset_index(drop=True)


def _build_tensor_payload_v3(scene_windows: pd.DataFrame, selected_tokens: pd.DataFrame, top_k: int) -> dict[str, Any]:
    scene_count = len(scene_windows)
    token_dim = len(TOKEN_MODEL_FEATURE_COLUMNS)
    global_dim = len(GLOBAL_MODEL_FEATURE_COLUMNS)
    floor_dim = len(SCENE_FLOOR_FEATURE_COLUMNS)
    token_features = np.zeros((scene_count, top_k, token_dim), dtype=np.float32)
    token_mask = np.zeros((scene_count, top_k), dtype=np.uint8)
    token_track_ids = np.full((scene_count, top_k), "", dtype=object)
    token_case_keys = np.full((scene_count, top_k), "", dtype=object)

    for _, row in selected_tokens.iterrows():
        scene_idx = int(row["scene_row_id"])
        slot = int(row["token_slot"])
        token_features[scene_idx, slot, :] = row[TOKEN_MODEL_FEATURE_COLUMNS].astype(float).to_numpy(dtype=np.float32)
        token_mask[scene_idx, slot] = 1
        token_track_ids[scene_idx, slot] = str(row["track_id"])
        token_case_keys[scene_idx, slot] = str(row["case_key"])

    global_features = scene_windows[GLOBAL_MODEL_FEATURE_COLUMNS].astype(float).to_numpy(dtype=np.float32)
    floor_features = scene_windows[SCENE_FLOOR_FEATURE_COLUMNS].astype(float).to_numpy(dtype=np.float32)
    return {
        "token_features": token_features,
        "token_mask": token_mask,
        "global_features": global_features,
        "floor_features": floor_features,
        "y_4cls": scene_windows["scene_y_original"].astype(np.int64).to_numpy(),
        "y_binary": scene_windows["scene_y_binary"].astype(np.int64).to_numpy(),
        "y_severity": scene_windows["scene_y_schemaC"].astype(np.int64).to_numpy(),
        "label_available": scene_windows["scene_label_available"].astype(np.uint8).to_numpy(),
        "split": scene_windows["split"].astype(str).to_numpy(dtype=object),
        "video_id": scene_windows["video_id"].astype(str).to_numpy(dtype=object),
        "scene_window_idx": scene_windows["scene_window_idx"].astype(np.int64).to_numpy(),
        "start_sec": scene_windows["start_sec"].astype(float).to_numpy(dtype=np.float32),
        "end_sec": scene_windows["end_sec"].astype(float).to_numpy(dtype=np.float32),
        "token_track_ids": token_track_ids,
        "token_case_keys": token_case_keys,
        "token_feature_columns": np.asarray(TOKEN_MODEL_FEATURE_COLUMNS, dtype=object),
        "global_feature_columns": np.asarray(GLOBAL_MODEL_FEATURE_COLUMNS, dtype=object),
        "floor_feature_columns": np.asarray(SCENE_FLOOR_FEATURE_COLUMNS, dtype=object),
        "top_k": np.asarray([top_k], dtype=np.int64),
    }


def _build_audit_v3(
    config: dict[str, Any],
    scene_windows: pd.DataFrame,
    selected_tokens: pd.DataFrame,
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    tensor_payload: dict[str, Any],
) -> dict[str, Any]:
    labeled = scene_windows[scene_windows["scene_label_available"] == 1].copy()
    annotation_manifest = scene_windows.attrs.get("annotation_manifest", {})
    audit = {
        "schema_version": "scene_risk.dataset_r52_v3.audit/v1",
        "run_id": config["run_id"],
        "label_strategy": config["scene"]["label_strategy"],
        "risk1_positive": bool(config["scene"]["risk1_positive"]),
        "train_predictions_kind": config["scene"]["train_predictions_kind"],
        "top_k": int(config["scene"]["top_k"]),
        "feature_rows": int(len(features)),
        "prediction_rows": int(len(predictions)),
        "scene_window_count": int(len(scene_windows)),
        "labeled_scene_window_count": int(len(labeled)),
        "unlabeled_scene_window_count": int((scene_windows["scene_label_available"] == 0).sum()),
        "selected_token_count": int(len(selected_tokens)),
        "tensor_shapes": {
            "token_features": list(tensor_payload["token_features"].shape),
            "global_features": list(tensor_payload["global_features"].shape),
            "floor_features": list(tensor_payload["floor_features"].shape),
            "token_mask": list(tensor_payload["token_mask"].shape),
        },
        "feature_groups": {
            "token_model_features": TOKEN_MODEL_FEATURE_COLUMNS,
            "global_model_features": GLOBAL_MODEL_FEATURE_COLUMNS,
            "scene_floor_features": SCENE_FLOOR_FEATURE_COLUMNS,
            "token_audit_only_features": TOKEN_AUDIT_ONLY_COLUMNS,
            "removed_from_v2_learned_input": REMOVED_FROM_V2_LEARNED_INPUT,
        },
        "label_distribution_by_split": _label_distribution(labeled),
        "boundary_label_policy": config.get("scene", {}).get("boundary_label_policy", {"enabled": False}),
        "boundary_policy_summary": _boundary_policy_summary(scene_windows),
        "floor_coverage_by_split_label": _floor_coverage(labeled),
        "prediction_available_by_split": _prediction_available(scene_windows),
        "feature_stats": {
            "token_model_features": _numeric_stats(selected_tokens, TOKEN_MODEL_FEATURE_COLUMNS),
            "global_model_features": _numeric_stats(scene_windows, GLOBAL_MODEL_FEATURE_COLUMNS),
            "scene_floor_features": _numeric_stats(scene_windows, SCENE_FLOOR_FEATURE_COLUMNS),
        },
        "annotation_manifest": annotation_manifest,
        "notes": _audit_notes_v3(config["scene"]["train_predictions_kind"]),
    }
    return audit


def _audit_notes_v3(train_predictions_kind: str) -> list[str]:
    notes = [
        "V3 token/global tensors contain learned_input features only; hard predictions and support/policy flags are excluded from model input.",
        "Floor features are stored separately in floor_features and scene CSV; use them in Hybrid/Postprocess, not as learned input.",
        "Scene labels use annotation first_overlapping_window_id~last_overlapping_window_id and choose max risk on boundary overlaps.",
    ]
    if "oof" in str(train_predictions_kind).lower():
        notes.insert(
            2,
            "Train predictions are video-group OOF predictions; each train scene window uses single-vehicle predictions from a fold model that excluded the same video_id.",
        )
    else:
        notes.insert(
            2,
            "Train predictions are fast in-sample selected-checkpoint predictions, not OOF; do not use train metrics as final evidence.",
        )
    return notes


def render_audit_md_v3(audit: dict[str, Any]) -> str:
    lines = [
        "# Scene Dataset R52 V3 Feature-Clean Audit",
        "",
        f"- scene windows: {audit['scene_window_count']}",
        f"- labeled scene windows: {audit['labeled_scene_window_count']}",
        f"- unlabeled scene windows: {audit['unlabeled_scene_window_count']}",
        f"- selected token rows: {audit['selected_token_count']}",
        f"- top_k: {audit['top_k']}",
        f"- train prediction kind: {audit['train_predictions_kind']}",
        "",
        "## Feature Groups",
        "",
        f"- token model features: {len(audit['feature_groups']['token_model_features'])}",
        f"- global model features: {len(audit['feature_groups']['global_model_features'])}",
        f"- scene floor features: {len(audit['feature_groups']['scene_floor_features'])}",
        f"- removed from v2 learned input: {len(audit['feature_groups']['removed_from_v2_learned_input'])}",
        "",
        "## Split Label Distribution",
        "",
        "| split | risk0 | risk1 | risk2 | risk3 | positive |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in ["train", "val", "test", "all"]:
        item = audit["label_distribution_by_split"][split]
        lines.append(
            f"| {split} | {item.get('0', 0)} | {item.get('1', 0)} | {item.get('2', 0)} | {item.get('3', 0)} | {item.get('positive', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Boundary Policy",
            "",
            f"- enabled: {bool(audit.get('boundary_label_policy', {}).get('enabled', False))}",
            f"- hard_min_overlap_ratio: {audit.get('boundary_label_policy', {}).get('hard_min_overlap_ratio', '')}",
            f"- exclude_from_training_and_primary_eval: {audit.get('boundary_label_policy', {}).get('exclude_from_training_and_primary_eval', '')}",
            "",
            "| split | raw labeled | clean labeled | ambiguous | raw positive ambiguous | dominant-lower ambiguous |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in audit.get("boundary_policy_summary", []):
        lines.append(
            f"| {row['split']} | {row['raw_labeled_rows']} | {row['clean_labeled_rows']} | {row['ambiguous_rows']} | "
            f"{row['raw_positive_ambiguous_rows']} | {row['dominant_lower_ambiguous_rows']} |"
        )
    lines.extend(
        [
            "",
            "## Floor Coverage By Label",
            "",
            "| split | risk | rows | single floor >= label % | default floor >= label % | default floor FP rows |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in audit["floor_coverage_by_split_label"]:
        lines.append(
            f"| {row['split']} | {row['risk']} | {row['rows']} | {row['single_floor_ge_label_rate']:.4f} | "
            f"{row['default_floor_ge_label_rate']:.4f} | {row['default_floor_fp_rows']} |"
        )
    lines.extend(["", "## Notes", ""])
    for note in audit["notes"]:
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def _label_distribution(labeled: pd.DataFrame) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for split in ["train", "val", "test"]:
        out[split] = _one_label_distribution(labeled[labeled["split"].eq(split)])
    out["all"] = _one_label_distribution(labeled)
    return out


def _boundary_policy_summary(scene_windows: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw_available = pd.to_numeric(scene_windows.get("scene_label_available_raw", scene_windows["scene_label_available"]), errors="coerce").fillna(0).astype(int)
    clean_available = pd.to_numeric(scene_windows["scene_label_available"], errors="coerce").fillna(0).astype(int)
    raw_y = pd.to_numeric(scene_windows.get("scene_y_original_raw", scene_windows["scene_y_original"]), errors="coerce").fillna(-1).astype(int)
    clean_y = pd.to_numeric(scene_windows["scene_y_original"], errors="coerce").fillna(-1).astype(int)
    ambiguous = pd.to_numeric(scene_windows.get("scene_boundary_ambiguous", 0), errors="coerce").fillna(0).astype(int)
    dominant = pd.to_numeric(scene_windows.get("scene_label_dominant_risk", -1), errors="coerce").fillna(-1).astype(int)
    work = scene_windows.copy()
    work["_raw_available"] = raw_available
    work["_clean_available"] = clean_available
    work["_raw_y"] = raw_y
    work["_clean_y"] = clean_y
    work["_ambiguous"] = ambiguous
    work["_dominant"] = dominant
    for split in ["train", "val", "test", "all"]:
        sub = work if split == "all" else work[work["split"].eq(split)]
        amb = sub["_ambiguous"].astype(int) > 0
        rows.append(
            {
                "split": split,
                "raw_labeled_rows": int((sub["_raw_available"] == 1).sum()),
                "clean_labeled_rows": int((sub["_clean_available"] == 1).sum()),
                "ambiguous_rows": int(amb.sum()),
                "raw_positive_ambiguous_rows": int((amb & (sub["_raw_y"] > 0)).sum()),
                "dominant_lower_ambiguous_rows": int((amb & (sub["_dominant"] >= 0) & (sub["_raw_y"] > sub["_dominant"])).sum()),
                "raw_risk1_ambiguous_rows": int((amb & (sub["_raw_y"] == 1)).sum()),
                "raw_risk2_ambiguous_rows": int((amb & (sub["_raw_y"] == 2)).sum()),
                "raw_risk3_ambiguous_rows": int((amb & (sub["_raw_y"] == 3)).sum()),
                "clean_positive_rows": int(((sub["_clean_available"] == 1) & (sub["_clean_y"] > 0)).sum()),
            }
        )
    return rows


def _one_label_distribution(df: pd.DataFrame) -> dict[str, int]:
    counts = {str(k): int(v) for k, v in df["scene_y_original"].astype(int).value_counts().sort_index().items()}
    counts["positive"] = int((df["scene_y_original"].astype(int) > 0).sum())
    return counts


def _floor_coverage(labeled: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        split_df = labeled[labeled["split"].eq(split)]
        for risk in [0, 1, 2, 3]:
            sub = split_df[split_df["scene_y_original"].astype(int).eq(risk)]
            if sub.empty:
                single_rate = 0.0
                default_rate = 0.0
                default_fp = 0
            else:
                single_rate = float((sub["single_vehicle_floor_risk"] >= sub["scene_y_original"]).mean())
                default_rate = float((sub["default_floor_risk"] >= sub["scene_y_original"]).mean())
                default_fp = int(((sub["scene_y_original"] == 0) & (sub["default_floor_risk"] > 0)).sum())
            rows.append(
                {
                    "split": split,
                    "risk": risk,
                    "rows": int(len(sub)),
                    "single_floor_ge_label_rate": single_rate,
                    "default_floor_ge_label_rate": default_rate,
                    "default_floor_fp_rows": default_fp,
                }
            )
    return rows


def _prediction_available(scene_windows: pd.DataFrame) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for split in ["train", "val", "test"]:
        sub = scene_windows[scene_windows["split"].eq(split)]
        out[split] = {
            "rows": int(len(sub)),
            "with_predictions": int((sub["prediction_available_count"] > 0).sum()),
        }
    return out


def _numeric_stats(df: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float | int]]:
    stats: dict[str, dict[str, float | int]] = {}
    for col in columns:
        values = pd.to_numeric(df[col], errors="coerce")
        stats[col] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "nunique": int(values.nunique(dropna=True)),
            "zero_rate": float((values == 0).mean()),
        }
    return stats


def _nth_largest(values: pd.Series, n: int) -> float:
    arr = np.sort(pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=float))[::-1]
    if len(arr) < n:
        return 0.0
    return float(arr[n - 1])


def _schema_c_to_original_floor(value: int) -> int:
    if value <= 0:
        return 0
    if value == 1:
        return 2
    return 3


def _join_sorted_unique(values: pd.Series) -> str:
    unique = sorted({str(v) for v in values.dropna()}, key=_natural_key)
    return ",".join(unique)


def _natural_key(value: str) -> tuple[int, str]:
    try:
        return (0, f"{int(value):012d}")
    except ValueError:
        return (1, value)


def _require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"{name} missing columns: {missing}")


def _unique_columns(columns: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for col in columns:
        if col not in seen:
            seen.add(col)
            out.append(col)
    return out
