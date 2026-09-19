#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build scene-level vehicle interaction sidecar v2 with IPM world distance and lane sanity checks."
    )
    parser.add_argument("--scene-windows-csv", required=True, type=Path)
    parser.add_argument("--scene-tokens-csv", required=True, type=Path)
    parser.add_argument("--video-fps-csv", required=True, type=Path)
    parser.add_argument("--bbox-dir", required=True, type=Path)
    parser.add_argument("--ipm-json", required=True, type=Path)
    parser.add_argument("--lane-map-json", required=True, type=Path)
    parser.add_argument("--trajectory-sidecar-csv", type=Path, default=None)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--manifest-json", required=True, type=Path)
    parser.add_argument("--close-norm-threshold", type=float, default=1.05)
    parser.add_argument("--world-close-meter-threshold", type=float, default=7.5)
    parser.add_argument("--world-lateral-meter-threshold", type=float, default=3.2)
    parser.add_argument("--world-closing-speed-threshold", type=float, default=0.7)
    parser.add_argument("--cpa-horizon-sec", type=float, default=2.5)
    parser.add_argument("--cpa-danger-distance-m", type=float, default=2.2)
    parser.add_argument("--cpa-min-relative-speed-mps", type=float, default=1.0)
    parser.add_argument("--cpa-min-closing-speed-mps", type=float, default=0.7)
    parser.add_argument("--cpa-min-course-score", type=float, default=0.25)
    parser.add_argument("--min-collision-course-ratio", type=float, default=0.12)
    parser.add_argument("--min-world-close-ratio", type=float, default=0.35)
    parser.add_argument("--min-world-quality-ratio", type=float, default=0.70)
    parser.add_argument("--max-world_jitter_ratio", "--max-world-jitter-ratio", dest="max_world_jitter_ratio", type=float, default=0.25)
    parser.add_argument("--min-common-frames", type=int, default=3)
    parser.add_argument("--min-risk-floor", type=int, default=2)
    parser.add_argument(
        "--risk-source-floor-mode",
        choices=["max_all", "single_strict_only", "single_vehicle_only"],
        default="max_all",
        help="Which token floor columns define risk tracks for pair interaction extraction.",
    )
    parser.add_argument(
        "--pair-risk-scope",
        choices=["video_global", "scene_window"],
        default="video_global",
        help=(
            "video_global keeps the legacy behavior: all tracks that are risk anywhere in the video are excluded "
            "from the other-vehicle side. scene_window uses a fast source-only exclusion: a risk source is only "
            "prevented from pairing with itself, so a vehicle that becomes risk later can still be an other vehicle earlier."
        ),
    )
    args = parser.parse_args()

    scene_windows = pd.read_csv(args.scene_windows_csv, low_memory=False)
    scene_tokens = pd.read_csv(args.scene_tokens_csv, low_memory=False)
    fps_table = pd.read_csv(args.video_fps_csv, low_memory=False)
    trajectory_sidecar = (
        pd.read_csv(args.trajectory_sidecar_csv, low_memory=False)
        if args.trajectory_sidecar_csv and args.trajectory_sidecar_csv.exists()
        else None
    )

    _require_columns(scene_windows, ["scene_row_id", "video_id", "scene_window_idx", "start_sec", "end_sec"], "scene_windows")
    _require_columns(
        scene_tokens,
        [
            "scene_row_id",
            "case_key",
            "video_id",
            "track_id",
            "token_single_vehicle_floor_risk",
            "token_strict_policy_floor_risk",
            "token_support_floor_risk",
        ],
        "scene_tokens",
    )
    _require_columns(fps_table, ["video_id", "fps"], "video_fps")

    H_img_to_world, ipm_payload = load_homography(args.ipm_json)
    lane_map = load_lane_map(args.lane_map_json)
    trajectory_lookup = build_trajectory_lookup(trajectory_sidecar)
    risk_by_row = build_risk_track_lookup(scene_tokens, args.min_risk_floor, args.risk_source_floor_mode)
    fps_lookup = build_fps_lookup(fps_table)

    output_rows: list[dict[str, Any]] = []
    video_status: dict[str, dict[str, Any]] = {}
    for video_id, video_windows in scene_windows.sort_values(["video_id", "scene_window_idx"]).groupby("video_id", sort=False):
        video_key = str(video_id)
        bbox_path = args.bbox_dir / f"{video_key}.csv"
        fps = fps_lookup.get(video_key)
        if fps is None or fps <= 0:
            status = "missing_fps"
            bbox = None
        elif not bbox_path.exists():
            status = "missing_bbox_csv"
            bbox = None
        else:
            try:
                bbox = load_bbox_csv_v2(
                    bbox_path,
                    fps=fps,
                    video_id=video_key,
                    H_img_to_world=H_img_to_world,
                    ipm_payload=ipm_payload,
                    lane_map=lane_map,
                    trajectory_lookup=trajectory_lookup,
                )
                status = "ok"
            except Exception as exc:  # noqa: BLE001 - keep batch build robust and record failed videos.
                status = f"bbox_load_failed:{exc}"
                bbox = None

        frame_pair_rows = 0
        if bbox is not None:
            video_risk_tracks = sorted(
                {
                    track_id
                    for scene_row_id in video_windows["scene_row_id"].astype(int).tolist()
                    for track_id in risk_by_row.get(int(scene_row_id), {"risk_track_ids": []})["risk_track_ids"]
                }
            )
            frame_pairs = precompute_video_frame_pairs_v2(
                bbox,
                video_risk_tracks,
                cpa_horizon_sec=float(args.cpa_horizon_sec),
                cpa_danger_distance_m=float(args.cpa_danger_distance_m),
                cpa_min_relative_speed_mps=float(args.cpa_min_relative_speed_mps),
                cpa_min_closing_speed_mps=float(args.cpa_min_closing_speed_mps),
                cpa_min_course_score=float(args.cpa_min_course_score),
                other_vehicle_scope=(
                    "exclude_self_only" if args.pair_risk_scope == "scene_window" else "exclude_global_risk"
                ),
            )
            frame_pair_rows = int(len(frame_pairs))
        else:
            frame_pairs = None

        video_status[video_key] = {
            "status": status,
            "fps": fps if fps is not None else "",
            "bbox_path": str(bbox_path),
            "scene_windows": int(len(video_windows)),
            "frame_pair_rows": int(frame_pair_rows),
            "pair_risk_scope": str(args.pair_risk_scope),
        }

        for row in video_windows.itertuples(index=False):
            row_frame_pairs = frame_pairs
            output_rows.append(
                summarize_window_interaction_v2(
                    row=row,
                    frame_pairs=row_frame_pairs,
                    fps=fps,
                    risk_by_row=risk_by_row,
                    status=status,
                    close_norm_threshold=float(args.close_norm_threshold),
                    world_close_meter_threshold=float(args.world_close_meter_threshold),
                    world_lateral_meter_threshold=float(args.world_lateral_meter_threshold),
                    world_closing_speed_threshold=float(args.world_closing_speed_threshold),
                    min_collision_course_ratio=float(args.min_collision_course_ratio),
                    min_world_close_ratio=float(args.min_world_close_ratio),
                    min_world_quality_ratio=float(args.min_world_quality_ratio),
                    max_world_jitter_ratio=float(args.max_world_jitter_ratio),
                    min_common_frames=int(args.min_common_frames),
                )
            )
        video_status[video_key]["frame_pair_rows"] = int(frame_pair_rows)

    out = pd.DataFrame(output_rows).sort_values(["video_id", "scene_window_idx"]).reset_index(drop=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    manifest = {
        "schema_version": "scene_risk.scene_interaction_sidecar_v2.manifest/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "scene_windows_csv": str(args.scene_windows_csv),
            "scene_tokens_csv": str(args.scene_tokens_csv),
            "video_fps_csv": str(args.video_fps_csv),
            "bbox_dir": str(args.bbox_dir),
            "ipm_json": str(args.ipm_json),
            "lane_map_json": str(args.lane_map_json),
            "trajectory_sidecar_csv": str(args.trajectory_sidecar_csv) if args.trajectory_sidecar_csv else "",
        },
        "thresholds": {
            "close_norm_threshold": float(args.close_norm_threshold),
            "world_close_meter_threshold": float(args.world_close_meter_threshold),
            "world_lateral_meter_threshold": float(args.world_lateral_meter_threshold),
            "world_closing_speed_threshold": float(args.world_closing_speed_threshold),
            "cpa_horizon_sec": float(args.cpa_horizon_sec),
            "cpa_danger_distance_m": float(args.cpa_danger_distance_m),
            "cpa_min_relative_speed_mps": float(args.cpa_min_relative_speed_mps),
            "cpa_min_closing_speed_mps": float(args.cpa_min_closing_speed_mps),
            "cpa_min_course_score": float(args.cpa_min_course_score),
            "min_collision_course_ratio": float(args.min_collision_course_ratio),
            "min_world_close_ratio": float(args.min_world_close_ratio),
            "min_world_quality_ratio": float(args.min_world_quality_ratio),
            "max_world_jitter_ratio": float(args.max_world_jitter_ratio),
            "min_common_frames": int(args.min_common_frames),
            "min_risk_floor": int(args.min_risk_floor),
            "risk_source_floor_mode": str(args.risk_source_floor_mode),
            "pair_risk_scope": str(args.pair_risk_scope),
        },
        "outputs": {
            "output_csv": str(args.output_csv),
            "manifest_json": str(args.manifest_json),
        },
        "ipm": {
            "homography_key": "homography_img_to_world",
            "bev": ipm_payload.get("bev", {}),
            "image_size": ipm_payload.get("image_size", {}),
        },
        "lane_map": {
            "lane_count": len(lane_map["lanes"]),
            "source": str(args.lane_map_json),
        },
        "row_count": int(len(out)),
        "v1_flagged_rows": int(pd.to_numeric(out["interaction_risk2_near_other_flag"], errors="coerce").fillna(0).sum()),
        "world_flagged_rows": int(pd.to_numeric(out["interaction_world_risk2_near_other_flag"], errors="coerce").fillna(0).sum()),
        "video_status_counts": {
            str(key): int(value)
            for key, value in pd.Series([item["status"] for item in video_status.values()]).value_counts().sort_index().items()
        },
        "video_status": video_status,
    }
    args.manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output_csv} rows={len(out)} v1_flagged={manifest['v1_flagged_rows']} world_flagged={manifest['world_flagged_rows']}")
    print(f"manifest={args.manifest_json}")


def build_interaction_sidecar_from_frames(
    *,
    scene_windows: pd.DataFrame,
    scene_tokens: pd.DataFrame,
    fps_table: pd.DataFrame,
    bbox_dir: Path,
    ipm_json: Path,
    lane_map_json: Path,
    trajectory_sidecar: pd.DataFrame | None = None,
    close_norm_threshold: float = 1.05,
    world_close_meter_threshold: float = 7.5,
    world_lateral_meter_threshold: float = 3.2,
    world_closing_speed_threshold: float = 0.7,
    cpa_horizon_sec: float = 2.5,
    cpa_danger_distance_m: float = 2.2,
    cpa_min_relative_speed_mps: float = 1.0,
    cpa_min_closing_speed_mps: float = 0.7,
    cpa_min_course_score: float = 0.25,
    min_collision_course_ratio: float = 0.12,
    min_world_close_ratio: float = 0.35,
    min_world_quality_ratio: float = 0.70,
    max_world_jitter_ratio: float = 0.25,
    min_common_frames: int = 3,
    min_risk_floor: int = 2,
    risk_source_floor_mode: str = "max_all",
    pair_risk_scope: str = "video_global",
    H_img_to_world: np.ndarray | None = None,
    ipm_payload: dict[str, Any] | None = None,
    lane_map: dict[str, Any] | None = None,
    trajectory_lookup: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    scene_windows = scene_windows.copy()
    scene_tokens = scene_tokens.copy()
    fps_table = fps_table.copy()
    _require_columns(scene_windows, ["scene_row_id", "video_id", "scene_window_idx", "start_sec", "end_sec"], "scene_windows")
    _require_columns(
        scene_tokens,
        [
            "scene_row_id",
            "case_key",
            "video_id",
            "track_id",
            "token_single_vehicle_floor_risk",
            "token_strict_policy_floor_risk",
            "token_support_floor_risk",
        ],
        "scene_tokens",
    )
    _require_columns(fps_table, ["video_id", "fps"], "video_fps")

    if H_img_to_world is None or ipm_payload is None:
        H_img_to_world, ipm_payload = load_homography(ipm_json)
    if lane_map is None:
        lane_map = load_lane_map(lane_map_json)
    if trajectory_lookup is None:
        trajectory_lookup = build_trajectory_lookup(trajectory_sidecar)
    risk_by_row = build_risk_track_lookup(scene_tokens, int(min_risk_floor), str(risk_source_floor_mode))
    fps_lookup = build_fps_lookup(fps_table)
    if pair_risk_scope not in {"video_global", "scene_window"}:
        raise ValueError(f"invalid pair_risk_scope: {pair_risk_scope}")

    output_rows: list[dict[str, Any]] = []
    video_status: dict[str, dict[str, Any]] = {}
    for video_id, video_windows in scene_windows.sort_values(["video_id", "scene_window_idx"]).groupby("video_id", sort=False):
        video_key = str(video_id)
        bbox_path = bbox_dir / f"{video_key}.csv"
        fps = fps_lookup.get(video_key)
        if fps is None or fps <= 0:
            status = "missing_fps"
            bbox = None
        elif not bbox_path.exists():
            status = "missing_bbox_csv"
            bbox = None
        else:
            try:
                bbox = load_bbox_csv_v2(
                    bbox_path,
                    fps=fps,
                    video_id=video_key,
                    H_img_to_world=H_img_to_world,
                    ipm_payload=ipm_payload,
                    lane_map=lane_map,
                    trajectory_lookup=trajectory_lookup,
                )
                status = "ok"
            except Exception as exc:  # noqa: BLE001 - mirror CLI robustness and record failed videos.
                status = f"bbox_load_failed:{exc}"
                bbox = None

        frame_pair_rows = 0
        if bbox is not None:
            video_risk_tracks = sorted(
                {
                    track_id
                    for scene_row_id in video_windows["scene_row_id"].astype(int).tolist()
                    for track_id in risk_by_row.get(int(scene_row_id), {"risk_track_ids": []})["risk_track_ids"]
                }
            )
            frame_pairs = precompute_video_frame_pairs_v2(
                bbox,
                video_risk_tracks,
                cpa_horizon_sec=float(cpa_horizon_sec),
                cpa_danger_distance_m=float(cpa_danger_distance_m),
                cpa_min_relative_speed_mps=float(cpa_min_relative_speed_mps),
                cpa_min_closing_speed_mps=float(cpa_min_closing_speed_mps),
                cpa_min_course_score=float(cpa_min_course_score),
                other_vehicle_scope=(
                    "exclude_self_only" if pair_risk_scope == "scene_window" else "exclude_global_risk"
                ),
            )
            frame_pair_rows = int(len(frame_pairs))
        else:
            frame_pairs = None

        video_status[video_key] = {
            "status": status,
            "fps": fps if fps is not None else "",
            "bbox_path": str(bbox_path),
            "scene_windows": int(len(video_windows)),
            "frame_pair_rows": int(frame_pair_rows),
            "pair_risk_scope": str(pair_risk_scope),
        }

        for row in video_windows.itertuples(index=False):
            row_frame_pairs = frame_pairs
            output_rows.append(
                summarize_window_interaction_v2(
                    row=row,
                    frame_pairs=row_frame_pairs,
                    fps=fps,
                    risk_by_row=risk_by_row,
                    status=status,
                    close_norm_threshold=float(close_norm_threshold),
                    world_close_meter_threshold=float(world_close_meter_threshold),
                    world_lateral_meter_threshold=float(world_lateral_meter_threshold),
                    world_closing_speed_threshold=float(world_closing_speed_threshold),
                    min_collision_course_ratio=float(min_collision_course_ratio),
                    min_world_close_ratio=float(min_world_close_ratio),
                    min_world_quality_ratio=float(min_world_quality_ratio),
                    max_world_jitter_ratio=float(max_world_jitter_ratio),
                    min_common_frames=int(min_common_frames),
                )
            )
        video_status[video_key]["frame_pair_rows"] = int(frame_pair_rows)

    out = pd.DataFrame(output_rows).sort_values(["video_id", "scene_window_idx"]).reset_index(drop=True)
    manifest = {
        "schema_version": "scene_risk.scene_interaction_sidecar_v2.manifest/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "bbox_dir": str(bbox_dir),
            "ipm_json": str(ipm_json),
            "lane_map_json": str(lane_map_json),
            "trajectory_sidecar_in_memory": trajectory_sidecar is not None,
        },
        "thresholds": {
            "close_norm_threshold": float(close_norm_threshold),
            "world_close_meter_threshold": float(world_close_meter_threshold),
            "world_lateral_meter_threshold": float(world_lateral_meter_threshold),
            "world_closing_speed_threshold": float(world_closing_speed_threshold),
            "cpa_horizon_sec": float(cpa_horizon_sec),
            "cpa_danger_distance_m": float(cpa_danger_distance_m),
            "cpa_min_relative_speed_mps": float(cpa_min_relative_speed_mps),
            "cpa_min_closing_speed_mps": float(cpa_min_closing_speed_mps),
            "cpa_min_course_score": float(cpa_min_course_score),
            "min_collision_course_ratio": float(min_collision_course_ratio),
            "min_world_close_ratio": float(min_world_close_ratio),
            "min_world_quality_ratio": float(min_world_quality_ratio),
            "max_world_jitter_ratio": float(max_world_jitter_ratio),
            "min_common_frames": int(min_common_frames),
            "min_risk_floor": int(min_risk_floor),
            "risk_source_floor_mode": str(risk_source_floor_mode),
            "pair_risk_scope": str(pair_risk_scope),
        },
        "ipm": {
            "homography_key": "homography_img_to_world",
            "bev": ipm_payload.get("bev", {}) if ipm_payload else {},
            "image_size": ipm_payload.get("image_size", {}) if ipm_payload else {},
        },
        "lane_map": {
            "lane_count": len(lane_map["lanes"]) if lane_map else 0,
            "source": str(lane_map_json),
        },
        "row_count": int(len(out)),
        "v1_flagged_rows": int(pd.to_numeric(out["interaction_risk2_near_other_flag"], errors="coerce").fillna(0).sum()),
        "world_flagged_rows": int(pd.to_numeric(out["interaction_world_risk2_near_other_flag"], errors="coerce").fillna(0).sum()),
        "video_status_counts": {
            str(key): int(value)
            for key, value in pd.Series([item["status"] for item in video_status.values()]).value_counts().sort_index().items()
        },
        "video_status": video_status,
    }
    return out, manifest


def load_homography(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    H = np.asarray(payload.get("homography_img_to_world", []), dtype=np.float64)
    if H.shape != (3, 3) or not np.all(np.isfinite(H)):
        raise ValueError(f"invalid homography_img_to_world in {path}")
    return H, payload


def load_lane_map(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    lanes = []
    for lane in payload.get("lanes", []):
        polygon = np.asarray(lane.get("polygon", []), dtype=np.float64)
        direction = np.asarray(lane.get("direction", []), dtype=np.float64)
        if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
            continue
        if direction.shape != (2,) or not np.all(np.isfinite(direction)):
            direction = np.array([np.nan, np.nan], dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if np.isfinite(norm) and norm > 1e-6:
            direction = direction / norm
        lanes.append(
            {
                "lane_id": str(lane.get("lane_id", "")),
                "polygon": polygon,
                "direction_img": direction,
            }
        )
    return {
        "source": str(path),
        "image_size": payload.get("resolution") or payload.get("image_size") or {"width": 1920, "height": 1080},
        "lanes": lanes,
    }


def build_trajectory_lookup(sidecar: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    if sidecar is None or sidecar.empty:
        return {}
    required = [
        "video_id",
        "track_id",
        "ts_window_idx",
        "selected_lane_known",
        "occupied_lane_flow_cos",
        "opposite_lane_occupancy_ratio",
        "artifact_bbox_area_cv",
        "artifact_max_center_step",
        "heading_motion_gate",
    ]
    available = [column for column in required if column in sidecar.columns]
    if not {"video_id", "track_id", "ts_window_idx"}.issubset(set(available)):
        return {}
    frame = sidecar[available].copy()
    frame["video_id"] = frame["video_id"].astype(str)
    for col in ["track_id", "ts_window_idx"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["video_id", "track_id", "ts_window_idx"]).copy()
    frame["track_id"] = frame["track_id"].astype(int)
    frame["approx_ts_window_idx"] = frame["ts_window_idx"].astype(int)
    for col in available:
        if col in {"video_id", "track_id", "ts_window_idx"}:
            continue
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
        frame[f"traj_{col}"] = frame[col].astype(float)
    keep_cols = ["track_id", "approx_ts_window_idx"] + [
        f"traj_{col}" for col in available if col not in {"video_id", "track_id", "ts_window_idx"}
    ]
    lookup: dict[str, pd.DataFrame] = {}
    for video_id, group in frame.groupby("video_id", sort=False):
        lookup[str(video_id)] = group[keep_cols].drop_duplicates(["track_id", "approx_ts_window_idx"], keep="last")
    return lookup


def build_risk_track_lookup(tokens: pd.DataFrame, min_risk_floor: int, risk_source_floor_mode: str) -> dict[int, dict[str, Any]]:
    frame = tokens.copy()
    for col in ["token_single_vehicle_floor_risk", "token_strict_policy_floor_risk", "token_support_floor_risk"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0).astype(int)
    frame["track_id_int"] = pd.to_numeric(frame["track_id"], errors="coerce")
    if risk_source_floor_mode == "single_vehicle_only":
        source_floor = frame["token_single_vehicle_floor_risk"]
    elif risk_source_floor_mode == "single_strict_only":
        source_floor = frame[["token_single_vehicle_floor_risk", "token_strict_policy_floor_risk"]].max(axis=1)
    else:
        source_floor = frame[
            ["token_single_vehicle_floor_risk", "token_strict_policy_floor_risk", "token_support_floor_risk"]
        ].max(axis=1)
    risk = frame[
        (frame["track_id_int"].notna())
        & (source_floor >= min_risk_floor)
    ].copy()
    lookup: dict[int, dict[str, Any]] = {}
    for scene_row_id, group in risk.groupby("scene_row_id", sort=False):
        track_ids = sorted({int(value) for value in group["track_id_int"].astype(int).tolist()})
        case_keys = sorted({str(value) for value in group["case_key"].tolist() if str(value)})
        lookup[int(scene_row_id)] = {"risk_track_ids": track_ids, "risk_case_keys": case_keys}
    return lookup


def build_fps_lookup(fps_table: pd.DataFrame) -> dict[str, float]:
    status_col = "status" if "status" in fps_table.columns else "probe_status" if "probe_status" in fps_table.columns else None
    ok = fps_table.copy()
    if status_col is not None:
        ok = ok[ok[status_col].astype(str).eq("ok")].copy()
    ok["fps"] = pd.to_numeric(ok["fps"], errors="coerce")
    return {str(row.video_id): float(row.fps) for row in ok.itertuples(index=False) if pd.notna(row.fps) and float(row.fps) > 0}


def load_bbox_csv_v2(
    path: Path,
    *,
    fps: float,
    video_id: str,
    H_img_to_world: np.ndarray,
    ipm_payload: dict[str, Any],
    lane_map: dict[str, Any],
    trajectory_lookup: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    bbox = pd.read_csv(path, low_memory=False)
    _require_columns(bbox, ["frame", "track_id", "x1", "y1", "x2", "y2"], f"bbox[{path.name}]")
    for col in ["frame", "track_id", "x1", "y1", "x2", "y2"]:
        bbox[col] = pd.to_numeric(bbox[col], errors="coerce")
    bbox = bbox.dropna(subset=["frame", "track_id", "x1", "y1", "x2", "y2"]).copy()
    bbox["frame"] = bbox["frame"].astype(int)
    bbox["track_id"] = bbox["track_id"].astype(int)
    if "timestamp_sec" in bbox.columns:
        timestamp = pd.to_numeric(bbox["timestamp_sec"], errors="coerce")
        bbox["time_sec"] = timestamp
    else:
        bbox["time_sec"] = bbox["frame"].astype(float) / float(fps)
    bbox["time_sec"] = bbox["time_sec"].fillna(bbox["frame"].astype(float) / float(fps))
    bbox["cx"] = (bbox["x1"] + bbox["x2"]) * 0.5
    bbox["cy"] = (bbox["y1"] + bbox["y2"]) * 0.5
    bbox["foot_x"] = bbox["cx"]
    bbox["foot_y"] = bbox["y2"]
    bbox["w"] = (bbox["x2"] - bbox["x1"]).clip(lower=1.0)
    bbox["h"] = (bbox["y2"] - bbox["y1"]).clip(lower=1.0)
    bbox["diag"] = np.hypot(bbox["w"], bbox["h"]).clip(lower=1.0)

    bbox = project_bbox_to_world(bbox, H_img_to_world, ipm_payload)
    bbox = assign_lane_features(bbox, lane_map, H_img_to_world)
    bbox = attach_trajectory_window_quality(bbox, video_id, trajectory_lookup)
    bbox = add_track_motion_features(bbox)
    return bbox.sort_values(["frame", "track_id"]).reset_index(drop=True)


def project_bbox_to_world(bbox: pd.DataFrame, H_img_to_world: np.ndarray, ipm_payload: dict[str, Any]) -> pd.DataFrame:
    out = bbox.copy()
    points = np.stack([out["foot_x"].to_numpy(dtype=float), out["foot_y"].to_numpy(dtype=float)], axis=1)
    homogeneous = np.concatenate([points, np.ones((len(points), 1), dtype=np.float64)], axis=1)
    projected = homogeneous @ H_img_to_world.T
    denom = projected[:, 2]
    valid_denom = np.isfinite(denom) & (np.abs(denom) >= 1e-9)
    world_x = np.full(len(out), np.nan, dtype=np.float64)
    world_y = np.full(len(out), np.nan, dtype=np.float64)
    world_x[valid_denom] = projected[valid_denom, 0] / denom[valid_denom]
    world_y[valid_denom] = projected[valid_denom, 1] / denom[valid_denom]
    out["world_x_m"] = world_x
    out["world_y_m"] = world_y
    bottom_left = project_points(
        np.stack([out["x1"].to_numpy(dtype=float), out["y2"].to_numpy(dtype=float)], axis=1),
        H_img_to_world,
    )
    bottom_right = project_points(
        np.stack([out["x2"].to_numpy(dtype=float), out["y2"].to_numpy(dtype=float)], axis=1),
        H_img_to_world,
    )
    width_m = np.linalg.norm(bottom_right - bottom_left, axis=1)
    valid_width = np.isfinite(width_m) & (width_m >= 0.3) & (width_m <= 8.0)
    out["world_bbox_width_m"] = np.where(valid_width, width_m, np.nan)
    object_radius = np.where(valid_width, np.clip(width_m * 0.5 + 0.6, 1.1, 2.1), 1.5)
    out["world_collision_object_radius_m"] = object_radius
    ipm_valid = np.isfinite(world_x) & np.isfinite(world_y)
    bev = ipm_payload.get("bev", {}) or {}
    if all(key in bev for key in ["min_x", "max_x", "min_y", "max_y"]):
        margin_m = 5.0
        in_bev = (
            (world_x >= float(bev["min_x"]) - margin_m)
            & (world_x <= float(bev["max_x"]) + margin_m)
            & (world_y >= float(bev["min_y"]) - margin_m)
            & (world_y <= float(bev["max_y"]) + margin_m)
        )
    else:
        in_bev = np.ones(len(out), dtype=bool)
    out["ipm_point_valid"] = (ipm_valid & in_bev).astype(np.int8)

    image_size = ipm_payload.get("image_size", {}) or {}
    width = float(image_size.get("width", 1920.0))
    height = float(image_size.get("height", 1080.0))
    edge_distance = np.minimum.reduce(
        [
            out["foot_x"].to_numpy(dtype=float),
            width - out["foot_x"].to_numpy(dtype=float),
            out["foot_y"].to_numpy(dtype=float),
            height - out["foot_y"].to_numpy(dtype=float),
        ]
    )
    out["image_edge_distance_px"] = edge_distance
    out["image_edge_low_quality"] = ((edge_distance < 20.0) | (edge_distance < 0.0)).astype(np.int8)
    return out


def assign_lane_features(bbox: pd.DataFrame, lane_map: dict[str, Any], H_img_to_world: np.ndarray) -> pd.DataFrame:
    out = bbox.copy()
    n = len(out)
    out["lane_id"] = ""
    out["lane_known"] = 0
    out["lane_dir_img_x"] = np.nan
    out["lane_dir_img_y"] = np.nan
    out["lane_dir_world_x"] = np.nan
    out["lane_dir_world_y"] = np.nan
    if n == 0 or not lane_map.get("lanes"):
        return out
    x = out["foot_x"].to_numpy(dtype=float)
    y = out["foot_y"].to_numpy(dtype=float)
    assigned = np.zeros(n, dtype=bool)
    for lane in lane_map["lanes"]:
        mask = point_in_polygon_vectorized(x, y, lane["polygon"]) & ~assigned
        if not bool(mask.any()):
            continue
        direction_img = lane["direction_img"]
        out.loc[mask, "lane_id"] = lane["lane_id"]
        out.loc[mask, "lane_known"] = 1
        out.loc[mask, "lane_dir_img_x"] = float(direction_img[0])
        out.loc[mask, "lane_dir_img_y"] = float(direction_img[1])
        assigned |= mask

    known = out["lane_known"].to_numpy(dtype=int) > 0
    if bool(known.any()):
        world_dir = local_lane_direction_world(
            out.loc[known, "foot_x"].to_numpy(dtype=float),
            out.loc[known, "foot_y"].to_numpy(dtype=float),
            out.loc[known, "lane_dir_img_x"].to_numpy(dtype=float),
            out.loc[known, "lane_dir_img_y"].to_numpy(dtype=float),
            H_img_to_world,
        )
        out.loc[known, "lane_dir_world_x"] = world_dir[:, 0]
        out.loc[known, "lane_dir_world_y"] = world_dir[:, 1]
    return out


def attach_trajectory_window_quality(
    bbox: pd.DataFrame,
    video_id: str,
    trajectory_lookup: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    out = bbox.copy()
    # Scene / single-vehicle windows use 0.6667 sec window and 0.3333 sec stride.
    out["approx_ts_window_idx"] = np.floor((out["time_sec"].to_numpy(dtype=float) + 1e-6) / 0.3333).astype(int)
    defaults = {
        "selected_lane_known": 0.0,
        "occupied_lane_flow_cos": 0.0,
        "opposite_lane_occupancy_ratio": 0.0,
        "artifact_bbox_area_cv": 0.0,
        "artifact_max_center_step": 0.0,
        "heading_motion_gate": 0.0,
    }
    for key in defaults:
        out[f"traj_{key}"] = defaults[key]
    if not trajectory_lookup:
        return out
    features = trajectory_lookup.get(str(video_id))
    if features is None or features.empty:
        return out
    original_index = out.index
    out = out.merge(features, on=["track_id", "approx_ts_window_idx"], how="left", suffixes=("", "_lookup"))
    out.index = original_index
    for key, default in defaults.items():
        col = f"traj_{key}"
        lookup_col = f"{col}_lookup"
        if lookup_col in out.columns:
            out[col] = pd.to_numeric(out[lookup_col], errors="coerce").fillna(out[col])
            out = out.drop(columns=[lookup_col])
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(default)
    return out


def add_track_motion_features(bbox: pd.DataFrame) -> pd.DataFrame:
    out = bbox.sort_values(["track_id", "frame"], kind="mergesort").copy()
    out["world_x_smooth_m"] = np.nan
    out["world_y_smooth_m"] = np.nan
    out["world_vx_mps"] = np.nan
    out["world_vy_mps"] = np.nan
    out["world_speed_kmh"] = np.nan
    out["world_motion_extreme"] = 0
    for _track_id, group in out.groupby("track_id", sort=False):
        idx = group.index
        x = group["world_x_m"].rolling(window=5, center=True, min_periods=1).median()
        y = group["world_y_m"].rolling(window=5, center=True, min_periods=1).median()
        t = group["time_sec"].to_numpy(dtype=float)
        vx = np.full(len(group), np.nan, dtype=np.float64)
        vy = np.full(len(group), np.nan, dtype=np.float64)
        speed = np.full(len(group), np.nan, dtype=np.float64)
        dx = np.diff(x.to_numpy(dtype=float))
        dy = np.diff(y.to_numpy(dtype=float))
        dt = np.diff(t)
        valid = np.isfinite(dx) & np.isfinite(dy) & np.isfinite(dt) & (dt > 1e-8)
        vx[1:][valid] = dx[valid] / dt[valid]
        vy[1:][valid] = dy[valid] / dt[valid]
        speed[1:][valid] = np.hypot(vx[1:][valid], vy[1:][valid]) * 3.6
        out.loc[idx, "world_x_smooth_m"] = x.to_numpy(dtype=float)
        out.loc[idx, "world_y_smooth_m"] = y.to_numpy(dtype=float)
        out.loc[idx, "world_vx_mps"] = vx
        out.loc[idx, "world_vy_mps"] = vy
        out.loc[idx, "world_speed_kmh"] = speed
        out.loc[idx, "world_motion_extreme"] = (np.isfinite(speed) & (speed > 220.0)).astype(np.int8)
    return out.sort_values(["frame", "track_id"], kind="mergesort").reset_index(drop=True)


def precompute_video_frame_pairs_v2(
    bbox: pd.DataFrame,
    risk_track_ids: list[int],
    *,
    cpa_horizon_sec: float,
    cpa_danger_distance_m: float,
    cpa_min_relative_speed_mps: float,
    cpa_min_closing_speed_mps: float,
    cpa_min_course_score: float,
    other_vehicle_scope: str = "exclude_global_risk",
) -> pd.DataFrame:
    columns = [
        "frame",
        "time_sec",
        "risk_track_id",
        "nearest_other_track_id",
        "norm_center_distance",
        "center_distance_px",
        "iou",
        "world_nearest_other_track_id",
        "world_distance_m",
        "world_longitudinal_m",
        "world_lateral_m",
        "world_closing_speed_mps",
        "world_ttc_sec",
        "world_cpa_other_track_id",
        "world_cpa_time_sec",
        "world_cpa_min_distance_m",
        "world_cpa_collision_ttc_sec",
        "world_cpa_collision_radius_m",
        "world_cpa_relative_speed_mps",
        "world_cpa_closing_speed_mps",
        "world_cpa_bearing_closure_score",
        "world_cpa_crossing_angle_deg",
        "world_cpa_collision_course_score",
        "world_cpa_collision_course_flag",
        "world_pair_valid",
        "world_pair_quality",
        "world_same_lane",
        "world_lane_known_pair",
        "world_edge_low_quality_pair",
        "world_motion_extreme_pair",
        "risk_world_speed_kmh",
        "other_world_speed_kmh",
        "cpa_risk_world_speed_kmh",
        "cpa_other_world_speed_kmh",
    ]
    if bbox.empty or not risk_track_ids:
        return pd.DataFrame(columns=columns)
    if other_vehicle_scope not in {"exclude_global_risk", "exclude_self_only"}:
        raise ValueError(f"invalid other_vehicle_scope: {other_vehicle_scope}")
    risk_set = {int(value) for value in risk_track_ids}
    records: list[dict[str, Any]] = []
    for frame, group in bbox.groupby("frame", sort=False):
        risk_boxes = group[group["track_id"].isin(risk_set)]
        if other_vehicle_scope == "exclude_self_only":
            other_boxes = group
        else:
            other_boxes = group[~group["track_id"].isin(risk_set)]
        if risk_boxes.empty or other_boxes.empty:
            continue
        time_sec = float(group["time_sec"].iloc[0])
        risk_arr = risk_boxes[
            [
                "track_id",
                "cx",
                "cy",
                "diag",
                "x1",
                "y1",
                "x2",
                "y2",
                "world_x_smooth_m",
                "world_y_smooth_m",
                "world_vx_mps",
                "world_vy_mps",
                "world_speed_kmh",
                "ipm_point_valid",
                "lane_known",
                "lane_dir_world_x",
                "lane_dir_world_y",
                "image_edge_low_quality",
                "world_motion_extreme",
                "world_collision_object_radius_m",
            ]
        ].to_numpy(dtype=float)
        other_arr = other_boxes[
            [
                "track_id",
                "cx",
                "cy",
                "diag",
                "x1",
                "y1",
                "x2",
                "y2",
                "world_x_smooth_m",
                "world_y_smooth_m",
                "world_vx_mps",
                "world_vy_mps",
                "world_speed_kmh",
                "ipm_point_valid",
                "lane_known",
                "lane_dir_world_x",
                "lane_dir_world_y",
                "image_edge_low_quality",
                "world_motion_extreme",
                "world_collision_object_radius_m",
            ]
        ].to_numpy(dtype=float)
        risk_lane_ids = risk_boxes["lane_id"].astype(str).to_numpy()
        other_lane_ids = other_boxes["lane_id"].astype(str).to_numpy()

        img_dx = risk_arr[:, 1:2] - other_arr[:, 1][None, :]
        img_dy = risk_arr[:, 2:3] - other_arr[:, 2][None, :]
        px_dist = np.hypot(img_dx, img_dy)
        denom = np.maximum((risk_arr[:, 3:4] + other_arr[:, 3][None, :]) * 0.5, 1.0)
        norm_dist = px_dist / denom
        self_pair = risk_arr[:, 0:1] == other_arr[:, 0][None, :]
        if other_vehicle_scope == "exclude_self_only":
            norm_dist = np.where(self_pair, np.inf, norm_dist)
        has_other = np.isfinite(norm_dist).any(axis=1)
        if not bool(has_other.any()):
            continue
        img_nearest_idx = norm_dist.argmin(axis=1)

        world_dx = other_arr[:, 8][None, :] - risk_arr[:, 8:9]
        world_dy = other_arr[:, 9][None, :] - risk_arr[:, 9:10]
        world_dist = np.hypot(world_dx, world_dy)
        valid_world = (
            np.isfinite(world_dist)
            & (risk_arr[:, 13:14] >= 1.0)
            & (other_arr[:, 13][None, :] >= 1.0)
        )
        if other_vehicle_scope == "exclude_self_only":
            valid_world = valid_world & (~self_pair)
        world_score = np.where(valid_world, world_dist, np.inf)
        world_nearest_idx = world_score.argmin(axis=1)
        row_indices = np.arange(risk_arr.shape[0])

        selected_img_other = other_arr[img_nearest_idx]
        selected_norm = norm_dist[row_indices, img_nearest_idx]
        selected_px = px_dist[row_indices, img_nearest_idx]
        selected_iou = bbox_iou_arrays(risk_arr, selected_img_other)

        selected_world_other = other_arr[world_nearest_idx]
        selected_world_valid = np.isfinite(world_score[row_indices, world_nearest_idx])
        selected_world_dist = world_dist[row_indices, world_nearest_idx]
        rel_x = selected_world_other[:, 8] - risk_arr[:, 8]
        rel_y = selected_world_other[:, 9] - risk_arr[:, 9]
        dir_x = risk_arr[:, 15].copy()
        dir_y = risk_arr[:, 16].copy()
        motion_dir = np.stack([risk_arr[:, 10], risk_arr[:, 11]], axis=1)
        motion_norm = np.linalg.norm(motion_dir, axis=1)
        fallback = (~np.isfinite(dir_x)) | (~np.isfinite(dir_y)) | (np.hypot(dir_x, dir_y) < 1e-6)
        moving = np.isfinite(motion_norm) & (motion_norm > 0.3)
        use_motion = fallback & moving
        dir_x[use_motion] = motion_dir[use_motion, 0] / motion_norm[use_motion]
        dir_y[use_motion] = motion_dir[use_motion, 1] / motion_norm[use_motion]
        dir_valid = np.isfinite(dir_x) & np.isfinite(dir_y) & (np.hypot(dir_x, dir_y) > 1e-6)
        longitudinal = np.full(risk_arr.shape[0], np.nan, dtype=np.float64)
        lateral = np.full(risk_arr.shape[0], np.nan, dtype=np.float64)
        longitudinal[dir_valid] = rel_x[dir_valid] * dir_x[dir_valid] + rel_y[dir_valid] * dir_y[dir_valid]
        lateral[dir_valid] = np.abs(rel_x[dir_valid] * (-dir_y[dir_valid]) + rel_y[dir_valid] * dir_x[dir_valid])

        rel_dist = np.maximum(selected_world_dist, 1e-6)
        unit_x = rel_x / rel_dist
        unit_y = rel_y / rel_dist
        closing_speed = (
            (risk_arr[:, 10] - selected_world_other[:, 10]) * unit_x
            + (risk_arr[:, 11] - selected_world_other[:, 11]) * unit_y
        )
        closing_speed = np.where(np.isfinite(closing_speed), closing_speed, np.nan)
        ttc = np.full_like(selected_world_dist, np.nan, dtype=np.float64)
        ttc_mask = (closing_speed > 1e-6) & np.isfinite(selected_world_dist)
        np.divide(selected_world_dist, closing_speed, out=ttc, where=ttc_mask)

        (
            cpa_idx,
            cpa_valid,
            cpa_time,
            cpa_distance,
            cpa_collision_ttc,
            cpa_collision_radius,
            cpa_relative_speed,
            cpa_closing_speed,
            cpa_bearing_closure,
            cpa_crossing_angle,
            cpa_score,
            cpa_flag,
        ) = compute_cpa_selection(
            risk_arr,
            other_arr,
            valid_world=valid_world,
            cpa_horizon_sec=cpa_horizon_sec,
            cpa_danger_distance_m=cpa_danger_distance_m,
            cpa_min_relative_speed_mps=cpa_min_relative_speed_mps,
            cpa_min_closing_speed_mps=cpa_min_closing_speed_mps,
            cpa_min_course_score=cpa_min_course_score,
        )
        selected_cpa_other = other_arr[cpa_idx]

        selected_other_lane_ids = other_lane_ids[world_nearest_idx]
        selected_risk_lane_ids = risk_lane_ids
        lane_known_pair = (
            (risk_arr[:, 14] >= 1.0)
            & (selected_world_other[:, 14] >= 1.0)
            & (selected_risk_lane_ids != "")
            & (selected_other_lane_ids != "")
        )
        same_lane = lane_known_pair & (selected_risk_lane_ids == selected_other_lane_ids)
        edge_low_quality_pair = (risk_arr[:, 17] >= 1.0) | (selected_world_other[:, 17] >= 1.0)
        motion_extreme_pair = (risk_arr[:, 18] >= 1.0) | (selected_world_other[:, 18] >= 1.0)
        pair_quality = (
            selected_world_valid.astype(float)
            * (1.0 - edge_low_quality_pair.astype(float) * 0.35)
            * (1.0 - motion_extreme_pair.astype(float) * 0.75)
        )

        for idx in range(risk_arr.shape[0]):
            if not bool(has_other[idx]):
                continue
            records.append(
                {
                    "frame": int(frame),
                    "time_sec": time_sec,
                    "risk_track_id": int(risk_arr[idx, 0]),
                    "nearest_other_track_id": int(selected_img_other[idx, 0]),
                    "norm_center_distance": float(selected_norm[idx]),
                    "center_distance_px": float(selected_px[idx]),
                    "iou": float(selected_iou[idx]),
                    "world_nearest_other_track_id": int(selected_world_other[idx, 0]) if selected_world_valid[idx] else "",
                    "world_distance_m": float(selected_world_dist[idx]) if selected_world_valid[idx] else np.nan,
                    "world_longitudinal_m": float(longitudinal[idx]) if selected_world_valid[idx] else np.nan,
                    "world_lateral_m": float(lateral[idx]) if selected_world_valid[idx] else np.nan,
                    "world_closing_speed_mps": float(closing_speed[idx]) if selected_world_valid[idx] else np.nan,
                    "world_ttc_sec": float(ttc[idx]) if selected_world_valid[idx] and np.isfinite(ttc[idx]) else np.nan,
                    "world_cpa_other_track_id": int(selected_cpa_other[idx, 0]) if cpa_valid[idx] else "",
                    "world_cpa_time_sec": float(cpa_time[idx]) if cpa_valid[idx] and np.isfinite(cpa_time[idx]) else np.nan,
                    "world_cpa_min_distance_m": float(cpa_distance[idx]) if cpa_valid[idx] and np.isfinite(cpa_distance[idx]) else np.nan,
                    "world_cpa_collision_ttc_sec": (
                        float(cpa_collision_ttc[idx]) if cpa_valid[idx] and np.isfinite(cpa_collision_ttc[idx]) else np.nan
                    ),
                    "world_cpa_collision_radius_m": (
                        float(cpa_collision_radius[idx]) if cpa_valid[idx] and np.isfinite(cpa_collision_radius[idx]) else np.nan
                    ),
                    "world_cpa_relative_speed_mps": (
                        float(cpa_relative_speed[idx]) if cpa_valid[idx] and np.isfinite(cpa_relative_speed[idx]) else np.nan
                    ),
                    "world_cpa_closing_speed_mps": (
                        float(cpa_closing_speed[idx]) if cpa_valid[idx] and np.isfinite(cpa_closing_speed[idx]) else np.nan
                    ),
                    "world_cpa_bearing_closure_score": (
                        float(cpa_bearing_closure[idx]) if cpa_valid[idx] and np.isfinite(cpa_bearing_closure[idx]) else np.nan
                    ),
                    "world_cpa_crossing_angle_deg": (
                        float(cpa_crossing_angle[idx]) if cpa_valid[idx] and np.isfinite(cpa_crossing_angle[idx]) else np.nan
                    ),
                    "world_cpa_collision_course_score": float(cpa_score[idx]) if cpa_valid[idx] and np.isfinite(cpa_score[idx]) else 0.0,
                    "world_cpa_collision_course_flag": int(cpa_flag[idx]),
                    "world_pair_valid": int(selected_world_valid[idx]),
                    "world_pair_quality": float(pair_quality[idx]),
                    "world_same_lane": int(same_lane[idx]),
                    "world_lane_known_pair": int(lane_known_pair[idx]),
                    "world_edge_low_quality_pair": int(edge_low_quality_pair[idx]),
                    "world_motion_extreme_pair": int(motion_extreme_pair[idx]),
                    "risk_world_speed_kmh": float(risk_arr[idx, 12]) if np.isfinite(risk_arr[idx, 12]) else np.nan,
                    "other_world_speed_kmh": float(selected_world_other[idx, 12]) if np.isfinite(selected_world_other[idx, 12]) else np.nan,
                    "cpa_risk_world_speed_kmh": float(risk_arr[idx, 12]) if cpa_valid[idx] and np.isfinite(risk_arr[idx, 12]) else np.nan,
                    "cpa_other_world_speed_kmh": (
                        float(selected_cpa_other[idx, 12]) if cpa_valid[idx] and np.isfinite(selected_cpa_other[idx, 12]) else np.nan
                    ),
                }
            )
    return pd.DataFrame(records, columns=columns)


def compute_cpa_selection(
    risk_arr: np.ndarray,
    other_arr: np.ndarray,
    *,
    valid_world: np.ndarray,
    cpa_horizon_sec: float,
    cpa_danger_distance_m: float,
    cpa_min_relative_speed_mps: float,
    cpa_min_closing_speed_mps: float,
    cpa_min_course_score: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    risk_count = risk_arr.shape[0]
    if risk_count == 0 or other_arr.shape[0] == 0:
        empty_idx = np.zeros(risk_count, dtype=int)
        empty_bool = np.zeros(risk_count, dtype=bool)
        empty_float = np.full(risk_count, np.nan, dtype=np.float64)
        empty_score = np.zeros(risk_count, dtype=np.float64)
        return (
            empty_idx,
            empty_bool,
            empty_float,
            empty_float,
            empty_float,
            empty_float,
            empty_float,
            empty_float,
            empty_float,
            empty_float,
            empty_score,
            empty_bool,
        )

    rel_x = other_arr[:, 8][None, :] - risk_arr[:, 8:9]
    rel_y = other_arr[:, 9][None, :] - risk_arr[:, 9:10]
    rel_vx = other_arr[:, 10][None, :] - risk_arr[:, 10:11]
    rel_vy = other_arr[:, 11][None, :] - risk_arr[:, 11:12]
    dist = np.hypot(rel_x, rel_y)
    rel_speed = np.hypot(rel_vx, rel_vy)
    rel_speed_sq = rel_speed * rel_speed
    dot = rel_x * rel_vx + rel_y * rel_vy
    with np.errstate(divide="ignore", invalid="ignore"):
        t_cpa_raw = -dot / rel_speed_sq
    t_eval = np.clip(t_cpa_raw, 0.0, float(cpa_horizon_sec))
    cpa_x = rel_x + rel_vx * t_eval
    cpa_y = rel_y + rel_vy * t_eval
    cpa_dist = np.hypot(cpa_x, cpa_y)

    closing_speed = np.full_like(dist, np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        closing_speed = -dot / np.maximum(dist, 1e-6)
    bearing_closure = np.clip(closing_speed / np.maximum(rel_speed, 1e-6), 0.0, 1.0)

    risk_speed_matrix = np.hypot(risk_arr[:, 10:11], risk_arr[:, 11:12])
    other_speed_matrix = np.hypot(other_arr[:, 10][None, :], other_arr[:, 11][None, :])
    heading_dot_matrix = risk_arr[:, 10:11] * other_arr[:, 10][None, :] + risk_arr[:, 11:12] * other_arr[:, 11][None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        heading_cos_matrix = heading_dot_matrix / (risk_speed_matrix * other_speed_matrix)
    heading_cos_matrix = np.clip(heading_cos_matrix, -1.0, 1.0)
    crossing_angle_matrix = np.degrees(np.arccos(heading_cos_matrix))
    angle_known_matrix = (risk_speed_matrix >= 0.3) & (other_speed_matrix >= 0.3) & np.isfinite(crossing_angle_matrix)
    angle_score = np.where(angle_known_matrix, np.clip(crossing_angle_matrix / 45.0, 0.0, 1.0), 0.5)

    risk_radius = risk_arr[:, 19:20]
    other_radius = other_arr[:, 19][None, :]
    dynamic_radius = risk_radius + other_radius
    collision_radius = np.minimum(np.clip(dynamic_radius, 2.0, 4.2), float(cpa_danger_distance_m))
    collision_ttc = solve_collision_ttc(rel_x, rel_y, rel_vx, rel_vy, collision_radius, float(cpa_horizon_sec))
    rear_end_exception = (
        angle_known_matrix
        & (crossing_angle_matrix < 15.0)
        & (cpa_dist <= 1.0)
        & np.isfinite(collision_ttc)
        & (collision_ttc <= 1.2)
    )
    angle_support = (
        (angle_known_matrix & (crossing_angle_matrix >= 15.0))
        | rear_end_exception
        | ((~angle_known_matrix) & (cpa_dist <= 1.2) & (closing_speed >= float(cpa_min_closing_speed_mps)))
    )

    edge_pair = (risk_arr[:, 17:18] >= 1.0) | (other_arr[:, 17][None, :] >= 1.0)
    motion_extreme_pair = (risk_arr[:, 18:19] >= 1.0) | (other_arr[:, 18][None, :] >= 1.0)
    pair_quality = (
        valid_world.astype(float)
        * (1.0 - edge_pair.astype(float) * 0.35)
        * (1.0 - motion_extreme_pair.astype(float) * 0.75)
    )

    future_valid = (
        valid_world
        & np.isfinite(t_cpa_raw)
        & np.isfinite(cpa_dist)
        & np.isfinite(rel_speed)
        & (rel_speed >= float(cpa_min_relative_speed_mps))
        & (t_cpa_raw >= 0.10)
        & (t_cpa_raw <= float(cpa_horizon_sec))
        & (closing_speed >= float(cpa_min_closing_speed_mps))
        & (pair_quality >= 0.65)
        & angle_support
    )
    distance_score = np.clip((float(cpa_danger_distance_m) - cpa_dist) / max(float(cpa_danger_distance_m), 1e-6), 0.0, 1.0)
    time_score = np.clip((float(cpa_horizon_sec) - t_cpa_raw) / max(float(cpa_horizon_sec), 1e-6), 0.0, 1.0)
    speed_score = np.clip(rel_speed / 12.0, 0.0, 1.0)
    score = pair_quality * (0.40 * distance_score + 0.20 * time_score + 0.20 * bearing_closure + 0.10 * speed_score + 0.10 * angle_score)
    score = np.where(future_valid & (cpa_dist <= float(cpa_danger_distance_m)), score, 0.0)
    collision_flag = score >= float(cpa_min_course_score)

    score_for_select = np.where(np.isfinite(score), score, -np.inf)
    best_score_idx = score_for_select.argmax(axis=1)
    best_score = score_for_select[np.arange(risk_count), best_score_idx]
    candidate_dist = np.where(future_valid, cpa_dist, np.inf)
    best_dist_idx = candidate_dist.argmin(axis=1)
    has_future_candidate = np.isfinite(candidate_dist[np.arange(risk_count), best_dist_idx])
    nearest_idx = np.where(valid_world, dist, np.inf).argmin(axis=1)
    best_idx = np.where(best_score > 0.0, best_score_idx, np.where(has_future_candidate, best_dist_idx, nearest_idx))
    row_idx = np.arange(risk_count)

    selected_valid = valid_world[row_idx, best_idx] & np.isfinite(rel_speed[row_idx, best_idx])
    risk_speed = np.hypot(risk_arr[:, 10], risk_arr[:, 11])
    selected_other = other_arr[best_idx]
    other_speed = np.hypot(selected_other[:, 10], selected_other[:, 11])
    heading_dot = risk_arr[:, 10] * selected_other[:, 10] + risk_arr[:, 11] * selected_other[:, 11]
    with np.errstate(divide="ignore", invalid="ignore"):
        heading_cos = heading_dot / (risk_speed * other_speed)
    heading_cos = np.clip(heading_cos, -1.0, 1.0)
    crossing_angle = np.degrees(np.arccos(heading_cos))
    crossing_angle = np.where((risk_speed >= 0.3) & (other_speed >= 0.3) & np.isfinite(crossing_angle), crossing_angle, np.nan)

    return (
        best_idx.astype(int),
        selected_valid,
        t_cpa_raw[row_idx, best_idx],
        cpa_dist[row_idx, best_idx],
        collision_ttc[row_idx, best_idx],
        collision_radius[row_idx, best_idx],
        rel_speed[row_idx, best_idx],
        closing_speed[row_idx, best_idx],
        bearing_closure[row_idx, best_idx],
        crossing_angle,
        score[row_idx, best_idx],
        collision_flag[row_idx, best_idx],
    )


def solve_collision_ttc(
    rel_x: np.ndarray,
    rel_y: np.ndarray,
    rel_vx: np.ndarray,
    rel_vy: np.ndarray,
    radius: np.ndarray,
    horizon_sec: float,
) -> np.ndarray:
    a = rel_vx * rel_vx + rel_vy * rel_vy
    b = 2.0 * (rel_x * rel_vx + rel_y * rel_vy)
    c = rel_x * rel_x + rel_y * rel_y - radius * radius
    disc = b * b - 4.0 * a * c
    out = np.full_like(a, np.nan, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(c) & (a > 1e-8) & (disc >= 0.0)
    if not bool(valid.any()):
        return out
    sqrt_disc = np.zeros_like(a, dtype=np.float64)
    sqrt_disc[valid] = np.sqrt(disc[valid])
    t1 = np.full_like(a, np.nan, dtype=np.float64)
    t2 = np.full_like(a, np.nan, dtype=np.float64)
    t1[valid] = (-b[valid] - sqrt_disc[valid]) / (2.0 * a[valid])
    t2[valid] = (-b[valid] + sqrt_disc[valid]) / (2.0 * a[valid])
    intersects_future = valid & (t2 >= 0.0) & (t1 <= horizon_sec)
    entry = np.where(t1 >= 0.0, t1, 0.0)
    out[intersects_future & (entry <= horizon_sec)] = entry[intersects_future & (entry <= horizon_sec)]
    return out


def summarize_window_interaction_v2(
    *,
    row: Any,
    frame_pairs: pd.DataFrame | None,
    fps: float | None,
    risk_by_row: dict[int, dict[str, Any]],
    status: str,
    close_norm_threshold: float,
    world_close_meter_threshold: float,
    world_lateral_meter_threshold: float,
    world_closing_speed_threshold: float,
    min_collision_course_ratio: float,
    min_world_close_ratio: float,
    min_world_quality_ratio: float,
    max_world_jitter_ratio: float,
    min_common_frames: int,
) -> dict[str, Any]:
    scene_row_id = int(row.scene_row_id)
    video_id = str(row.video_id)
    scene_window_idx = int(row.scene_window_idx)
    risk_payload = risk_by_row.get(scene_row_id, {"risk_track_ids": [], "risk_case_keys": []})
    risk_track_ids = [int(value) for value in risk_payload["risk_track_ids"]]
    base = empty_window_row_v2(
        scene_row_id=scene_row_id,
        video_id=video_id,
        scene_window_idx=scene_window_idx,
        status=status,
        fps=fps,
        risk_track_ids=risk_track_ids,
        risk_case_keys=risk_payload["risk_case_keys"],
    )
    if frame_pairs is None or not risk_track_ids:
        return base

    start_sec = float(row.start_sec)
    end_sec = float(row.end_sec)
    risk_set = {int(value) for value in risk_track_ids}
    sub = frame_pairs[
        (frame_pairs["time_sec"] >= start_sec)
        & (frame_pairs["time_sec"] < end_sec)
        & (frame_pairs["risk_track_id"].isin(risk_set))
    ].copy()
    if sub.empty:
        return base

    frame_min = sub.groupby("frame", sort=False)["norm_center_distance"].min()
    nearest_idx = int(sub["norm_center_distance"].idxmin())
    nearest = sub.loc[nearest_idx]
    close_frames = sub.loc[sub["norm_center_distance"] <= close_norm_threshold, "frame"].nunique()
    overlap_frames = sub.loc[sub["iou"] > 0, "frame"].nunique()
    common_frames = sub["frame"].nunique()
    base["interaction_other_vehicle_count"] = int(sub["nearest_other_track_id"].nunique())
    base["interaction_common_frame_count"] = int(common_frames)
    base["interaction_close_frame_count"] = int(close_frames)
    base["interaction_overlap_frame_count"] = int(overlap_frames)
    base["interaction_pair_observation_count"] = int(len(sub))
    base["interaction_min_norm_center_distance"] = float(nearest["norm_center_distance"])
    base["interaction_median_frame_min_norm_center_distance"] = float(frame_min.median()) if len(frame_min) else np.nan
    base["interaction_min_center_distance_px"] = float(nearest["center_distance_px"])
    base["interaction_min_iou"] = float(sub["iou"].min())
    base["interaction_max_iou"] = float(sub["iou"].max())
    base["interaction_nearest_risk_track_id"] = str(int(nearest["risk_track_id"]))
    base["interaction_nearest_other_track_id"] = str(int(nearest["nearest_other_track_id"]))
    base["interaction_risk2_near_other_flag"] = int(
        common_frames >= min_common_frames and (close_frames >= 3 or overlap_frames >= 1)
    )

    world = sub[sub["world_pair_valid"].astype(float) >= 1.0].copy()
    if world.empty:
        return base
    world_nearest_idx = int(world["world_distance_m"].idxmin())
    world_nearest = world.loc[world_nearest_idx]
    world_common_frames = world["frame"].nunique()
    world_close_mask = world["world_distance_m"].astype(float) <= world_close_meter_threshold
    world_lateral_close_mask = world["world_lateral_m"].abs().astype(float) <= world_lateral_meter_threshold
    world_same_lane_mask = world["world_same_lane"].astype(float) >= 1.0
    world_lane_support_mask = world_same_lane_mask | world_lateral_close_mask | (world["world_lane_known_pair"].astype(float) < 1.0)
    world_close_supported = world_close_mask & world_lane_support_mask
    world_close_frames = world.loc[world_close_supported, "frame"].nunique()
    world_same_lane_frames = world.loc[world_same_lane_mask, "frame"].nunique()
    world_lateral_close_frames = world.loc[world_lateral_close_mask, "frame"].nunique()
    world_closing_frames = world.loc[
        world["world_closing_speed_mps"].astype(float) >= world_closing_speed_threshold,
        "frame",
    ].nunique()
    cpa = world[world["world_cpa_collision_course_score"].astype(float) > 0].copy()
    cpa_flag_frames = world.loc[world["world_cpa_collision_course_flag"].astype(float) >= 1.0, "frame"].nunique()
    world_quality_frames = world.loc[world["world_pair_quality"].astype(float) >= 0.65, "frame"].nunique()
    world_jitter_frames = world.loc[world["world_motion_extreme_pair"].astype(float) >= 1.0, "frame"].nunique()
    world_edge_low_quality_frames = world.loc[world["world_edge_low_quality_pair"].astype(float) >= 1.0, "frame"].nunique()
    world_other_moving_frames = world.loc[world["other_world_speed_kmh"].astype(float) >= 5.0, "frame"].nunique()
    world_other_stationary_frames = world.loc[world["other_world_speed_kmh"].astype(float) < 3.0, "frame"].nunique()
    denom = max(int(common_frames), 1)
    world_close_ratio = float(world_close_frames / denom)
    cpa_flag_ratio = float(cpa_flag_frames / denom)
    world_quality_ratio = float(world_quality_frames / denom)
    world_jitter_ratio = float(world_jitter_frames / denom)
    world_edge_ratio = float(world_edge_low_quality_frames / denom)
    world_other_moving_ratio = float(world_other_moving_frames / denom)
    world_other_stationary_ratio = float(world_other_stationary_frames / denom)
    lane_known_frames = world.loc[world["world_lane_known_pair"].astype(float) >= 1.0, "frame"].nunique()

    base["interaction_world_common_frame_count"] = int(world_common_frames)
    base["interaction_world_close_frame_count"] = int(world_close_frames)
    base["interaction_world_close_frame_ratio"] = world_close_ratio
    base["interaction_world_quality_frame_ratio"] = world_quality_ratio
    base["interaction_world_jitter_frame_ratio"] = world_jitter_ratio
    base["interaction_world_edge_low_quality_frame_ratio"] = world_edge_ratio
    base["interaction_world_other_moving_frame_ratio"] = world_other_moving_ratio
    base["interaction_world_other_stationary_frame_ratio"] = world_other_stationary_ratio
    base["interaction_world_same_lane_frame_count"] = int(world_same_lane_frames)
    base["interaction_world_lateral_close_frame_count"] = int(world_lateral_close_frames)
    base["interaction_world_lane_known_frame_count"] = int(lane_known_frames)
    base["interaction_world_closing_frame_count"] = int(world_closing_frames)
    base["interaction_world_cpa_frame_count"] = int(cpa_flag_frames)
    base["interaction_world_cpa_frame_ratio"] = cpa_flag_ratio
    base["interaction_world_pair_observation_count"] = int(len(world))
    base["interaction_world_min_distance_m"] = float(world_nearest["world_distance_m"])
    base["interaction_world_median_frame_min_distance_m"] = float(world.groupby("frame", sort=False)["world_distance_m"].min().median())
    base["interaction_world_min_abs_lateral_m"] = float(np.nanmin(np.abs(world["world_lateral_m"].to_numpy(dtype=float))))
    base["interaction_world_min_abs_longitudinal_m"] = float(np.nanmin(np.abs(world["world_longitudinal_m"].to_numpy(dtype=float))))
    nearest_risk_speed = world_nearest["risk_world_speed_kmh"]
    nearest_other_speed = world_nearest["other_world_speed_kmh"]
    base["interaction_world_nearest_risk_speed_kmh"] = float(nearest_risk_speed) if np.isfinite(nearest_risk_speed) else np.nan
    base["interaction_world_nearest_other_speed_kmh"] = float(nearest_other_speed) if np.isfinite(nearest_other_speed) else np.nan
    closing_values = world["world_closing_speed_mps"].to_numpy(dtype=float)
    closing_values = closing_values[np.isfinite(closing_values)]
    base["interaction_world_max_closing_speed_mps"] = float(np.max(closing_values)) if closing_values.size else np.nan
    finite_ttc = world["world_ttc_sec"].to_numpy(dtype=float)
    finite_ttc = finite_ttc[np.isfinite(finite_ttc) & (finite_ttc >= 0.0)]
    base["interaction_world_min_ttc_sec"] = float(np.min(finite_ttc)) if finite_ttc.size else np.nan
    if not cpa.empty:
        cpa_best_idx = int(cpa["world_cpa_collision_course_score"].astype(float).idxmax())
        cpa_best = cpa.loc[cpa_best_idx]
        cpa_distance_values = cpa["world_cpa_min_distance_m"].to_numpy(dtype=float)
        cpa_distance_values = cpa_distance_values[np.isfinite(cpa_distance_values)]
        cpa_ttc_values = cpa["world_cpa_collision_ttc_sec"].to_numpy(dtype=float)
        cpa_ttc_values = cpa_ttc_values[np.isfinite(cpa_ttc_values) & (cpa_ttc_values >= 0.0)]
        cpa_time_values = cpa["world_cpa_time_sec"].to_numpy(dtype=float)
        cpa_time_values = cpa_time_values[np.isfinite(cpa_time_values) & (cpa_time_values >= 0.0)]
        base["interaction_world_cpa_min_distance_m"] = float(np.min(cpa_distance_values)) if cpa_distance_values.size else np.nan
        base["interaction_world_cpa_best_time_sec"] = float(cpa_best["world_cpa_time_sec"])
        base["interaction_world_cpa_min_time_sec"] = float(np.min(cpa_time_values)) if cpa_time_values.size else np.nan
        base["interaction_world_cpa_min_collision_ttc_sec"] = float(np.min(cpa_ttc_values)) if cpa_ttc_values.size else np.nan
        base["interaction_world_cpa_best_collision_radius_m"] = float(cpa_best["world_cpa_collision_radius_m"])
        base["interaction_world_cpa_max_relative_speed_mps"] = float(np.nanmax(cpa["world_cpa_relative_speed_mps"].to_numpy(dtype=float)))
        base["interaction_world_cpa_max_closing_speed_mps"] = float(np.nanmax(cpa["world_cpa_closing_speed_mps"].to_numpy(dtype=float)))
        base["interaction_world_cpa_best_bearing_closure_score"] = float(cpa_best["world_cpa_bearing_closure_score"])
        base["interaction_world_cpa_best_crossing_angle_deg"] = float(cpa_best["world_cpa_crossing_angle_deg"])
        base["interaction_world_cpa_max_collision_course_score"] = float(cpa_best["world_cpa_collision_course_score"])
        base["interaction_world_cpa_nearest_risk_track_id"] = str(int(cpa_best["risk_track_id"]))
        base["interaction_world_cpa_nearest_other_track_id"] = str(int(cpa_best["world_cpa_other_track_id"]))
        cpa_risk_speed = cpa_best["cpa_risk_world_speed_kmh"]
        cpa_other_speed = cpa_best["cpa_other_world_speed_kmh"]
        base["interaction_world_cpa_nearest_risk_speed_kmh"] = float(cpa_risk_speed) if np.isfinite(cpa_risk_speed) else np.nan
        base["interaction_world_cpa_nearest_other_speed_kmh"] = float(cpa_other_speed) if np.isfinite(cpa_other_speed) else np.nan
    base["interaction_world_nearest_risk_track_id"] = str(int(world_nearest["risk_track_id"]))
    base["interaction_world_nearest_other_track_id"] = str(int(world_nearest["world_nearest_other_track_id"]))
    base["interaction_world_collision_course_flag"] = int(
        world_common_frames >= min_common_frames
        and cpa_flag_ratio >= min_collision_course_ratio
        and world_quality_ratio >= min_world_quality_ratio
        and world_jitter_ratio <= max_world_jitter_ratio
    )
    base["interaction_world_risk2_near_other_flag"] = int(
        world_common_frames >= min_common_frames
        and world_close_ratio >= min_world_close_ratio
        and world_quality_ratio >= min_world_quality_ratio
        and world_jitter_ratio <= max_world_jitter_ratio
    )
    return base


def empty_window_row_v2(
    *,
    scene_row_id: int,
    video_id: str,
    scene_window_idx: int,
    status: str,
    fps: float | None,
    risk_track_ids: list[int],
    risk_case_keys: list[str],
) -> dict[str, Any]:
    return {
        "scene_row_id": scene_row_id,
        "video_id": video_id,
        "scene_window_idx": scene_window_idx,
        "interaction_status": status,
        "interaction_fps": float(fps) if fps is not None else np.nan,
        "interaction_risk_track_ids": ",".join(str(value) for value in risk_track_ids),
        "interaction_risk_case_keys": ",".join(risk_case_keys),
        "interaction_risk_vehicle_count": len(risk_track_ids),
        "interaction_other_vehicle_count": 0,
        "interaction_common_frame_count": 0,
        "interaction_close_frame_count": 0,
        "interaction_overlap_frame_count": 0,
        "interaction_pair_observation_count": 0,
        "interaction_min_norm_center_distance": np.nan,
        "interaction_median_frame_min_norm_center_distance": np.nan,
        "interaction_min_center_distance_px": np.nan,
        "interaction_min_iou": 0.0,
        "interaction_max_iou": 0.0,
        "interaction_nearest_risk_track_id": "",
        "interaction_nearest_other_track_id": "",
        "interaction_risk2_near_other_flag": 0,
        "interaction_world_common_frame_count": 0,
        "interaction_world_close_frame_count": 0,
        "interaction_world_close_frame_ratio": 0.0,
        "interaction_world_quality_frame_ratio": 0.0,
        "interaction_world_jitter_frame_ratio": 0.0,
        "interaction_world_edge_low_quality_frame_ratio": 0.0,
        "interaction_world_other_moving_frame_ratio": 0.0,
        "interaction_world_other_stationary_frame_ratio": 0.0,
        "interaction_world_same_lane_frame_count": 0,
        "interaction_world_lateral_close_frame_count": 0,
        "interaction_world_lane_known_frame_count": 0,
        "interaction_world_closing_frame_count": 0,
        "interaction_world_cpa_frame_count": 0,
        "interaction_world_cpa_frame_ratio": 0.0,
        "interaction_world_pair_observation_count": 0,
        "interaction_world_min_distance_m": np.nan,
        "interaction_world_median_frame_min_distance_m": np.nan,
        "interaction_world_min_abs_lateral_m": np.nan,
        "interaction_world_min_abs_longitudinal_m": np.nan,
        "interaction_world_nearest_risk_speed_kmh": np.nan,
        "interaction_world_nearest_other_speed_kmh": np.nan,
        "interaction_world_max_closing_speed_mps": np.nan,
        "interaction_world_min_ttc_sec": np.nan,
        "interaction_world_cpa_min_distance_m": np.nan,
        "interaction_world_cpa_best_time_sec": np.nan,
        "interaction_world_cpa_min_time_sec": np.nan,
        "interaction_world_cpa_min_collision_ttc_sec": np.nan,
        "interaction_world_cpa_best_collision_radius_m": np.nan,
        "interaction_world_cpa_max_relative_speed_mps": np.nan,
        "interaction_world_cpa_max_closing_speed_mps": np.nan,
        "interaction_world_cpa_best_bearing_closure_score": np.nan,
        "interaction_world_cpa_best_crossing_angle_deg": np.nan,
        "interaction_world_cpa_max_collision_course_score": 0.0,
        "interaction_world_cpa_nearest_risk_speed_kmh": np.nan,
        "interaction_world_cpa_nearest_other_speed_kmh": np.nan,
        "interaction_world_nearest_risk_track_id": "",
        "interaction_world_nearest_other_track_id": "",
        "interaction_world_cpa_nearest_risk_track_id": "",
        "interaction_world_cpa_nearest_other_track_id": "",
        "interaction_world_collision_course_flag": 0,
        "interaction_world_risk2_near_other_flag": 0,
    }


def point_in_polygon_vectorized(x: np.ndarray, y: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    inside = np.zeros_like(x, dtype=bool)
    xj, yj = polygon[-1]
    for xi, yi in polygon:
        intersects = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi)
        inside ^= intersects
        xj, yj = xi, yi
    return inside


def local_lane_direction_world(
    x: np.ndarray,
    y: np.ndarray,
    dx_img: np.ndarray,
    dy_img: np.ndarray,
    H_img_to_world: np.ndarray,
) -> np.ndarray:
    p1 = project_points(np.stack([x, y], axis=1), H_img_to_world)
    p2 = project_points(np.stack([x + dx_img * 20.0, y + dy_img * 20.0], axis=1), H_img_to_world)
    direction = p2 - p1
    norm = np.linalg.norm(direction, axis=1, keepdims=True)
    return direction / np.where(norm < 1e-9, np.nan, norm)


def project_points(points_xy: np.ndarray, H_img_to_world: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate([points_xy, np.ones((len(points_xy), 1), dtype=np.float64)], axis=1)
    projected = homogeneous @ H_img_to_world.T
    denom = projected[:, 2:3]
    denom = np.where(np.abs(denom) < 1e-9, np.nan, denom)
    return projected[:, :2] / denom


def bbox_iou_arrays(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    left = np.maximum(a[:, 4], b[:, 4])
    top = np.maximum(a[:, 5], b[:, 5])
    right = np.minimum(a[:, 6], b[:, 6])
    bottom = np.minimum(a[:, 7], b[:, 7])
    inter_w = np.maximum(0.0, right - left)
    inter_h = np.maximum(0.0, bottom - top)
    inter = inter_w * inter_h
    area_a = np.maximum(0.0, a[:, 6] - a[:, 4]) * np.maximum(0.0, a[:, 7] - a[:, 5])
    area_b = np.maximum(0.0, b[:, 6] - b[:, 4]) * np.maximum(0.0, b[:, 7] - b[:, 5])
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0)


def _require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"{name} missing columns: {missing}")


if __name__ == "__main__":
    main()
