from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SceneSegment:
    video_id: str
    segment_idx: int
    start_sec: float
    end_sec: float
    risk: int
    source: str
    first_overlapping_window_id: int | None = None
    last_overlapping_window_id: int | None = None
    primary_window_id: int | None = None


def load_scene_segments(annotation_dir: str | Path) -> tuple[dict[str, list[SceneSegment]], dict[str, Any]]:
    root = Path(annotation_dir)
    segments_by_video: dict[str, list[SceneSegment]] = {}
    missing_videos: list[str] = []
    invalid_segments: list[dict[str, Any]] = []
    json_files = sorted(root.glob("*.json"))

    for path in json_files:
        video_id = path.stem
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        node = data.get("annotations", {}).get(video_id, {})
        raw_segments = node.get("scene_segments", []) or []
        parsed: list[SceneSegment] = []
        for idx, raw in enumerate(raw_segments):
            try:
                start = float(raw["timestamp_start_sec"])
                end = float(raw["timestamp_end_sec"])
                risk = int(raw["risk"])
            except (KeyError, TypeError, ValueError) as exc:
                invalid_segments.append(
                    {
                        "video_id": video_id,
                        "segment_idx": idx,
                        "error": str(exc),
                        "raw": raw,
                    }
                )
                continue
            if end <= start:
                invalid_segments.append(
                    {
                        "video_id": video_id,
                        "segment_idx": idx,
                        "error": "end_sec <= start_sec",
                        "raw": raw,
                    }
                )
                continue
            parsed.append(
                SceneSegment(
                    video_id=video_id,
                    segment_idx=idx,
                    start_sec=start,
                    end_sec=end,
                    risk=risk,
                    source=str(path),
                    first_overlapping_window_id=_optional_int(raw.get("first_overlapping_window_id")),
                    last_overlapping_window_id=_optional_int(raw.get("last_overlapping_window_id")),
                    primary_window_id=_optional_int(raw.get("primary_window_id")),
                )
            )
        if parsed:
            segments_by_video[video_id] = parsed
        else:
            missing_videos.append(video_id)

    manifest = {
        "annotation_dir": str(root),
        "annotation_file_count": len(json_files),
        "videos_with_scene_segments": len(segments_by_video),
        "videos_without_scene_segments": missing_videos,
        "invalid_segment_count": len(invalid_segments),
        "invalid_segments": invalid_segments[:50],
    }
    return segments_by_video, manifest


def label_window_by_index(
    segments: list[SceneSegment] | None,
    scene_window_idx: int,
    start_sec: float,
    end_sec: float,
) -> dict[str, Any]:
    if not segments:
        return {
            "scene_label_available": 0,
            "scene_y_original": -1,
            "scene_y_schemaC": -1,
            "scene_y_binary": -1,
            "scene_label_overlap_sec": 0.0,
            "scene_label_segment_idx": -1,
            "scene_label_source": "no_scene_segments",
            "scene_label_strategy": "first_last",
            "scene_label_primary_match": 0,
        }

    range_matches: list[tuple[int, int, float, SceneSegment]] = []
    for seg in segments:
        if seg.first_overlapping_window_id is None or seg.last_overlapping_window_id is None:
            continue
        if seg.first_overlapping_window_id <= scene_window_idx <= seg.last_overlapping_window_id:
            primary_match = int(seg.primary_window_id == scene_window_idx)
            overlap = max(0.0, min(end_sec, seg.end_sec) - max(start_sec, seg.start_sec))
            range_matches.append((seg.risk, primary_match, overlap, seg))

    if range_matches:
        # If annotation ranges overlap at boundaries, use the more severe label.
        # Primary-window matches and overlap duration are tie-breakers only.
        risk, primary_match, overlap, seg = max(range_matches, key=lambda item: (item[0], item[1], item[2]))
        return {
            "scene_label_available": 1,
            "scene_y_original": int(risk),
            "scene_y_schemaC": scene_schema_c(risk),
            "scene_y_binary": int(risk > 0),
            "scene_label_overlap_sec": float(overlap),
            "scene_label_segment_idx": int(seg.segment_idx),
            "scene_label_source": seg.source,
            "scene_label_strategy": "first_last",
            "scene_label_primary_match": int(primary_match),
        }

    fallback = label_window(segments, start_sec, end_sec)
    fallback["scene_y_binary"] = int(fallback["scene_y_original"] > 0) if fallback["scene_label_available"] else -1
    fallback["scene_label_strategy"] = "overlap_fallback"
    fallback["scene_label_primary_match"] = 0
    return fallback


def label_window(
    segments: list[SceneSegment] | None,
    start_sec: float,
    end_sec: float,
) -> dict[str, Any]:
    if not segments:
        return {
            "scene_label_available": 0,
            "scene_y_original": -1,
            "scene_y_schemaC": -1,
            "scene_label_overlap_sec": 0.0,
            "scene_label_segment_idx": -1,
            "scene_label_source": "no_scene_segments",
        }

    best: tuple[float, int, SceneSegment] | None = None
    for seg in segments:
        overlap = max(0.0, min(end_sec, seg.end_sec) - max(start_sec, seg.start_sec))
        if overlap <= 0:
            continue
        key = (overlap, seg.risk)
        if best is None or key > (best[0], best[1]):
            best = (overlap, seg.risk, seg)

    if best is None:
        return {
            "scene_label_available": 0,
            "scene_y_original": -1,
            "scene_y_schemaC": -1,
            "scene_label_overlap_sec": 0.0,
            "scene_label_segment_idx": -1,
            "scene_label_source": "no_overlap",
        }

    overlap, _, seg = best
    return {
        "scene_label_available": 1,
        "scene_y_original": int(seg.risk),
        "scene_y_schemaC": scene_schema_c(seg.risk),
        "scene_label_overlap_sec": float(overlap),
        "scene_label_segment_idx": int(seg.segment_idx),
        "scene_label_source": seg.source,
    }


def scene_schema_c(original_risk: int) -> int:
    if original_risk <= 1:
        return 0
    if original_risk == 2:
        return 1
    return 2


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if str(value).strip() == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None
