"""Small, testable annotation data model compatible with the research JSON."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WINDOW_SEC = 0.6667
STRIDE_SEC = 0.3333


@dataclass(frozen=True)
class Segment:
    start_frame: int
    end_frame: int
    risk: int
    fps: float

    def __post_init__(self) -> None:
        if self.start_frame < 0 or self.end_frame < self.start_frame:
            raise ValueError("segment frame range is invalid")
        if self.risk not in {0, 1, 2, 3}:
            raise ValueError("risk must be 0, 1, 2, or 3")
        if self.fps <= 0:
            raise ValueError("fps must be positive")

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "frame_start": self.start_frame,
                "frame_end": self.end_frame,
                "timestamp_start_sec": round(self.start_frame / self.fps, 6),
                "timestamp_end_sec": round((self.end_frame + 1) / self.fps, 6),
                "duration_sec": round((self.end_frame - self.start_frame + 1) / self.fps, 6),
                "window_policy_id": "window06667_stride03333_s3_fixed20",
                "window_sec": WINDOW_SEC,
                "stride_sec": STRIDE_SEC,
                "timestamp_mode": True,
                "resample_policy": "uniform_timestamp_20",
                "sample_count": 20,
                "sample_times_policy": "linspace_endpoint_false",
                "window_overlap_policy": "max_overlap_ratio",
            }
        )
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any], default_fps: float) -> "Segment":
        return cls(
            start_frame=int(payload.get("start_frame", payload.get("frame_start", 0))),
            end_frame=int(payload.get("end_frame", payload.get("frame_end", 0))),
            risk=int(payload["risk"]),
            fps=float(payload.get("fps", default_fps)),
        )


class AnnotationDocument:
    def __init__(self, video_id: str, fps: float) -> None:
        self.video_id = str(video_id)
        self.fps = float(fps)
        self.scene_segments: list[Segment] = []
        self.track_segments: dict[str, list[Segment]] = {}

    def add(self, segment: Segment, track_id: str | None = None) -> None:
        target = self.scene_segments if track_id is None else self.track_segments.setdefault(str(track_id), [])
        target.append(segment)
        target.sort(key=lambda item: (item.start_frame, item.end_frame))

    def remove(self, index: int, track_id: str | None = None) -> None:
        target = self.scene_segments if track_id is None else self.track_segments.get(str(track_id), [])
        del target[index]

    def segments(self, track_id: str | None = None) -> list[Segment]:
        return self.scene_segments if track_id is None else self.track_segments.get(str(track_id), [])

    def to_payload(self) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        tracks = {
            track_id: {"segments": [segment.to_payload() for segment in segments]}
            for track_id, segments in sorted(self.track_segments.items())
        }
        return {
            "schema_version": "tcn_scene_annotation.timestamp_fixed20.compact.v1",
            "version": 4,
            "updated_at": now,
            "export_mode": "per_video",
            "annotation_source": "traffic-risk annotate",
            "window_policy": {
                "window_policy_id": "window06667_stride03333_s3_fixed20",
                "window_sec": WINDOW_SEC,
                "stride_sec": STRIDE_SEC,
                "timestamp_mode": True,
                "resample_policy": "uniform_timestamp_20",
                "sample_count": 20,
                "sample_times_policy": "linspace_endpoint_false",
                "window_overlap_policy": "max_overlap_ratio",
            },
            "annotations": {
                self.video_id: {
                    "tracks": tracks,
                    "scene_segments": [segment.to_payload() for segment in self.scene_segments],
                }
            },
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_payload(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path, video_id: str, fps: float) -> "AnnotationDocument":
        document = cls(video_id, fps)
        if not path.exists():
            return document
        payload = json.loads(path.read_text(encoding="utf-8"))
        video = payload.get("annotations", {}).get(str(video_id), {})
        document.scene_segments = [
            Segment.from_payload(segment, fps) for segment in video.get("scene_segments", [])
        ]
        for track_id, track in video.get("tracks", {}).items():
            document.track_segments[str(track_id)] = [
                Segment.from_payload(segment, fps) for segment in track.get("segments", [])
            ]
        return document
