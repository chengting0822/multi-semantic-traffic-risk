"""Prepare timestamped detector rows and formal per-track reference windows.

This is the path-independent extraction of the accepted online P1 stage.  The
large detector CSV is processed in chunks; only per-track time bounds remain
in memory while the timestamp source is written.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from traffic_risk.identifiers import canonical_id
from .car_tracks import car_track_ids, update_class_counts


WINDOW_SEC = 0.6667
STRIDE_SEC = 0.3333
DETECTOR_COLUMNS = {"frame", "track_id", "x1", "y1", "x2", "y2", "conf", "cls"}


@dataclass(frozen=True)
class TrackingPreparation:
    video_id: str
    fps: float
    timestamp_csv: Path
    reference_windows_csv: Path
    detector_rows: int
    track_count: int
    reference_window_count: int


def validate_video_id(value: Any) -> str:
    video_id = canonical_id(value)
    if not video_id:
        raise ValueError("video_id must not be empty")
    if video_id in {".", ".."} or "/" in video_id or "\\" in video_id:
        raise ValueError("video_id must be a portable label without path separators")
    return video_id


def infer_video_id(video_path: Path, explicit_video_id: str | None = None) -> str:
    return validate_video_id(explicit_video_id if explicit_video_id is not None else video_path.stem)


def read_video_fps(video_path: Path) -> float:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dependency checked by CLI doctor
        raise RuntimeError("OpenCV is required to read video metadata; alternatively pass --fps") from exc

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        capture.release()
    if fps <= 0:
        raise ValueError(f"video has invalid FPS: {video_path}")
    return fps


def _numeric_video_id(video_id: str) -> int | None:
    try:
        number = float(video_id)
    except ValueError:
        return None
    return int(number) if number.is_integer() else None


def _timestamp_chunks(
    detector_csv: Path,
    *,
    video_id: str,
    video_path: Path,
    fps: float,
    chunksize: int,
) -> Iterator[pd.DataFrame]:
    numeric_video_id = _numeric_video_id(video_id)
    for chunk in pd.read_csv(detector_csv, chunksize=chunksize, low_memory=False):
        missing = sorted(DETECTOR_COLUMNS - set(chunk.columns))
        if missing:
            raise ValueError(f"detector CSV is missing columns: {missing}")
        output = chunk.copy()
        output["frame"] = pd.to_numeric(output["frame"], errors="coerce").fillna(-1).astype(int)
        output["track_id"] = pd.to_numeric(output["track_id"], errors="coerce").fillna(-1).astype(int)
        output["timestamp_sec"] = output["frame"].astype(float) / fps
        output["source_type"] = "anomaly"
        output["source_video_name"] = video_path.name
        output["source_video_id"] = video_id
        output["numeric_video_id"] = numeric_video_id
        output["canonical_case_key"] = [
            f"{video_id}/{track_id}" for track_id in output["track_id"].to_numpy()
        ]
        output["fps"] = fps
        preferred = [
            "frame", "track_id", "x1", "y1", "x2", "y2", "conf", "cls",
            "tl_state", "tl_prob_green", "tl_prob_red", "timestamp_sec",
            "source_type", "source_video_name", "source_video_id",
            "numeric_video_id", "canonical_case_key", "fps",
        ]
        ordered = [column for column in preferred if column in output.columns]
        ordered.extend(column for column in output.columns if column not in ordered)
        yield output[ordered]


def _reference_windows(track_bounds: dict[int, tuple[float, float]], video_id: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for track_id in sorted(track_bounds):
        first_ts, last_ts = track_bounds[track_id]
        index = 0
        start = round(first_ts, 4)
        while start <= last_ts + 1e-9:
            rows.append(
                {
                    "case_key": f"{video_id}/{track_id}",
                    "source_type": "anomaly_csv",
                    "source_id": video_id,
                    "video_id": video_id,
                    "track_id": str(track_id),
                    "ts_window_idx": index,
                    "start_sec": start,
                    "end_sec": round(start + WINDOW_SEC, 4),
                }
            )
            index += 1
            start = round(first_ts + index * STRIDE_SEC, 4)
    return pd.DataFrame(
        rows,
        columns=[
            "case_key", "source_type", "source_id", "video_id", "track_id",
            "ts_window_idx", "start_sec", "end_sec",
        ],
    )


def prepare_tracking_csv(
    *,
    detector_csv: Path,
    video_path: Path,
    output_dir: Path,
    video_id: str | None = None,
    fps: float | None = None,
    chunksize: int = 250_000,
) -> TrackingPreparation:
    """Convert any detector CSV into the formal timestamp/window contract."""

    detector_csv = Path(detector_csv)
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    if not detector_csv.is_file():
        raise FileNotFoundError(detector_csv)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")

    resolved_video_id = infer_video_id(video_path, video_id)
    resolved_fps = float(fps) if fps is not None else read_video_fps(video_path)
    if resolved_fps <= 0:
        raise ValueError("fps must be positive")

    timestamp_dir = output_dir / "timestamps"
    timestamp_dir.mkdir(parents=True, exist_ok=True)
    timestamp_csv = timestamp_dir / f"{resolved_video_id}.csv"
    temporary_csv = timestamp_csv.with_suffix(".csv.part")
    reference_csv = output_dir / "reference_windows.csv"

    row_count = 0
    track_bounds: dict[int, tuple[float, float]] = {}
    class_counts: dict[int, Counter[int]] = {}
    wrote_header = False
    try:
        for chunk in _timestamp_chunks(
            detector_csv,
            video_id=resolved_video_id,
            video_path=video_path,
            fps=resolved_fps,
            chunksize=chunksize,
        ):
            chunk.to_csv(
                temporary_csv,
                mode="a",
                header=not wrote_header,
                index=False,
                quoting=csv.QUOTE_MINIMAL,
            )
            wrote_header = True
            row_count += len(chunk)
            update_class_counts(class_counts, chunk)
            valid = chunk.loc[chunk["track_id"].ge(0), ["track_id", "timestamp_sec"]]
            if not valid.empty:
                bounds = valid.groupby("track_id", sort=False)["timestamp_sec"].agg(["min", "max"])
                for track_id, row in bounds.iterrows():
                    current = track_bounds.get(int(track_id))
                    low, high = float(row["min"]), float(row["max"])
                    track_bounds[int(track_id)] = (
                        low if current is None else min(low, current[0]),
                        high if current is None else max(high, current[1]),
                    )
        if not wrote_header:
            raise ValueError(f"detector CSV has no rows: {detector_csv}")
        temporary_csv.replace(timestamp_csv)
    except Exception:
        temporary_csv.unlink(missing_ok=True)
        raise

    accepted_ids = car_track_ids(class_counts)
    windows = _reference_windows(
        {track_id: bounds for track_id, bounds in track_bounds.items() if track_id in accepted_ids},
        resolved_video_id,
    )
    windows.to_csv(reference_csv, index=False)
    return TrackingPreparation(
        video_id=resolved_video_id,
        fps=resolved_fps,
        timestamp_csv=timestamp_csv,
        reference_windows_csv=reference_csv,
        detector_rows=row_count,
        track_count=len(accepted_ids),
        reference_window_count=len(windows),
    )


__all__ = [
    "DETECTOR_COLUMNS", "STRIDE_SEC", "WINDOW_SEC", "TrackingPreparation",
    "infer_video_id", "prepare_tracking_csv", "read_video_fps", "validate_video_id",
]
