"""Choose car trajectories by track identity, not by each noisy frame class."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

CAR_CLASS_ID = 2


def update_class_counts(counts: dict[int, Counter[int]], frame: pd.DataFrame) -> None:
    """Accumulate only small per-track class counts from a CSV chunk."""

    if "track_id" not in frame or "cls" not in frame:
        raise ValueError("tracking table requires track_id and cls")
    track_ids = pd.to_numeric(frame["track_id"], errors="coerce").fillna(-1).astype(int)
    classes = pd.to_numeric(frame["cls"], errors="coerce").fillna(-1).astype(int)
    pairs = pd.DataFrame({"track_id": track_ids, "cls": classes})
    pairs = pairs.loc[pairs["track_id"].ge(0)]
    for (track_id, class_id), count in pairs.groupby(["track_id", "cls"], sort=False).size().items():
        counts.setdefault(int(track_id), Counter())[int(class_id)] += int(count)


def car_track_ids(counts: Mapping[int, Counter[int]]) -> set[int]:
    """Accept a track only when more than half its detections are cars.

    All frames of an accepted ID are kept, including temporary motorcycle,
    truck, or bus labels.  Ambiguous ties are excluded rather than treating a
    non-car track with a few false car detections as a car.
    """

    return {
        int(track_id)
        for track_id, class_counts in counts.items()
        if class_counts.get(CAR_CLASS_ID, 0) > sum(class_counts.values()) / 2
    }


def scan_car_track_ids(path: str | Path, *, chunksize: int = 250_000) -> set[int]:
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    counts: dict[int, Counter[int]] = defaultdict(Counter)
    for chunk in pd.read_csv(path, usecols=["track_id", "cls"], chunksize=chunksize, low_memory=False):
        update_class_counts(counts, chunk)
    return car_track_ids(counts)


__all__ = ["CAR_CLASS_ID", "car_track_ids", "scan_car_track_ids", "update_class_counts"]
