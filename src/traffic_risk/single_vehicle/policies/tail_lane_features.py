#!/usr/bin/env python3
"""Clean v48 tail-lane backtrack sidecar generation.

This module distills the accepted v48 upstream generation step:

    C4O window source + v28 lane sidecar + v28 frame-level lane candidate cache
    -> v48 tail-lane backtrack sidecar columns

It does not apply any risk rule.  v49/v50c consume these generated columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


KEY_COLUMNS = ["case_key", "video_id", "track_id", "ts_window_idx"]


@dataclass(frozen=True)
class CleanV48TailLaneSidecarColumns:
    tail_lane: str = "v92_v48_tail_lane"
    tail_lane_ratio: str = "v92_v48_tail_lane_ratio"
    tail_frame_count: str = "v92_v48_tail_frame_count"
    tail_appended_sequence: str = "v92_v48_tail_appended_sequence"
    tail_appended_backtrack: str = "v92_v48_tail_appended_backtrack"
    tail_lane_change_count_after_append: str = "v92_v48_tail_lane_change_count_after_append"
    tail_bbox_stable: str = "v92_v48_tail_bbox_stable"
    tail_lane_usable: str = "v92_v48_tail_lane_usable"
    tail_lane_stable_majority: str = "v92_v48_tail_lane_stable_majority"
    tail_transition_onset: str = "v92_v48_tail_transition_onset"
    tail_lane_backtrack_weaving_onset: str = "v92_v48_tail_lane_backtrack_weaving_onset"
    tail_lane_backtrack_weaving_onset_score: str = "v92_v48_tail_lane_backtrack_weaving_onset_score"


COLS = CleanV48TailLaneSidecarColumns()


HISTORICAL_V48_TAIL_LANE_SIDECAR_COLUMNS = {
    COLS.tail_lane: "v48_tail_lane",
    COLS.tail_lane_ratio: "v48_tail_lane_ratio",
    COLS.tail_frame_count: "v48_tail_frame_count",
    COLS.tail_appended_sequence: "v48_tail_appended_sequence",
    COLS.tail_appended_backtrack: "v48_tail_appended_backtrack",
    COLS.tail_lane_change_count_after_append: "v48_tail_lane_change_count_after_append",
    COLS.tail_bbox_stable: "v48_tail_bbox_stable",
    COLS.tail_lane_usable: "v48_tail_lane_usable",
    COLS.tail_lane_stable_majority: "v48_tail_lane_stable_majority",
    COLS.tail_transition_onset: "v48_tail_transition_onset",
    COLS.tail_lane_backtrack_weaving_onset: "v48_tail_lane_backtrack_weaving_onset",
    COLS.tail_lane_backtrack_weaving_onset_score: "v48_tail_lane_backtrack_weaving_onset_score",
}

TEXT_CLEAN_COLUMNS = {
    COLS.tail_lane,
    COLS.tail_appended_sequence,
}


def norm_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return text


def normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "case_key" in out.columns:
        out["case_key"] = out["case_key"].map(norm_id)
    for col in ["video_id", "track_id"]:
        if col in out.columns:
            out[col] = out[col].map(norm_id)
    if "ts_window_idx" in out.columns:
        out["ts_window_idx"] = pd.to_numeric(out["ts_window_idx"], errors="coerce").fillna(-1).astype(int)
    return out


def num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default).astype(float)


def compress(seq: list[str]) -> list[str]:
    out: list[str] = []
    for value in seq:
        if not value:
            continue
        if not out or out[-1] != value:
            out.append(value)
    return out


def has_backtrack(seq: list[str]) -> bool:
    comp = compress(seq)
    for i in range(len(comp) - 2):
        if comp[i] == comp[i + 2] and comp[i] != comp[i + 1]:
            return True
    return False


def append_tail_sequence(base_seq_text: str, tail_lane: str) -> list[str]:
    # The historical v48 artifact was built from CSV-loaded v28 columns, so an
    # empty v28 sequence appeared as the literal string "nan" before tail append.
    # Preserve that encoding for exact artifact equivalence; risk rules consume
    # the derived boolean/score columns, not this display string directly.
    text = str(base_seq_text or "")
    if text == "":
        text = "nan"
    base = [x for x in text.split(">") if x]
    if tail_lane:
        base.append(tail_lane)
    return compress(base)


def load_frame_cache(cache_dir: Path, video_id: str) -> pd.DataFrame:
    path = cache_dir / f"{video_id}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["track_id"] = df["track_id"].map(norm_id)
    return df.sort_values(["track_id", "timestamp_sec", "frame"], kind="mergesort")


def build_clean_v48_tail_lane_sidecar(
    c4o_source: pd.DataFrame,
    v28_sidecar: pd.DataFrame,
    *,
    frame_cache_dir: Path,
) -> pd.DataFrame:
    c4o = normalize_keys(c4o_source)
    v28 = normalize_keys(v28_sidecar)

    base_cols = [
        "case_key",
        "video_id",
        "track_id",
        "ts_window_idx",
        "start_sec",
        "end_sec",
        "split",
    ]
    missing = [c for c in base_cols if c not in c4o.columns]
    if missing:
        raise KeyError(f"c4o source missing columns: {missing}")
    out = c4o[base_cols].copy()

    v28_cols = [
        "v28_lane_sequence_smooth_6w",
        "v28_transition_type_6w",
        "v28_lane_valid_ratio",
        "v28_bbox_area_cv_raw",
        "v28_bbox_area_log_jump_max",
    ]
    out = out.merge(v28[KEY_COLUMNS + [c for c in v28_cols if c in v28.columns]], on=KEY_COLUMNS, how="left")

    out[COLS.tail_lane] = ""
    out[COLS.tail_lane_ratio] = 0.0
    out[COLS.tail_frame_count] = 0.0
    out[COLS.tail_appended_sequence] = ""

    for video_id, idx in out.groupby("video_id", sort=False).groups.items():
        frames = load_frame_cache(frame_cache_dir, str(video_id))
        if frames.empty:
            continue
        by_track = {tid: part for tid, part in frames.groupby("track_id", sort=False)}
        for i in idx:
            row = out.loc[i]
            part = by_track.get(str(row["track_id"]))
            if part is None or part.empty:
                continue
            start = float(row["start_sec"])
            end = float(row["end_sec"])
            mid = start + (end - start) * 0.5
            tail = part[(part["timestamp_sec"] >= mid) & (part["timestamp_sec"] < end)].copy()
            tail = tail[tail["cand_top1"].fillna("").astype(str).ne("")]
            if tail.empty:
                continue
            counts = tail["cand_top1"].astype(str).value_counts()
            lane = str(counts.index[0])
            ratio = float(counts.iloc[0] / len(tail))
            appended = append_tail_sequence(str(row.get("v28_lane_sequence_smooth_6w", "")), lane)
            out.at[i, COLS.tail_lane] = lane
            out.at[i, COLS.tail_lane_ratio] = ratio
            out.at[i, COLS.tail_frame_count] = float(len(tail))
            out.at[i, COLS.tail_appended_sequence] = ">".join(appended)

    appended_backtrack = out[COLS.tail_appended_sequence].map(lambda s: has_backtrack(str(s).split(">")))
    lane_changes_after_tail = out[COLS.tail_appended_sequence].map(
        lambda s: max(len(compress(str(s).split(">"))) - 1, 0)
    )
    bbox_stable = (num(out, "v28_bbox_area_cv_raw") < 0.26) & (num(out, "v28_bbox_area_log_jump_max") < 0.16)
    lane_usable = num(out, "v28_lane_valid_ratio", 1.0) >= 0.85
    tail_stable = (num(out, COLS.tail_lane_ratio) >= 0.90) & (num(out, COLS.tail_frame_count) >= 8)
    transition_onset = out.get("v28_transition_type_6w", pd.Series("", index=out.index)).fillna("").astype(str).eq(
        "single_lane_change_or_stable"
    )

    out[COLS.tail_appended_backtrack] = appended_backtrack.astype(int)
    out[COLS.tail_lane_change_count_after_append] = lane_changes_after_tail.astype(float)
    out[COLS.tail_bbox_stable] = bbox_stable.astype(int)
    out[COLS.tail_lane_usable] = lane_usable.astype(int)
    out[COLS.tail_lane_stable_majority] = tail_stable.astype(int)
    out[COLS.tail_transition_onset] = transition_onset.astype(int)
    out[COLS.tail_lane_backtrack_weaving_onset] = (
        bbox_stable
        & lane_usable
        & tail_stable
        & appended_backtrack
        & (lane_changes_after_tail >= 2)
        & transition_onset
    ).astype(int)
    out[COLS.tail_lane_backtrack_weaving_onset_score] = (
        num(out, COLS.tail_lane_ratio)
        * out[COLS.tail_appended_backtrack].astype(float)
        * out[COLS.tail_bbox_stable].astype(float)
        * out[COLS.tail_lane_usable].astype(float)
        * out[COLS.tail_transition_onset].astype(float)
    ).clip(0.0, 1.0)

    keep = KEY_COLUMNS + [
        COLS.tail_lane,
        COLS.tail_lane_ratio,
        COLS.tail_frame_count,
        COLS.tail_appended_sequence,
        COLS.tail_appended_backtrack,
        COLS.tail_lane_change_count_after_append,
        COLS.tail_bbox_stable,
        COLS.tail_lane_usable,
        COLS.tail_lane_stable_majority,
        COLS.tail_transition_onset,
        COLS.tail_lane_backtrack_weaving_onset,
        COLS.tail_lane_backtrack_weaving_onset_score,
    ]
    return out[keep]


def add_clean_v48_tail_lane_sidecar(
    frame: pd.DataFrame,
    *,
    c4o_source_path: Path,
    frame_cache_dir: Path,
    v28_sidecar_path: Path | None = None,
    v28_sidecar_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = normalize_keys(frame)
    if not c4o_source_path.exists():
        raise FileNotFoundError(c4o_source_path)
    if not frame_cache_dir.exists():
        raise FileNotFoundError(frame_cache_dir)
    if v28_sidecar_frame is None:
        if v28_sidecar_path is None:
            raise ValueError("v28_sidecar_path or v28_sidecar_frame is required")
        if not v28_sidecar_path.exists():
            raise FileNotFoundError(v28_sidecar_path)
        v28_sidecar = pd.read_csv(v28_sidecar_path, low_memory=False)
    else:
        v28_sidecar = v28_sidecar_frame

    c4o_source = pd.read_csv(c4o_source_path, low_memory=False)
    sidecar = normalize_keys(
        build_clean_v48_tail_lane_sidecar(
            c4o_source,
            v28_sidecar,
            frame_cache_dir=frame_cache_dir,
        )
    )
    if sidecar.duplicated(KEY_COLUMNS).any():
        dup = sidecar.loc[sidecar.duplicated(KEY_COLUMNS, keep=False), KEY_COLUMNS].head()
        raise ValueError(f"duplicated v48 sidecar keys: {dup.to_dict('records')}")
    return out.merge(sidecar, on=KEY_COLUMNS, how="left", validate="many_to_one")


def overwrite_historical_v48_tail_lane_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for clean_col, historical_col in HISTORICAL_V48_TAIL_LANE_SIDECAR_COLUMNS.items():
        if clean_col in out.columns:
            out[historical_col] = out[clean_col]
    return out
