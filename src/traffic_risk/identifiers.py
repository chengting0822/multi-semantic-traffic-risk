"""Stable identifiers shared by the public pipeline.

Video and track IDs are labels, not measurements.  Keeping them as canonical
strings supports both the original numeric cases and arbitrary user videos.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def canonical_id(value: Any) -> str:
    """Normalize CSV number formatting while preserving textual identifiers."""

    if pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else text


def normalize_window_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize identifiers used to join per-vehicle semantic windows."""

    out = frame.copy()
    for column in ("split", "case_key"):
        if column in out.columns:
            out[column] = out[column].astype(str).str.strip()
    for column in ("video_id", "track_id"):
        if column in out.columns:
            out[column] = out[column].map(canonical_id)
    if "ts_window_idx" in out.columns:
        out["ts_window_idx"] = pd.to_numeric(
            out["ts_window_idx"], errors="coerce"
        ).fillna(-1).astype(int)
    return out


__all__ = ["canonical_id", "normalize_window_keys"]
