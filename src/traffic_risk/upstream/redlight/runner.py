"""Path-independent runner around the accepted red-light formula module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .feature_contract import JOIN_KEY_FIELDNAMES, MODULE_V0_FIELD_ORDER, MODULE_V0_SIDECAR_FIELD_ORDER
from .formulas import (
    aggregate_window,
    build_reference_case_index,
    ensure_timestamp_sec,
    extract_case_key,
    extract_fps_value,
    extract_video_id,
    make_failure_row,
    make_failure_sidecar_row,
    normalize_reference_windows,
    validate_input_columns,
)
from .geometry import compute_track_stopline_profile, load_stop_lines
from .traffic_light import build_frame_traffic_light_table


@dataclass(frozen=True)
class RedlightFeatures:
    windows: pd.DataFrame
    sidecar: pd.DataFrame


def _ordered_frame(rows: list[dict], columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=[*JOIN_KEY_FIELDNAMES, *columns])


def build_redlight_features(
    *,
    timestamp_csv: Path,
    reference_windows_csv: Path,
    stop_lines_json: Path,
) -> RedlightFeatures:
    """Build window-level red-light evidence with the research-time formulas."""

    timestamp_csv = Path(timestamp_csv)
    reference_windows_csv = Path(reference_windows_csv)
    stop_lines_json = Path(stop_lines_json)
    for path in (timestamp_csv, reference_windows_csv, stop_lines_json):
        if not path.is_file():
            raise FileNotFoundError(path)

    raw = pd.read_csv(timestamp_csv, low_memory=False)
    validate_input_columns(raw)
    references = normalize_reference_windows(pd.read_csv(reference_windows_csv, low_memory=False))
    references = references.loc[references["source_type"].eq("anomaly_csv")].copy()

    video_id = extract_video_id(raw, timestamp_csv)
    references = references.loc[references["video_id"].eq(video_id)].copy()
    if references.empty:
        raise ValueError(f"reference windows contain no rows for video_id={video_id!r}")
    reference_index = build_reference_case_index(references).get(video_id, {})

    fps = extract_fps_value(raw)
    raw, timestamp_column = ensure_timestamp_sec(raw)
    stop_lines, stopline_failure = load_stop_lines(stop_lines_json)
    if not stop_lines:
        raise ValueError(stopline_failure)
    traffic_light_table, _metadata = build_frame_traffic_light_table(raw)

    module_rows: list[dict] = []
    sidecar_rows: list[dict] = []
    processed: set[str] = set()
    for track_id_value, track in raw.groupby("track_id", sort=True):
        track_id = int(float(track_id_value))
        if track_id < 0:
            continue
        case_key = extract_case_key(track, video_id, track_id)
        windows = reference_index.get(case_key)
        if windows is None:
            continue
        processed.add(case_key)
        evidence, stats = compute_track_stopline_profile(track, stop_lines)
        for _, window in windows.iterrows():
            module_row, sidecar_row = aggregate_window(
                window,
                evidence,
                stats,
                traffic_light_table,
                source_csv=str(timestamp_csv),
                timestamp_column_used=timestamp_column,
                fps_value=fps,
                stopline_geometry_source=str(stop_lines_json),
            )
            module_rows.append(module_row)
            sidecar_rows.append(sidecar_row)

    for case_key, windows in reference_index.items():
        if case_key in processed:
            continue
        for _, window in windows.iterrows():
            reason = "case_not_found_in_timestamp_csv"
            module_rows.append(make_failure_row(window, reason))
            sidecar_rows.append(
                make_failure_sidecar_row(
                    window,
                    source_csv=str(timestamp_csv),
                    timestamp_column_used=timestamp_column,
                    fps_value=fps,
                    stopline_geometry_source=str(stop_lines_json),
                    failure_reason=reason,
                )
            )

    return RedlightFeatures(
        windows=_ordered_frame(module_rows, MODULE_V0_FIELD_ORDER),
        sidecar=_ordered_frame(sidecar_rows, MODULE_V0_SIDECAR_FIELD_ORDER),
    )


def write_redlight_features(features: RedlightFeatures, output_dir: Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    windows_path = output_dir / "redlight_features.csv"
    sidecar_path = output_dir / "redlight_sidecar.csv"
    features.windows.to_csv(windows_path, index=False)
    features.sidecar.to_csv(sidecar_path, index=False)
    return windows_path, sidecar_path


__all__ = ["RedlightFeatures", "build_redlight_features", "write_redlight_features"]
