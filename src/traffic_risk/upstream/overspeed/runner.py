"""Path-independent runner around the accepted overspeed formula module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .feature_contract import JOIN_KEY_FIELDNAMES, MODULE_V0_FIELD_ORDER, MODULE_V0_SIDECAR_FIELD_ORDER
from .formulas import (
    aggregate_window,
    build_reference_case_index,
    extract_case_key,
    extract_fps_value,
    extract_video_id,
    make_failure_row,
    make_failure_sidecar_row,
    normalize_reference_windows,
    validate_input_columns,
)
from .ipm_speed_utils import (
    GUI_HOMOGRAPHY_KEY,
    compute_track_speed_profile,
    ensure_timestamp_sec,
    load_homography_img_to_world,
)


@dataclass(frozen=True)
class OverspeedFeatures:
    windows: pd.DataFrame
    sidecar: pd.DataFrame


def _ordered_frame(rows: list[dict], columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=[*JOIN_KEY_FIELDNAMES, *columns])


def build_overspeed_features(
    *,
    timestamp_csv: Path,
    reference_windows_csv: Path,
    ipm_json: Path,
) -> OverspeedFeatures:
    """Build window-level overspeed evidence with the research-time formulas."""

    timestamp_csv = Path(timestamp_csv)
    reference_windows_csv = Path(reference_windows_csv)
    ipm_json = Path(ipm_json)
    for path in (timestamp_csv, reference_windows_csv, ipm_json):
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

    fps, _metadata = extract_fps_value(raw, timestamp_csv, video_id, "")
    raw, timestamp_column = ensure_timestamp_sec(raw, fps)
    homography, _failure = load_homography_img_to_world(ipm_json)
    ipm_loaded = homography is not None

    module_rows: list[dict] = []
    sidecar_rows: list[dict] = []
    processed: set[str] = set()
    for track_id_value, track in raw.groupby("track_id", sort=True):
        track_id = int(track_id_value)
        case_key = extract_case_key(track, video_id, track_id)
        windows = reference_index.get(case_key)
        if windows is None:
            continue
        processed.add(case_key)
        evidence, stats = compute_track_speed_profile(
            track,
            homography,
            video_id=video_id,
            track_id=track_id,
            ipm_source_path=str(ipm_json),
            ipm_source_type=ipm_json.name,
            homography_key=GUI_HOMOGRAPHY_KEY,
            timestamp_column_used=timestamp_column,
        )
        for _, window in windows.iterrows():
            module_row, sidecar_row = aggregate_window(
                window,
                evidence,
                stats,
                source_csv=str(timestamp_csv),
                timestamp_column_used=timestamp_column,
                fps_value=fps,
                ipm_loaded=ipm_loaded,
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
                    failure_reason=reason,
                )
            )

    return OverspeedFeatures(
        windows=_ordered_frame(module_rows, MODULE_V0_FIELD_ORDER),
        sidecar=_ordered_frame(sidecar_rows, MODULE_V0_SIDECAR_FIELD_ORDER),
    )


def write_overspeed_features(features: OverspeedFeatures, output_dir: Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    windows_path = output_dir / "overspeed_features.csv"
    sidecar_path = output_dir / "overspeed_sidecar.csv"
    features.windows.to_csv(windows_path, index=False)
    features.sidecar.to_csv(sidecar_path, index=False)
    return windows_path, sidecar_path


__all__ = ["OverspeedFeatures", "build_overspeed_features", "write_overspeed_features"]
