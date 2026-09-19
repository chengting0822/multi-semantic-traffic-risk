import argparse
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .utils import load_scene_config, point_in_polygon


REQUIRED_COLUMNS = ["frame", "track_id", "x1", "y1", "x2", "y2", "conf", "cls"]


@dataclass
class TrackCsvReport:
    rows: int
    frames_min: int
    frames_max: int
    n_unique_frames: int
    n_unique_tracks: int
    invalid_bbox_rows: int
    negative_coord_rows: int
    classes: List[int]
    conf_min: float
    conf_mean: float
    conf_max: float
    per_frame_count_quantiles: Dict[str, float]
    track_len_quantiles: Dict[str, float]
    tracks_len_le_5: int
    tracks_len_le_10: int
    tracks_len_le_20: int
    tracks_with_frame_gaps: int
    max_frame_gap_inside_track: int
    long_track_count: int
    long_track_angle_hist_12: List[int]
    long_track_angle_quantiles: Dict[str, float]
    roi_ground_inside_ratio: Optional[float]


def _quantile_dict(s: pd.Series, qs=None):
    if qs is None:
        qs = [0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
    qv = s.quantile(qs)
    return {str(k): float(v) for k, v in qv.items()}


def analyze_tracks(df: pd.DataFrame, roi_poly=None, long_track_len: int = 50) -> TrackCsvReport:
    inv_bbox = ((df["x2"] <= df["x1"]) | (df["y2"] <= df["y1"])).sum()
    neg_coord = (df[["x1", "y1", "x2", "y2"]] < 0).any(axis=1).sum()

    per_frame = df.groupby("frame").size()
    lens = df.groupby("track_id").size()

    # gap stats
    gap_tracks = 0
    max_gap = 0
    for _, g in df.groupby("track_id"):
        arr = np.sort(g["frame"].values.astype(np.int64))
        if arr.size < 2:
            continue
        d = np.diff(arr)
        if np.any(d > 1):
            gap_tracks += 1
            md = int(d.max())
            if md > max_gap:
                max_gap = md

    # direction stats on long tracks
    g = df.sort_values("frame").groupby("track_id")
    starts = g[["x1", "x2", "y2"]].first()
    ends = g[["x1", "x2", "y2"]].last()
    sx = (starts["x1"] + starts["x2"]) * 0.5
    sy = starts["y2"]
    ex = (ends["x1"] + ends["x2"]) * 0.5
    ey = ends["y2"]
    dx = ex - sx
    dy = ey - sy
    angles = np.degrees(np.arctan2(dy, dx))
    long_mask = lens >= int(long_track_len)
    ang_long = angles[long_mask]
    if len(ang_long) > 0:
        hist12 = np.histogram(ang_long, bins=12, range=(-180, 180))[0].astype(int).tolist()
        aq = np.quantile(ang_long, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
        aq_dict = {
            "0.0": float(aq[0]),
            "0.1": float(aq[1]),
            "0.25": float(aq[2]),
            "0.5": float(aq[3]),
            "0.75": float(aq[4]),
            "0.9": float(aq[5]),
            "1.0": float(aq[6]),
        }
    else:
        hist12 = [0] * 12
        aq_dict = {"0.0": 0.0, "0.1": 0.0, "0.25": 0.0, "0.5": 0.0, "0.75": 0.0, "0.9": 0.0, "1.0": 0.0}

    roi_inside_ratio = None
    if roi_poly is not None:
        cx = (df["x1"].values + df["x2"].values) * 0.5
        cy = df["y2"].values
        in_roi = np.fromiter((point_in_polygon(float(x), float(y), roi_poly) for x, y in zip(cx, cy)), dtype=bool)
        roi_inside_ratio = float(in_roi.mean()) if in_roi.size > 0 else 0.0

    return TrackCsvReport(
        rows=int(len(df)),
        frames_min=int(df["frame"].min()),
        frames_max=int(df["frame"].max()),
        n_unique_frames=int(df["frame"].nunique()),
        n_unique_tracks=int(df["track_id"].nunique()),
        invalid_bbox_rows=int(inv_bbox),
        negative_coord_rows=int(neg_coord),
        classes=sorted(int(x) for x in df["cls"].unique().tolist()),
        conf_min=float(df["conf"].min()),
        conf_mean=float(df["conf"].mean()),
        conf_max=float(df["conf"].max()),
        per_frame_count_quantiles=_quantile_dict(per_frame.astype(float)),
        track_len_quantiles=_quantile_dict(lens.astype(float)),
        tracks_len_le_5=int((lens <= 5).sum()),
        tracks_len_le_10=int((lens <= 10).sum()),
        tracks_len_le_20=int((lens <= 20).sum()),
        tracks_with_frame_gaps=int(gap_tracks),
        max_frame_gap_inside_track=int(max_gap),
        long_track_count=int(long_mask.sum()),
        long_track_angle_hist_12=hist12,
        long_track_angle_quantiles=aq_dict,
        roi_ground_inside_ratio=roi_inside_ratio,
    )


def print_report(report: TrackCsvReport):
    print("=== tracks.csv 檢查報告 ===")
    print(f"rows={report.rows}")
    print(
        f"frames=[{report.frames_min},{report.frames_max}] "
        f"unique_frames={report.n_unique_frames} unique_tracks={report.n_unique_tracks}"
    )
    print(
        f"invalid_bbox_rows={report.invalid_bbox_rows} negative_coord_rows={report.negative_coord_rows} "
        f"classes={report.classes}"
    )
    print(f"conf(min/mean/max)=({report.conf_min:.4f}/{report.conf_mean:.4f}/{report.conf_max:.4f})")
    print(f"track_len_quantiles={report.track_len_quantiles}")
    print(
        f"short_tracks <=5:{report.tracks_len_le_5} <=10:{report.tracks_len_le_10} "
        f"<=20:{report.tracks_len_le_20}"
    )
    print(
        f"tracks_with_frame_gaps={report.tracks_with_frame_gaps} "
        f"max_gap_inside_track={report.max_frame_gap_inside_track}"
    )
    print(f"long_track_count={report.long_track_count}")
    print(f"long_track_angle_hist_12={report.long_track_angle_hist_12}")
    if report.roi_ground_inside_ratio is not None:
        print(f"roi_ground_inside_ratio={report.roi_ground_inside_ratio:.4f}")

    warnings = get_report_warnings(report)

    if warnings:
        print("warnings:")
        for w in warnings:
            print(f"- {w}")
    else:
        print("warnings: none")


def get_report_warnings(report: TrackCsvReport) -> List[str]:
    warnings = []
    if report.invalid_bbox_rows > 0:
        warnings.append("存在無效 bbox (x2<=x1 或 y2<=y1)")
    if report.tracks_len_le_10 / max(1, report.n_unique_tracks) > 0.4:
        warnings.append("短軌跡比例偏高（可能追蹤碎片化）")
    if report.tracks_with_frame_gaps / max(1, report.n_unique_tracks) > 0.2:
        warnings.append("斷軌比例偏高（track_id 不連續）")
    if report.roi_ground_inside_ratio is not None and report.roi_ground_inside_ratio < 0.95:
        warnings.append("很多接地點落在 ROI 外，請檢查 ROI 或過濾設定")
    return warnings


def main():
    parser = argparse.ArgumentParser(description="check quality of tracks.csv")
    parser.add_argument("--csv", default="tracks.csv")
    parser.add_argument("--scene-config", default="")
    parser.add_argument("--long-track-len", type=int, default=50)
    parser.add_argument("--save-json", default="")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少必要欄位: {missing}")

    roi_poly = None
    if args.scene_config:
        cfg = load_scene_config(args.scene_config)
        roi_poly = cfg["roi_polygon"]

    report = analyze_tracks(df, roi_poly=roi_poly, long_track_len=args.long_track_len)
    print_report(report)

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, ensure_ascii=False, indent=2)
        print(f"saved_json={args.save_json}")


if __name__ == "__main__":
    main()
