import json
import logging
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np


def setup_logging(level: str = "INFO") -> None:
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def resolve_inference_device(
    preferred_device: int = 0,
    force_gpu_if_available: bool = True,
    logger: logging.Logger = None,
) -> Tuple[object, bool]:
    """
    Return (device_for_ultralytics, use_half_precision).
    If force_gpu_if_available=True and CUDA is available, always use GPU.
    """
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        n_gpu = int(torch.cuda.device_count()) if cuda_available else 0
    except Exception as e:
        if logger is not None:
            logger.warning("CUDA probe failed (%s), fallback to CPU", e)
        return "cpu", False

    if cuda_available and n_gpu > 0:
        if preferred_device < 0 and force_gpu_if_available:
            use_device = 0
        elif 0 <= preferred_device < n_gpu:
            use_device = preferred_device
        else:
            use_device = 0

        try:
            torch.cuda.set_device(use_device)
            if logger is not None:
                logger.info(
                    "CUDA detected (%s GPU). Force using GPU[%s]: %s",
                    n_gpu,
                    use_device,
                    torch.cuda.get_device_name(use_device),
                )
            return use_device, True
        except Exception as e:
            if logger is not None:
                logger.warning("set_device failed (%s), fallback to CPU", e)
            return "cpu", False

    if logger is not None:
        logger.warning("No CUDA GPU detected, using CPU")
    return "cpu", False


def load_scene_config(scene_config_path: str) -> Dict:
    with open(scene_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "roi_polygon" not in cfg:
        raise ValueError("scene_config.json 必須包含 roi_polygon")
    cfg["roi_polygon"] = [(float(x), float(y)) for x, y in cfg["roi_polygon"]]
    return cfg


def polygon_bbox(poly: Sequence[Tuple[float, float]]) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def point_in_polygon(x: float, y: float, poly: Sequence[Tuple[float, float]]) -> bool:
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= 0


def build_crop_mask(
    roi_poly: Sequence[Tuple[float, float]],
    xmin: int,
    ymin: int,
    crop_w: int,
    crop_h: int,
) -> np.ndarray:
    pts = np.asarray(roi_poly, dtype=np.float32).copy()
    pts[:, 0] -= float(xmin)
    pts[:, 1] -= float(ymin)
    pts[:, 0] = np.clip(pts[:, 0], 0, max(0, crop_w - 1))
    pts[:, 1] = np.clip(pts[:, 1], 0, max(0, crop_h - 1))
    pts = pts.astype(np.int32).reshape(-1, 1, 2)

    mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def apply_crop_mask(crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cv2.bitwise_and(crop, crop, mask=mask)


def resample_polyline(points: np.ndarray, n: int) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((n, 2), dtype=np.float32)
    if len(points) == 1:
        return np.repeat(points[:1], n, axis=0).astype(np.float32)

    diffs = np.diff(points, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    total_len = float(np.sum(seg_lens))
    if total_len < 1e-12:
        return np.repeat(points[:1], n, axis=0).astype(np.float32)

    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    target = np.linspace(0.0, total_len, n)

    out = np.zeros((n, 2), dtype=np.float32)
    j = 0
    for i, t in enumerate(target):
        while j < len(cum) - 2 and cum[j + 1] < t:
            j += 1
        t0, t1 = cum[j], cum[j + 1]
        p0, p1 = points[j], points[j + 1]
        alpha = (t - t0) / (t1 - t0 + 1e-12)
        out[i] = (1.0 - alpha) * p0 + alpha * p1
    return out


def drop_redundant_points(points: np.ndarray, min_dist: float) -> np.ndarray:
    if points.shape[0] <= 1 or min_dist <= 0:
        return points

    keep_idx = [0]
    last = points[0]
    for i in range(1, points.shape[0]):
        if float(np.linalg.norm(points[i] - last)) >= min_dist:
            keep_idx.append(i)
            last = points[i]

    if keep_idx[-1] != points.shape[0] - 1:
        keep_idx.append(points.shape[0] - 1)
    return points[np.asarray(keep_idx, dtype=np.int64)]


def trajectory_path_length(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def max_turn_angle_deg(points: np.ndarray) -> float:
    if points.shape[0] < 3:
        return 0.0

    max_deg = 0.0
    for i in range(1, points.shape[0] - 1):
        v1 = points[i] - points[i - 1]
        v2 = points[i + 1] - points[i]
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1e-12 or n2 < 1e-12:
            continue
        cosv = float(np.dot(v1, v2) / (n1 * n2 + 1e-12))
        cosv = max(-1.0, min(1.0, cosv))
        ang = float(np.degrees(np.arccos(cosv)))
        if ang > max_deg:
            max_deg = ang
    return max_deg
