import argparse
import csv
import json
import logging
import os
import queue
import shutil
import subprocess
import threading
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Iterable, Optional, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from ultralytics import YOLO

from .quality import analyze_tracks, get_report_warnings
from .utils import (
    apply_crop_mask,
    build_crop_mask,
    load_scene_config,
    polygon_bbox,
    resolve_inference_device,
    setup_logging,
)

CSV_COLUMNS = [
    "frame",
    "track_id",
    "x1",
    "y1",
    "x2",
    "y2",
    "conf",
    "cls",
    "tl_state",
    "tl_prob_green",
    "tl_prob_red",
]
LOGGER = logging.getLogger("yolotest_tl")

# 不再用 deque(maxlen=...) 造成軌跡「往前消失」
# 改成無上限 list，並用「離開 ROI / 超過寬限期」來決定清除
KEEP_MISSING_FRAMES_DEFAULT = 30  # 允許漏偵測的幀數（避免遮擋就消失）


def _parse_classes(classes: Optional[Sequence[int]]) -> Optional[Iterable[int]]:
    return list(classes) if classes is not None else None


def _append_run_record_jsonl(path: str, record: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _parse_points_str(points_text: str) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    for token in (points_text or "").split(";"):
        token = token.strip()
        if not token:
            continue
        xy = [x.strip() for x in token.split(",")]
        if len(xy) != 2:
            raise ValueError(f"tl-roi 點位格式錯誤: {token}")
        pts.append((int(round(float(xy[0]))), int(round(float(xy[1])))))
    if len(pts) < 3:
        raise ValueError("tl-roi 至少需要 3 個點，格式: x,y;x,y;x,y;...")
    return pts


def _clip_polygon_points(poly: list[tuple[int, int]], w: int, h: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for x, y in poly:
        xx = int(np.clip(x, 0, max(0, w - 1)))
        yy = int(np.clip(y, 0, max(0, h - 1)))
        out.append((xx, yy))
    return out


class TinyCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        out_dim = max(2, int(num_classes))
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TrafficLightCnnInfer:
    def __init__(
        self,
        model_path: str,
        infer_device: object,
        img_size_override: int = 0,
        min_prob: float = 0.5,
        unknown_label: str = "unknown",
    ):
        self.model_path = model_path
        self.min_prob = float(np.clip(min_prob, 0.0, 1.0))
        self.unknown_label = str(unknown_label or "unknown")

        if isinstance(infer_device, str) and infer_device == "cpu":
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        ckpt_obj = torch.load(model_path, map_location=self.device)
        state_dict, ckpt_meta = self._extract_state_dict_and_meta(ckpt_obj)
        num_classes = self._infer_num_classes(state_dict)

        class_names_raw = ckpt_meta.get("class_names")
        class_names = [str(x) for x in class_names_raw] if isinstance(class_names_raw, (list, tuple)) else []
        if len(class_names) != num_classes:
            if class_names:
                LOGGER.warning(
                    "traffic_light_cnn class_names length mismatch (%s != %s), fallback to defaults",
                    len(class_names),
                    num_classes,
                )
            class_names = self._default_class_names(num_classes)
        self.class_names = class_names

        img_size_from_ckpt = ckpt_meta.get("img_size", 64)
        try:
            img_size_from_ckpt = int(img_size_from_ckpt)
        except Exception:
            img_size_from_ckpt = 64
        self.img_size = int(max(16, img_size_override if int(img_size_override) > 0 else img_size_from_ckpt))

        model = TinyCNN(num_classes=num_classes).to(self.device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        self.model = model

        self.green_idx = self.class_names.index("green") if "green" in self.class_names else 0
        self.red_idx = self.class_names.index("red") if "red" in self.class_names else min(1, len(self.class_names) - 1)
        LOGGER.info(
            "traffic_light_cnn loaded model=%s device=%s img_size=%s classes=%s min_prob=%.3f",
            model_path,
            self.device,
            self.img_size,
            self.class_names,
            self.min_prob,
        )

    @staticmethod
    def _strip_prefix_if_all(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
        if state_dict and all(k.startswith(prefix) for k in state_dict):
            return {k[len(prefix):]: v for k, v in state_dict.items()}
        return state_dict

    @classmethod
    def _extract_state_dict_and_meta(cls, ckpt_obj: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        state_dict: Optional[dict[str, torch.Tensor]] = None
        meta: dict[str, Any] = {}

        if isinstance(ckpt_obj, dict):
            meta = ckpt_obj
            for k in ("state_dict", "model_state_dict", "weights", "net"):
                v = ckpt_obj.get(k)
                if isinstance(v, dict):
                    state_dict = v
                    break
            if state_dict is None and isinstance(ckpt_obj.get("model"), nn.Module):
                state_dict = ckpt_obj["model"].state_dict()
            if state_dict is None:
                # 直接 torch.save(model.state_dict(), path) 的格式
                if ckpt_obj and all(isinstance(k, str) and torch.is_tensor(v) for k, v in ckpt_obj.items()):
                    state_dict = ckpt_obj
                    meta = {}
        elif isinstance(ckpt_obj, nn.Module):
            state_dict = ckpt_obj.state_dict()

        if state_dict is None:
            if isinstance(ckpt_obj, dict):
                keys = list(ckpt_obj.keys())
                head = keys[:8]
                raise ValueError(f"traffic_light checkpoint 格式不支援，keys={head}")
            raise ValueError(f"traffic_light checkpoint 格式不支援，type={type(ckpt_obj).__name__}")

        state_dict = dict(state_dict)
        state_dict = cls._strip_prefix_if_all(state_dict, "module.")
        state_dict = cls._strip_prefix_if_all(state_dict, "model.")
        return state_dict, meta

    @staticmethod
    def _find_tensor_by_suffix(state_dict: dict[str, torch.Tensor], suffixes: Sequence[str]) -> Optional[torch.Tensor]:
        for key, tensor in state_dict.items():
            if not torch.is_tensor(tensor):
                continue
            for suffix in suffixes:
                if key == suffix or key.endswith("." + suffix):
                    return tensor
        return None

    @classmethod
    def _infer_num_classes(cls, state_dict: dict[str, torch.Tensor]) -> int:
        weight = cls._find_tensor_by_suffix(
            state_dict, ("net.10.weight", "classifier.weight", "fc.weight", "head.weight")
        )
        if weight is not None and weight.ndim >= 2:
            return int(weight.shape[0])
        bias = cls._find_tensor_by_suffix(state_dict, ("net.10.bias", "classifier.bias", "fc.bias", "head.bias"))
        if bias is not None and bias.ndim >= 1:
            return int(bias.shape[0])
        raise ValueError("無法從 traffic_light checkpoint 推斷類別數（找不到最後分類層）")

    @staticmethod
    def _default_class_names(num_classes: int) -> list[str]:
        if num_classes == 2:
            return ["green", "red"]
        if num_classes == 3:
            return ["green", "yellow", "red"]
        return [f"class_{i}" for i in range(int(max(2, num_classes)))]

    def _preprocess(self, crop_bgr: np.ndarray, mask: Optional[np.ndarray]) -> torch.Tensor:
        if mask is not None and mask.shape[:2] == crop_bgr.shape[:2]:
            crop_bgr = cv2.bitwise_and(crop_bgr, crop_bgr, mask=mask)
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        arr = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)
        x = torch.from_numpy(arr).unsqueeze(0).to(self.device, non_blocking=True)
        return x

    def predict(self, crop_bgr: np.ndarray, mask: Optional[np.ndarray]) -> tuple[str, float, float]:
        if crop_bgr.size == 0:
            return self.unknown_label, float("nan"), float("nan")
        with torch.inference_mode():
            x = self._preprocess(crop_bgr, mask)
            logits = self.model(x)
            prob = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

        pred_idx = int(np.argmax(prob))
        pred_prob = float(prob[pred_idx])
        p_green = float(prob[self.green_idx]) if self.green_idx < prob.shape[0] else float("nan")
        p_red = float(prob[self.red_idx]) if self.red_idx < prob.shape[0] else float("nan")
        state = self.class_names[pred_idx] if pred_idx < len(self.class_names) else self.unknown_label
        if pred_prob < self.min_prob:
            state = self.unknown_label
        return state, p_green, p_red


class FramePrefetchReader:
    def __init__(self, video_path: str, max_prefetch: int = 64):
        self.video_path = video_path
        self.max_prefetch = max(4, int(max_prefetch))
        self.q: "queue.Queue[Optional[tuple[int, np.ndarray]]]" = queue.Queue(maxsize=self.max_prefetch)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self._err: Optional[Exception] = None

    def start(self):
        self.thread.start()
        return self

    def _worker(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self._err = FileNotFoundError(f"無法開啟影片: {self.video_path}")
            self.q.put(None)
            return
        frame_id = 0
        try:
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                while not self.stop_event.is_set():
                    try:
                        self.q.put((frame_id, frame), timeout=0.1)
                        break
                    except queue.Full:
                        continue
                frame_id += 1
        except Exception as e:
            self._err = e
        finally:
            cap.release()
            while True:
                try:
                    self.q.put(None, timeout=0.1)
                    break
                except queue.Full:
                    if self.stop_event.is_set():
                        break

    def get(self):
        item = self.q.get()
        if item is None and self._err is not None:
            raise self._err
        return item

    def close(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)


class AsyncCsvWriter:
    """Background CSV writer to reduce main-thread I/O stalls."""

    def __init__(self, out_csv: str, columns: Sequence[str], max_queue_chunks: int = 8):
        self.out_csv = out_csv
        self.columns = list(columns)
        self.max_queue_chunks = max(2, int(max_queue_chunks))
        self.q: "queue.Queue[Optional[list[list[Any]]]]" = queue.Queue(maxsize=self.max_queue_chunks)
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self._err: Optional[Exception] = None
        self.rows_written = 0

    def start(self):
        self.thread.start()
        return self

    def _raise_if_error(self):
        if self._err is not None:
            raise RuntimeError(f"async csv writer failed: {self._err}") from self._err

    def _worker(self):
        try:
            with open(self.out_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.columns)
                chunks_since_flush = 0
                while True:
                    item = self.q.get()
                    if item is None:
                        break
                    if item:
                        writer.writerows(item)
                        self.rows_written += len(item)
                        chunks_since_flush += 1
                        if chunks_since_flush >= 4:
                            # Balance performance and durability.
                            f.flush()
                            chunks_since_flush = 0
                f.flush()
        except Exception as e:
            self._err = e

    def enqueue(self, rows: list[list[Any]]) -> None:
        if not rows:
            return
        while True:
            self._raise_if_error()
            try:
                self.q.put(rows, timeout=0.5)
                return
            except queue.Full:
                continue

    def close(self) -> None:
        while True:
            self._raise_if_error()
            try:
                self.q.put(None, timeout=0.5)
                break
            except queue.Full:
                continue
        if self.thread.is_alive():
            self.thread.join()
        self._raise_if_error()


class GpuSafetyController:
    """Soft GPU guard: ramp-up + optional FPS cap + GPU metrics backoff."""

    def __init__(
        self,
        infer_device: object,
        max_infer_fps: float = 0.0,
        ramp_frames: int = 240,
        ramp_max_sleep_ms: float = 20.0,
        gpu_guard: bool = True,
        gpu_util_high: float = 90.0,
        gpu_power_high_ratio: float = 0.85,
        gpu_guard_sleep_ms: float = 8.0,
        metrics_poll_sec: float = 0.8,
    ):
        self.infer_device = infer_device
        self.is_gpu = infer_device != "cpu"
        self.max_infer_fps = float(max(0.0, max_infer_fps))
        self.ramp_frames = int(max(0, ramp_frames))
        self.ramp_max_sleep_ms = float(max(0.0, ramp_max_sleep_ms))
        self.gpu_guard = bool(gpu_guard and self.is_gpu)
        self.gpu_util_high = float(max(0.0, gpu_util_high))
        self.gpu_power_high_ratio = float(max(0.0, gpu_power_high_ratio))
        self.gpu_guard_sleep_ms = float(max(0.0, gpu_guard_sleep_ms))
        self.metrics_poll_sec = float(max(0.2, metrics_poll_sec))

        self.total_sleep_sec = 0.0
        self.sleep_events = 0
        self.infer_calls = 0
        self.last_gpu_util: Optional[float] = None
        self.last_gpu_power_ratio: Optional[float] = None
        self._last_infer_end: Optional[float] = None
        self._last_metrics_ts = 0.0
        self._cached_metrics: Optional[tuple[Optional[float], Optional[float]]] = None
        self._metric_read_failures = 0

        self._nvml = None
        self._nvml_handle = None
        self._nvml_ready = False
        self._nvml_device_index = 0
        self.metrics_backend = "disabled"
        self.metrics_available = False
        if self.gpu_guard:
            self._init_metrics_backend()

    @staticmethod
    def _parse_device_index(infer_device: object) -> int:
        if isinstance(infer_device, int):
            return int(max(0, infer_device))
        if isinstance(infer_device, str):
            s = infer_device.strip().lower()
            if s.isdigit():
                return int(s)
            if s.startswith("cuda:"):
                right = s.split(":", 1)[1].strip()
                if right.isdigit():
                    return int(right)
        return 0

    @staticmethod
    def _parse_metric_float(value: str) -> Optional[float]:
        s = str(value).strip()
        if not s or s.upper() in {"N/A", "[N/A]"}:
            return None
        s = s.replace("W", "").replace("%", "").strip()
        try:
            return float(s)
        except Exception:
            return None

    def _init_nvml(self) -> bool:
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            n = int(pynvml.nvmlDeviceGetCount())
            idx = self._parse_device_index(self.infer_device)
            if idx < 0 or idx >= n:
                idx = 0
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            self._nvml = pynvml
            self._nvml_ready = True
            self._nvml_device_index = idx
            self.metrics_backend = "nvml"
            self.metrics_available = True
            LOGGER.info(
                "gpu_guard metrics_backend=nvml device=%s util_high=%.1f%% power_ratio_high=%.2f guard_sleep=%.1fms poll=%.2fs",
                idx,
                self.gpu_util_high,
                self.gpu_power_high_ratio,
                self.gpu_guard_sleep_ms,
                self.metrics_poll_sec,
            )
            return True
        except Exception:
            self._nvml_ready = False
            self._nvml = None
            self._nvml_handle = None
            return False

    def _query_nvidia_smi_metrics(self) -> Optional[tuple[Optional[float], Optional[float]]]:
        smi_path = shutil.which("nvidia-smi")
        if not smi_path:
            return None

        idx = self._parse_device_index(self.infer_device)
        cmd = [
            smi_path,
            "--query-gpu=utilization.gpu,power.draw,power.limit",
            "--format=csv,noheader,nounits",
            "-i",
            str(idx),
        ]
        try:
            out = subprocess.check_output(
                cmd,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=1.5,
            )
        except Exception:
            return None

        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        if not lines:
            return None
        parts = [x.strip() for x in lines[0].split(",")]
        if len(parts) < 3:
            return None

        util = self._parse_metric_float(parts[0])
        power_draw = self._parse_metric_float(parts[1])
        power_limit = self._parse_metric_float(parts[2])
        power_ratio: Optional[float] = None
        if power_draw is not None and power_limit is not None and power_limit > 1e-6:
            power_ratio = float(power_draw / power_limit)
        return util, power_ratio

    def _init_nvidia_smi(self) -> bool:
        sample = self._query_nvidia_smi_metrics()
        if sample is None:
            return False
        self.metrics_backend = "nvidia-smi"
        self.metrics_available = True
        self._cached_metrics = sample
        self._last_metrics_ts = time.perf_counter()
        LOGGER.info(
            "gpu_guard metrics_backend=nvidia-smi device=%s util_high=%.1f%% power_ratio_high=%.2f guard_sleep=%.1fms poll=%.2fs",
            self._parse_device_index(self.infer_device),
            self.gpu_util_high,
            self.gpu_power_high_ratio,
            self.gpu_guard_sleep_ms,
            self.metrics_poll_sec,
        )
        return True

    def _init_metrics_backend(self) -> None:
        if self._init_nvml():
            return
        LOGGER.warning("gpu_guard NVML unavailable, try fallback metrics backend=nvidia-smi")
        if self._init_nvidia_smi():
            return
        self.metrics_backend = "none"
        self.metrics_available = False
        LOGGER.warning("gpu_guard metrics unavailable, fallback to ramp/fps only")

    def _read_gpu_metrics(self) -> Optional[tuple[Optional[float], Optional[float]]]:
        if self.metrics_backend == "nvml":
            if (not self._nvml_ready) or (self._nvml is None) or (self._nvml_handle is None):
                return None
            try:
                util = float(self._nvml.nvmlDeviceGetUtilizationRates(self._nvml_handle).gpu)
                power_w = float(self._nvml.nvmlDeviceGetPowerUsage(self._nvml_handle)) / 1000.0
                try:
                    limit_w = float(self._nvml.nvmlDeviceGetEnforcedPowerLimit(self._nvml_handle)) / 1000.0
                except Exception:
                    limit_w = float(self._nvml.nvmlDeviceGetPowerManagementLimit(self._nvml_handle)) / 1000.0
                power_ratio = None if limit_w <= 1e-6 else float(power_w / limit_w)
                return util, power_ratio
            except Exception as e:
                LOGGER.warning("gpu_guard NVML read failed (%s), switch to nvidia-smi fallback", e)
                self._nvml_ready = False
                if self._init_nvidia_smi():
                    return self._cached_metrics
                self.metrics_backend = "none"
                self.metrics_available = False
                return None

        if self.metrics_backend == "nvidia-smi":
            return self._query_nvidia_smi_metrics()
        return None

    def _get_metrics_with_cache(self, now_ts: float) -> Optional[tuple[Optional[float], Optional[float]]]:
        if not self.metrics_available:
            return None
        if self._cached_metrics is not None and (now_ts - self._last_metrics_ts) < self.metrics_poll_sec:
            return self._cached_metrics

        metrics = self._read_gpu_metrics()
        if metrics is None:
            self._metric_read_failures += 1
            if self._metric_read_failures in {1, 10, 100}:
                LOGGER.warning(
                    "gpu metrics read failed backend=%s fail_count=%s",
                    self.metrics_backend,
                    self._metric_read_failures,
                )
            return self._cached_metrics
        self._metric_read_failures = 0
        self._cached_metrics = metrics
        self._last_metrics_ts = now_ts
        return metrics

    def before_infer(self, frame_id: int) -> None:
        if not self.is_gpu:
            return
        sleep_sec = 0.0
        now = time.perf_counter()

        if self.max_infer_fps > 0.0 and self._last_infer_end is not None:
            min_interval = 1.0 / self.max_infer_fps
            elapsed = now - self._last_infer_end
            if elapsed < min_interval:
                s = min_interval - elapsed
                time.sleep(s)
                sleep_sec += s
                now += s

        if self.ramp_frames > 0 and frame_id < self.ramp_frames and self.ramp_max_sleep_ms > 0.0:
            ratio = 1.0 - (float(frame_id) / float(max(1, self.ramp_frames)))
            s = (self.ramp_max_sleep_ms / 1000.0) * ratio
            if s > 0.0:
                time.sleep(s)
                sleep_sec += s
                now += s

        if self.gpu_guard and self.metrics_available:
            metrics = self._get_metrics_with_cache(now)
            if metrics is not None:
                util, power_ratio = metrics
                if util is not None:
                    self.last_gpu_util = util
                if power_ratio is not None:
                    self.last_gpu_power_ratio = power_ratio
                over_util = (util is not None) and (util >= self.gpu_util_high)
                over_power = (power_ratio is not None) and (power_ratio >= self.gpu_power_high_ratio)
                if over_util or over_power:
                    s = self.gpu_guard_sleep_ms / 1000.0
                    if s > 0.0:
                        time.sleep(s)
                        sleep_sec += s

        if sleep_sec > 0.0:
            self.total_sleep_sec += sleep_sec
            self.sleep_events += 1

    def after_infer(self) -> None:
        if not self.is_gpu:
            return
        self._last_infer_end = time.perf_counter()
        self.infer_calls += 1

    def snapshot(self) -> dict[str, Any]:
        avg_sleep_ms = (self.total_sleep_sec * 1000.0 / self.infer_calls) if self.infer_calls > 0 else 0.0
        return {
            "metrics_backend": self.metrics_backend,
            "metrics_available": bool(self.metrics_available),
            "avg_sleep_ms": float(avg_sleep_ms),
            "sleep_events": float(self.sleep_events),
            "last_gpu_util": self.last_gpu_util,
            "last_gpu_power_ratio": self.last_gpu_power_ratio,
        }

    def close(self) -> None:
        if self._nvml_ready and self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_ready = False


def _draw_roi_overlay(frame: np.ndarray, roi_poly: list, color=(0, 255, 100), alpha=0.15):
    pts = np.array(roi_poly, dtype=np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)


def _color_from_id(track_id: int):
    # 固定且亮一點的顏色（避免太暗）
    return tuple(int(c) for c in (
        (track_id * 37 + 80) % 200 + 55,
        (track_id * 97 + 40) % 200 + 55,
        (track_id * 61 + 120) % 200 + 55,
    ))


def _draw_tracks(frame: np.ndarray, track_history: dict[int, list[tuple[int, int]]]):
    for track_id, points in track_history.items():
        if len(points) < 2:
            continue
        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=False, color=_color_from_id(track_id), thickness=2)


def run_tracking_to_csv_roi_crop(
    scene_config_path: str,
    video_path: str,
    out_csv: str,
    model_path: str = "yolo26x.pt",
    tracker_cfg: str = "botsort.yaml",
    device: int = 0,
    imgsz: int = 640,
    conf: float = 0.20,
    iou: float = 0.50,
    half: bool = True,
    classes=(2, 3, 5, 7),
    use_polygon_mask: bool = True,
    filter_ground_in_roi: bool = True,
    # 新增：讓軌跡只要還在 ROI 就維持（遮擋短暫漏偵測也不清）
    keep_missing_frames: int = KEEP_MISSING_FRAMES_DEFAULT,
    # 新增：離開 ROI 是否立刻清掉軌跡；False = 離開後仍保留軌跡（直到影片結束）
    drop_track_when_leave_roi: bool = True,
    log_every: int = 300,
    show: bool = False,
    show_scale: float = 0.6,
    save_video: Optional[str] = None,
    prefetch_frames: int = 64,
    cpu_threads: int = 0,
    tune_gpu_backend: bool = True,
    max_infer_fps: float = 0.0,
    ramp_frames: int = 240,
    ramp_max_sleep_ms: float = 20.0,
    gpu_guard: bool = True,
    gpu_util_high: float = 90.0,
    gpu_power_high_ratio: float = 0.85,
    gpu_guard_sleep_ms: float = 8.0,
    csv_flush_every: int = 50000,
    log_run_params: bool = True,
    params_log_path: str = "run_logs/csv_run_params.jsonl",
    run_tag: str = "",
    args_snapshot: Optional[dict] = None,
    run_quality_check: bool = True,
    quality_json_path: str = "",
    tl_cnn_model: str = "",
    tl_roi: str = "",
    tl_img_size: int = 0,
    tl_min_prob: float = 0.50,
    tl_unknown_label: str = "unknown",
    tl_every_n_frames: int = 1,
):
    t_start = time.perf_counter()
    start_at = datetime.now().isoformat(timespec="seconds")

    # ── CPU/OpenCV 調校 ──
    cv2.setUseOptimized(True)
    if cpu_threads > 0:
        cv2.setNumThreads(int(cpu_threads))
    else:
        # 0 = OpenCV 自動
        cv2.setNumThreads(0)

    # ── GPU 選擇：有 GPU 就強制使用 ──
    infer_device, half_ok = resolve_inference_device(
        preferred_device=device,
        force_gpu_if_available=True,
        logger=LOGGER,
    )
    if infer_device == "cpu":
        half = False
    elif not half_ok:
        half = False
    elif tune_gpu_backend:
        # 不改模型輸入尺寸與閾值，不影響偵測設定
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ── 讀 scene_config（roi_polygon = ROI 多邊形） ──
    cfg = load_scene_config(scene_config_path)
    roi_poly = cfg["roi_polygon"]
    xmin, ymin, xmax, ymax = polygon_bbox(roi_poly)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"無法開啟影片: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(w, xmax)
    ymax = min(h, ymax)
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("ROI bbox 無效，請檢查 scene_config 的 roi_polygon")

    crop_w = xmax - xmin
    crop_h = ymax - ymin
    crop_mask = build_crop_mask(roi_poly, xmin, ymin, crop_w, crop_h) if use_polygon_mask else None
    # Full-frame ROI mask for fast foot-point in/out check (faster than pointPolygonTest per box).
    roi_pts_i32 = np.asarray(roi_poly, dtype=np.int32).reshape((-1, 1, 2))
    roi_mask_full = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(roi_mask_full, [roi_pts_i32], 255)
    LOGGER.info("roi_filter_backend=mask mask_size=%sx%s", w, h)

    # ── 紅綠燈 ROI + CNN 設定（可選） ──
    tl_enabled = bool((tl_cnn_model or "").strip()) and bool((tl_roi or "").strip())
    tl_model: Optional[TrafficLightCnnInfer] = None
    tl_mask_local: Optional[np.ndarray] = None
    tl_xmin = 0
    tl_ymin = 0
    tl_xmax = 0
    tl_ymax = 0
    tl_state_curr = str(tl_unknown_label or "unknown")
    tl_p_green_curr = float("nan")
    tl_p_red_curr = float("nan")
    tl_infer_calls = 0
    tl_rows_emitted = 0
    tl_track_id = -9

    if tl_enabled:
        tl_poly = _parse_points_str(tl_roi)
        tl_poly = _clip_polygon_points(tl_poly, w=w, h=h)
        xs = [p[0] for p in tl_poly]
        ys = [p[1] for p in tl_poly]
        tl_xmin = int(min(xs))
        tl_ymin = int(min(ys))
        tl_xmax = int(max(xs)) + 1
        tl_ymax = int(max(ys)) + 1
        tl_xmax = int(np.clip(tl_xmax, 0, w))
        tl_ymax = int(np.clip(tl_ymax, 0, h))
        if tl_xmax <= tl_xmin or tl_ymax <= tl_ymin:
            raise ValueError("traffic light ROI 無效，請檢查 --tl-roi")

        tl_poly_local = np.asarray([(x - tl_xmin, y - tl_ymin) for x, y in tl_poly], dtype=np.int32).reshape((-1, 1, 2))
        tl_mask_local = np.zeros((tl_ymax - tl_ymin, tl_xmax - tl_xmin), dtype=np.uint8)
        cv2.fillPoly(tl_mask_local, [tl_poly_local], 255)
        tl_model = TrafficLightCnnInfer(
            model_path=tl_cnn_model,
            infer_device=infer_device,
            img_size_override=int(tl_img_size),
            min_prob=float(tl_min_prob),
            unknown_label=str(tl_unknown_label),
        )
        LOGGER.info(
            "tl_cnn enabled roi=(%s,%s,%s,%s) every_n=%s model=%s",
            tl_xmin,
            tl_ymin,
            tl_xmax,
            tl_ymax,
            int(max(1, tl_every_n_frames)),
            tl_cnn_model,
        )
    else:
        LOGGER.info("tl_cnn disabled (set both --tl-cnn-model and --tl-roi to enable)")

    LOGGER.info("scene_id=%s roi_bbox=(%s,%s,%s,%s)", cfg.get("scene_id"), xmin, ymin, xmax, ymax)
    LOGGER.info(
        "video=%s frames=%s fps=%.2f size=%sx%s model=%s tracker=%s",
        video_path, total_frames, fps, w, h, model_path, tracker_cfg,
    )
    LOGGER.info(
        "keep_missing_frames=%s drop_track_when_leave_roi=%s filter_ground_in_roi=%s prefetch=%s cpu_threads=%s",
        keep_missing_frames, drop_track_when_leave_roi, filter_ground_in_roi, prefetch_frames, cpu_threads,
    )
    LOGGER.info(
        "gpu_safety max_infer_fps=%.2f ramp_frames=%s ramp_max_sleep_ms=%.1f gpu_guard=%s util_high=%.1f power_ratio_high=%.2f guard_sleep_ms=%.1f",
        max_infer_fps,
        ramp_frames,
        ramp_max_sleep_ms,
        gpu_guard,
        gpu_util_high,
        gpu_power_high_ratio,
        gpu_guard_sleep_ms,
    )

    # ── YOLO 模型 ──
    model_ext = os.path.splitext(str(model_path))[1].lower()
    is_trt_engine = model_ext == ".engine"
    model = YOLO(model_path)
    if infer_device != "cpu" and not is_trt_engine:
        model = model.to("cuda")
    elif is_trt_engine:
        LOGGER.info("detected TensorRT engine model (%s), skip model.to() and use runtime device=%s", model_path, infer_device)

    gpu_safety = GpuSafetyController(
        infer_device=infer_device,
        max_infer_fps=max_infer_fps,
        ramp_frames=ramp_frames,
        ramp_max_sleep_ms=ramp_max_sleep_ms,
        gpu_guard=gpu_guard,
        gpu_util_high=gpu_util_high,
        gpu_power_high_ratio=gpu_power_high_ratio,
        gpu_guard_sleep_ms=gpu_guard_sleep_ms,
    )
    init_gpu_stats = gpu_safety.snapshot()
    LOGGER.info(
        "gpu_metrics backend=%s available=%s",
        init_gpu_stats.get("metrics_backend"),
        init_gpu_stats.get("metrics_available"),
    )

    # ── 視覺化影片輸出 ──
    video_writer = None
    if save_video:
        os.makedirs(os.path.dirname(save_video) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_w = int(w * show_scale)
        out_h = int(h * show_scale)
        video_writer = cv2.VideoWriter(save_video, fourcc, fps, (out_w, out_h))

    # ★優化後的軌跡維護狀態
    track_history: dict[int, list[tuple[int, int]]] = defaultdict(list)
    last_seen_frame: dict[int, int] = {}
    last_in_roi: dict[int, bool] = {}

    rows: list[list[Any]] = []
    pbar = tqdm(total=total_frames if total_frames > 0 else None, desc="track", unit="frame")
    skipped_empty_crop = 0
    kept_dets = 0
    reader = FramePrefetchReader(video_path=video_path, max_prefetch=prefetch_frames).start()
    frame_id = -1
    stop_early = False
    total_rows_written = 0

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    csv_writer = AsyncCsvWriter(
        out_csv=out_csv,
        columns=CSV_COLUMNS,
        max_queue_chunks=max(4, min(16, int(prefetch_frames) // 8 if int(prefetch_frames) > 0 else 8)),
    ).start()
    LOGGER.info("csv_writer=async queue_chunks=%s csv_flush_every=%s", csv_writer.max_queue_chunks, csv_flush_every)

    def flush_rows(force: bool = False):
        nonlocal rows, total_rows_written
        if not rows:
            return
        if (not force) and len(rows) < max(1, int(csv_flush_every)):
            return
        csv_writer.enqueue(rows)
        total_rows_written += int(len(rows))
        rows = []

    while True:
        item = reader.get()
        if item is None:
            break
        frame_id, frame = item

        # ── 每幀（或每 N 幀）先做紅綠燈 CNN 狀態判定 ──
        if tl_enabled and tl_model is not None and tl_mask_local is not None:
            if (frame_id % max(1, int(tl_every_n_frames))) == 0:
                tl_patch = frame[tl_ymin:tl_ymax, tl_xmin:tl_xmax]
                if tl_patch.size > 0:
                    tl_state_curr, tl_p_green_curr, tl_p_red_curr = tl_model.predict(tl_patch, tl_mask_local)
                    tl_infer_calls += 1

        # ── 用 ROI bbox 裁切，並用 ROI 多邊形做 mask ──
        crop = frame[ymin:ymax, xmin:xmax]
        if crop.size == 0:
            skipped_empty_crop += 1
            pbar.update(1)
            continue

        crop_infer = apply_crop_mask(crop, crop_mask) if crop_mask is not None else crop

        gpu_safety.before_infer(frame_id)
        with torch.inference_mode():
            results = model.track(
                source=crop_infer,
                tracker=tracker_cfg,
                persist=True,
                stream=False,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                augment=False,
                device=infer_device,
                half=half,
                classes=_parse_classes(classes),
                verbose=False,
            )
        gpu_safety.after_infer()

        r = results[0]
        boxes = r.boxes

        # 本幀看到了哪些 track id
        seen_this_frame: set[int] = set()
        frame_has_tl_box_row = False

        # ── 收集 tracking 結果 ──
        if boxes is not None and boxes.id is not None:
            b_xyxy = boxes.xyxy.detach().cpu().numpy()
            b_conf = boxes.conf.detach().cpu().numpy()
            b_cls = boxes.cls.detach().cpu().numpy()
            b_ids = boxes.id.detach().cpu().numpy()

            for i in range(len(b_ids)):
                x1, y1, x2, y2 = b_xyxy[i]
                # 還原到全圖座標
                x1 += xmin
                x2 += xmin
                y1 += ymin
                y2 += ymin

                x1 = float(np.clip(x1, 0, w - 1))
                y1 = float(np.clip(y1, 0, h - 1))
                x2 = float(np.clip(x2, 0, w - 1))
                y2 = float(np.clip(y2, 0, h - 1))
                if x2 <= x1 or y2 <= y1:
                    continue

                tid = int(b_ids[i])
                seen_this_frame.add(tid)

                # 用「腳點」判斷是否在 ROI（使用 mask 索引，避免 pointPolygonTest 開銷）
                footx = (x1 + x2) * 0.5
                footy = y2
                foot_ix = int(footx)
                foot_iy = int(footy)
                in_roi = bool(
                    0 <= foot_ix < w and 0 <= foot_iy < h and roi_mask_full[foot_iy, foot_ix] != 0
                )

                last_seen_frame[tid] = frame_id
                last_in_roi[tid] = in_roi

                # 原本的 CSV 過濾邏輯保留：
                if filter_ground_in_roi and not in_roi:
                    continue

                # ★軌跡更新策略：只要這次在 ROI，就把點加進去（不會被 maxlen 擦掉）
                # 如果你希望「只要曾在 ROI，就算短暫出界也繼續畫」，可把條件改成 True
                if in_roi:
                    cx_full = int((x1 + x2) / 2)
                    cy_full = int((y1 + y2) / 2)
                    track_history[tid].append((cx_full, cy_full))

                rows.append([
                    frame_id,
                    tid,
                    round(x1, 2),
                    round(y1, 2),
                    round(x2, 2),
                    round(y2, 2),
                    round(float(b_conf[i]), 4),
                    int(b_cls[i]),
                    str(tl_state_curr),
                    ("" if not np.isfinite(tl_p_green_curr) else round(float(tl_p_green_curr), 4)),
                    ("" if not np.isfinite(tl_p_red_curr) else round(float(tl_p_red_curr), 4)),
                ])
                if int(b_cls[i]) == 9:
                    frame_has_tl_box_row = True
                kept_dets += 1
                if len(rows) >= max(1, int(csv_flush_every)):
                    flush_rows(force=False)

        # 強制補一筆紅綠燈框到 CSV，確保 GUI 一定能畫出紅綠燈框（不受 vehicle ROI 過濾影響）
        if tl_enabled and not frame_has_tl_box_row and tl_xmax > tl_xmin and tl_ymax > tl_ymin:
            if np.isfinite(tl_p_green_curr) and np.isfinite(tl_p_red_curr):
                tl_conf = float(max(tl_p_green_curr, tl_p_red_curr))
            elif np.isfinite(tl_p_green_curr):
                tl_conf = float(tl_p_green_curr)
            elif np.isfinite(tl_p_red_curr):
                tl_conf = float(tl_p_red_curr)
            else:
                tl_conf = 0.0
            rows.append([
                frame_id,
                tl_track_id,
                round(float(tl_xmin), 2),
                round(float(tl_ymin), 2),
                round(float(tl_xmax - 1), 2),
                round(float(tl_ymax - 1), 2),
                round(float(np.clip(tl_conf, 0.0, 1.0)), 4),
                9,
                str(tl_state_curr),
                ("" if not np.isfinite(tl_p_green_curr) else round(float(tl_p_green_curr), 4)),
                ("" if not np.isfinite(tl_p_red_curr) else round(float(tl_p_red_curr), 4)),
            ])
            tl_rows_emitted += 1
            if len(rows) >= max(1, int(csv_flush_every)):
                flush_rows(force=False)

        # ── 軌跡清除規則（關鍵優化）：離開 ROI 或太久沒看到才清 ──
        if drop_track_when_leave_roi:
            # 1) 如果本幀有看到且 in_roi=False → 立刻清
            # 2) 如果沒看到 → 超過 keep_missing_frames 且 last_in_roi=False/未知 → 清
            to_delete = []
            for tid in list(track_history.keys()):
                if tid in seen_this_frame:
                    if last_in_roi.get(tid, True) is False:
                        to_delete.append(tid)
                else:
                    # 沒看到：可能漏偵測，給寬限期
                    last_seen = last_seen_frame.get(tid, -10**9)
                    missing = frame_id - last_seen
                    # 若最後一次仍在 ROI 且在寬限期內，保留
                    if last_in_roi.get(tid, False) and missing <= keep_missing_frames:
                        continue
                    # 否則超過寬限期就清
                    if missing > keep_missing_frames:
                        to_delete.append(tid)

            for tid in to_delete:
                track_history.pop(tid, None)
                last_seen_frame.pop(tid, None)
                last_in_roi.pop(tid, None)

        # 額外清理：避免 ROI 外或早已消失的 ID 讓 dict 無限成長
        if frame_id % 60 == 0 and last_seen_frame:
            stale_meta = [
                tid
                for tid, lf in last_seen_frame.items()
                if (frame_id - lf) > max(keep_missing_frames * 4, 120) and tid not in track_history
            ]
            for tid in stale_meta:
                last_seen_frame.pop(tid, None)
                last_in_roi.pop(tid, None)

        # ── 畫即時畫面（整張圖上疊 ROI + 軌跡） ──
        if show or save_video:
            annotated_crop = r.plot()
            vis_frame = frame.copy()
            vis_frame[ymin:ymax, xmin:xmax] = annotated_crop
            _draw_roi_overlay(vis_frame, roi_poly)
            _draw_tracks(vis_frame, track_history)
            if tl_enabled:
                cv2.rectangle(vis_frame, (tl_xmin, tl_ymin), (tl_xmax - 1, tl_ymax - 1), (0, 255, 255), 2)
                tl_txt = f"TL={tl_state_curr} g={tl_p_green_curr:.2f} r={tl_p_red_curr:.2f}" if np.isfinite(tl_p_green_curr) and np.isfinite(tl_p_red_curr) else f"TL={tl_state_curr}"
                cv2.putText(
                    vis_frame,
                    tl_txt,
                    (tl_xmin, max(20, tl_ymin - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            det_count = len(boxes.id) if (boxes is not None and boxes.id is not None) else 0
            cv2.putText(
                vis_frame,
                f"Frame: {frame_id}  Dets: {det_count}  Tracks: {len(track_history)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 255), 2,
            )

            disp = cv2.resize(vis_frame, (int(w * show_scale), int(h * show_scale)))
            if show:
                cv2.imshow("YOLO Tracking", disp)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    LOGGER.info("使用者按下 q，提前結束")
                    stop_early = True
                elif key == ord("p"):
                    LOGGER.info("暫停中，按任意鍵繼續...")
                    cv2.waitKey(0)
            if video_writer is not None:
                video_writer.write(disp)

        pbar.update(1)
        if log_every > 0 and frame_id % log_every == 0:
            elapsed = max(1e-9, time.perf_counter() - t_start)
            avg_fps = (frame_id + 1) / elapsed
            gpu_stats = gpu_safety.snapshot()
            LOGGER.info(
                "progress frame=%s/%s kept_dets=%s active_rows=%s active_tracks=%s avg_fps=%.2f gpu_backend=%s gpu_sleep_ms=%.2f gpu_util=%s gpu_power_ratio=%s tl_state=%s tl_rows=%s",
                frame_id, total_frames if total_frames > 0 else "?",
                kept_dets, len(rows), len(track_history),
                avg_fps,
                gpu_stats.get("metrics_backend"),
                gpu_stats["avg_sleep_ms"] or 0.0,
                (
                    "n/a"
                    if gpu_stats["last_gpu_util"] is None
                    else f"{float(gpu_stats['last_gpu_util']):.1f}%"
                ),
                (
                    "n/a"
                    if gpu_stats["last_gpu_power_ratio"] is None
                    else f"{float(gpu_stats['last_gpu_power_ratio']):.2f}"
                ),
                tl_state_curr if tl_enabled else "n/a",
                tl_rows_emitted if tl_enabled else 0,
            )
        if stop_early:
            break

    pbar.close()
    reader.close()
    if video_writer:
        video_writer.release()
    if show:
        cv2.destroyAllWindows()
    gpu_safety.close()

    flush_rows(force=True)
    csv_writer.close()
    total_rows_written = int(csv_writer.rows_written)

    LOGGER.info(
        "done tracks_csv=%s rows=%s skipped_empty_crop=%s tl_enabled=%s tl_infer_calls=%s tl_rows=%s",
        out_csv, total_rows_written, skipped_empty_crop, tl_enabled, tl_infer_calls, tl_rows_emitted,
    )

    quality_summary = None
    quality_warnings = []
    if run_quality_check:
        try:
            df_check = pd.read_csv(out_csv)
            rep = analyze_tracks(df_check, roi_poly=roi_poly, long_track_len=50)
            quality_warnings = get_report_warnings(rep)
            quality_summary = {
                "rows": int(rep.rows),
                "n_unique_tracks": int(rep.n_unique_tracks),
                "n_unique_frames": int(rep.n_unique_frames),
                "invalid_bbox_rows": int(rep.invalid_bbox_rows),
                "tracks_len_le_10": int(rep.tracks_len_le_10),
                "tracks_with_frame_gaps": int(rep.tracks_with_frame_gaps),
                "roi_ground_inside_ratio": (
                    None if rep.roi_ground_inside_ratio is None else float(rep.roi_ground_inside_ratio)
                ),
                "warnings": quality_warnings,
            }
            LOGGER.info("quality_check summary=%s", quality_summary)
            if quality_json_path:
                os.makedirs(os.path.dirname(quality_json_path) or ".", exist_ok=True)
                with open(quality_json_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "summary": quality_summary,
                            "full_report": rep.__dict__,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                LOGGER.info("saved quality report=%s", quality_json_path)
        except Exception as e:
            LOGGER.warning("quality_check failed: %s", e)

    if log_run_params:
        duration_sec = float(time.perf_counter() - t_start)
        record = {
            "run_tag": run_tag,
            "start_at": start_at,
            "end_at": datetime.now().isoformat(timespec="seconds"),
            "duration_sec": round(duration_sec, 3),
            "scene_id": cfg.get("scene_id"),
            "scene_config_path": scene_config_path,
            "video_path": video_path,
            "out_csv": out_csv,
            "model_path": model_path,
            "tracker_cfg": tracker_cfg,
            "infer_device": infer_device,
            "half": bool(half),
            "imgsz": int(imgsz),
            "conf": float(conf),
            "iou": float(iou),
            "classes": list(classes) if classes is not None else None,
            "use_polygon_mask": bool(use_polygon_mask),
            "filter_ground_in_roi": bool(filter_ground_in_roi),
            "keep_missing_frames": int(keep_missing_frames),
            "drop_track_when_leave_roi": bool(drop_track_when_leave_roi),
            "prefetch_frames": int(prefetch_frames),
            "cpu_threads": int(cpu_threads),
            "tune_gpu_backend": bool(tune_gpu_backend),
            "max_infer_fps": float(max_infer_fps),
            "ramp_frames": int(ramp_frames),
            "ramp_max_sleep_ms": float(ramp_max_sleep_ms),
            "gpu_guard": bool(gpu_guard),
            "gpu_util_high": float(gpu_util_high),
            "gpu_power_high_ratio": float(gpu_power_high_ratio),
            "gpu_guard_sleep_ms": float(gpu_guard_sleep_ms),
            "csv_flush_every": int(csv_flush_every),
            "video_total_frames": int(total_frames),
            "video_fps": float(fps),
            "video_width": int(w),
            "video_height": int(h),
            "rows_written": int(total_rows_written),
            "skipped_empty_crop": int(skipped_empty_crop),
            "tl_enabled": bool(tl_enabled),
            "tl_cnn_model": tl_cnn_model,
            "tl_roi": tl_roi,
            "tl_img_size": int(tl_img_size),
            "tl_min_prob": float(tl_min_prob),
            "tl_unknown_label": tl_unknown_label,
            "tl_every_n_frames": int(max(1, tl_every_n_frames)),
            "tl_last_state": tl_state_curr,
            "tl_infer_calls": int(tl_infer_calls),
            "tl_rows_emitted": int(tl_rows_emitted),
            "gpu_safety_stats": gpu_safety.snapshot(),
            "quality_summary": quality_summary,
            "quality_warnings": quality_warnings,
            "args_snapshot": args_snapshot or {},
        }
        _append_run_record_jsonl(params_log_path, record)
        LOGGER.info("saved run params log=%s", params_log_path)


if __name__ == "__main__":
    setup_logging("INFO")
    parser = argparse.ArgumentParser(description="ROI crop YOLO26+BoT-SORT/ByteTrack + traffic-light CNN state to tracks.csv")
    parser.add_argument("--scene-config", default="scene_config.json")
    parser.add_argument("--video", default="MAH02324.MP4")
    parser.add_argument("--out-csv", default="tracks.csv")
    parser.add_argument("--model", default="yolo26x.pt")
    parser.add_argument("--tracker", default="botsort.yaml")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--half", action="store_true", default=False)
    parser.add_argument("--classes", default="2,3,5,7")
    parser.add_argument("--no-polygon-mask", action="store_true")
    parser.add_argument("--keep-outside-roi", action="store_true")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--log-every", type=int, default=300)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--show-scale", type=float, default=0.6)
    parser.add_argument("--save-video", default=None)
    parser.add_argument("--prefetch-frames", type=int, default=64,
                        help="影片讀幀預取佇列大小（建議 32~256）")
    parser.add_argument("--cpu-threads", type=int, default=0,
                        help="OpenCV CPU thread 數；0=自動")
    parser.add_argument("--no-gpu-tune", action="store_true",
                        help="關閉 GPU backend 調優（cudnn benchmark/tf32）")
    parser.add_argument("--max-infer-fps", type=float, default=0.0,
                        help="推論 FPS 上限；0=不限制（若需保護 GPU 可設 30~90）")
    parser.add_argument("--ramp-frames", type=int, default=240,
                        help="啟動緩升幀數（前 N 幀會額外 sleep，負載逐步拉升）")
    parser.add_argument("--ramp-max-sleep-ms", type=float, default=20.0,
                        help="啟動第一幀最多額外 sleep 毫秒，之後線性下降到 0")
    parser.add_argument("--no-gpu-guard", action="store_true",
                        help="關閉 NVML 使用率/功率守門（預設開啟）")
    parser.add_argument("--gpu-util-high", type=float, default=90.0,
                        help="GPU 使用率高水位（%%），超過就短暫 backoff")
    parser.add_argument("--gpu-power-high-ratio", type=float, default=0.85,
                        help="GPU 功率高水位（相對 power limit 比例），超過就 backoff")
    parser.add_argument("--gpu-guard-sleep-ms", type=float, default=8.0,
                        help="觸發 GPU 守門時每次額外 sleep 毫秒")
    parser.add_argument("--csv-flush-every", type=int, default=50000,
                        help="每累積多少筆 bbox 就分批寫入 CSV（降低記憶體/GC 開銷）")

    # 新增參數
    parser.add_argument("--keep-missing-frames", type=int, default=KEEP_MISSING_FRAMES_DEFAULT,
                        help="允許短暫漏偵測仍保留軌跡的幀數（預設 30）")
    parser.add_argument("--keep-tracks-after-leave-roi", action="store_true",
                        help="離開 ROI 後也不要清掉軌跡（預設離開就清）")
    parser.add_argument("--params-log-path", default="run_logs/csv_run_params.jsonl",
                        help="每次執行參數紀錄檔（jsonl）")
    parser.add_argument("--no-params-log", action="store_true",
                        help="不寫入參數紀錄檔")
    parser.add_argument("--run-tag", default="",
                        help="這次執行的自訂標籤（例如 expA_v1）")
    parser.add_argument("--no-quality-check", action="store_true",
                        help="完成 CSV 後不執行品質檢查")
    parser.add_argument("--quality-json-path", default="",
                        help="品質檢查完整報告輸出路徑（json）")
    parser.add_argument("--tl-cnn-model", default="",
                        help="紅綠燈 CNN 權重路徑（例如 red_green_cnn.pt）")
    parser.add_argument("--tl-roi", default="",
                        help="紅綠燈 ROI 多邊形點位，格式 x,y;x,y;x,y;...")
    parser.add_argument("--tl-img-size", type=int, default=0,
                        help="CNN 輸入尺寸，0=使用模型內建設定")
    parser.add_argument("--tl-min-prob", type=float, default=0.50,
                        help="CNN 最低信心值，低於此值輸出 unknown")
    parser.add_argument("--tl-unknown-label", default="unknown",
                        help="低信心時寫入 CSV 的紅綠燈狀態文字")
    parser.add_argument("--tl-every-n-frames", type=int, default=1,
                        help="每 N 幀做一次紅綠燈 CNN 推論（其餘幀沿用上一個結果）")

    args = parser.parse_args()

    setup_logging(args.log_level)
    LOGGER.info("yolotest script started, parsing arguments done")
    LOGGER.info(
        "args video=%s out_csv=%s model=%s tracker=%s device=%s imgsz=%s conf=%.3f iou=%.3f tl_model=%s tl_roi=%s",
        args.video,
        args.out_csv,
        args.model,
        args.tracker,
        args.device,
        args.imgsz,
        args.conf,
        args.iou,
        args.tl_cnn_model if args.tl_cnn_model else "<off>",
        args.tl_roi if args.tl_roi else "<off>",
    )
    classes = tuple(int(x.strip()) for x in args.classes.split(",") if x.strip()) if args.classes else None

    run_tracking_to_csv_roi_crop(
        scene_config_path=args.scene_config,
        video_path=args.video,
        out_csv=args.out_csv,
        model_path=args.model,
        tracker_cfg=args.tracker,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        half=args.half,
        classes=classes,
        use_polygon_mask=not args.no_polygon_mask,
        filter_ground_in_roi=not args.keep_outside_roi,
        keep_missing_frames=args.keep_missing_frames,
        drop_track_when_leave_roi=(not args.keep_tracks_after_leave_roi),
        log_every=args.log_every,
        show=args.show,
        show_scale=args.show_scale,
        save_video=args.save_video,
        prefetch_frames=args.prefetch_frames,
        cpu_threads=args.cpu_threads,
        tune_gpu_backend=not args.no_gpu_tune,
        max_infer_fps=args.max_infer_fps,
        ramp_frames=args.ramp_frames,
        ramp_max_sleep_ms=args.ramp_max_sleep_ms,
        gpu_guard=not args.no_gpu_guard,
        gpu_util_high=args.gpu_util_high,
        gpu_power_high_ratio=args.gpu_power_high_ratio,
        gpu_guard_sleep_ms=args.gpu_guard_sleep_ms,
        csv_flush_every=args.csv_flush_every,
        log_run_params=not args.no_params_log,
        params_log_path=args.params_log_path,
        run_tag=args.run_tag,
        args_snapshot=dict(vars(args)),
        run_quality_check=not args.no_quality_check,
        quality_json_path=args.quality_json_path,
        tl_cnn_model=args.tl_cnn_model,
        tl_roi=args.tl_roi,
        tl_img_size=args.tl_img_size,
        tl_min_prob=args.tl_min_prob,
        tl_unknown_label=args.tl_unknown_label,
        tl_every_n_frames=args.tl_every_n_frames,
    )
