# 車輛偵測與追蹤

[回到首頁](../README.md) · [研究方法](methodology.md) · [資料介面](data.md)

這個階段將影片轉成逐幀車輛軌跡 CSV，同時記錄號誌 CNN 的狀態。它是風險流程的影像前端，不會單獨產生整體風險等級。

## 最短執行方式

```bash
./scripts/run_detection.sh input.mp4 /path/to/yolo26x.pt outputs/tracks.csv
```

這個腳本會套用論文期間的影像尺寸、confidence、IoU、追蹤、GPU 與號誌預設。可透過 `PYTHON_BIN=/path/to/python` 指定 Python。

## 完整新版 CLI

```bash
traffic-risk detect -- \
  --scene-config configs/camera_roi.json \
  --video input.mp4 \
  --out-csv outputs/tracks.csv \
  --model /path/to/yolo26x.pt \
  --tracker configs/bytetrack.yaml \
  --device 0 --half \
  --imgsz 1280 --conf 0.18 --iou 0.55 \
  --classes 2,3,5,7,9 \
  --keep-outside-roi \
  --prefetch-frames 256 --cpu-threads 20 \
  --max-infer-fps 0 --ramp-frames 0 --ramp-max-sleep-ms 0 --no-gpu-guard \
  --tl-cnn-model models/traffic_light_classifier.pt \
  --tl-roi "1079,426;1119,426;1119,443;1079,443" \
  --tl-min-prob 0.55 \
  --no-quality-check --log-level INFO --log-every 100
```

`--video` 也可改為第一個 positional argument；`--model`/`--yolo-model`、`--out-csv`/`--output`、`--scene-config`/`--camera-config` 及 `--tracker`/`--tracker-config` 都是同義介面。

<details>
<summary><strong>展開：原始 yolotest.py 的等價呼叫形式</strong></summary>

```bash
python yolotest.py \
  --scene-config scene_config.json \
  --video car_video_8_hours.mp4 \
  --out-csv car_video_8_hours_20260509.csv \
  --model yolo26x.pt \
  --tracker trackers/bytetrack_b7_tune_a.yaml \
  --device 0 --half \
  --imgsz 1280 --conf 0.18 --iou 0.55 \
  --classes 2,3,5,7,9 \
  --keep-outside-roi \
  --prefetch-frames 256 --cpu-threads 20 \
  --max-infer-fps 0 --ramp-frames 0 --ramp-max-sleep-ms 0 --no-gpu-guard \
  --tl-cnn-model traffic_light.pt \
  --tl-roi "1079,426;1119,426;1119,443;1079,443" \
  --tl-min-prob 0.55 \
  --no-quality-check --log-level INFO --log-every 100
```

精簡版的 `src/traffic_risk/detection/` 從這個正式入口萃取，並保留上述舊參數名稱作為 alias。

</details>

## 關鍵參數

| 參數 | 預設 | 作用 |
|---|---:|---|
| `--imgsz` | 1280 | YOLO 推論影像尺寸 |
| `--conf` | 0.18 | 偵測置信度門檻 |
| `--iou` | 0.55 | NMS IoU 門檻 |
| `--classes` | 2,3,5,7,9 | 保留的交通工具類別 ID |
| `--prefetch-frames` | 256 | CPU 預取幀數 |
| `--cpu-threads` | 20 | OpenCV／前處理可用執行緒 |
| `--tl-min-prob` | 0.55 | 號誌 CNN 最低接受機率 |
| `--device 0 --half` | GPU 0, FP16 | NVIDIA GPU 推論設定 |

`--max-infer-fps 0` 代表不主動限速；`--no-gpu-guard` 關閉溫度／使用率節流。這些參數是實驗預設，不代表任何電腦都應盲目複製；顯存不足時可降低 `--imgsz` 或 `--prefetch-frames`。

## 輸出

CSV 至少包含：

```text
frame, track_id, x1, y1, x2, y2, conf, cls,
tl_state, tl_prob_green, tl_prob_red
```

同時會產生參數快照 `*.run.jsonl`；若啟用 quality check，會另存 `*.quality.json`。完整交通語意與風險推論還需軌跡特徵建立與後續模型。

## 權重與環境

- 輕量號誌 CNN 權重已放在 `models/traffic_light_classifier.pt`。
- YOLO 大型權重與 TensorRT engine 未納入 Git，請以 `--model` 指定。
- 建議 Ubuntu 22.04、Python 3.10、與本機 CUDA 相容的 PyTorch；先執行 `traffic-risk doctor` 檢查環境。
