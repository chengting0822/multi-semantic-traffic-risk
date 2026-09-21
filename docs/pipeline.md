# 執行流程

本專案把研究期間分散的正式模型整理為可測試的 Python package。所有路徑均可由參數指定，不會讀取原始專案的絕對路徑。

## 目前可直接執行的部分

```text
影片
  ↓ YOLO + ByteTrack + 號誌 CNN
逐幀追蹤 CSV
  ↓ 分批時間轉換 + 每車時間視窗
時間追蹤表 + reference windows

C4O 上游時間窗 + 軌跡語意時間窗
  ↓ 車道、紅燈區、軌跡脈絡與尾端特徵重建
16 維單一車輛特徵
  ↓ Causal GRU + 完整語意政策鏈
單一車輛風險
  ↓ Top-16 車輛聚合 + 場景統計 + CPA/TTC 互動
59 維場景特徵
  ↓ Scene Token Pooling + Hybrid
整體交通風險
```

偵測由 `traffic-risk detect` 執行，時間轉換由 `traffic-risk prepare-tracks` 執行，後半段由 `traffic-risk run-risk` 執行。四支 Demo 只是可重現範例，三個命令的輸入路徑與影片 ID 均可替換。「時間視窗 → C4O／軌跡語意」仍在去除歷史依賴，現階段保留成明確邊界，避免把舊基準版誤稱為論文正式版。

## 安裝與影片

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[detection]'
python scripts/download_demo_videos.py
```

影片會放在 `data/demo_videos/`，不會提交到 Git history。

## 1. 車輛偵測與追蹤

```bash
traffic-risk detect -- \
  data/demo_videos/14.mp4 \
  --yolo-model /path/to/yolo26x.pt \
  --output outputs/14/tracks.csv
```

完整參數見 [偵測與追蹤](detection.md)。

## 2. 建立時間追蹤表與每車視窗

```bash
traffic-risk prepare-tracks \
  outputs/intersection-a/tracks.csv \
  /path/to/intersection-a.mp4 \
  outputs/intersection-a/upstream \
  --video-id intersection-a
```

輸出 `timestamps/intersection-a.csv` 與 `reference_windows.csv`。讀取預設每批 250,000 列，可用 `--chunksize` 調整；影片 ID 可為 `14` 之類數字，也可為 `intersection-a` 之類文字。

## 3. 完整風險推論

準備下列當次執行產物：

- `c4o_features.csv`：每台車每個時間窗的多語意融合特徵。
- `trajectory_features.csv`：異常軌跡與道路幾何 sidecar。
- `tracks/14.csv`：逐幀追蹤表；目錄內檔名使用影片 ID。

執行：

```bash
traffic-risk run-risk \
  outputs/upstream/c4o_features.csv \
  outputs/upstream/trajectory_features.csv \
  outputs/tracks \
  outputs/case14 \
  --annotations annotations \
  --device auto
```

主要輸出：

```text
outputs/case14/
├── manifest.json
├── traffic_risk.csv
├── single_vehicle/
│   ├── single_vehicle_learned_windows.csv
│   ├── single_vehicle_risk.csv
│   └── semantic_features/
└── scene/
    ├── scene_features.npz
    ├── scene_windows.csv
    ├── scene_tokens.csv
    ├── interaction_features.csv
    └── scene_risk.csv
```

## 4. 時間標註

```bash
traffic-risk annotate 14 --tracks outputs/14/tracks.csv
```

操作方式：

- `Space`：播放／暫停。
- `←`、`→`：前後一幀。
- `[`：將目前畫面設為區間起點。
- `0`～`3`：以目前畫面為終點，加入對應風險區間。
- `Ctrl+S`：儲存。

工具支援「整體場景」與「指定車輛」兩層標註。`14`、`76`、`96`、`115` 會從 Demo 設定取得預設路徑；其他 ID 傳入 `--video`、可選的 `--tracks` 與 `--output` 即可：

```bash
traffic-risk annotate intersection-a --video /path/to/intersection-a.mp4
```

JSON 使用研究原本的 `timestamp_fixed20.compact.v1` 結構。

## GPU 與 CPU

- YOLO 偵測建議使用 NVIDIA GPU。
- 單一車輛與場景模型可使用 `--device cpu` 驗證，或以 `--device auto` 自動使用 CUDA。
- 這是離線批次流程，不宣稱為串流即時版本；推論時間應分別報告偵測、語意建置與風險模型階段。
