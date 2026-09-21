# 執行流程

本專案把研究期間分散的正式模型整理為可測試的 Python package。所有路徑均可由參數指定，不會讀取原始專案的絕對路徑。

## 完整資料流

```text
影片
  ↓ YOLO + ByteTrack + 號誌 CNN
逐幀追蹤 CSV
  ↓ 分批時間轉換 + 每車時間視窗
時間追蹤表 + reference windows
  ├─ 固定 20 點重採樣 + GMM + 車道幾何 → 異常軌跡語意
  ├─ IPM 底部中心投影 + 時間平滑 + 可靠度 → 超速語意
  └─ 號誌時序 + 停止線跨越 + 跨線後前進 → 闖紅燈語意
      ↓ 三分支 v0 中間表 → v8 政策輸出與融合 → S3.3 前綴重播（離線）
12 維 C4O 核心特徵 + 4 維車道／紅燈區特徵
  ↓ 車道、紅燈區、軌跡脈絡與尾端特徵重建
16 維單一車輛特徵
  ↓ Causal GRU + 完整語意政策鏈
單一車輛風險
  ↓ Top-16 車輛聚合 + 場景統計 + CPA/TTC 互動
59 維場景特徵
  ↓ Scene Token Pooling + Hybrid
整體交通風險
```

離線一鍵入口為 `traffic-risk run-video`；分段入口仍保留 `detect`、`prepare-tracks`、`build-semantics-v8`、`run-risk`，方便查核。四支 Demo 不是白名單。v0 表仍只是中間表，不能直接餵給 GRU；正式輸入由 v8 與 C4O 建立。此流程會讀取完整影片與軌跡，尚不是串流／即時推論版本。

```bash
traffic-risk run-video /path/to/video.mp4 outputs/my-video \
  --yolo-model /path/to/yolo26x.pt \
  --video-id intersection-a \
  --lane-map configs/lane_map.json \
  --double-yellow configs/double_yellow.json \
  --stop-lines configs/stop_lines.json \
  --ipm configs/ipm.json
```

若已有同一影片的逐幀偵測表，可用 `--tracks-csv /path/to/tracks.csv` 取代 `--yolo-model`。輸出在 `outputs/my-video/risk/traffic_risk.csv`。攝影機位置改變時，ROI、號誌 ROI、車道、停止線與 IPM 都需要重新標定；預設設定只對研究場景有效。

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

本研究只評估汽車。`reference_windows.csv` 只包含同一軌跡過半數偵測幀為 `cls=2` 的汽車 ID；同一 ID 若偶爾被誤判為其他類別，所有幀仍保留。平手或非汽車佔多數的軌跡不進入風險分析。號誌與非汽車列仍留在 timestamp CSV，供號誌分析及稽核。

YOLO 預設追蹤類別為 `2,3,5,7,9`，沿用研究期設定；類別跳動只有在偵測器有輸出且沿用同一 ID 時才能補回。如需涵蓋其他誤分類別，可用 `run-video --classes ...` 擴充偵測範圍，但會改變追蹤結果，不能當作與研究期完全相同的輸入。

### 2.0 一次建立三分支、v8 與 C4O

```bash
traffic-risk build-semantics-v8 \
  outputs/intersection-a/upstream/timestamps/intersection-a.csv \
  outputs/intersection-a/semantics \
  --lane-map configs/lane_map.json \
  --double-yellow configs/double_yellow.json \
  --trajectory-model models/trajectory_gmm_b5.json \
  --ipm configs/ipm.json \
  --stop-lines configs/stop_lines.json
```

輸出 `trajectory/`、`overspeed/`、`redlight/`、`fusion/` 的 v0 表，以及 `v8/` 的三分支政策表、融合表、S3.3 前綴與 `c4o_features.csv`。此命令可用於任意影片 ID，幾何設定仍須與攝影機視角相符。只需要舊版 15 維稽核表時可使用 `build-semantics-v0`。

若要分步檢查，可使用下面各分支指令。

### 2.1 建立超速語意

```bash
traffic-risk build-overspeed \
  outputs/intersection-a/upstream/timestamps/intersection-a.csv \
  outputs/intersection-a/upstream/reference_windows.csv \
  outputs/intersection-a/upstream/overspeed \
  --ipm configs/ipm.json
```

這是研究期接受的公式：車輛框底部中心經 IPM 轉為世界座標，依時間差計算 km/h，再做 rolling median、time-aware EMA、視窗 p95、可靠度與超速等級判定。輸出 `overspeed_features.csv` 與 `overspeed_sidecar.csv`。IPM 與影片必須對應同一相機視角。

### 2.2 建立闖紅燈語意

```bash
traffic-risk build-redlight \
  outputs/intersection-a/upstream/timestamps/intersection-a.csv \
  outputs/intersection-a/upstream/reference_windows.csv \
  outputs/intersection-a/upstream/redlight \
  --stop-lines configs/stop_lines.json
```

此分支對齊每幀號誌狀態與車輛軌跡，檢查停止線跨越、跨線時是否紅燈，以及跨線後是否持續前進。輸出 `redlight_features.csv` 與 `redlight_sidecar.csv`。停止線設定必須對應影片的相機視角。

### 2.3 建立異常軌跡語意

```bash
traffic-risk build-trajectory \
  outputs/intersection-a/upstream/timestamps/intersection-a.csv \
  outputs/intersection-a/upstream/trajectory \
  --lane-map configs/lane_map.json \
  --double-yellow configs/double_yellow.json
```

此分支分批讀取追蹤 CSV，以固定 20 點重採樣、凍結 GMM、因果 prefix 與正式 v8 子類型公式建立 `trajectory_features.csv` 和 `trajectory_sidecar.csv`。如需手動執行 v0 融合，三分支必須使用同一批汽車時間窗，再執行 `traffic-risk build-fusion-v0 TRAJECTORY_SIDECAR OVERSPEED_FEATURES REDLIGHT_FEATURES OUTPUT_DIR`；缺少對應時間窗時會報錯，不會補零。

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
