# 資料介面

公開 repository 不包含原始影片與大型訓練資料。下列介面用來區分「資料本身」與「可公開的推論程式」。

## 1. 追蹤輸出

`traffic-risk detect` 產生逐幀 CSV，核心欄位為：

```text
frame, track_id, x1, y1, x2, y2, conf, cls,
tl_state, tl_prob_green, tl_prob_red
```

道路 ROI、車道、停止線與 IPM 設定置於 `configs/`。

## 2. 單車 GRU 輸入

每列是一個車輛時間窗，必須有：

```text
case_key, video_id, track_id, ts_window_idx
```

以及 `models/single_vehicle_normalization.json` 中列出的 16 個語意特徵。正式後處理另外需要 `r46_risk3_policy`、`r46_risk2_policy`、`computed_speed_kmh_p95` 與 `speed_smoothed_kmh_p95` 等五個 side-channel 欄位。推論輸出 `prob_class0..2`、`high_risk_prob`、`critical_prob`、`pred_gru_raw` 與 `pred_v37e`。

`pred_gru_raw` 是純 GRU argmax；`pred_v37e` 加上正式軌跡級抑噪與超速語意 rescue；論文最終結果還需要後續交通語意規則鏈，不能把其中任一中間欄位冒充最終 pipeline 結果。

## 3. 單車規則鏈輸入

`apply-single-policies` 以 window table 加上下列 prepared sidecars 執行：

- source features：速度、號誌、軌跡與規則證據。
- trajectory features：逆向、蛇行、路線偏移與轉彎情境。
- lane / lane-family features：車道歸屬、方向、切換與尾端穩定性。
- red-light-zone features：紅燈期間的停止線區域與前進量。
- tail-lane / wrong-way-tail features：短期尾端車道與逆向證據。
- double-yellow features：雙黃線接觸與跨越證據。

最終 `risk_level` 採 Schema C：`0 = 無風險`、`1 = 中風險`、`2 = 高風險`，並在同一條軌跡內做累積不降級。

## 4. 場景模型輸入

NPZ 的主要 tensor：

```text
token_features   N × 16 × 14
token_mask       N × 16
global_features  N × 59
```

每個場景窗最多 16 台車，每車 14 維；場景與互動資訊為 59 維。模型將每台車編碼成 64 維，Max/Mean Pooling 後得到 128 維，與場景分支 64 維拼接為 192 維，再融合為 96 維。

## 為什麼不附完整資料集

- 原始影片與逐幀軌跡容量很大。
- 部分影像的公開授權需另行確認。
- 交通影像可能涉及車牌、地點與隱私資訊。
- 作品集的重點是方法、核心實作、凍結模型與可驗證結果，而不是把私有研究工作目錄整包公開。
