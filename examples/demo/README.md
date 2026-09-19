# Four-case runnable demo

這個目錄放的不是虛構資料，而是從論文正式場景資料集中擷取的四個展示案例：

| Case | 原始 video ID | 事件 |
|---:|---:|---|
| 1 | 14 | 等待後逆向闖紅燈 |
| 2 | 76 | 壅塞車流蛇行 |
| 3 | 96 | 嚴重超速與逆向 |
| 4 | 115 | 闖紅燈與逆向複合事件 |

## 執行

在 repository 根目錄執行：

```bash
./scripts/run_demo.sh --device cpu
```

程式會重新執行凍結場景模型與正式 Hybrid 融合，寫出 `outputs/demo_scene_risk.csv`，再對 183 個有正式標籤的時間窗逐列驗證。

## 檔案

- `scene_features.npz`：186 個場景時間窗的模型 tensor。
- `scene_windows.csv`：場景層統計與單一車輛風險下限。
- `scene_tokens.csv`：848 筆車輛 token 與語意來源。
- `interaction_features.csv`：車對車距離、CPA、TTC 與品質特徵。
- `expected_scene_risk.csv`：原始正式 pipeline 的選定 Hybrid 輸出，僅用於驗證。
- `cases.json`：案例名稱、video ID 與 GitHub Release 影片檔名。

## 邊界

這個範例驗證的是「準備好的車輛／場景特徵 → 場景風險」正式流程。原始 MP4 和加上辨識結果的 MP4 會放在 `demo-v1` GitHub Release，不直接寫入 Git 歷史。從原始影片重新執行 YOLO 時，仍需另行提供當時使用的 YOLO 權重。
