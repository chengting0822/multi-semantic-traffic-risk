# Demo 指南

[回到首頁](../README.md) · [偵測與追蹤](detection.md) · [等價驗證](parity.md)

## 四個展示案例

| Case | 來源影片 | 主要情境 | YouTube | Release 結果影片 |
|---:|---:|---|---|---|
| 1 | 14.mp4 | 停等紅燈後逆向行駛並闖紅燈 | [觀看](https://youtu.be/NUPHokZSsbk) | `case01-wrong-way-red-light-violation.mp4` |
| 2 | 76.mp4 | 高車流情境下蛇行穿梭 | [觀看](https://youtu.be/SLkpqrQS3C8) | `case02-weaving-dense-traffic.mp4` |
| 3 | 96.mp4 | 逆向行駛且嚴重超速 | [觀看](https://youtu.be/vVxrHDQVQmU) | `case03-severe-speeding-wrong-way.mp4` |
| 4 | 115.mp4 | 逆向、超速與闖紅燈的複合違規 | [觀看](https://youtu.be/iLS613jvpMI) | `case04-red-light-wrong-way-combined.mp4` |

原始與結果影片放在 GitHub `demo-v1` Release；檔名、大小與 SHA-256 記錄在 [`examples/demo/cases.json`](../examples/demo/cases.json)。

## 執行內建風險 Demo

```bash
./scripts/run_demo.sh --device cpu
```

有 NVIDIA GPU 且 PyTorch/CUDA 環境正確時：

```bash
./scripts/run_demo.sh --device cuda
```

這個 Demo 會執行：

1. 讀取四個案例擷取的場景 tensor、scene windows、vehicle tokens 與互動特徵。
2. 載入凍結的場景權重。
3. 執行場景神經模型與論文選定的 Hybrid 融合。
4. 將每列 `risk_level` 與正式結果比對。
5. 輸出 `outputs/demo_scene_risk.csv`。

成功執行應顯示：

```text
verified_rows=183 mismatches=0
```

## 這個 Demo 做了什麼、沒做什麼？

**它確實會執行**凍結的場景模型和 Hybrid 決策，不是寫死的結果列表。

為了讓 repository 能夠快速下載與在 CPU 測試，內建 Demo 從「已準備好的場景特徵」開始，不會在每次執行時重跑整段 YOLO 影片偵測。如需從影片開始，請先使用 [`run_detection.sh`](../scripts/run_detection.sh)，再依 [資料介面](data.md) 建立後續特徵。

## 為什麼影片放 Release？

MP4 寫入 Git commit 後，即使後來刪除，仍可能長期留在 Git 歷史並擴大每次 clone。因此：

- Git repository：程式、設定、模型、縮圖與小型可執行資料。
- GitHub Release：四部原始影片、四部結果影片及模擬對照影片。
- 不公開：完整訓練影片、大型逐幀 CSV、cache 與歷史實驗輸出。
