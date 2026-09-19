# 正式輸出等價驗證

精簡版不是依照流程圖重新實作，而是先鎖定正式 checkpoint 與程式，再以原始正式輸出逐列驗證。

## 已通過項目

| 元件 | 比對資料量 | 結果 |
|---|---:|---|
| 單車 causal GRU + 正式 track postprocess，validation | 13,892 windows | raw/final 類別差異皆 0；機率最大絕對差約 3.99e-6 |
| 單車 causal GRU + 正式 track postprocess，test | 12,384 windows | raw/final 類別差異皆 0；機率最大絕對差約 2.68e-6 |
| 單車完整語意規則鏈 | 26,276 windows | current risk 差異 0；final cumulative risk 差異 0 |
| 場景神經模型 | 16,446 scene windows | 四組預測類別差異皆 0；機率最大絕對差約 4.77e-7 |
| 場景 hybrid 融合 | 16,446 scene windows、176 欄 | 所有 prediction/reason 欄一致；正式選定輸出差異 0 |
| `predict-scene` 公開 CLI | 16,446 scene windows | 讀檔、模型、hybrid 與寫檔完整執行；`risk_level` 差異 0 |

單車最終風險分布亦一致：risk 0 = 24,646、risk 1 = 709、risk 2 = 921。

## 重跑驗證

這項驗證不需要把大型資料複製進精簡 repository，但執行環境必須還能存取原始研究目錄：

```bash
python scripts/verify_original_parity.py \
  --original-root /path/to/original-project \
  --check all
```

`--check` 也可單獨指定 `single-model`、`single-policy`、`scene-model` 或 `scene-hybrid`。驗證器會重跑精簡版程式，再以 case、影片、軌跡與時間窗 key 對齊正式輸出，不是只比對總體準確率。

## 驗證邊界

- 上表驗證的是論文凍結模型與正式 prepared tensors/features 的推論等價性。
- 影片解碼、YOLO 權重版本、GPU/TensorRT、OpenCV 與追蹤器版本可能影響最前端偵測結果；因此前端另外保留參數快照與模型雜湊。
- 浮點機率允許 `1e-5` 以內誤差，但離散預測要求完全相同。
- 私有資料集不會提交到 GitHub；完整等價測試需在原始研究資料仍可存取的環境執行。
