# 正式輸出等價驗證

精簡版不是依照流程圖重新實作，而是先鎖定正式 checkpoint 與程式，再以原始正式輸出逐列驗證。

## 已通過項目

| 元件 | 比對資料量 | 結果 |
|---|---:|---|
| IPM 超速語意分支，video 115 | 110 vehicle windows | 全部語意欄、可靠度與 sidecar 差異 0 |
| 闖紅燈語意分支，video 115 | 110 vehicle windows | 全部事件欄、停止線證據與 sidecar 差異 0 |
| 異常軌跡與三分支 v0 融合，video 115 | 110 vehicle windows | 與研究期正式融合表共通欄位逐列一致（容許 CSV 浮點誤差 `1e-6`） |
| 單一車輛 causal GRU + 正式 track postprocess，validation | 13,892 windows | raw/final 類別差異皆 0；機率最大絕對差約 3.99e-6 |
| 單一車輛 causal GRU + 正式 track postprocess，test | 12,384 windows | raw/final 類別差異皆 0；機率最大絕對差約 2.68e-6 |
| 單一車輛完整語意規則鏈 | 26,276 windows | current risk 差異 0；final cumulative risk 差異 0 |
| 場景神經模型 | 16,446 scene windows | 四組預測類別差異皆 0；機率最大絕對差約 4.77e-7 |
| 場景 hybrid 融合 | 16,446 scene windows、176 欄 | 所有 prediction/reason 欄一致；正式選定輸出差異 0 |
| `predict-scene` 公開 CLI | 16,446 scene windows | 讀檔、模型、hybrid 與寫檔完整執行；`risk_level` 差異 0 |

單一車輛最終風險分布亦一致：risk 0 = 24,646、risk 1 = 709、risk 2 = 921。

## 汽車類別跳動與歷史表的差異

研究期的中間表曾逐幀只保留 `cls=2`；現在為避免同一輛汽車短暫被 YOLO 判成其他類別而斷軌，先以整條 `track_id` 的過半數類別判定是否為汽車，再保留該 ID 的全部幀。這是刻意的輸入政策變更，因此含類別跳動的舊表不能再要求逐列完全相同。以現存的四支展示影片追蹤 CSV 重跑三分支 v0 融合後：

| 影片 | 新版／研究期視窗數 | 差異原因 |
|---|---:|---|
| 14 | 406／406 | 汽車 ID 10、12 有少量其他類別幀；視窗 key 相同，部分數值因補回這些幀而改變。 |
| 76 | 163／164 | ID 12 的 4 幀類別跳動已補回；ID 51 實際為 176 幀機車、僅 3 幀誤判汽車，新版不再納入。 |
| 96 | 175／174 | 汽車 ID 8、30 有其他類別幀；補回後 ID 30 多形成一個尾端視窗。 |
| 115 | 110／110 | 沒有汽車類別跳動，對齊結果一致。 |

這些比較只涵蓋「追蹤 CSV → 15 維 v0 融合」階段；不能用來宣稱從任意 MP4 到最終風險已完成等價驗證。v8 融合與正式 C4O 特徵仍待接通。

## 重跑驗證

這項驗證不需要把大型資料複製進精簡 repository，但執行環境必須還能存取原始研究目錄：

```bash
python scripts/verify_original_parity.py \
  --original-root /path/to/original-project \
  --check all
```

`--check` 也可單獨指定 `single-model`、`single-policy`、`scene-model` 或 `scene-hybrid`。驗證器會重跑精簡版程式，再以 case、影片、軌跡與時間窗 key 對齊正式輸出，不是只比對總體準確率。

## 驗證邊界

- 超速分支從原始 timestamp tracking rows 重算；其餘模型項目驗證論文凍結模型與正式 prepared tensors/features 的推論等價性。
- 影片解碼、YOLO 權重版本、GPU/TensorRT、OpenCV 與追蹤器版本可能影響最前端偵測結果；因此前端另外保留參數快照與模型雜湊。
- 浮點機率允許 `1e-5` 以內誤差，但離散預測要求完全相同。
- 私有資料集不會提交到 GitHub；完整等價測試需在原始研究資料仍可存取的環境執行。
