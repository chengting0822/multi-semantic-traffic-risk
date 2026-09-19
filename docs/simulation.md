# 模擬資料：OpenStreetMap → SUMO → CARLA

[回到首頁](../README.md) · [研究方法](methodology.md) · [Demo 指南](demo.md)

模擬環境建置於 **Ubuntu 22.04**。目的是在可重複的條件下取得多車流與高風險行為，並保留道路幾何、號誌與車道語意。

```text
OpenStreetMap            SUMO                         CARLA
道路幾何與路網   →   背景交通流與路徑   →   視覺場景、感測器與影片
                                                   └→ 人工操控高風險車輛
```

## 建置流程

1. 從 OpenStreetMap 取得道路形狀、路口與連接關係。
2. 將路網轉換到 SUMO，用來產生一般車流、行駛路徑與背景交通。
3. 將場景匯入 CARLA，配置攝影機視角、號誌、車輛與影像輸出。
4. 對一般交通流不易自然產生的高風險行為，研究人員使用 CARLA 內建鍵盤駕駛功能操控指定車輛。

## 高風險行為如何產生？

人工操控的目的不是隨意駕駛，而是有意識地重現常見與不常見事件，包含逆向、蛇行、超速、闖紅燈與多種行為同時出現的複合情境。每段影片仍會經過影像偵測、追蹤與後續風險流程，而不是直接將操作指令當成模型答案。

| 真實道路參考 | CARLA 模擬影像 |
|---|---|
| ![Real road](../assets/simulation/real-road-reference.jpg) | ![CARLA simulation](../assets/simulation/carla-road-simulation.jpg) |

完整影片放在 GitHub `simulation-v1` Release，不寫入 Git 歷史：

- [真實道路參考影片](../../releases/download/simulation-v1/real-road-reference.mp4)
- [CARLA 模擬影片](../../releases/download/simulation-v1/carla-road-simulation.mp4)

## 資料使用原則與限制

- 模擬資料可精確控制事件，但不等於真實交通分布。
- 攝影機、材質、光影與車流模型會造成模擬到真實的領域差異。
- 高風險樣本由研究人員設計，適合測試特定語意，不用來聲稱事故發生率。
- 公開 repository 只放示範影片與小型可執行資料，不放完整訓練資料。
