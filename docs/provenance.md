# 正式版本來源與取捨

本文件記錄精簡專案如何從原始工作目錄選出正式內容。選擇依據是論文評估文件、正式 entrypoint、checkpoint manifest 與逐列輸出比對，不是檔案名稱中的版本號或最後修改時間。

## 採用來源

### 影像前端

- 正式來源：`/home/chengting/project0225/yolotest.py`
- 同步相依：`trajectory_utils.py`、`check_tracks_csv.py`、`scene_config.json`、`trackers/bytetrack_b7_tune_a.yaml`、`traffic_light.pt`
- 精簡後位置：`src/traffic_risk/detection/`
- 偵測器大權重不納入 Git；執行時以 `--yolo-model` 指定。

### 單一車輛風險

- 正式公開包裝層：`v93_clean_production_pipeline_20260622`
- 已驗證核心：`v92_clean_root_cause_modules_20260621`
- 凍結模型：original-split、fold 00、`formal_focal_sqrt_gamma1_gru64_policyguided_seed42_best.pt`
- 架構：16 維輸入、單層 causal GRU、hidden size 64、三級輸出。
- v94 只更動論文評估標籤邊界，不是新的推論模型，因此沒有當成 runtime 版本。

### 上游超速語意

- 公式來源：`overspeed_module_v0.py` 與 `ipm_speed_utils.py`。
- 精簡後位置：`src/traffic_risk/upstream/overspeed/`。
- 保留 IPM 投影、rolling median + time-aware EMA、p95 視窗速度、異常極速排除、可靠度與分級公式；移除舊 CLI 的工作區絕對路徑與報表程式。

### 上游闖紅燈語意

- 公式來源：`redlight_module_v0.py`、`redlight_geometry_utils.py` 與 `traffic_light_utils.py`。
- 精簡後位置：`src/traffic_risk/upstream/redlight/`。
- 保留號誌時序聚合、停止線符號距離、跨線事件、跨線後移動與證據分數；移除舊 CLI 的絕對路徑與報表程式。

### 場景風險

- 論文鎖定基線：`scene_v93_retrain_hybrid_select_v2_learned_high`
- 凍結模型：`scene_token_pool_v1_best_hybrid_selected.pt`
- 選定融合規則：`single_floor_severity_gate_strict_pure_overspeed_rescue_learned_interaction_track_sticky_cpa_onset_high_confidence`
- 架構：車輛 token 14 維、場景特徵 59 維、hidden 64、融合層 96、Max + Mean Pooling。

## 明確排除的歷史或實驗內容

- `online/run_full_mainline_streaming.py` 與 `online/single_vehicle_mainline_inference.py`：仍指向 2026-06-03 至 2026-06-05 的舊基線，且含失效的絕對路徑，只能作為舊串接行為參考。
- 場景模組舊 `CURRENT_BASELINE.md`：指向 6 月 4 日基線，晚於此文件的名稱不代表論文採用。
- `scene_pair_stateful_interaction_v93_v1`：研究實驗，不是論文正式基線。
- `scene_v93_retrain_hybrid_select_v3_train_hard_samples`：後續 hard-sample 實驗，不取代論文鎖定的 v2 learned-high。
- v94 label overlay：只用於評估標籤，不更動模型輸出。
- 所有 `runs/`、`reports/`、`cache/`、fold 重複 checkpoint、除錯輸出與大型 CSV：不納入作品集 repository。

## 相容性策略

公開函式使用 `predict-single`、`apply-single-policies`、`predict-scene-model` 等穩定名稱。部分內部欄位仍保留研究期版本前綴，因為它們是規則鏈的資料契約；任意改名會增加破壞逐列等價的風險。這些名稱不會被當成使用者必須理解的研究概念。
