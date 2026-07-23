# PLAN2：Phase 5-7 後續增強實作藍圖

> **定位**：主線 Phase 0-4 已完成並發布（開發藍圖與全部決策記錄見 [PLAN.md](PLAN.md)）。本檔規劃三項後續增強，實作時照本檔逐步執行；全部完成後，本檔即為 Phase 5-7 的歷史記錄。
> **Decision Log 仍統一記在 [PLAN.md](PLAN.md) 第 2 章**（append-only 慣例不變），本檔不另設決策表；每個 Phase 的關鍵取捨在完成時補一條 D40+ 條目。

---

## 0. 三項增強與執行順序

| Phase | 項目 | 為什麼值得做 | 預估工作量 |
|---|---|---|---|
| 5 | 評估數字加信賴區間 | P3-P5 折的測試集只有 6 段影片，現在報的 Sensitivity 全是點估計；不揭露變異幅度，懂統計的讀者會扣分 | 半天內 |
| 6 | VLM 描述品質對照 | README 表3 的備援欄寫著「本專案主線未呼叫」——一個沒被驗證過的設計決定；補上實測比較才算數 | 半天 |
| 7 | Le2i 跨資料集泛化 | **效益最大**。目前所有評估都在 URFD 單一資料集（同一種攝影機、同一間房間）上做，「換個場景還行不行」完全沒有答案；這是有經驗的讀者必問的第一個問題 | 1-2 天 |

**執行順序：Phase 5 → 6 → 7**，按外部依賴的摩擦力遞增排序——5 零依賴、6 需要 `OPENAI_API_KEY`、7 需要 Kaggle token 加上較長的 pose 抽取時間。三者彼此獨立，可跳做或並行。

**每個 Phase 完成時的共同收尾**（不重複寫在各 Phase 裡）：
1. PLAN.md Decision Log 補 D40+ 條目（含驗證證據）
2. PROGRESS.md 快速回憶區更新
3. `uv run python scripts/check_public_text.py <本次動到的公開檔>` 全綠
4. `uv run pytest -q` 全綠
5. Conventional Commit（正體中文訊息），git 操作由使用者本人執行

---

## Phase 5：評估數字的統計誠實度（信賴區間）✅ 已完成（2026-07-22）

### 目標

事件級指標的樣本量極小——P3/P4/P5 折的測試集各只有 6 段 fall 影片，報出來的 Sensitivity 0.83/0.67/0.50 每一個都是 n=6 的點估計。加上 Wilson score 95% 信賴區間，讓讀者直接看見「這些數字的變異區間本身就很寬」，比在文字裡籠統警告一句更誠實也更專業。

### 現況（2026-07-22 盤點）

- `scripts/evaluate.py` 全檔沒有任何信賴區間/變異數計算（grep 過 `bootstrap|confidence|interval` 零命中）
- 事件級表格由 `write_report()`（rule 路徑）產出到 `docs/results/rule_baseline.md`
- README「事件級指標」表格同步呈現這些數字

### 實作步驟（全部完成）

1. [x] 新增純函式 `wilson_interval(successes, n, z=1.96) -> tuple[float, float]`——**實作時放到 `src/fallguard/stats.py`（新檔）而非 `scripts/evaluate.py`**：`scripts/` 不是可安裝套件、既有測試從不直接 import `scripts/` 內的函式，放進 `src/fallguard/` 才能用專案既有的 `from fallguard.x import y` 模式正常單元測試，架構上更一致。手寫公式，不加新套件依賴，n≤0 時回傳 `(0.0, 1.0)`。
2. [x] `write_report()` 的兩個事件級表格（文獻預設、折內調參後）各加「Sensitivity 95% CI」「Specificity 95% CI」兩欄獨立呈現（未用併欄格式）；**視窗級 F1 表格不加 CI**——此取捨已記入 Decision Log D41。
3. [x] 重跑 `uv run python scripts/evaluate.py --model rule --protocol loso` 再生 `rule_baseline.md`——`diff` 驗收：既有 F1/Sensitivity/Specificity/延遲數字逐格一致，diff 為零，只新增 CI 欄與一段說明文字。
4. [x] README 事件級表格**不加欄**，改在表格下方加一句「樣本量提醒」白話段落，指向 `rule_baseline.md` 看完整 CI（見上方 DoD 的風險評估說明）。
5. [x] `tests/test_stats.py` 新增 8 項單元測試：0/6、6/6、5/6 已知值對照（[0.44, 0.97]，與人工試算一致）、n=0、以及 4 組邊界情境的 `0≤lo≤hi≤1` 通用性檢查。

### DoD

- [x] `uv run pytest -q` 全綠 → **51 passed**（新增 `tests/test_stats.py` 8 項）
- [x] `rule_baseline.md` 舊數字完全不變、只多 CI 欄 → `diff` 改版前後確認：F1/Sensitivity/Specificity/延遲數字逐格比對完全一致，只新增「Sensitivity 95% CI」「Specificity 95% CI」兩欄與一段說明文字
- [x] README 不擠版 → **實作時改變做法**：README 只加了一句「樣本量提醒」白話段落（指向 rule_baseline.md 看完整 CI），沒有加表格欄位，故不存在 D35 那種窄欄擠壓風險，不需要跑重量級截圖驗證；完整的 CI 欄位改放在 `docs/results/rule_baseline.md`（10 欄的細節表格，GitHub 表格 CSS 對超寬表格是加水平捲軸而非硬擠壓儲存格換行，風險剖面跟 README 的緊湊表格不同）

### 風險

實作結果：無意外風險。決定把詳細 CI 數字全放進 `docs/results/rule_baseline.md`、README 只留一句話指過去，從根本上避開了表格擠版風險，不需要用到 D35 的截圖驗證流程。

---

## Phase 6：VLM 描述品質對照

### 目標

README 表3 設計了「主力 `GEMINI_MODEL` / 備援 `OPENAI_MODEL`」雙模型分工，但備援那路從未被呼叫過，形同紙上設計。用 `events/` 現有的真實跌倒截圖對兩個模型各跑一輪描述，產出併排比較表，把這個設計決定變成有實測依據的結論。

### 現況（2026-07-22 盤點）

- `src/fallguard/vlm.py` 的 `_build_model()` 無參數、寫死 `settings.gemini_model` + `model_provider="google_genai"`（含放寬 DANGEROUS_CONTENT 的 safety_settings）
- `config.openai_model`（env `OPENAI_MODEL`，預設 `gpt-5-mini`）在整個 `src/` 從未被引用
- `events/` 有 12 張真實事件截圖（6 次事件 × impact/confirm 對），是現成素材

### 待使用者人工處理

- 確認 `.env` 已設 `OPENAI_API_KEY`（實作時只檢查鍵存在與否，不讀值）

### 實作步驟

1. [x] `src/fallguard/vlm.py`：拆出 `_describe_scene_raw()`（不吞例外的核心邏輯）+ `describe_scene()`（生產路徑包一層 try/except）；`_build_model()`/`describe_scene()` 加可選參數 `(model, provider)`，零參數呼叫維持跟改動前完全相同的呼叫方式——**`detect.py` 呼叫端零改動、既有測試不改仍全綠**（第一版實作漏了這點，被既有測試的 monkeypatch 當場抓到，已修正，見 D42）。google 專屬的 `safety_settings` 只在預設 Gemini 路徑帶入，OpenAI 路徑不帶。
2. [x] 新增 `langchain-openai>=1.3.5,<1.4.0` 依賴（鎖版本避免連帶升級 `langchain-core`，見 D42）。
3. [x] 新腳本 `scripts/compare_vlm.py`：
   - 迭代 `events/*.jpg`（12 張），對每張分別用 `GEMINI_MODEL` 與 `OPENAI_MODEL` 跑描述（直接呼叫 `_describe_scene_raw()` 取得真實失敗原因，比生產路徑的統一 FALLBACK_TEXT 更有診斷價值）
   - 輸出 `docs/results/vlm_comparison.md`：併排表 + 檔尾評比維度提示
   - **成本估算門檻改用 `--yes` 旗標**（不用原計畫的互動式 `input()`——這支腳本會由 AI 透過工具呼叫執行，無法回應互動提示，旗標更穩妥）：不帶 `--yes` 只印估算不執行
4. [ ] **待實際執行**（需使用者確認金額後加 `--yes` 跑）：`uv run python scripts/compare_vlm.py --yes`
5. [ ] 使用者人工閱讀併排表、寫幾行結論 → README 表3 備援欄的「本專案主線未呼叫」改為一句實測結論 + 指向 `vlm_comparison.md`。
6. [x] **刻意不做 LLM-as-judge**：`OPENAI_MODEL` 本身是受測者，不能兼任裁判；樣本只有 12 張，人工判讀比自動評分更可信。此取捨已記入 D42。
7. [x] `tests/test_vlm.py` 補 2 項測試：帶參數呼叫時模型字串/供應商有正確傳遞、不帶參數呼叫方式不變（monkeypatch 假物件，不真呼叫 API）。

**附帶決定**：實作過程中使用者提議把 `GEMINI_MODEL` 預設值從 `gemini-3.1-flash-lite` 升級到 `gemini-3.5-flash-lite`（WebFetch 查證官方模型清單確認兩者皆為 Stable），已同步更新 `.env`/`.env.example`/`config.py`/README/PLAN.md（見 D43）。

### DoD

- [ ] `uv run python scripts/compare_vlm.py --yes` 產出 `docs/results/vlm_comparison.md`，24 格描述無缺漏（被安全過濾者如實標記）
- [x] `describe_scene()` 無參數呼叫行為與現行完全一致（既有 pytest 5 項不改仍全綠，`uv run pytest -q` 53 passed）
- [ ] README 表3 備援欄更新（待步驟 4-5 實際執行完成後才動）；`check_public_text.py` 全綠

### 風險

- OpenAI 對「人倒地」影像的安全過濾行為未知 → 沿用 vlm.py 既有 fallback 設計（失敗回傳固定文字、不拋例外）；若真被擋，「會被安全過濾」本身就是有價值的對照發現，照實寫進 vlm_comparison.md

---

## Phase 7：Le2i 跨資料集泛化（效益最大）

### 目標

兌現 PLAN.md §7.1 早已定義的 P3 協定：「URFD 全量訓練 → Le2i 當純測試集（受試者天然不相交），只報事件級指標」。用一個完全沒看過的資料集（不同攝影機、不同房間、320×240/25fps）回答「換一個場景這套系統還行不行」。

### 現況（2026-07-22 盤點）

- `scripts/download_data.py` 的 `download_le2i()` 已寫好：Kaggle API 下載 `tuyenldvn/falldataset-imvia` → `data/raw/le2i/` 自動解壓；但對下載後的版面**零斷言**，實際目錄結構/標註格式從未驗證過
- `scripts/prepare_data.py` 四處 URFD 硬編碼：枚舉式影片清單（fall-01..30/adl-01..40）、檔名模板 `{vid}-cam0.mp4`、`urfall-cam0-{kind}s.csv` 標籤格式、D12 的 kind 覆寫規則；fps 處理與 npz schema 則是通用的
- `scripts/evaluate.py` 的 `run_fold()` 對任何含 train/test 清單的 fold dict 都能跑，但 `load_all_videos()` 只 glob `data/processed/*.npz`、`window_ground_truth()` 假設 URFD 標籤模型
- 特徵已做 25Hz 重採樣 fps 無關化（PLAN.md §7.3），Le2i 25fps vs URFD 30fps 的差異在設計上已被吸收

### 待使用者人工處理

- Kaggle 帳號 + API token 放 `~/.kaggle/kaggle.json`
- 執行 `uv sync --extra le2i` 與下載指令

### 實作步驟（依序）

1. **下載與盤點**：
   ```bash
   uv sync --extra le2i
   uv run python scripts/download_data.py --fallback le2i
   ```
   下載完成後**先盤點實際版面再往下做**：目錄結構、影片數（預期原始 ~221 段、含標註可用子集 ~191 段，見 PLAN.md §4）、標註檔格式（預期是「每段影片附 fall 起訖幀 + 逐幀 bbox」的 txt，但以實際盤點為準）。盤點結論記入 Decision Log；格式與預期不符時回報使用者選替代方案，不硬做。
2. **前處理**：新腳本 `scripts/prepare_le2i.py`（不動 `prepare_data.py` 的 URFD 路徑）：
   - 影片清單用 glob 探索（取代枚舉）
   - 解析 Le2i 標註 → 映射成與 URFD 相同的 npz schema（`xyn/conf/bbox_xywh/raw_label/label_present/fps/timestamps` + kind），輸出到 **`data/processed_le2i/`**（與 URFD 的 `processed/` 分開，避免 `load_all_videos()` 混撈）
   - pose 抽取邏輯（YOLO 載入、track 參數、npz 寫出）從 `prepare_data.py` 抽成共用函式或直接 import 重用——遵循 D18/D20「同一件事共用同一份邏輯」的教訓，不複製貼上第二份
   - 標籤語意規約（哪些幀算正例、無人幀怎麼標）在盤點後定案，記 Decision Log
   - 印每段影片的 pose 偵測率統計（沿用 URFD 慣例）；偵測率過低的影片誠實排除並記錄數量
3. **評估**：`scripts/evaluate.py` 加 `--protocol cross`：
   - train = 全部 70 段 URFD（規則式門檻與 FSM 時間參數用既有 `tune_thresholds`/`tune_fsm_timing` 在全 URFD 上調，機制不重寫）
   - test = 全部可用的 Le2i 影片
   - **只報事件級指標**（Sensitivity / Specificity / false alarms per hour，沿用 §7.1 P3 與 §7.2 定義），並套用 Phase 5 的 `wilson_interval`
   - 載入端：`load_all_videos()` 加目錄參數或另立函式讀 `processed_le2i/`
   - 輸出 `docs/results/cross_dataset.md`
4. **README**：
   - 評估結果章節新增「跨資料集泛化」小節：數字表 + 誠實討論（預期成績會下滑；下滑多少、可能原因——視角/解析度/場景差異）
   - 資料集與授權章節補 Le2i 的實際授權條款與正式引用（下載後從資料集附帶文件確認）
5. **測試**：Le2i 標註解析函式單元測試（合成標註檔 → 已知 label 序列）；依 `test_prepare_data.py` 慣例，`processed_le2i/` 不存在時 `pytest.skip` 而非失敗。

### DoD

- [ ] `uv run python scripts/evaluate.py --model rule --protocol cross` 跑通、產出 `docs/results/cross_dataset.md`
- [ ] README 泛化小節有實際數字與討論
- [ ] `uv run pytest -q` 全綠（含新標註解析測試）
- [ ] URFD 既有數字完全不受影響（`rule_baseline.md` diff 為零）

### 風險與對策

| 風險 | 對策 |
|---|---|
| Kaggle 鏡像版面/標註格式與預期不符 | 步驟 1 先盤點再定案；異常時回報使用者選替代方案（別的鏡像或縮小子集），不硬做 |
| Le2i 解析度低（320×240），pose 品質差 | prepare 時印逐段偵測率；低於門檻的影片誠實排除並記錄數量，不混入充數 |
| 成績大幅下滑 | 這本身就是有價值的發現——照實報告 + 歸因討論。**絕不在 Le2i 上調任何參數救數字**，那會污染「純測試集」的定位 |
| pose 抽取時間 | URFD 70 段實測 287s → Le2i ~191 段估 13-15 分鐘（RTX 4090），可接受 |

---

## 附：完成後的收尾

三個 Phase 全部完成後：本檔開頭的定位說明改為「已全部完成」，PLAN.md §13 的文件清單補上本檔連結，PROGRESS.md 記錄最終狀態。是否合併回 PLAN.md 或保留獨立檔案，屆時由使用者決定。
