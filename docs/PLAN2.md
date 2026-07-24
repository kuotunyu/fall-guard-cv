# PLAN2：Phase 5-7 後續增強實作藍圖 ✅ 全部完成（2026-07-24）

> **定位**：主線 Phase 0-4 已完成並發布（開發藍圖與全部決策記錄見 [PLAN.md](PLAN.md)）。本檔規劃三項後續增強；**Phase 5、6、7 已於 2026-07-24 全部完成**，本檔現在是歷史記錄。Phase 7 由使用者授權在其離開電腦期間自行完成到底（唯一限制：不執行 git push，避免 GitHub Contributors 出現非本人的紀錄）。
> **Decision Log 仍統一記在 [PLAN.md](PLAN.md) 第 2 章**（append-only 慣例不變），本檔不另設決策表；D40-D46 記錄了三個 Phase 的完整過程。

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

## Phase 6：VLM 描述品質對照 ✅ 已完成（2026-07-23）

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
4. [x] 使用者確認成本後執行 `uv run python scripts/compare_vlm.py --yes`：24 次呼叫（12 張圖 × 2 模型）全數成功，無一次被安全過濾擋下。
5. [x] 閱讀併排結果、寫結論 → README 表3 備援欄「本專案主線未呼叫」改為實測結論 + 指向 `docs/results/vlm_comparison.md`。**中途發現隱私問題並修正**：初版把逐圖描述直接寫進要公開的檔案，使用者明確表示「在自己家裡跌倒測試」這部分不想公開——雖然這件事本身已在 D24 抽象揭露過，但逐圖敘事式描述暴露的細節程度不同。修正：拆成 `vlm_comparison_detail.md`（逐圖完整內容，加進 `.gitignore`，只留本機）+ `vlm_comparison.md`（公開版，只留自動統計數字 + 不含具體居家細節的彙總分析）；`compare_vlm.py` 本身也同步修改，讓這個拆分成為腳本預設行為而非每次手動處理，見 D44。
6. [x] **刻意不做 LLM-as-judge**：`OPENAI_MODEL` 本身是受測者，不能兼任裁判；樣本只有 12 張，人工判讀比自動評分更可信。此取捨已記入 D42。
7. [x] `tests/test_vlm.py` 補 2 項測試：帶參數呼叫時模型字串/供應商有正確傳遞、不帶參數呼叫方式不變（monkeypatch 假物件，不真呼叫 API）。

**附帶決定**：實作過程中使用者提議把 `GEMINI_MODEL` 預設值從 `gemini-3.1-flash-lite` 升級到 `gemini-3.5-flash-lite`（WebFetch 查證官方模型清單確認兩者皆為 Stable），已同步更新 `.env`/`.env.example`/`config.py`/README/PLAN.md（見 D43）。

### DoD

- [x] `uv run python scripts/compare_vlm.py --yes` 產出兩份檔案：`vlm_comparison_detail.md`（本機限定，24 格描述無缺漏）+ `vlm_comparison.md`（公開彙總版）
- [x] `describe_scene()` 無參數呼叫行為與現行完全一致（既有 pytest 5 項不改仍全綠，`uv run pytest -q` 53 passed）
- [x] README 表3 備援欄更新；`check_public_text.py` 全綠；`git status` 確認 `vlm_comparison_detail.md` 不在追蹤範圍內（D44）

### 風險

- OpenAI 對「人倒地」影像的安全過濾行為未知 → 沿用 vlm.py 既有 fallback 設計（失敗回傳固定文字、不拋例外）；若真被擋，「會被安全過濾」本身就是有價值的對照發現，照實寫進 vlm_comparison.md

---

## Phase 7：Le2i 跨資料集泛化（效益最大）✅ 已完成（2026-07-24）

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

### 實作步驟（依序，全部完成）

1. [x] **下載與盤點**：實際版面跟計畫階段的網路二手資訊有落差，**依實測結果重新定案，不照原假設硬做**（詳見 docs/PLAN.md D45）：
   - 6 個場景資料夾（Coffee_room_01/02、Home_01/02、Office、Lecture_room），只有前 4 個有標註檔案可驗證，**Office/Lecture_room 完全沒有任何標註、整批排除**——不編造 ground truth
   - 4 個有標註的資料夾合計 130 段（不是原估的 ~191）：127 段跌倒、3 段無跌倒
   - 標註格式：有跌倒的檔案前兩行是純數字（起訖幀）；無跌倒的檔案沒有這兩行、直接是逐幀資料——這個「省略表頭」的細節官方 README 完全沒提
   - 舊版 .avi 容器的 cv2 幀數常態性少報 1，改以 YOLO 實際解碼幀數為對齊基準
2. [x] **前處理**：新腳本 `scripts/prepare_le2i.py`：
   - 影片清單用 glob 探索
   - npz schema 與 URFD 相同，輸出到 `data/processed_le2i/`
   - pose 抽取邏輯抽成 `src/fallguard/pose.py::extract_video_pose()` 共用函式，`prepare_data.py` 也改呼叫這份，並用重抽 fall-01 + 逐欄比對確認**改動前後 100% 位元級相同**，證實重構未影響 URFD
   - 130/130 成功、耗時 434s、平均偵測率 96.4%（最低 42.7%）
3. [x] **評估**：`scripts/evaluate.py` 加 `--protocol cross`：`run_fold()` 完全不用改（本來就是協定無關的通用函式，證實了規劃階段的判斷），只需要 `load_all_videos()` 加目錄參數 + 新增 `run_cross_evaluation()`/`write_cross_report()`。**過程中撞見一個從未在 URFD 上出現過的真 bug 並修正**：`rules.py::window_score()` 對「整段視窗無偵測」回傳 `float("-inf")`，Le2i 較嚴苛的偵測條件第一次觸發這個分支，導致 sklearn PR-AUC 計算崩潰；改成有限值哨兵常數 `NO_DETECTION_SCORE = -1e6`，`diff` 確認 URFD 既有結果完全不受影響（見 docs/PLAN.md D46）
4. [x] **README**：評估結果新增「跨資料集泛化」小節（數字表 + 根因假說，不只是丟數字）；資料集與授權補上 Le2i 正式引用（Charfi et al. 2013, JEI）與授權狀態（Kaggle 鏡像 license 欄位 Unknown，依資料集要求引用來源論文）
5. [x] **測試**：`tests/test_prepare_le2i.py` 7 項（3 項合成標註檔測試不需要真的下載資料、4 項 schema/label 測試在資料存在時才跑）；新增 `tests/conftest.py` 讓測試能 import `scripts/` 內的模組

### DoD

- [x] `uv run python scripts/evaluate.py --model rule --protocol cross` 跑通、產出 `docs/results/cross_dataset.md`
- [x] README 泛化小節有實際數字與討論（含根因假說）
- [x] `uv run pytest -q` 全綠 → **60 passed**
- [x] URFD 既有數字完全不受影響 → `rule_baseline.md` 在 pose.py 重構前後、rules.py 修正前後皆 `diff` 為零

### 實際結果

| 指標 | 數值 | 95% CI |
|---|---|---|
| Sensitivity | 0.559 | [0.47, 0.64] |
| Specificity | 0.000 | [0.00, 0.56] |
| FP/小時 | 110.5 | — |

Sensitivity 掉到跟 URFD 自己最差兩折差不多的量級，可接受；Specificity 崩到 0，但只有 3 段 adl 樣本、CI 極寬。最可能的根因：URFD 折內調參後 `confirm_seconds` 只有 0.3 秒（因應 URFD 片段過短調出來的極端值），套到 Le2i 日常活動上門檻過低——不是模型能力不足，是評估用的極短確認秒數不該被誤用成跨資料集判定門檻，真實部署預設 10 秒敏感度低很多。

### 風險與對策（實際發生的部分）

| 風險 | 實際結果 |
|---|---|
| Kaggle 鏡像版面/標註格式與預期不符 | **真的發生**：原估 ~191 段可用，實際盤點只有 130 段有標註可驗證；已依實測結果重新定案，見上方步驟 1 |
| Le2i 解析度低（320×240），pose 品質差 | 平均偵測率 96.4%，比 URFD 的 90.1% 還高，比預期樂觀；最低單支 42.7% |
| 成績大幅下滑 | Specificity 真的崩到 0，照實報告 + 根因假說（見上方「實際結果」），**沒有在 Le2i 上調任何參數救數字** |
| pose 抽取時間 | 130 段實測 434s(~7 分鐘)，比估計的 13-15 分鐘快 |
| （未預期）評估時的 infinity 崩潰 | 計畫階段沒料到的真 bug，Le2i 的嚴苛偵測條件才第一次觸發，已修正（見上方步驟 3、docs/PLAN.md D46） |

---

## 附：完成後的收尾

三個 Phase 全部完成（2026-07-24）。是否合併回 PLAN.md 或保留獨立檔案，由使用者決定；目前先保留獨立檔案，PROGRESS.md 已記錄最終狀態。

**收尾附帶修正的 2 個小 bug**（過程中發現，跟三項增強本身無直接關係，但值得記錄）：
1. `describe_scene()` 加可選參數的第一版漏了「零參數呼叫要跟改動前完全一樣」，被既有測試當場抓到（Phase 6）
2. `prepare_data.py`/`prepare_le2i.py` 的批次抽取摘要統計，在「先用 `--limit` 小量測試、正式全量跑時部分影片被略過」的情境下會漏算——已修正成統一從磁碟重新統計（Phase 7）
