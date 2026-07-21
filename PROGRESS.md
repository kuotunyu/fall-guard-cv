# fall-guard-cv 進度追蹤

<!-- 使用規則：
  🧭 快速回憶區：直接改寫、不留歷史、全區 ≤30 行。每次收工前更新（用 /update-progress skill）。
  📜 Phase 日誌：只增不改。每 Phase 收尾寫一條，附驗證證據（指令+輸出數字）與 git tag。
  回來時的 10 分鐘流程：讀快速回憶區 → git log --oneline -10 → uv run pytest -q → 讀 docs/PLAN.md 當前 Phase DoD。
  （或直接用 /resume-context skill 自動跑完上述流程。） -->

## 🧭 快速回憶區

**上次收工日期**：2026-07-21
**現在做到哪**：**Phase 2 全部完成**：features.py/fsm.py/rules.py/evaluate.py 已由 AI commit 進版控（tag phase-1 已打）；**最後一批（error_analysis.py + README 回填 + PLAN.md/PROGRESS.md 收尾文件）尚未 commit，改由使用者自行操作** git（本輪起使用者要求 git 操作自己來）。`uv run pytest -q` 25 passed。
**下一步（開機第一件事）**：使用者先自行 `git add`/`git commit`/`git tag phase-2` 這次的異動（error_analysis.py、README.md、docs/PLAN.md、PROGRESS.md、pyproject.toml/uv.lock 的 matplotlib、docs/results/error_analysis.md、docs/assets/error_analysis_triplet.png）；之後開 Phase 3——npz 打包上傳 Colab、寫 `notebooks/train_colab.ipynb`（XGBoost 必做，窗口統計特徵），動工前先查 Context7/官方文件確認 xgboost 現行 API。
**未決問題**：站姿/坐姿跌倒的逐段對照表找不到官方來源，README/評估結果已誠實註記從缺（見 PLAN.md §7.2）。
**待使用者人工處理**：(1) 建立 Discord Webhook 並填入 .env 的 `DISCORD_WEBHOOK_URL`（Phase 4 前完成即可）。(2) **本輪（誤報分析+README+收尾）的 git commit/tag 由使用者自行執行**——建議 commit 訊息與分批方式見對話紀錄底部建議。
**已知坑**：torch 必須走 cu128 index（已寫進 pyproject，重建環境直接 `uv sync` 即可）。clone 後要跑一次 `git config core.hooksPath .githooks` 守門才會生效。公開文件裡不要寫出「磁碟機:\Users」的字面路徑示例——會被自家 hook 擋下，用文字描述代替。ultralytics `half=True` 已棄用，一律用 `quantize=16`（D14）。ultralytics 下載的 `.pt` 權重會先落在 CWD，`prepare_data.py` 已自動搬進 `models/pretrained/`（已 gitignore）。**cv2 GUI 視窗失焦時按鍵會被 OS 排隊，恢復焦點瞬間可能暴衝連續觸發**——`annotate_urfd.py` 已加 400ms 按鍵沖刷防護。**LOSO 的 P3/P4/P5 折沒有 ADL 測試樣本**（D15，evaluate.py/README 已處理成 N/A 不硬平均）。**任何狀態機/計時邏輯寫 NaN 防呆時，純時間判斷(逾時/確認/冷卻)必須獨立於特徵值是否缺失都照常執行**（D16 血淚教訓，未來寫 detect.py 部署版時要記得）。**評估用的跌倒確認秒數不是寫死 2s，是折內調參出來的**（D16 取代 D11 假設，grid 見 evaluate.py TUNE_CONFIRM_SECONDS_GRID）。matplotlib 中文圖表記得設定 `font.family` 為系統中文字型（msjh.ttc），預設字型沒有中文字形。

## 📜 Phase 日誌（append-only）

### 計畫階段（2026-07-20 完成）

- 完成：前置研究（URFD 現況實測、YOLO26-pose 選型、LangChain 1.x 多模態寫法、Discord webhook 規格、既有專案慣例盤點）→ 四項決策拍板（XGBoost 必做+GRU 選做；LOSO 人工標註；巢狀 repo；先文件後實作）→ 產出 docs/PLAN.md + PROGRESS.md + 三支 skills + CLAUDE.md 複本。
- 關鍵驗證證據（2026-07-20 實測）：
  - URFD 官方頁與下載 URL pattern 全部 HTTP 200（`https://fenix.ur.edu.pl/~mkepski/ds/`）
  - YOLO26-pose 為 ultralytics 官方 latest/recommended（2026-01-14 發布；ultralytics PyPI 8.4.102）
  - langchain 1.3.14 / langchain-google-genai 4.2.7；1.x 影像 content block 標準格式確認
- 決策記錄：PLAN.md 第 2 章 D1–D11。
- 尚未執行：git init（本條目完成時 repo 還不是 git repo，故無 commit 範圍/tag）。

### Phase 0 — 環境與骨架（2026-07-21 完成，tag: phase-0）

- 完成：git init（main）+ hooksPath 設定 + 3 個 Conventional Commits；pyproject（cu128 explicit index、torch+torchvision 同鎖、src layout + hatchling）；`uv sync`；.env 補齊專案變數（`DISCORD_WEBHOOK_URL` 留白待使用者填）+ `.env.example`；自既有專案移植 `scripts/check_public_text.py` + `.githooks/` + `.claude/private/redlist.txt`（不進 git）；README 骨架（十章節標題 + TODO 標記）；`src/fallguard/`（`config.py` 含金鑰 mapping D7）+ 冒煙測試。
- 驗證證據（2026-07-21 實測）：
  - `uv run python -c "import torch; print(torch.cuda.is_available())"` → `True`（RTX 4090、torch 2.11.0+cu128）
  - `uv run pytest -q` → `3 passed`
  - 含本機路徑樣式的測試 commit 被 pre-commit 擋下（exit 1）；正式 3 commits 全數通過 public-copy-check
- 決策連結：PLAN.md D7（金鑰 mapping）、D9（cu128 同鎖 torchvision）、D10（src layout）。
- commit 範圍：`69f7c09..0835351`
- 插曲：public-copy-check SKILL.md 原文含字面路徑示例被自家 hook 擋下，已改為文字描述（教訓收進「已知坑」）。

### Phase 1 — 資料下載 + 關鍵點抽取 + 人工標註（2026-07-21 完成，tag: phase-1）

- 完成：
  - `scripts/download_data.py`：URFD 下載（斷點續傳、summary），實測 70 mp4 + 2 CSV 全數 200、共 0.10 GB。
  - CSV 格式偵查：`urfall-cam0-{falls,adls}.csv` 無標頭，欄位 = video_id, frame_num(1-indexed), label(-1/0/1), 8 個深度衍生特徵；frame_num 可能有缺口（非連續）。
  - **D12 發現**：ADL 40 段中 16 段含 label=1 幀（姿態水平幾何特徵，非跌倒事件），已定案 kind 覆寫規則並回頭修正 §7.2。
  - Context7 MCP 在本環境未連接，依 CLAUDE.md 改讀官方文件 + 實測 API 確認 ultralytics 8.4.102 用法；**發現 `half=True` 已棄用，改用 `quantize=16`（D14）**，順手全文改正。
  - `scripts/prepare_data.py`：YOLO26m-pose + ByteTrack 抽取 70 支影片關鍵點 → `data/processed/*.npz`；權重快取到 `models/pretrained/`（gitignore）。
  - `scripts/annotate_urfd.py`：互動標註工具（縮圖蒙太奇、已標受試者參考列、PIL 中文疊字、播放/回上一段/備註/複查模式），含 ADL 躺姿比例提示（用 D12 的發現輔助人工判斷）。
  - `scripts/subject_sheet.py`、`scripts/compare_subjects.py`、`scripts/peek_video.py`：標註輔助小工具（全受試者對照圖、兩人放大比對、獨立播放不影響進度）。
  - `scripts/make_splits.py` + `tests/test_splits.py`：GroupKFold + LOSO 皆已產生。
  - `tests/test_prepare_data.py`：npz schema + fall-01/adl-01 標籤對齊(含已知缺口)驗證。
  - **D13**：修正 `.gitignore`，`urfd_meta.csv` 與 `splits.json` 例外進 git（不可重現的人工標註）。
  - **人工標註完成**：70/70 段，unknown 0；過程中發現 cv2 視窗失焦累積按鍵、恢復焦點瞬間暴衝連續觸發的 bug（40 段 ADL 曾被誤標成只有 P1/P2 兩種且分佈不合理），已修正工具加入 400ms 按鍵沖刷（D15）並用 `--review --kind adl` 只複查受影響的 40 段（30 段 fall 未受影響、未重做）。複查後確認 ADL **真的只有 P1、P2 兩位受試者出現**（使用者親眼確認、官方頁未反駁此分佈），P3/P4/P5 只在 fall 中出現，連帶造成 LOSO 折指標可用性不對稱（記入 D15，Phase 2 需處理）。
- 驗證證據（2026-07-21 實測）：
  - `uv run python scripts/prepare_data.py` → `70/70 ok`，耗時 287s，平均偵測率 90.0%（最低 fall-19 53.0%，跌倒瞬間遮擋，符合預期）
  - `uv run python scripts/make_splits.py` → GroupKFold 5 折 + LOSO 5 折（P1 test=30 段 fall6+adl24、P2 test=22 段 fall6+adl16、P3/P4/P5 test 皆 6 段 fall6+adl0）已寫入 `data/splits.json`
  - `uv run pytest -q` → `12 passed`（含 LOSO 測試，原本 skip 現轉 pass）
- 決策連結：PLAN.md D12（ADL label 語意）、D13（.gitignore 例外）、D14（quantize 取代 half）、D15（ADL 僅 2 受試者 + LOSO 指標不對稱 + 按鍵暴衝 bug 修正）。
- commit 範圍：`de1f321..`（本 Phase 內多次小 commit，見 `git log --oneline`）

### Phase 2 — 特徵工程 + 規則式 baseline（2026-07-21 完成，tag: phase-2 待使用者自行打）

- 完成：
  - `src/fallguard/features.py`：軀幹角/角速度/質心垂直速度/bbox 比/頭踝差/髖高/y 離散度/缺失率,含短缺口內插、25Hz 重採樣。
  - `src/fallguard/fsm.py`：完整狀態機(NORMAL→FALLING→ON_GROUND→CONFIRMED→ALERTED)。
  - `src/fallguard/rules.py`：視窗級無狀態規則分類器(供 P/R/F1/PR-AUC 用)。
  - `scripts/evaluate.py`：LOSO/GroupKFold 評估,折內調參(v_y/θ/falling_timeout/confirm_seconds),結果落 `docs/results/rule_baseline.md`。
  - `scripts/error_analysis.py`：動作類別 × 誤報率表 + 跌倒/躺床/蹲下三聯特徵曲線圖(`docs/assets/error_analysis_triplet.png`)。
  - README 回填：系統架構(兩張 mermaid)、資料集授權、快速開始四步指令、評估結果(視窗級+事件級雙表、誤報分析、已知限制)。
  - **D16 兩項發現**：(1) fsm.py 的 NaN 防呆過寬,連純時間判斷都被跳過,狀態機永久卡死,事件級 Sensitivity 曾恆為 0——已修正(只有需要當下特徵值的判斷才跳過)。(2) 修完 bug 後仍發現文獻預設的 `falling_timeout_s=1.0` 與 D11 的評估用 `confirm_seconds=2s` 對 URFD 短片段系統性過嚴(25/25 進到 ON_GROUND 的影片剩餘時長中位數僅 0.77s)——`evaluate.py` 改為兩者皆納入折內調參範圍,取代 D11 原假設。
- 驗證證據（2026-07-21 實測）：
  - `uv run pytest -q` → `25 passed`
  - `uv run python scripts/evaluate.py --model rule --protocol loso` → LOSO 事件級 Sensitivity(調參後)：P1=1.00 P2=1.00 P3=0.83 P4=0.67 P5=0.50；Specificity(僅 P1/P2 可算)：0.92/0.94
  - `uv run python scripts/error_analysis.py` → 40 段 ADL 中 3 段誤報(7.5%)，動作類別以「躺床」(14.3%)最高，跟 LOSO specificity 數字互相印證
  - `--protocol groupkfold` 亦跑過confirm 正常運作(未持久化為主報告,LOSO 才是 §7.1 主協定)
- 決策連結：PLAN.md D16（NaN 防呆修正 + confirm_seconds/falling_timeout_s 折內調參取代 D11 固定值）。
- **本輪 git commit/tag 由使用者自行執行**：features.py/fsm.py/rules.py/evaluate.py 與相關修正已由 AI 分批 commit(commit 範圍 `aea6096..22a9f86` 附近,見 `git log --oneline`)；error_analysis.py + README 回填 + 本篇文件更新這批**尚未 commit**，建議 commit 訊息：`feat: add error analysis and fill in README with rule baseline results`，完成後 `git tag phase-2`。
