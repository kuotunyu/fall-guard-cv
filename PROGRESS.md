# fall-guard-cv 進度追蹤

<!-- 使用規則：
  🧭 快速回憶區：直接改寫、不留歷史、全區 ≤30 行。每次收工前更新（用 /update-progress skill）。
  📜 Phase 日誌：只增不改。每 Phase 收尾寫一條，附驗證證據（指令+輸出數字）與 git tag。
  回來時的 10 分鐘流程：讀快速回憶區 → git log --oneline -10 → uv run pytest -q → 讀 docs/PLAN.md 當前 Phase DoD。
  （或直接用 /resume-context skill 自動跑完上述流程。） -->

## 🧭 快速回憶區

**上次收工日期**：2026-07-22
**現在做到哪**：**Phase 4 全部完成，PLAN.md 第 14 章收尾清單全數打勾，repo 已公開上線並通過兩輪完整度稽核**（D25 找到 2 個真缺口已補；D33 使用者最終決定拿掉「為什麼做這個專案」整段；D34 使用者看實際發布頁截圖回饋「不要一堆括號」，README 全文改寫成用逗號/破折號銜接，只留 markdown 連結語法與學術引用格式等必要語法括號）。demo.gif 疊字排版與 FPS 顯示邏輯先前經使用者多輪實際畫面回饋修正 7 次（D26-D32）。`uv run pytest -q` 43 passed；`check_public_text.py` 全綠。**目前沒有已知的待辦事項**。
**下一步（開機第一件事）**：commit 本輪異動（README 括號改寫、docs/PLAN.md D34）並推上 GitHub，即可視為正式完整發布。之後若要繼續開發，可考慮 PLAN.md「如果被問接下來會怎麼優化」列出的方向（特徵計算增量化、站/坐姿分層資料、通報管道多元化、GRU 延伸實驗）。
**未決問題**：站姿/坐姿跌倒的逐段對照表找不到官方來源，README/評估結果已誠實註記從缺（見 PLAN.md §7.2，D25 稽核確認此限制屬誠實揭露、非疏漏，不阻擋發布）。
**待使用者人工處理**：無（commit + push 本輪異動即可）。
**已知坑**：torch 必須走 cu128 index（已寫進 pyproject，`uv sync` 即可）。clone 後要跑 `git config core.hooksPath .githooks`。公開文件不寫死本機絕對路徑（會被 hook 擋）。ultralytics 用 `quantize=16` 不用已棄用的 `half=True`（D14）。**NaN 防呆只能擋「需要當下特徵值」的判斷，純時間判斷(逾時/確認/冷卻)要照常執行**（D16 血淚教訓）。**LOSO P3/P4/P5 折沒有 ADL 測試樣本**，指標誠實標 N/A（D15）。**1-slot 即時佇列(丟舊幀設計)拿去測固定長度短影片檔會失效**，量吞吐量要用獨立單執行緒迴圈(`--benchmark`，D21)。**VLM 回應的 `.content` 可能是純字串或 content block list，一律用 `.text` 屬性取值**（D22）。**cv2 疊字/overlay 一次性 demo 腳本容易踩的坑**：同一畫面緩衝區重複疊字時底色矩形要固定寬度、半透明底色蓋不住舊字要每次疊字用乾淨畫面複製、字級縮放公式不要寫成二次縮放會被壓到肉眼難辨（D26/D29）；ffmpeg 轉 GIF 用 `palettegen stats_mode=full + dither=sierra2_4a` 才能保住大面積純色文字不失真變色，肉眼驗收要看「轉出來的 GIF」不是轉檔前的 mp4（D28）；畫面上顯示 FPS 用滑動視窗而非累積平均才不會有暖機爬升假象，且**不同程式(demo 腳本 vs 正式 detect.py)天生速度不同，不能不配速就直接比較數字**（D30-D32）。matplotlib 中文圖表要設 `font.family` 為系統中文字型（msjh.ttc）。

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

### Phase 3 — Colab 訓練 XGBoost + 權重回程 + 上傳 HF（2026-07-21 完成，tag: phase-3 待打）

- 完成：
  - `src/fallguard/features.py`：新增 `window_stat_vector`/`STAT_FEATURE_NAMES`（54 維視窗統計特徵：9 基礎特徵 × 6 統計量）。
  - `scripts/prepare_train_export.py`：關鍵點特徵 → 54 維視窗統計向量匯出，223KB，供上傳 Colab（D17，比整批 npz 小很多、不含影像）。
  - `notebooks/fall-guard-cv_train_xgboost_colab.ipynb`：XGBoost LOSO 五折訓練 + 全資料最終部署模型 + SHAP 特徵重要度，xgboost 鎖 3.2.0（D17，本專案 Python 3.11 限制）。過程中因使用者操作卡關兩次，改成分批（三次獨立）上傳檔案的設計，並把檔名從 `train_colab.ipynb` 改成 `fall-guard-cv_train_xgboost_colab.ipynb`（辨識度考量）。
  - 使用者於 Colab（Pro+，T4）執行完成，帶回 `xgb_fold_{P1..P5}.json` + `xgb_final.json` + `xgb_loso_results.json` + `shap_summary.png`，AI 協助解壓並放進 `models/xgboost/`（已 gitignore）。
  - `scripts/evaluate.py --model xgb`：本機重現驗證。**首次驗收失敗**（P2 precision 差 0.025、P4 recall 差 0.017），追出根因是 `prepare_train_export.py` 與 `evaluate.py` 各自維護一份視窗排除邏輯、互相飄移（D18）。修正：新增共用函式 `build_xgb_stat_samples`，兩邊都改呼叫它，不再各自實作。
  - README「模型選型」章節回填 rule vs XGBoost 對照表 + SHAP 交叉驗證觀察 + HF 連結。
  - **GRU（3b）使用者決定不做**，直接進上傳階段。
  - `scripts/upload_to_hf.py` + 模型卡（`models/xgboost/README.md`，過 public-copy-check）：上傳到 `steven0226/fall-guard-cv-xgboost`（公開，CC BY-NC-SA 4.0，D19）。`config.py` 補 `hf_token` 欄位。
- 驗證證據（2026-07-21 實測）：
  - 修正前：`uv run python scripts/evaluate.py --model xgb --protocol loso` → ❌ P2/P4 誤差超出 ±0.01
  - 修正後：同指令 → ✅ 全部 15 項指標(5 折 × P/R/F1)與 Colab 完全重現，誤差 0.000；本機重現視窗數 802/598/30/33/36 與 Colab n_test 完全一致
  - `uv run pytest -q` → `26 passed`
  - `uv run python scripts/upload_to_hf.py --repo-id steven0226/fall-guard-cv-xgboost` → 上傳成功；WebFetch 確認 HF 頁面檔案清單與模型卡渲染正確
- 決策連結：PLAN.md D17（視窗統計特徵匯出設計、xgboost 版本鎖定、模型回傳而非重訓練的理由）、D18（視窗篩選邏輯飄移 bug 與修正）、D19（HF 上傳、CC BY-NC-SA 4.0 授權選擇）。
- 尚未完成：`git tag phase-3`（待使用者 commit 本輪後補上）。
- **本輪 git commit/tag 由使用者自行執行**。

### Phase 4 — 即時偵測 + 通報 + demo 收尾（2026-07-21 功能面完成，tag: phase-4 待打）

- 完成：
  - `src/fallguard/pose.py`：`PoseEstimator` 包裝 `model.track()` 逐幀即時推論（`persist=True`/`quantize=16`/`max_det=1`），與 `prepare_data.py` 共用權重快取邏輯。
  - `src/fallguard/vlm.py`：`describe_scene()` 呼叫 Gemini VLM 描述現場，`safety_settings` 放寬 DANGEROUS_CONTENT；任何失敗（安全過濾/例外/空回應）一律回傳預留文字不拋例外。
  - `src/fallguard/notify.py`：`send_fall_alert()` Discord webhook multipart 送出，429 依 `retry_after` 重送一次，`SEND_IMAGE=false` 只送文字。
  - `src/fallguard/detect.py`：三執行緒（capture 1-slot 佇列/main 推論+特徵+狀態機+疊加/alert worker 非同步 VLM→Discord）；特徵計算重用 `features.py` 對滑動緩衝區重跑（不寫增量版，呼應 D18 教訓，見 D20）；`--benchmark` 獨立單執行緒吞吐量測試（見 D21）。
  - **範圍決策（D20）**：不建立 `classifier.py`，即時偵測只用規則式 FSM——XGBoost 推論邏輯已存在於 `evaluate.py`，即時路徑另建一份沒有呼叫端的重複程式碼屬於過早抽象。
  - `fsm.py` 新增 `lying_elapsed_s` 屬性，供 UI 顯示 ON_GROUND 倒數秒數。
  - `tests/test_notify.py`（6 項，mock webhook：成功/附圖/429 重送/未設定）、`tests/test_vlm.py`（4 項，含 D22 迴歸測試）、`tests/test_fsm.py` 新增 `lying_elapsed_s` 測試、新建 `tests/test_docs.py`（README/PLAN 必含章節守門，5 項）。
  - `docs/assets/demo.gif`：用真實 URFD fall-01 影格時間戳跑完整管線錄製，480x360/12fps/2.55MB/7.25s，因資料集限制做了幾個取捨（D23）。
  - README 全章節填完：動機段草稿、pose/VLM 選型表、隱私設計、成本估算、關鍵套件版本表、即時偵測章節。
- 過程中發現並修正的 bug（均已記入 Decision Log）：
  - **D21**：1-slot 即時佇列（丟舊幀設計）測固定長度短影片檔會失效——capture thread 不配速會在 main thread 準備好之前把整支片讀完丟棄。修正：影片檔來源依原生 fps 配速；新增 `--benchmark` 獨立單執行緒迴圈量真實吞吐量。
  - **D22**：`vlm.py` 的 `response.content` 有時是 content block list 而非純字串，直接 `str()` 會把整包 dict/list 結構原樣塞進通報文字；改用 `.text` 正規化屬性。用真實 GEMINI_MODEL 呼叫實測驗證修正前後差異。
  - overlay 疊字底色矩形若依當下文字寬度動態縮小，同一畫面緩衝區連續疊字時會蓋不住前一幀較寬的殘留文字（demo 錄製凍結延伸階段發現）——改回固定寬度矩形。
- 驗證證據（2026-07-21 實測）：
  - `uv run python -m fallguard.detect --source data/raw/urfd/fall-01-cam0.mp4 --benchmark` → `[benchmark] 共處理 300 幀，耗時 7.02s，平均 FPS=42.8`
  - `uv run python -m fallguard.detect --source data/raw/urfd/fall-01-cam0.mp4 --no-display --confirm-seconds 0.3` → 完整事件鏈跑通：`[detect] 確認跌倒` → VLM 回傳正確繁中描述（含姿態/環境/嚴重程度分析）→ `[notify] DISCORD_WEBHOOK_URL 未設定，略過送出`（優雅失敗，不中斷主迴圈）；`events/` 正確存下 impact+confirm 兩張截圖（人工檢視截圖內容正確：URFD 房間、人物倒地）
  - `uv run pytest -q` → `43 passed`
  - `uv run python scripts/check_public_text.py` → 全綠
- 決策連結：PLAN.md D20（detect.py 範圍決策：不建 classifier.py）、D21（1-slot 佇列與 --benchmark 設計）、D22（VLM `.text` bug 修正）、D23（demo.gif 取捨）。
- **本輪 git commit/tag 由使用者自行執行**。

### Phase 4 收尾：webcam 實機驗證 + repo 公開上線（2026-07-21 完成，tag: phase-4）

- 使用者自行完成三次 commit + `git tag phase-4`。
- **D24：webcam + Discord 真實送達驗證**——使用者用實體攝影機執行 `uv run python -m fallguard.detect --source 0`，在鏡頭前真人實際倒地（非影片檔模擬）。畫面疊加、狀態機判定、FPS 讀數皆正常；VLM 用真實 `GEMINI_MODEL` 產出兩次現場描述（姿態/環境/意識狀態/嚴重程度 1-5 分，內容合理可判讀）；Discord 頻道實際收到兩則通報。至此 Phase 4 最後兩項「只能靠程式碼審查/mock 驗證」的項目（webcam 路徑、Discord 真實送達）都補上真實環境證據。
- **repo 建立與公開上線**：協助使用者釐清本機 `gh` CLI 誤登入 `tun0000`（測試帳號）而非 `kuotunyu`（求職用帳號）的問題，且 git 的 GitHub credential helper 委派給 `gh auth git-credential`，兩者需一致才能正確推送。使用者改用 `gh auth login` + `gh auth switch` 切換到 kuotunyu 後，由 AI 執行 `gh repo create fall-guard-cv --public`（在確認帳號正確後），使用者自行完成 `git push`。上線後逐項核對：Contributors 僅 kuotunyu 一人（無 tun0000/Claude）、phase-0~phase-4 共 5 個 tag 都在、`data/`/`models/` 沒有大型檔案或誤上傳的原始資料、`.env` 未被 commit、兩張 mermaid 圖用瀏覽器工具（非 WebFetch，因其不執行 JS 會誤判為「Loading」）確認實際渲染正常、demo.gif 直接打 raw 檔案 URL 確認 HTTP 200 且大小/型別正確。
- 過程中修正的環境問題：使用者一開始把 `DISCORD_WEBHOOK_URL` 貼到了外層 `4_fall-guard-cv/.env`（巢狀資料夾容易混淆的兩份 .env 之一），而非專案實際讀取的 `4_fall-guard-cv/fall-guard-cv/.env`；AI 用不印出網址內容的方式把值搬到正確位置並清空外層那份。
- 驗證證據（2026-07-21 實測）：
  - 使用者實測回報：「我故意跌倒在地上，有收到 discord 通知」，附上兩則實際收到的完整通報文字
  - `curl -sI https://raw.githubusercontent.com/kuotunyu/fall-guard-cv/main/docs/assets/demo.gif` → `200 OK`，`Content-Length: 2551406`，`Content-Type: image/gif`
  - 瀏覽器 `read_page` 確認兩處 mermaid 圖渲染為 `region "mermaid rendered output container"`（非原始程式碼區塊）
- 決策連結：PLAN.md D24（webcam/Discord 真實環境驗證）。
- commit 範圍：`70d6685..d43f7a9`（三個 Phase 4 commit + docs 收尾）。
- **`git tag phase-4` 已由使用者完成；repo 已推送至 <https://github.com/kuotunyu/fall-guard-cv>（public）**。

### 收尾：37-agent 完整度稽核 + 補漏（2026-07-21）

- 使用者要求「看看還有哪邊沒做」，用 Workflow 跑一次完整度稽核（非臨時肉眼檢查）：4 個平行掃描 agent 分別讀 PLAN.md 全文（Phase 0-4 DoD + 第 14 章收尾清單）、PROGRESS.md 快速回憶區與 Phase 日誌、README.md 對照 PLAN §10 應含章節、即時 repo 狀態（`git status`/`pytest`/`check_public_text.py`/TODO 掃描/`.env` 設定狀態），找出 33 個候選缺口；再用 33 個獨立 verify agent 逐一重新對照當下 repo 實際內容查證（不信任掃描 agent 的引言，直接讀當下檔案/跑當下指令）。
- 結果：**10 個真缺口**（其中 5 個是同一件事在不同檔案的不同措辭，實際是 2 個真正的新發現 + 3 個既知項目的重複提及）、**18 個「文件記帳落後於實際進度」假警報**（PLAN.md 第 14 章一堆 checkbox 因為單純忘記勾選而顯示未完成，實際工作早就做完並有證據，例如 mermaid 渲染/demo.gif/MIT 授權/公開文案掃描/URFD 引用/HF 連結/pytest 全綠/PROGRESS 發布狀態）、**5 個純誤判**（例如 GRU 選做被使用者婉拒不算缺口、`events/` 測試截圖已正確 gitignore 不算問題）。
- **兩個真正被漏掉的項目已修正**：
  1. README 標題下缺 badges（Python 3.11 / uv / MIT，PLAN §10 item 1 要求）→ 已補上 shields.io 徽章。
  2. `scripts/evaluate.py` 的 `window_level_metrics()` 一直都有計算混淆矩陣（`confusion_matrix(...).tolist()`），但從沒有任何報告實際顯示過（只用衍生出來的 P/R/F1 呈現）→ `write_report()` 新增「混淆矩陣（視窗級，折內調參後）」章節（5 折逐一 + 加總），README 評估結果新增對應加總表（TN=1207/FP=46/FN=29/TP=115）。
- **「家人沒有 Discord」的討論**：使用者主動提出，AI 研究後發現 LINE Notify 已於 2025-03-31 停止服務（替代方案 LINE Messaging API 需官方帳號、超過免費額度要付費，設定複雜度不亞於 Discord），提供四個選項讓使用者決定；使用者選擇「維持現狀，反正這個專案只是放在 GitHub 上展示的」——明確定案，不再是待辦事項。
- 驗證證據（2026-07-21 實測）：
  - `uv run python scripts/evaluate.py --model rule --protocol loso` → 重新產生 `docs/results/rule_baseline.md`，15 項 F1 數字與修正前完全一致（0.778/0.764/0.800/0.700/0.636），確認新增混淆矩陣邏輯未影響既有評估結果
  - `uv run pytest -q` → `43 passed`
  - `uv run python scripts/check_public_text.py` → 全綠
- 決策連結：PLAN.md D25（稽核方法與兩個真缺口修正）。
- **本輪 git commit 由使用者自行執行**。
