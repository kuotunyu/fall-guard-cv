---
name: update-progress
description: 每次收工前、或收掉一個 Phase 時，依固定格式更新 PROGRESS.md；收 Phase 時另跑三檔一致 checklist 並打 git tag。使用者說「收工」「更新進度」「這個 Phase 做完了」時使用。
---

# update-progress：收工與收 Phase 的固定程序

## A. 每次收工（必做）

更新 `PROGRESS.md` 🧭 快速回憶區——**直接改寫、不留歷史、全區 ≤30 行**。首行 = **上次收工日期**（今天日期），其下五欄缺一不可：

1. **現在做到哪**：一兩句話，含當前 Phase 與最近完成的具體東西（附數字，如「F1=0.xx」）
2. **下一步（開機第一件事）**：必須是**一條可直接執行的指令或具體動作**，不是方向描述。壞例：「繼續做特徵工程」；好例：「跑 `uv run python scripts/evaluate.py --model rule --protocol loso`，結果落 docs/results/rule_baseline.md」
3. **未決問題**：還沒拍板的技術決策（指向 PLAN.md 章節）
4. **待使用者人工處理**：需要使用者本人動手的事（建 webhook、人工標註、HF 授權等）
5. **已知坑**：只留「還會踩」的坑；已修掉的移進 Phase 日誌

## B. 收 Phase（Phase DoD 全勾時做）

1. 在 📜 Phase 日誌 **append** 一條（只增不改），必含：
   - 完成內容清單（對照 PLAN.md 第 6 章該 Phase DoD）
   - **驗證證據：可重跑的指令 + 當時的輸出數字**（例：`uv run pytest -q` → `12 passed`；`prepare_data.py` → `70/70 ok`）——這是隔月回來能「信任現狀」的關鍵
   - 相關決策連結（PLAN.md D#）
   - commit 範圍（`git log --oneline` 首尾 hash）
2. **三檔一致 checklist**：
   - [ ] PLAN.md Decision Log：本 Phase 產生的新決策已 append（含 supersedes 標記）
   - [ ] PLAN.md 第 6 章：該 Phase DoD 逐項勾選
   - [ ] README：該 Phase 應回填的章節已更新（見 PLAN.md 各 Phase DoD 的 README 項）
   - [ ] 涉及公開文案的變更已過 `/public-copy-check`
3. `git tag phase-N` 並 commit（Conventional Commits，例：`docs: close phase 1, tag phase-1`）
4. 向使用者回報該 Phase 結果，**經確認後才進下一個 Phase**（CLAUDE.md 規範）
