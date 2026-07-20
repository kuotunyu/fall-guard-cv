---
name: public-copy-check
description: 任何公開產出（README/PLAN/PROGRESS/commit 訊息/GIF/截圖/HF 模型卡/Space 文案）發布或 commit 前的守門程序——擋公司名、金鑰、本機路徑、個資。產出要公開的內容前一律使用。
---

# public-copy-check：公開產出守門

**適用對象**：README.md、docs/PLAN.md、PROGRESS.md、commit 訊息、docs/assets/ 的 GIF 與截圖、HF 模型卡、任何會進 git 或對外發布的文字與影像。

## 文字檢查

1. **跑腳本**：`uv run python scripts/check_public_text.py <檔案...>`
   （腳本與禁詞清單 `.claude/private/redlist.txt` 於 Phase 0 自既有專案移植；**移植前**改用下方人工清單逐項檢查。redlist.txt 不進 git。）
2. **人工清單**（腳本存在時仍抽查）：
   - [ ] 無特定公司名與「標案」「評鑑」等職業情境字眼（redlist.txt 明確枚舉）——動機一律以個人/家庭經驗、高齡社會議題、台灣開源生態興趣撰寫（CLAUDE.md 公開文案守則）。註：技術段落中不可避免的技術供應商/開源產品名（Gemini、Discord、Hugging Face、YOLO、Colab、Kaggle 等）允許出現；禁令對象是與使用者職業背景相關的公司/機構名與圈內字眼
   - [ ] 無 API 金鑰值、token、webhook URL（`DISCORD_WEBHOOK_URL` 的實際值絕不可出現）
   - [ ] 無本機絕對路徑（`C:\` 開頭的使用者目錄路徑）——文件內一律用 repo 相對路徑
   - [ ] 無私人規劃文件的檔名或內容引用
   - [ ] 資料授權註記在場：URFD（CC BY-NC-SA 4.0，引 Kwolek & Kepski 2014）；若用 Le2i 亦同

## 影像/GIF 檢查（docs/assets/）

- [ ] 畫面內無終端機/編輯器帶出的本機路徑、金鑰、webhook URL
- [ ] Discord 通知截圖：遮蔽 server 名/頻道名/使用者名以外，不露任何個資
- [ ] 無非資料集受試者的人臉（自錄 demo 只能出現作者本人，且作者同意）
- [ ] URFD 素材處附授權註記（PLAN.md 第 9 章規格）

## commit 前

- pre-commit / commit-msg hook（Phase 0 起生效）會自動跑 check_public_text.py；hook 擋下時**修正內容，不得 --no-verify 繞過**。
- 發現已 commit 的洩漏：立即回報使用者，討論 history 處理方式，不要自行 force push。
