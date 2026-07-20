# fall-guard-cv:居家即時跌倒偵測與家人通報

> 🚧 開發中。進度見 [PROGRESS.md](PROGRESS.md),完整開發藍圖見 [docs/PLAN.md](docs/PLAN.md)。

<!-- TODO(Phase 4): demo GIF 置此(docs/assets/demo.gif,≤8MB,規格見 PLAN.md 第 9 章) -->

## 為什麼做這個專案

<!-- TODO(Phase 4): 第一人稱動機段(家中長輩獨處的跌倒憂慮 × 電腦視覺研究背景),草稿經作者修改後定稿 -->

## 系統架構

<!-- TODO(Phase 2): 自 docs/PLAN.md 第 3 章複用兩張 mermaid(資料流 + 狀態機),並附狀態機白話說明 -->

## 模型選型

<!-- TODO(Phase 3/4): 表1 pose 模型對照(YOLO26 n/s/m,mAP/延遲/選型理由/版本號);表2 分類器成績(rule vs XGBoost (vs GRU));表3 VLM 分工;台灣模型生態觀察註記 -->

## 資料集與授權

<!-- TODO(Phase 1): URFD 來源與 CC BY-NC-SA 4.0 授權、Kwolek & Kepski 2014 引用;Le2i 備案;本 repo 不重佈原始資料、只附下載腳本 -->

## 快速開始

<!-- TODO(Phase 1 起逐步回填): uv sync(Windows 需 cu128 index,已寫入 pyproject)、.env 設定表(見 .env.example)、download → prepare → evaluate → detect 四步指令 -->

## 評估結果

<!-- TODO(Phase 2/3): 切分方式與防洩漏聲明(LOSO 主協定)、視窗級+事件級指標、混淆矩陣、站/坐姿分層、誤報案例分析 -->

## 隱私設計

<!-- TODO(Phase 4): 平時零上傳(推論與特徵全地端)、僅確認跌倒事件送單張截圖、LOCAL_ONLY 完全離線模式、SEND_IMAGE 開關 -->

## 成本估算

<!-- TODO(Phase 4): 訓練 $0(Colab T4)/推論 $0(地端)/VLM 每次通報成本與月估(實測 token 數回填) -->

## 關鍵套件版本

<!-- TODO(Phase 4): 與 uv.lock 一致的版本表(ultralytics/torch/langchain/langchain-google-genai/xgboost) -->

## 開發紀錄與授權

- 進度追蹤:[PROGRESS.md](PROGRESS.md);每階段驗收以 git tag `phase-N` 標記
- License:[MIT](LICENSE)
<!-- TODO(Phase 1): 資料授權註記(URFD CC BY-NC-SA 4.0) -->
