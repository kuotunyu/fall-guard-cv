# 跨資料集泛化：URFD 訓練 → Le2i 純測試

協定：docs/PLAN.md §7.1 P3。門檻與時間參數只用 URFD(70 段)train 資料調參，Le2i 完全沒被看過、也沒有參與任何調參——受試者/場景/攝影機皆天然不相交。只報事件級指標，不報視窗級：Le2i 的視窗級標籤語意（哪些幀算跌倒中）跟 URFD 是否完全對等，還沒有像事件級（整段影片是否判定跌倒）那樣經過同等程度的驗證。

URFD(train)：70 段。Le2i(test)：130 段。

## 事件級指標（套用 URFD 調參後的門檻，Wilson 95% 信賴區間見 docs/PLAN2.md Phase 5）

| 指標 | 數值 | 95% CI |
|---|---|---|
| Sensitivity | 0.559 | [0.47, 0.64] |
| Specificity | 0.000 | [0.00, 0.56] |
| FP/小時 | 110.475 | — |

調參後參數（只在 URFD train 上搜尋）：v_y=3.0、θ=45.0°、falling_timeout_s=1.5s、confirm_seconds=0.3s。

**對照**：URFD 內部 LOSO 各折事件級指標見 [rule_baseline.md](rule_baseline.md)（注意：那是同資料集內部交叉驗證，跟這裡的跨資料集純測試不是同一種協定，數字不可直接相減當「下滑幅度」，只能定性比較量級）。