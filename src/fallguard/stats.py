"""小型統計工具(docs/PLAN2.md Phase 5)。

目前只有 Wilson score 信賴區間，供 scripts/evaluate.py 的事件級指標使用——
LOSO 小樣本折(如 P3/P4/P5 每折僅 6 段測試影片)算出來的 Sensitivity/Specificity
只是點估計，Wilson score 區間在 n 小或 phat 靠近 0/1 時比常態近似(Wald)更穩健，
不會出現下界 <0 或上界 >1 的失真。
"""

from __future__ import annotations

import math


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 信賴區間(預設 z=1.96 對應 95%)。n<=0 時回傳全然未知的 (0.0, 1.0)。"""
    if n <= 0:
        return (0.0, 1.0)
    phat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = phat + z2 / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z2 / (4 * n * n))
    lo = (center - margin) / denom
    hi = (center + margin) / denom
    return (max(0.0, lo), min(1.0, hi))
