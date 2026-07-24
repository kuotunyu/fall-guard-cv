"""VLM 描述品質對照(docs/PLAN2.md Phase 6)。

對 `events/` 現有的真實跌倒事件截圖,分別用 GEMINI_MODEL(主力)與 OPENAI_MODEL(備援)
各跑一次現場描述,產出併排表供人工評註——把表3「備援/評審」欄從沒驗證過的設計,
變成有實測依據的結論。

會花錢的批次 API 呼叫,預設只印成本估算不會真的呼叫(CLAUDE.md 規範:先估算成本
給使用者確認)。看過估算後,加 --yes 才會真的執行。

**隱私設計(D44)**:`events/` 的截圖是使用者真人跌倒測試的畫面,逐圖描述涉及居家隱私。
輸出拆兩份——`docs/results/vlm_comparison_detail.md`(逐圖完整內容,.gitignore 排除,
只留本機)與 `docs/results/vlm_comparison.md`(公開版,只有自動統計的彙總數字,不含
任何逐圖描述文字)。公開版的「彙總結果」文字分析段落需要人工/AI 讀過 detail 檔案
後再手動補寫,腳本本身只自動填入客觀統計數字。

用法：
    uv run python scripts/compare_vlm.py          # 只印成本估算,不呼叫任何 API
    uv run python scripts/compare_vlm.py --yes     # 確認後執行,呼叫兩個模型並產出報告
"""

from __future__ import annotations

import argparse
import sys

from fallguard.config import REPO_ROOT, settings
from fallguard.vlm import FALLBACK_TEXT, _describe_scene_raw

RESULTS_DIR = REPO_ROOT / "docs" / "results"
PUBLIC_PATH = RESULTS_DIR / "vlm_comparison.md"
DETAIL_PATH = RESULTS_DIR / "vlm_comparison_detail.md"  # .gitignore 排除,只留本機(D44)

# 每次呼叫 ≈ 1 張 720p JPEG + 短 prompt + ~150 token 輸出,兩個模型都是 flash-lite/mini
# 級距(見 README「成本估算」的單次估算基礎),單次成本遠低於 $0.001。
EST_COST_PER_CALL_USD = 0.001


def _sanitize_cell(text: str) -> str:
    """markdown 表格儲存格不能有裸露的換行或 `|`,轉成安全字元。"""
    return text.replace("\n", " ").replace("|", "\\|").strip()


def _call_with_diagnostics(image_path, *, model: str | None = None, provider: str | None = None) -> str:
    """跟 describe_scene() 不同:不吞例外的實際失敗原因,只吞在這裡並回報成人看得懂的診斷文字。

    這樣即使被安全過濾擋下,vlm_comparison.md 也能誠實記錄「為什麼」,而不是兩邊
    都印出一樣看不出差異的 FALLBACK_TEXT。
    """
    try:
        return _describe_scene_raw(image_path, model, provider)
    except Exception as exc:  # noqa: BLE001 - 這裡就是要把原因記下來,不是要中斷流程
        return f"{FALLBACK_TEXT}（診斷：{type(exc).__name__}: {exc}）"


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true", help="確認成本估算後才加這個旗標,否則只印估算不執行")
    args = parser.parse_args()

    events_dir = settings.events_dir
    images = sorted(events_dir.glob("*.jpg"))
    if not images:
        print(f"{events_dir} 底下沒有任何 .jpg,無法比較——先讓 detect.py 實際跑出至少一次事件。")
        sys.exit(1)

    n_calls = len(images) * 2
    est_total = n_calls * EST_COST_PER_CALL_USD
    print(f"找到 {len(images)} 張事件截圖（{events_dir}）。")
    print(f"將對每張圖各用 `{settings.gemini_model}`(GEMINI_MODEL)與 `{settings.openai_model}`(OPENAI_MODEL)跑一次描述，")
    print(f"共 {n_calls} 次 API 呼叫，估計總花費 < ${est_total:.3f}。")

    if not args.yes:
        print("\n這只是成本估算，尚未呼叫任何 API。確認金額可接受後，加 --yes 重新執行才會真的送出請求。")
        return

    rows = []
    for i, img in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {img.name} ...")
        gemini_text = _call_with_diagnostics(img)
        openai_text = _call_with_diagnostics(img, model=settings.openai_model, provider="openai")
        rows.append((img.name, gemini_text, openai_text))

    def _is_ok(text: str) -> bool:
        return not text.startswith(FALLBACK_TEXT)

    n_gemini_ok = sum(1 for _, g, _ in rows if _is_ok(g))
    n_openai_ok = sum(1 for _, _, o in rows if _is_ok(o))

    header = (
        f"docs/PLAN2.md Phase 6。主力：`{settings.gemini_model}`（GEMINI_MODEL）"
        f"｜備援：`{settings.openai_model}`（OPENAI_MODEL）。\n"
        f"素材：`events/` 現有 {len(images)} 張真實事件截圖（跌倒測試截圖，因涉及居家隱私，"
        "逐圖詳細內容不公開，只公開彙總結果）。"
    )

    detail_lines = [
        "# VLM 描述品質對照（逐圖詳細版，僅本機保留）",
        "",
        header,
        "",
        "> **本檔僅供本機參考，不進 git**：內容是使用者真人跌倒測試截圖的逐張文字描述，涉及居家隱私。",
        f"> 公開版彙總結論見 [{PUBLIC_PATH.name}]({PUBLIC_PATH.name})。",
        "",
        "| 圖檔 | GEMINI_MODEL 描述 | OPENAI_MODEL 描述 |",
        "|---|---|---|",
    ]
    for name, gemini_text, openai_text in rows:
        detail_lines.append(f"| {name} | {_sanitize_cell(gemini_text)} | {_sanitize_cell(openai_text)} |")

    public_lines = [
        "# VLM 描述品質對照",
        "",
        header,
        "",
        "## 彙總結果（自動統計）",
        "",
        f"- GEMINI_MODEL：{n_gemini_ok}/{len(images)} 次成功（非 fallback）",
        f"- OPENAI_MODEL：{n_openai_ok}/{len(images)} 次成功（非 fallback）",
        "",
        "> **文字分析段落待人工/AI 補寫**：讀過 `vlm_comparison_detail.md`（本機限定）的逐圖內容後，"
        "在此手動補上品質、格式差異、嚴重程度評分合理性等分析，再同步更新 README 表3 備援欄。",
        "",
        "## 評比維度提示（人工評註用）",
        "",
        "- 姿態描述準確度：是否正確描述人物姿態與位置",
        "- 環境描述：是否有提到周遭環境/危險物",
        "- 嚴重程度評分合理性：1-5 分是否符合畫面實際情況",
        "- 繁中流暢度：是否通順、適合直接給家人看",
        "",
        "**刻意不做 LLM-as-judge**：`OPENAI_MODEL` 本身是受測者，不能兼任裁判；樣本只有"
        f"{len(images)} 張，人工判讀比自動評分更可信（見 docs/PLAN2.md Phase 6）。",
    ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DETAIL_PATH.write_text("\n".join(detail_lines), encoding="utf-8")
    PUBLIC_PATH.write_text("\n".join(public_lines), encoding="utf-8")
    print(f"已寫入 {DETAIL_PATH}（本機限定，不進 git）")
    print(f"已寫入 {PUBLIC_PATH}（公開版，彙總文字分析待人工/AI 補寫）")


if __name__ == "__main__":
    main()
