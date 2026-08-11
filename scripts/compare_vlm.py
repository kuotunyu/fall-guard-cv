"""VLM 描述品質對照。

對 `events/` 現有的真實跌倒事件截圖,分別用 GEMINI_MODEL(主力)與 OPENAI_MODEL(對照)
各跑一次現場描述,產出併排表供人工檢查。這是 API 整合紀錄,不是安全驗證。

會花錢的批次 API 呼叫預設只顯示呼叫數量，不會真的呼叫；加 --yes 才會執行。

**隱私設計**：`events/` 的截圖是使用者真人跌倒測試畫面，逐圖描述涉及居家隱私。
輸出拆兩份——`docs/results/vlm_comparison_detail.md`(逐圖完整內容,.gitignore 排除,
只留本機)與 `docs/results/vlm_comparison.md`(公開版,只有自動統計的彙總數字,不含
任何逐圖描述文字)。公開版的「彙總結果」文字分析段落需要人工/AI 讀過 detail 檔案
後再手動補寫,腳本本身只自動填入客觀統計數字。

用法：
    uv run python scripts/compare_vlm.py          # 只印呼叫範圍,不呼叫任何 API
    uv run python scripts/compare_vlm.py --yes     # 確認後執行,呼叫兩個模型並產出報告
"""

from __future__ import annotations

import argparse
import sys

from fallguard.config import REPO_ROOT, settings
from fallguard.vlm import FALLBACK_TEXT, _describe_scene_raw

RESULTS_DIR = REPO_ROOT / "docs" / "results"
PUBLIC_PATH = RESULTS_DIR / "vlm_comparison.md"
DETAIL_PATH = RESULTS_DIR / "vlm_comparison_detail.md"  # .gitignore 排除，只留本機

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
    parser.add_argument("--yes", action="store_true", help="明確允許 API 呼叫；未加時只印呼叫範圍")
    args = parser.parse_args()

    events_dir = settings.events_dir
    images = sorted(events_dir.glob("*.jpg"))
    if not images:
        print(f"{events_dir} 底下沒有任何 .jpg,無法比較——先讓 detect.py 實際跑出至少一次事件。")
        sys.exit(1)

    n_calls = len(images) * 2
    print(f"找到 {len(images)} 張事件截圖（{events_dir}）。")
    print(f"將對每張圖各用 `{settings.gemini_model}`(GEMINI_MODEL)與 `{settings.openai_model}`(OPENAI_MODEL)跑一次描述，")
    print(f"共 {n_calls} 次 API 呼叫；費用依供應商當期定價、模型與輸入大小而定。")

    if not args.yes:
        print("\n尚未呼叫任何 API。查閱供應商當期定價後，加 --yes 重新執行才會送出請求。")
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
        f"主力：`{settings.gemini_model}`（GEMINI_MODEL）"
        f"｜對照：`{settings.openai_model}`（OPENAI_MODEL）。\n"
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
        "# VLM 描述品質非正式對照",
        "",
        header,
        "",
        "## 可確認的結果（自動統計）",
        "",
        f"- GEMINI_MODEL：{n_gemini_ok}/{len(images)} 次成功（非 fallback）",
        f"- OPENAI_MODEL：{n_openai_ok}/{len(images)} 次成功（非 fallback）",
        "",
        "> 非 fallback 只代表 API 有回應，不代表描述正確、嚴重度校準或安全性。",
        "",
        "## 方法限制",
        "",
        f"- 只有 {len(images)} 張私人影像,且來自同一居家環境與測試者。",
        "- 沒有獨立標註者、盲評、預先註冊 rubric 或統計檢定。",
        "- 私人影像與逐圖輸出不公開,第三方無法完整重現內容評比。",
        "- 對照模型不是裁判,也不是能排除系統性錯誤的安全備援。",
        "",
        "這份結果只能作為 API 整合紀錄,不能作為醫療、臨床或部署安全證據。",
    ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DETAIL_PATH.write_text("\n".join(detail_lines), encoding="utf-8")
    PUBLIC_PATH.write_text("\n".join(public_lines), encoding="utf-8")
    print(f"已寫入 {DETAIL_PATH}（本機限定，不進 git）")
    print(f"已寫入 {PUBLIC_PATH}（公開版，只含呼叫統計與方法限制）")


if __name__ == "__main__":
    main()
