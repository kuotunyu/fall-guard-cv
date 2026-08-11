# fall-guard-cv Portfolio Credibility Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the existing public prototype into a release-ready, evidence-calibrated CV/AI Engineering portfolio repository without adding product scope or rerunning expensive GPU experiments.

**Architecture:** Preserve the pose-to-alert pipeline and its evidence artifacts. Strengthen executable repository gates, replace stale process narration with one canonical agent guardrail, and rewrite public claims so the README is a faithful index of committed evaluation results rather than a product promise.

**Tech Stack:** Python 3.11, uv, pytest, Ruff, GitHub Actions, YOLO26-pose/Ultralytics, XGBoost, LangChain, GitHub CLI.

## Global Constraints

- Scope is only `https://github.com/kuotunyu/fall-guard-cv`.
- Do not modify `stock-cockpit` or the separate legacy `fall-detection-pose` repository.
- Do not download datasets, download model weights, train models, run dataset-wide inference, use secrets, or call VLM/Discord APIs.
- Preserve numeric failures and limitations; improve interpretation, not appearance.
- Do not claim clinical validation, guaranteed detection, production emergency response, or medical-device status.
- CI must remain CPU-only and must not require external services or credentials.
- Use the formal Git identity `kuotunyu <61350295+kuotunyu@users.noreply.github.com>`.

---

## File Structure

- `scripts/check_public_text.py`: one implementation for staged, commit-message, explicit-file, and tracked-tree public-copy scans.
- `tests/test_public_copy.py`: isolated tests for tracked-tree scanning and binary/private-path behavior.
- `tests/test_docs.py`: executable contracts for README evidence, CI gates, and removal of stale public process references.
- `pyproject.toml`, `uv.lock`: explicit Ruff development dependency and lint configuration.
- `.github/workflows/test.yml`: locked install, lint, tests, and tracked public-copy scan.
- `AGENTS.md`: concise canonical-repository and safe-validation instructions for future agent sessions.
- `README.md`: portfolio-facing system, evidence, limitations, privacy, and reproduction entry point.
- `.env.example`, `src/fallguard/config.py`, `src/fallguard/vlm.py`, `src/fallguard/detect.py`: remove stale planning references and volatile cost copy; keep behavior unchanged unless an official model identifier is invalid.
- `docs/results/*.md`, `scripts/evaluate.py`, `scripts/error_analysis.py`, `scripts/compare_vlm.py`: retain evidence while removing private/stale plan references and overstrong interpretation.
- `docs/PLAN.md`: delete after all useful evidence references have migrated.
- `docs/superpowers/specs/*`, `docs/superpowers/plans/*`: remove from the release tip after implementation; their committed history remains available, but they are not portfolio content.

---

### Task 1: Make the public-copy check scan the actual tracked repository

**Files:**
- Modify: `scripts/check_public_text.py`
- Create: `tests/test_public_copy.py`

**Interfaces:**
- Produces: `tracked_paths() -> list[str]`
- Produces: `scan_tracked(redlist: list[str]) -> list[str]`
- CLI contract: `python scripts/check_public_text.py --tracked` scans every Git-tracked path and returns nonzero on a blocked path or textual finding.

- [ ] **Step 1: Write failing tracked-scan tests**

```python
from pathlib import Path

import scripts.check_public_text as public_copy


def test_scan_tracked_checks_text_and_private_paths(monkeypatch, tmp_path: Path):
    private_path = "C:" + "/Users/example/private"
    (tmp_path / "README.md").write_text(private_path, encoding="utf-8")
    monkeypatch.setattr(public_copy, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        public_copy,
        "tracked_paths",
        lambda: ["README.md", "events/private-frame.jpg"],
    )

    hits = public_copy.scan_tracked([])

    assert any("Windows 使用者絕對路徑" in hit for hit in hits)
    assert any("events/private-frame.jpg" in hit for hit in hits)


def test_scan_tracked_skips_binary_content(monkeypatch, tmp_path: Path):
    (tmp_path / "asset.bin").write_bytes(b"\xff\xfe\x00\x01")
    monkeypatch.setattr(public_copy, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(public_copy, "tracked_paths", lambda: ["asset.bin"])

    assert public_copy.scan_tracked([]) == []
```

- [ ] **Step 2: Run the focused tests and confirm the missing interfaces fail**

Run: `uv run pytest tests/test_public_copy.py -q`

Expected: FAIL because `tracked_paths` and `scan_tracked` do not exist.

- [ ] **Step 3: Implement tracked-tree scanning and an explicit CLI mode**

```python
def tracked_paths() -> list[str]:
    return [path for path in _git("ls-files", "-z").split("\0") if path]


def scan_tracked(redlist: list[str]) -> list[str]:
    paths = tracked_paths()
    hits = scan_blocked_paths("tracked-路徑黑名單", paths)
    for relative in paths:
        path = REPO_ROOT / relative
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        hits += scan_text(f"tracked:{relative}", text, redlist)
    return hits
```

Add the `--tracked` branch before `--staged`. With no arguments, print usage and return `2` instead of reporting a vacuous pass.

- [ ] **Step 4: Run focused and full lightweight tests**

Run: `uv run pytest tests/test_public_copy.py tests/test_docs.py -q`

Expected: PASS.

- [ ] **Step 5: Verify the real tracked tree**

Run: `uv run python scripts/check_public_text.py --tracked`

Expected: PASS or actionable findings that are handled in Tasks 3–4; never a zero-file pass.

- [ ] **Step 6: Commit the scanner contract**

```bash
git add scripts/check_public_text.py tests/test_public_copy.py
git commit -m "test: scan the complete public repository"
```

---

### Task 2: Enforce Ruff, tests, and public-copy hygiene in CI

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `.github/workflows/test.yml`
- Modify: `tests/test_docs.py`
- Modify: Python files reported by the agreed Ruff rules, only for mechanical lint fixes.

**Interfaces:**
- CI commands: `uv sync --locked`, `uv run ruff check .`, `uv run pytest -q`, and `uv run python scripts/check_public_text.py --tracked`.
- Ruff scope: Python 3.11, line length 120, lint rules `E4`, `E7`, `E9`, `F`, and `I`.

- [ ] **Step 1: Add a failing CI-contract test**

```python
def test_ci_enforces_locked_quality_gates():
    workflow = _read(REPO_ROOT / ".github" / "workflows" / "test.yml")
    for command in [
        "uv sync --locked",
        "uv run ruff check .",
        "uv run pytest -q",
        "uv run python scripts/check_public_text.py --tracked",
    ]:
        assert command in workflow, f"CI 缺少品質門檻：{command}"
```

- [ ] **Step 2: Run the focused test and confirm it fails**

Run: `uv run pytest tests/test_docs.py::test_ci_enforces_locked_quality_gates -q`

Expected: FAIL because the workflow currently runs only sync and pytest.

- [ ] **Step 3: Add Ruff and lock it**

Run: `uv add --dev "ruff>=0.14,<1"`

Add:

```toml
[tool.ruff]
target-version = "py311"
line-length = 120

[tool.ruff.lint]
select = ["E4", "E7", "E9", "F", "I"]
```

- [ ] **Step 4: Update the workflow**

Use read-only permissions, a 15-minute timeout, pinned existing actions, and these steps:

```yaml
      - run: uv sync --locked
      - run: uv run ruff check .
      - run: uv run pytest -q
      - run: uv run python scripts/check_public_text.py --tracked
```

- [ ] **Step 5: Fix only deterministic Ruff findings**

Run: `uv run ruff check . --fix`

Then inspect `git diff`; do not accept semantic rewrites or blanket `noqa` additions.

- [ ] **Step 6: Verify the quality-gate task**

Run: `uv lock --check`

Run: `uv run ruff check .`

Run: `uv run pytest -q`

Run: `uv run python scripts/check_public_text.py --tracked`

Expected: all PASS.

- [ ] **Step 7: Commit the CI gate**

```bash
git add pyproject.toml uv.lock .github/workflows/test.yml tests/test_docs.py src scripts tests
git commit -m "ci: enforce repository quality gates"
```

---

### Task 3: Replace stale public planning with one canonical session guardrail

**Files:**
- Create: `AGENTS.md`
- Delete: `docs/PLAN.md`
- Modify: `.gitignore`
- Modify: `.env.example`
- Modify: `tests/test_docs.py`
- Modify: `src/fallguard/config.py`
- Modify: `src/fallguard/vlm.py`
- Modify: `docs/results/rule_baseline.md`
- Modify: `docs/results/cross_dataset.md`
- Modify: `docs/results/vlm_comparison.md`
- Modify: `scripts/evaluate.py`
- Modify: `scripts/error_analysis.py`
- Modify: `scripts/compare_vlm.py`

**Interfaces:**
- `AGENTS.md` is the only agent-session entry point.
- Public evidence documents may reference code, commands, citations, or other public result files; they must not depend on private `PROGRESS.md`, `PLAN2.md`, `CLAUDE.md`, or phase-number lore.

- [ ] **Step 1: Replace the plan-presence test with repository-boundary tests**

```python
def test_agent_guardrail_identifies_the_canonical_repository():
    text = _read(REPO_ROOT / "AGENTS.md")
    assert "https://github.com/kuotunyu/fall-guard-cv" in text
    assert "fall-detection-pose" in text
    assert "不得修改" in text


def test_public_docs_do_not_depend_on_private_session_files():
    paths = [REPO_ROOT / "README.md", *sorted((REPO_ROOT / "docs" / "results").glob("*.md"))]
    forbidden = ["PROGRESS.md", "PLAN2.md", "CLAUDE.md"]
    for path in paths:
        text = _read(path)
        for term in forbidden:
            assert term not in text, f"{path.relative_to(REPO_ROOT)} 仍依賴 {term}"
```

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run: `uv run pytest tests/test_docs.py -q`

Expected: FAIL because `AGENTS.md` is absent and result files reference private planning documents.

- [ ] **Step 3: Create the canonical guardrail**

```markdown
# Repository Guardrails

- Canonical remote: `https://github.com/kuotunyu/fall-guard-cv.git`.
- Before editing, verify `git remote get-url origin` and `git status --short --branch`.
- The sibling/legacy project `fall-detection-pose` is a different benchmark and must not be modified, merged, or pushed from this repository.
- Never commit `.env`, `events/`, raw datasets, model weights, private images, or private VLM descriptions.
- Lightweight validation: `uv sync --locked`, `uv run ruff check .`, `uv run pytest -q`, and `uv run python scripts/check_public_text.py --tracked`.
- GPU inference, dataset downloads, VLM calls, Discord calls, releases, and history rewrites require explicit task scope.
```

- [ ] **Step 4: Delete stale planning and migrate useful references**

Delete `docs/PLAN.md`. Replace `Dxx`, `Phase N`, `docs/PLAN.md`, `PLAN2.md`, and `CLAUDE.md` references with direct explanations, code symbols, public report links, or citations. Update `.env.example` to describe current runtime behavior directly and correct the stale statement that evaluation is fixed at two seconds.

- [ ] **Step 5: Verify no broken process references remain**

Run: `rg -n "PROGRESS\.md|PLAN2\.md|CLAUDE\.md|docs/PLAN\.md|Phase [0-9]|D[0-9]+" README.md .env.example src scripts docs/results tests AGENTS.md`

Expected: no stale process references; legitimate mathematical notation must be reviewed manually rather than mechanically removed.

- [ ] **Step 6: Run quality gates**

Run: `uv run ruff check .`

Run: `uv run pytest -q`

Run: `uv run python scripts/check_public_text.py --tracked`

Expected: all PASS.

- [ ] **Step 7: Commit the repository-boundary cleanup**

```bash
git add -A AGENTS.md .gitignore .env.example docs src scripts tests
git commit -m "docs: replace stale planning with repository guardrails"
```

---

### Task 4: Rebuild the README around evidence and limitations

**Files:**
- Modify: `README.md`
- Modify: `tests/test_docs.py`
- Modify: `src/fallguard/config.py` only if an official primary source shows a default model identifier is invalid.
- Modify: `.env.example` only if the same model-default correction is required.
- Modify: `src/fallguard/detect.py`
- Modify: `docs/results/vlm_comparison.md`
- Modify: `scripts/compare_vlm.py`

**Interfaces:**
- README headline evidence links to `docs/results/rule_baseline.md`, `docs/results/cross_dataset.md`, `docs/results/xgb_baseline.md`, and `docs/results/error_analysis.md`.
- Runtime model IDs remain environment-configurable.
- Volatile price copy is replaced with provider-agnostic wording.

- [ ] **Step 1: Add failing public-claim contracts**

```python
def test_readme_surfaces_material_evaluation_limits():
    text = _read(REPO_ROOT / "README.md")
    for required in [
        "P3/P4/P5",
        "Specificity 無法估計",
        "0.559",
        "0.000",
        "3 段 ADL",
        "不是臨床驗證",
    ]:
        assert required in text, f"README 未揭露：{required}"


def test_readme_avoids_unsupported_product_and_cost_claims():
    text = _read(REPO_ROOT / "README.md")
    for forbidden in ["能精確排除", "黃金平衡點", "單次告警成本低於", "cross-validation 與描述品質比對"]:
        assert forbidden not in text


def test_runtime_copy_avoids_hard_coded_api_price_claims():
    text = _read(REPO_ROOT / "src" / "fallguard" / "detect.py")
    assert "遠低於 $0.001" not in text
```

- [ ] **Step 2: Run focused tests and confirm they fail**

Run: `uv run pytest tests/test_docs.py -q`

Expected: FAIL on missing limitations and existing overclaims.

- [ ] **Step 3: Verify volatile external facts using primary sources**

Check official Ultralytics, Google AI, OpenAI, LangChain, URFD, and Hugging Face model pages. Retain exact model IDs or pricing only when directly supported and materially useful. Record retrieval dates in prose only where needed; otherwise remove volatile claims.

- [ ] **Step 4: Rewrite the README in this order**

1. Prototype positioning and disclaimer.
2. Demo and implemented pipeline.
3. Architecture and privacy boundary.
4. Evidence summary with window-level and event-level labels kept distinct.
5. Prominent evaluation limitations, including asymmetric LOSO negatives and Le2i failure.
6. Reproduction commands and optional inference setup.
7. Dataset/license and third-party attribution.
8. Repository map and formal software license.

Do not present the 42.8 FPS observation as a benchmark unless a committed protocol artifact supports it. Describe VLM comparison as a 12-image informal provider comparison, not cross-validation or safety validation.

- [ ] **Step 5: Remove volatile runtime cost output**

Replace hard-coded price language in `src/fallguard/detect.py` with:

```python
print("VLM 呼叫會依供應商、模型與輸入大小產生費用；請以供應商現行定價為準")
print("LOCAL_ONLY=true 可完全跳過 VLM 呼叫")
```

Update the VLM result generator and committed summary to say the 12-image comparison is informal, manually reviewed, non-blinded, and not a safety or clinical evaluation.

- [ ] **Step 6: Run all repository gates**

Run: `uv run ruff check .`

Run: `uv run pytest -q`

Run: `uv run python scripts/check_public_text.py --tracked`

Run: `git diff --check`

Expected: all PASS.

- [ ] **Step 7: Commit the evidence-calibrated public story**

```bash
git add README.md tests/test_docs.py src/fallguard/config.py src/fallguard/detect.py .env.example docs/results/vlm_comparison.md scripts/compare_vlm.py
git commit -m "docs: align portfolio claims with evidence"
```

---

### Task 5: Remove temporary planning artifacts and close the public release

**Files:**
- Delete: `docs/superpowers/specs/2026-08-12-portfolio-credibility-closure-design.md`
- Delete: `docs/superpowers/plans/2026-08-12-portfolio-credibility-closure.md`
- Modify: repository metadata or release notes only if verification supports publication.

**Interfaces:**
- Release candidate is the formal `main` branch after all local gates and GitHub Actions pass.
- Intended semantic version: `v0.1.0`, matching `pyproject.toml`, only if no conflicting tag/release exists.

- [ ] **Step 1: Remove planning-only Markdown from the release tip**

Run: `git rm -r docs/superpowers`

These documents remain in Git history but do not appear as portfolio content.

- [ ] **Step 2: Run the complete local release gate**

Run: `uv lock --check`

Run: `uv sync --locked`

Run: `uv run ruff check .`

Run: `uv run pytest -q`

Run: `uv run python scripts/check_public_text.py --tracked`

Run: `git diff --check`

Expected: all PASS without network-dependent tests, datasets, GPU work, secrets, or external API calls.

- [ ] **Step 3: Audit final repository identity and contents**

Run: `git remote -v`

Expected: only the formal `kuotunyu/fall-guard-cv` origin.

Run: `git status --short --branch`

Expected before the cleanup commit: only the intended planning-file deletions.

Run: `git ls-files`

Review: no `.env`, `events/`, raw dataset, checkpoint, private image, private VLM detail, or stale plan file.

- [ ] **Step 4: Commit release cleanup**

```bash
git add -A
git commit -m "chore: prepare fall-guard-cv v0.1.0"
```

- [ ] **Step 5: Push and wait for GitHub Actions**

Run: `git push origin main`

Run: `gh run watch --repo kuotunyu/fall-guard-cv --exit-status`

Expected: the final `main` workflow succeeds.

- [ ] **Step 6: Publish only after green CI**

First run: `gh release view v0.1.0 --repo kuotunyu/fall-guard-cv`

If no release/tag exists and all evidence is coherent, run:

```bash
gh release create v0.1.0 \
  --repo kuotunyu/fall-guard-cv \
  --target main \
  --title "fall-guard-cv v0.1.0" \
  --notes "Portfolio-ready research prototype release. Includes the edge-first pose-to-alert pipeline, reproducible lightweight quality gates, URFD LOSO evidence with explicit limitations, and transparent Le2i cross-dataset failure analysis. This is not a clinical or emergency-response product."
```

- [ ] **Step 7: Verify final public state**

Run: `git fetch origin`

Run: `git status --short --branch`

Expected: clean and aligned with `origin/main`.

Run: `gh run list --repo kuotunyu/fall-guard-cv --limit 3`

Run: `gh release view v0.1.0 --repo kuotunyu/fall-guard-cv`

Expected: green final CI and a release pointing at the verified commit.
