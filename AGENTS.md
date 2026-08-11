# Repository Guardrails

- Canonical remote: `https://github.com/kuotunyu/fall-guard-cv.git`.
- Before editing, verify `git remote get-url origin` and `git status --short --branch`.
- The sibling/legacy project `fall-detection-pose` is a different benchmark and must not be modified, merged, or pushed from this repository（不得修改）.
- Never commit `.env`, `events/`, raw datasets, model weights, private images, or private VLM descriptions.
- Lightweight validation: `uv sync --locked`, `uv run ruff check .`, `uv run pytest -q`, and `uv run python scripts/check_public_text.py --tracked`.
- GPU inference, dataset downloads, VLM calls, Discord calls, releases, and history rewrites require explicit task scope.
