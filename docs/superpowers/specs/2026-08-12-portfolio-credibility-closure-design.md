# fall-guard-cv Portfolio Credibility Closure Design

**Date:** 2026-08-12
**Status:** Approved
**Scope:** `kuotunyu/fall-guard-cv` only

## 1. Problem

`fall-guard-cv` is already a substantial computer-vision and AI-engineering portfolio project, with a clean `main`, a public GitHub repository, 81 passing tests, a working public-copy scanner, evaluation reports, and a successful GitHub Actions run. The remaining risk is not missing product functionality. It is credibility drift: public claims, research limitations, dependency checks, and repository guidance are not yet aligned tightly enough for long-term public maintenance.

The repository must remain an engineering and research prototype. It must not imply clinical validation, guaranteed fall detection, or production emergency-response readiness.

## 2. Goals

- Present a clear Computer Vision / AI Engineering portfolio story.
- Keep the existing pose, feature, classifier, state-machine, VLM, and notification pipeline intact.
- Make public claims traceable to committed reports, configuration, code, or reproducible commands.
- Surface material evaluation limitations near the corresponding results.
- Make CI enforce lint, tests, and public-copy hygiene.
- Reduce stale planning material and instructions that can confuse future Codex sessions.
- Leave one obvious canonical local checkout and GitHub remote for this project.

## 3. Non-goals

- No new model architecture, UI, agent, RAG, or unrelated feature work.
- No merger with the separate legacy `fall-detection-pose` repository.
- No expensive GPU retraining or dataset-wide inference unless an existing published claim cannot otherwise be verified.
- No claim that the system is a medical device, clinical product, emergency service, or validated home-safety product.
- No rewrite of historical Git commits merely to cosmetically alter old authorship or documentation.

## 4. Public Positioning

The primary positioning is:

> An edge-first fall-detection research prototype demonstrating pose inference, interpretable temporal features, rule and XGBoost baselines, stateful alert logic, privacy-aware multimodal escalation, and honest cross-dataset evaluation.

The README must distinguish:

- implemented capability from demonstrated result;
- evaluation configuration from deployment defaults;
- in-dataset estimates from cross-dataset generalization;
- local processing from event-triggered external VLM or Discord transmission;
- software prototype status from clinical or operational readiness.

## 5. Repository Changes

### 5.1 Documentation and claims

- Audit README numbers and model/configuration names against code, lockfile, and committed reports.
- Replace categorical claims such as reliably or precisely excluding common activities with evidence-calibrated wording.
- Place the important URFD and Le2i limitations beside the result summary, including:
  - only five identified URFD subjects;
  - ADL negatives appearing only for two subjects, leaving three LOSO folds without specificity estimates;
  - the timing-search candidate range having been informed by full-dataset exploration;
  - weak Le2i cross-dataset specificity and the small number of Le2i ADL videos;
  - evaluation thresholds not being interchangeable with the deployment confirmation default.
- Keep dataset licensing and third-party attribution explicit.
- Retain result documents that provide evidence. Remove or replace public planning instructions that reference private files, stale session mechanics, or superseded implementation steps.

### 5.2 Quality gates

- Add Ruff as an explicit development dependency and define its configuration.
- Make GitHub Actions run, at minimum:
  1. locked dependency installation;
  2. Ruff;
  3. the full lightweight pytest suite;
  4. the public-copy scanner.
- Keep CI free of dataset downloads, model downloads, secrets, GPU requirements, and external API calls.
- Preserve the existing secret/path scanner and strengthen it only where tests can demonstrate the new rule.

### 5.3 Maintainability and session safety

- Ensure README quick-start commands match the actual package and dependency layout.
- Provide one concise contributor/session entry point containing canonical path, remote, safe validation commands, prohibited public artifacts, and the boundary with `fall-detection-pose`.
- Prefer executable checks over prose-only instructions.
- Do not retain redundant public Markdown files solely as AI-session memory.

### 5.4 Release closure

- Verify local `main` against `origin/main`, tracked files, repository metadata, CI state, and public links.
- Create a release-ready commit only after all local gates pass.
- Create a GitHub release only if the repository state, documentation, and evidence form a coherent versioned snapshot. A release must not contain datasets, private images, secrets, local artifacts, or unsupported model binaries.

## 6. Verification

The final implementation is acceptable only when all of the following are true:

- `git status` is clean after committed changes.
- Local branch and formal GitHub remote are unambiguous.
- Ruff passes.
- All lightweight tests pass without network, dataset, GPU, or secret dependencies.
- The public-copy scanner passes.
- README headline numbers match committed evidence.
- README limitations describe the actual evaluation design and failures.
- No tracked secret, private path, private planning file, raw dataset, checkpoint, or private home image is introduced.
- GitHub Actions passes on the final commit.

## 7. Risks and Controls

- **Risk: documentation cleanup removes useful evidence.** Keep result artifacts and citations; remove only redundant process narration or replace it with concise maintained guidance.
- **Risk: stronger CI installs the heavy CUDA stack.** Separate lightweight quality dependencies from optional inference dependencies where needed and verify CI remains CPU-only.
- **Risk: changing wording hides poor results.** Preserve numeric results and failure disclosures; improve interpretation, not appearance.
- **Risk: unverified current model names or pricing become stale claims.** Remove unnecessary volatile claims or verify them against primary sources before retaining them.
- **Risk: accidental collision with the legacy fall project.** Never change or push `fall-detection-pose` as part of this work.

## 8. Expected Outcome

The repository should be suitable for a CV Engineer, AI Engineer, ML Engineer, or AI Application Engineer application because it demonstrates a complete inference-to-alert pipeline, reproducible software practices, evaluation discipline, privacy boundaries, and candid failure analysis. Its value comes from trustworthy engineering evidence, not from presenting the prototype as a solved safety product.
