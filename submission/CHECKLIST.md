# Submission preparation

The main README is the canonical submission writeup. Remaining access and reproducibility work is listed below.

## Required before submission

- [x] **Add the author-provided opening paragraph.** Added to the main README, with light AI-assisted edits requested by the author.
- [x] **Create the private GitHub repository.** [mechanistic-intervention-benchmark](https://github.com/vaishak2future/mechanistic-intervention-benchmark) is verified private and connected as `origin`.
- [x] **Review the submission code for commit and push.** Lightweight tests and credential/history checks passed. Push only `master`; keep `private/`, `runs/`, and `build/` ignored and `local/private-holdout` local.
- [ ] **Resolve judge access to the data.** Six dev examples are public. Hidden cases and calibration artifacts are private and ignored, so a fresh clone cannot recreate the complete task. Choose controlled judge access to the existing Valkyrie dataset, or an explicit release plan for the private data. Do not silently publish it. If releasing later, state when and under what license.
- [x] **Close the Inkling attempts.** Both 45-minute attempts failed before grading; traces are retained, compute is stopped, and no retries are planned. They are excluded from scored averages; the chart covers the eight completed four-model trials.
- [x] **Promote the submission writeup.** The main README contains the current methods, results, limitations, and author-provided opening. Earlier training and pilot documentation is archived under `docs/archive/`.
- [ ] **Test the release from a clean checkout.** `prepare_agents.py --readable` currently assumes existing pinned v1 agent bundles in ignored `build/agents/`; that is not a complete bootstrap for an external user. Supply/document the official-package download and pinning sequence or distribute appropriately licensed self-contained agent releases. Supply dependencies for optional controller/Docent workflows if they are advertised.
- [ ] **Choose licenses and public materials.** No repository license was found. Confirm permissions for benchmark code/data and preserve third-party model/harness license requirements. Keep credentials and raw/private artifacts out of the public tree. Verify judges can open Docent or provide selected redacted public trace excerpts.

## Already checked for this draft

- Eight finished `submission-45m` trials, all valid, from saved verifier receipts; exact model IDs and run IDs included.
- Score formula and cohort sizes match `looping/scoring.py`; reported counts and rounded agent durations match saved results.
- Figure uses a full 0–100 axis, trial means and individual trial dots; it does not imply a confidence interval or statistical ranking.
- Failure example D02 comes from public dev data and its serial calibrated dev baseline. No private prompt text is included.
- Both the graph SVG and run-ID cross-check parse/pass. The documented local command passed **18 tests** on 2026-11-15: `python3 -m unittest test_eval test_looping test_looping_scoring test_live_messages`.
- `runs/looping-sol-clean-serial-10m/manifest.json` matches the current frozen inputs and is the calibration directory used in the draft package command. Older `runs/looping-clean-serial-calibration/manifest.json` does not match the current private holdout; do not substitute it.
- Cleanup receipts report no remaining comparison environments for the eight completed trials or the failed Inkling attempts.

## Draft files and provenance

- `../README.md`: canonical submission writeup. `README.draft.md` redirects readers there.
- `results.svg`: finished four-model cohort only.
- `render_results.py`: regenerates the graph from local ignored `runs/mechanistic-submission-45m/summary.json`; no private data, model downloads, or GPU work. Regeneration requires those saved aggregate receipts, but viewing the SVG does not.

The score tables derive from `runs/mechanistic-submission-45m/summary.json` and each trial's saved `collected.json`. The public D02 observation derives from `runs/looping-sol-clean-serial-10m/dev-baseline.json`. Raw provenance artifacts stay outside Git. For an independently auditable public release, consider a small reviewed aggregate results file containing only model IDs, run IDs, rewards, cohort counts, timings, and source hashes; this draft has not copied any raw run artifacts into the submission folder.
