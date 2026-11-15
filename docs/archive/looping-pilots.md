# Historical looping pilot notes

These notes describe earlier pilots and superseded harness versions. See the [main README](../../README.md) for the submitted comparison.

# Mechanistic Intervention Benchmark

Dataset: `mechanistic-intervention-benchmark`, containing task
`stop-reasoning-loops`. MobileLLM is the current model under repair, not the
benchmark's identity. Earlier runs retain their original dataset references.
The full Valkyrie reference is
`mechanistic-intervention-benchmark/mechanistic-intervention-benchmark@v1`:
the first component is the existing parent namespace; the second is the dataset.

An agent gets the original MobileLLM weights, six development examples, a warmed evaluator and a fixed time budget on one H100. It submits a small JSON intervention; a fresh verifier applies that intervention to pristine weights.

The intervention language in `patch.py` is unchanged from the first pilot: bounded gains on selected attention heads, MLP neurons or whole branches. Diagnostic gradients and searches over those gains are allowed. Arbitrary checkpoints, added parameters, runtime hooks and prompt changes are not accepted.

## What is measured

The frozen hidden set contains seven repair cases, six specificity controls and four transfer cases. The eleven failure cases exhibit repetitive nontermination at baseline; all six controls answer correctly. These cohorts were selected from baseline measurements before the agent ran.

The primary score is **100 × max(0, newly solved failures − newly broken controls) / 11**.
Doing nothing earns zero. Each repair or transfer success earns 9.09 points; each
broken control loses 9.09 points, down to zero. Scores range from 0 to 100; the
regression count still distinguishes harmful patches from unchanged ones. Invalid submissions
receive zero and are marked invalid, rather than treated as measured model behavior.
The three cohort accuracies below are diagnostics only.

- **Repair:** correct completion on the seven known failure cases.
- **Specificity:** retention of the six baseline-correct cases.
- **Transfer:** correct completion on four failures with different content and tool situations.

A looping diagnostic requires exhaustion of the 1,024-token budget, a 12-word phrase repeated at least three times in the final 384 words, and at least 30% repeated five-grams in that tail. Exhaustion alone is not called looping. Merely stopping earlier does not earn credit: the answer must also be correct. The report distinguishes reduced looping from correct recovery.

`prompts.py` supplies tool schemas in the system message and calls the native chat template **without** its `tools` argument. This avoids the checkpoint's injected mandatory-function-call instruction. Each case is generated separately with greedy decoding, so changing the evaluation batch does not change the case's output. The original batched pilot is retained only as a historical, confounded result.

Tool calls must have the expected name and arguments, text tasks use their requested exact answer, and arithmetic controls use the final boxed answer. The small set supports a feasibility comparison, not a statistically precise leaderboard.

## Custom pilot

```sh
python looping/run.py --run runs/new-pilot
```

The controller needs Modal authentication, the MCP SDK and an authenticated Codex CLI. It measures the hidden baseline in an operator-owned environment, checks cohort eligibility, then warms a separate agent environment. The timer begins after generation and edit-reset readiness checks pass. The 600-second budget reserves its final 15 seconds for shutdown and submission capture.

Messages, tool calls and results enter one ordered `trace.jsonl` during the run. All model downloads and GPU work happen on Modal. The controller terminates its environments and stops its Modal app in a `finally` block. No persistent workspace volume is created.

Publish the saved trace and scores with the Docent SDK environment:

```sh
python looping/docent_export.py --run runs/new-pilot
```

## One-hour hosted comparison

`harbor/looping/` defines the same task with a 3,600-second agent budget. Its readiness check warms the development engine before agent execution. Only `/workspace/patch.json` crosses into the separate verifier; no model checkpoint transfer or shared volume is needed. Harbor writes net improvement divided by 100 under Valkyrie's primary `reward` key, alongside diagnostic cohort accuracies, regression count and submission validity.

Valkyrie accepts a headline when `reward` is between zero and one; our bounded
score meets this contract. Existing runs on `hour-v1` have no
dataset metric script, and `run rescore` cannot select a replacement script or
dataset version. Updating those hosted headlines requires a Valkyrie change.

Future runs use `bounded-net-v2` and the `codex-looping-v2` / `claude-looping-v2`
harnesses. Build those harnesses with `python looping/prepare_agents.py --readable`.
They print public messages, tool calls and results to the TUI in stream order,
while preserving the complete original JSON events in the archived trajectory.
Existing runs and their pinned harnesses are not changed.

`python looping/rescore.py` recomputes scores from saved verifier rewards without
inference. It preserves original `results.json`, records its hash in
`net-improvement.json`, and updates the local summary receipt. Then run
`python looping/export_comparison.py` in the Docent environment to update primary
score metadata while keeping each unified trace intact.

`package.py` checks the calibrated input hashes and keeps private cases in the verifier folder. `prepare_agents.py` pins the official Codex and Claude Code harnesses, selects high effort and disables subagents. The comparison uses two independent runs each of `gpt-6.1-sol`, `gpt-6-astra`, `claude-opus-5-5` and `claude-sonnet-5-5`.

Run artifacts and private tasks remain ignored by Git. Dataset versions, agent releases, source hashes, exact model IDs and actual elapsed times are retained with the results.
