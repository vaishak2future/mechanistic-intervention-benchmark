# Improve MobileLLM on Modal

One agent objective: improve the starting model's weights for tool calling.
All model downloads, training, inference, compilation, and checkpoint storage
stay on Modal. The laptop only sends code and receives reports/status.

## Build once on an H100

```sh
pip install -r requirements-modal.txt
modal setup
python modal_environment.py --run runs/h100 build \
  --model robbiemu/MobileLLM-R1-950M-mlx-fp16
```

Modal builds and caches the image remotely. The download step resolves the
requested revision to a commit and records it in `/base/source.json`. Supply
`--revision <commit>` for an explicit immutable source. For gated repositories,
add `--hf-secret <Modal-secret-name>` containing `HF_TOKEN`; this secret is used
only by the download layer, never injected into the agent sandbox.

The image includes pinned vLLM, PEFT, and Accelerate, the base checkpoint, public
tasks and evaluation scripts. No training recipe or example training answers are included. Its H100 build step must pass:

1. Load vLLM, compile kernels, capture CUDA graphs, and generate.
2. Evaluate all development tasks, saving raw samples and failure reasons.
3. Sleep/wake the engine and generate again.

These are inference readiness checks; the agent chooses and implements its own
weight-improvement method. Training is not preconfigured or smoke-tested for it.

Failures abort the build. `/prepared/readiness.json` records versions, GPU,
source revision, cache file count, and preparation duration.
`/prepared/baseline.json` contains measured scores, never placeholder results.
Compilation caches under `/cache` are baked into the image for compatible H100
runs. Build time is billable preparation time, outside the agent budget.

## Start a prepared session

```sh
python modal_environment.py --run runs/h100 start --seconds 3600 --agent-model gpt-6.1-sol --agent-reasoning high
python run_agent.py --run runs/h100
```

Starting creates an offline H100 sandbox with a private workspace Volume. Before
returning control it starts a resident vLLM engine, waits for a real generation
and sleep-mode readiness, and retrieves the prepared reports. Only then does
it set `started_at` and `deadline` in the local state and launch a trusted cloud
watchdog. The watchdog runs outside the sandbox and terminates it at the
deadline even if the laptop disconnects. A longer platform timeout is a fallback
if preparation or the watchdog fails. Commands and late submissions are also
checked against the deadline by the controller.

`run_agent.py` uses an authenticated Codex CLI and passes the model ID and
reasoning effort from run state directly to it. It starts a fresh session in an
empty temporary directory, disables the host shell, web search and subagents,
and exposes the timed H100 command interface through `modal_agent_tool.py`.
The agent reads `/task/TASK.txt` remotely. Credentials and hidden data stay with
the operator. A backend snapshot revision remains unknown unless exposed by the runner.

Public messages are captured automatically from `codex exec --json` as each
message completes. The tool bridge records calls and results in the same
append-only `trace.jsonl`, with IDs to match results to calls. Writers are locked
so concurrent events cannot corrupt the log. Reasoning events are discarded;
the agent does not need to call a logging tool or repeat its messages.

`trace.html` refreshes on disk after each event; reload it to see progress before
grading. The final report uses the same single chronological transcript, with
messages inline and expandable tool details. Results appear when they finish,
so overlapping calls do not reorder messages. Runner errors are recorded too.
The CLI is stopped 15 seconds before the sandbox deadline to leave time for
operator submission; the cloud watchdog remains authoritative.

Earlier smoke runs used the desktop subagent interface, which did not supply a
consumable event stream to this recorder. Their manually recovered messages
retain an explicit unknown-original-timestamp label; no ordering is invented.
Use `run_agent.py` for subsequent runs to get automatic capture. It requires
the Codex CLI and the dependencies in `requirements-modal.txt` on the controller.
The runner is covered by local fixtures and has completed an H100 smoke run
with automatic message capture. Its remote command tool is explicitly approved;
other host execution tools remain disabled.

The capture format follows [Codex's JSONL event stream](https://developers.openai.com/codex/noninteractive/).
The unified transcript follows [Inspect's message-history view](https://inspect.aisi.org.uk/log-viewer.html#messages).

## Warmup limits

CUDA graphs belong to a live CUDA process; an ordinary image cannot preserve
them. We capture them again before the agent clock and keep that engine alive.
It uses vLLM level-1 sleep between evaluations so training can use the GPU.
`warm_eval.py` evaluates the **base checkpoint only**, using that live engine.
`eval.py --model /workspace/submission` evaluates changed weights in a fresh
process, reusing compatible disk compilation caches; loading and graph capture
can still cost time. Changing hardware, model shape, vLLM, or compiler settings
can invalidate caches. This setup does not claim zero startup time for arbitrary
new checkpoints or GPU-memory snapshot support.

## Cache the hidden baseline once

```sh
python modal_environment.py --run runs/h100 baseline
```

This operator-only command runs the hidden baseline on the prepared image and
stores its report in ignored `private/baselines`. It stops the temporary sandbox
when finished. No rebuild or agent session is needed. The cache is never uploaded
into the agent image or mounted in training.

The cache key includes the immutable image ID (which fixes model weights,
tokenizer, evaluator, and sampling settings), GPU type, and exact holdout bytes.
Changed inputs get separate entries. Successful reports are written atomically;
failed evaluations are not cached. Keep this directory private and trusted.

`grade` reuses a matching baseline and launches only the candidate sandbox. On a
cache miss it computes and saves the baseline first. Use `--baseline-cache` to
share a trusted operator cache across run folders. This reuses a fixed sampled
baseline; it does not promise bitwise reproducibility across different GPU hosts.

## Submit and grade

```sh
python modal_environment.py --run runs/h100 submit
python modal_environment.py --run runs/h100 grade
```

The agent must finish writing `/workspace/submission/model.safetensors` before
submission. The controller flushes the Volume and waits for training termination.
Weights stay in Modal; grading mounts that workspace read-only in fresh H100
sandboxes. Only the selected weight file is copied into a clean candidate model,
using trusted metadata from the original image. Submitted code, tokenizer,
configurations, and compilation caches are never used by the grader.

`private/tasks_holdout.json` is uploaded only to grading sandboxes. The image
builder uses an explicit public-file allowlist. Existing holdout examples were
visible during authoring: replace them via `grade --holdout /private/tasks.json`
for an uncontaminated evaluation. Never feed hidden results back into training.

Each grading run automatically writes a readable `reports/report.html` with scores,
per-task comparisons, expandable sample outputs, and the agent trace when present.
To render an existing run: `python report.py runs/smoke-10m --tasks private/tasks_holdout.json`.
No cloud resources are needed to render reports.

Reports land in `runs/h100/reports`; `summary.json` records baseline, submission,
and improvement on a 0–1 scale. Candidate grading uses the same settings as the cached baseline. Invalid weights fail; no successful score is invented.
Move aside a failed `reports` directory before retrying. To cancel training:

```sh
python modal_environment.py --run runs/h100 stop
```

Workspace Volumes and image caches remain after compute stops; delete unneeded
Volumes through Modal when finished. Their name is in `state.json`. Use a new
`--run` directory for another build/session. GPU resources are allocated only
by explicit build/start/grade commands, not module imports or local tests.

## Local verification

```sh
python -m unittest -v
```

These small tests check data, grading, isolation, and timer ordering.
Actual CUDA readiness and baseline numbers require the remote image build.
The earlier Docker launcher remains available in `environment.py`, but the
prepared H100 workflow above is the supported path for this experiment.

References: [Modal image builds](https://modal.com/docs/guide/images),
[Modal Volume persistence](https://modal.com/docs/guide/volumes),
[vLLM sleep mode](https://docs.vllm.ai/en/v0.11.0/features/sleep_mode.html),
[vLLM compilation caching](https://docs.vllm.ai/en/stable/configuration/optimization/).
