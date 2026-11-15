# Improve MobileLLM

One agent gets model weights, an H100 and ten minutes to improve tool calling.
Valkyrie runs the agent, records its trace, invokes our verifier and stores results.
This repository contains the task and grading logic—not a separate runner or viewer.

## Task

- **Input:** pinned MobileLLM-R1-950M weights, 10 dev examples and an evaluator.
- **Submission:** `/workspace/submission/model.safetensors`, containing full weights.
- **Evaluation:** 10 hidden tasks, each sampled four times.
- **Reward:** submitted success rate minus the cached baseline success rate.

Every sampled answer counts. This estimates average success, not pass@4.
The suite is a small smoke test; 100% does not imply broad tool-use ability.
The agent chooses its own training approach.

## Run

The team organization already has `codex-mobilellm-v1`, the `openai-api-key` secret
and a Modal provider configured with `modal-credentials`.

```sh
uv tool install valkyrie-cli
valkyrie whoami                 # Run valkyrie login only if needed.
valkyrie org use hackathon-studentt-565bbbff14

python3 package_harbor.py --output build/v2
valkyrie dataset push mobilellm/weight-improvement build/v2 --tag v2

valkyrie run start --agent codex-mobilellm-v1 \
  --dataset mobilellm/weight-improvement@v2 \
  --model gpt-6.1-sol --provider modal \
  -s OPENAI_API_KEY openai-api-key --concurrency 1 --connect
```

Use a fresh output directory and tag when publishing another version.
Watch shared traces with `valkyrie tui`; retrieve scores with
`valkyrie run results RUN_ID`.

## Files

| Path | Purpose |
| --- | --- |
| `harbor/mobilellm/instruction.md` | The agent's task |
| `harbor/mobilellm/task.toml` | H100 resources, time limits and isolated grading |
| `harbor/mobilellm/environment/` | Cloud image, weight download and GPU readiness |
| `harbor/mobilellm/tests/` | Weight-only submission validation and hidden grading |
| `eval.py`, `tasks.json` | Evaluator and public dev examples |
| `package_harbor.py` | Assemble a self-contained upload package |
| `prepare_valkyrie_agent.py` | Configure the official Codex package for our model |
| `test_*.py` | Lightweight local checks |

`private/`, `build/` and `runs/` are ignored by Git. Hidden tasks remain versioned
on `local/private-holdout`. Packaging includes them only in the verifier folder,
never the agent image. No weights are downloaded to this machine.

## Setup and grading

The cloud image pins the model revision and inference dependencies. The readiness
check generates a dev baseline, compiles kernels and warms a resident vLLM engine
before the agent timer. Changed checkpoints still require a new model load.
The portable image warms GPU caches during setup; it does not reuse the old native
Modal image's baked caches.

The hosted agent needs network access for installation and its model API.
The verifier uses a fresh, offline environment. Only the submitted weight file is
transferred; configuration, tokenizer and evaluator come from trusted sources.
The cached baseline is checked against the holdout and evaluator fingerprints.

The verifier writes per-sample results, a score summary and Harbor's `reward.txt`
under `/logs/verifier`. Valkyrie handles logs and presentation. Infrastructure
errors fail the run instead of producing an invented score. The hosted platform
may retain checkpoint artifacts separately from its temporary compute.

## Recreate the team agent

The official Codex package has an older model allowlist. This setup helper adds
`gpt-6.1-sol` and sets high reasoning; execution and trace collection remain in
the official package hosted by Valkyrie.

```sh
valkyrie agent install https://github.com/vals-ai/public-agent-registry/tree/main/examples/codex-v1.0.0
valkyrie agent download codex-v1.0.0 --output-dir build/agents
python3 prepare_valkyrie_agent.py build/agents/codex-v1.0.0 build/agents/codex-mobilellm-v1
valkyrie agent push build/agents/codex-mobilellm-v1
```

## Check locally

```sh
python3 -m unittest -q
```

No GPU is needed for these checks. The first hosted run verified H100 allocation
and pre-timer readiness, but Valkyrie rejected the 1.9 GB checkpoint during transfer
to the separate verifier (`size limit exceeded`). Hidden grading remains blocked
until that transfer limit is resolved; do not treat the package as end-to-end validated.

### Checkpoint handoff status

Valkyrie's public tracker source sets `MAX_ARTIFACT_BYTES` to 512 MiB. Its Modal
adapter supports volumes, but the Harbor dataset configuration and separate-verifier
creation path do not forward volume mounts. Adding a mount to `task.toml` alone
is not a supported fix.

A CPU-only Modal test transferred 1,899,395,112 random bytes through a v2 Volume
between two offline sandboxes in 21 seconds. The SHA-256 hashes matched; the reader
could not write or see the sibling run folder. Both sandboxes and the volume were
deleted. Evidence and the reproduction script are in ignored `runs/valkyrie/`.
This validates storage, not the hosted Harbor integration or GPU grading.

To use volumes, Valkyrie must expose mounts in the dataset environment schema and
forward them to both generation and separate-verifier creation. Use a unique
submission folder per attempt, persist writes and stop the writer before grading,
mount that folder read-only in the verifier, and delete it after results are saved.
The checkpoint must then be removed from `artifacts` to avoid the archive limit.
Keep hidden tests outside the shared volume and do not give the agent Modal credentials.

Source inspected: [Valkyrie tracker, b04b9fa](https://github.com/vals-ai/valkyrie/tree/b04b9fac1b5a3ef3dc2703d8e7bd038923a437f0/services/tracker/src/tracker/services/datasets).
After the provider integration is available, test it through a CPU-only hosted task
before running another H100 training task.

References: [Valkyrie](https://docs.valkyrie.vals.ai/get-started/quickstart),
[Harbor task format](https://docs.harborframework.com/tasks/overview),
[separate verification](https://docs.harborframework.com/tasks/separate-verifier).
