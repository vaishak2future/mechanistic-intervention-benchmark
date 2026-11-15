# Improve MobileLLM

One agent gets model weights, an H100 and ten minutes to improve tool calling.
Valkyrie runs the agent, records its trace, invokes our verifier and stores results.
This repository contains the task and grading logic—not a separate runner or viewer.

## Task

- **Input:** pinned MobileLLM-R1-950M weights, 10 dev examples and an evaluator.
- **Submission:** full safetensors weights, split into checksummed parts by `/task/submit.py`.
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

python3 package_harbor.py --output build/sharded-v3
valkyrie dataset push mobilellm/weight-improvement build/sharded-v3 --tag v3

valkyrie run start --agent codex-mobilellm-v1 \
  --dataset mobilellm/weight-improvement@v3 \
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
| `checks/transfer/` | Submit unchanged base weights to test hosted transfer and grading |

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
The verifier uses a fresh, offline environment. Only the submitted weight parts and checksum manifest are
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

No GPU is needed for these checks.

## Check transfer and grading without training

The small scripted agent below submits unchanged base weights. It uses no model
API or training, but runs the actual H100 readiness, artifact transfer and hidden
verifier through Valkyrie. All model files remain in the cloud.

```sh
valkyrie agent push checks/transfer
valkyrie run start --agent mobilellm-transfer-check-v1 \
  --dataset mobilellm/weight-improvement@v3 --provider modal \
  --concurrency 1 --connect
```

Valkyrie's 512 MiB limit applies to each artifact, not the entire submission.
The submission helper splits a single safetensors file into at most ten 400 MiB
transport parts. Each part is explicitly declared in `task.toml`; do not replace
those declarations with the containing directory, which would create one archive.
The verifier checks regular files, size limits and the reconstructed SHA-256,
then loads the original safetensors bytes with trusted model metadata.

Use a fresh `/workspace/submission` directory when packaging. The helper accepts
full weights, not an adapter or an already-sharded Transformers directory.
The maximum submission size is 4,000 MiB. Missing, corrupt and oversized parts fail
validation. No Modal volume integration is required.

Validated on hosted Valkyrie with dataset v3, run
`20f0f4c9-bbcf-4039-93a6-8d5d4c6e51d5`: ten parts transferred a 3,798,766,415-byte
unchanged checkpoint, reconstruction passed SHA-256 validation, and the isolated
H100 verifier scored 25%, matching the cached 25% baseline (reward 0). No training
or agent-model API calls ran. Both sandboxes were removed and the Modal app stopped.
The run took about 25 minutes; archive handling and transfer dominated, while
packaging took four seconds and generation of the 40 graded answers took 13 seconds.
Small result, trace and cleanup records are under ignored `runs/sharded-transfer/`.

References: [Valkyrie](https://docs.valkyrie.vals.ai/get-started/quickstart),
[Harbor task format](https://docs.harborframework.com/tasks/overview),
[separate verification](https://docs.harborframework.com/tasks/separate-verifier).
