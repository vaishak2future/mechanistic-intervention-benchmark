# Improve MobileLLM's tool calling

Improve the supplied model weights so the model performs better on unseen
tool-calling tasks. Choose your own approach within the 10-minute budget.

You have one H100, PyTorch, Transformers, PEFT, Accelerate and vLLM.

| Resource | Location |
| --- | --- |
| Starting weights and tokenizer | `/base` |
| Development tasks and expected results | `/task/tasks.json` |
| Evaluator | `/task/eval.py` |
| Development baseline | `/prepared/baseline.json` |
| Writable workspace | `/workspace` |

Evaluate a complete checkpoint with:

```sh
python /task/eval.py --model /workspace/checkpoint \
  --tasks /task/tasks.json --report /workspace/dev-results.json
```

The base model also has a warmed evaluator at `/task/warm_eval.py`. It sleeps
between requests to release GPU memory. Changed checkpoints load in a fresh process.

Save your final **full model weights** to:

```text
/workspace/submission/model.safetensors
```

Only this file is accepted. The original model configuration, tokenizer,
chat template and evaluator stay fixed. Adapters alone are not a submission.

After your run ends, an isolated verifier evaluates the submitted weights on
hidden tasks. Each task gets four sampled answers. Every answer is graded:
a tool request needs exactly one JSON call with the expected name and arguments;
an ordinary answer needs no call and the required answer text. Truncation fails.

Your reward is hidden success rate minus the starting model's hidden success
rate. This is an average success rate, not pass@4. No training recipe is provided.
