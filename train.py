"""Small LoRA starter: train on assistant answers, merge, and save full weights."""

import argparse
import json
from pathlib import Path


def encode_example(tokenizer, example, max_length):
    from eval import render_prompt

    prompt = render_prompt(tokenizer, example)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(example["answer"], add_special_tokens=False)
    answer_ids.append(tokenizer.eos_token_id)
    if len(prompt_ids) + len(answer_ids) > max_length:
        raise ValueError("Training example exceeds --max-length; do not silently drop its answer.")
    return {"input_ids": prompt_ids + answer_ids,
            "labels": [-100] * len(prompt_ids) + answer_ids}


def train(options):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Training requires a BF16-capable CUDA GPU.")
    set_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(options.model, local_files_only=True)
    examples = [json.loads(line) for line in options.data.read_text().splitlines() if line.strip()]
    if not examples:
        raise ValueError("Training data is empty.")
    encoded = [encode_example(tokenizer, example, options.max_length) for example in examples]
    model, loading = AutoModelForCausalLM.from_pretrained(
        options.model, torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa", output_loading_info=True,
    )
    if loading["missing_keys"] or loading["unexpected_keys"] or loading.get("mismatched_keys"):
        raise RuntimeError(f"Checkpoint does not match its Transformers architecture: {loading}")
    model = model.cuda()
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=8, lora_alpha=16, target_modules="all-linear",
    ))
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=options.learning_rate,
    )
    for step in range(options.steps):
        example = encoded[step % len(encoded)]
        batch = {key: torch.tensor([value], device="cuda") for key, value in example.items()}
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        if not torch.isfinite(loss):
            raise RuntimeError("Training produced a nonfinite loss.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        print(f"step {step + 1}/{options.steps}: loss={loss.item():.4f}", flush=True)
    model = model.merge_and_unload()
    model.config.use_cache = True
    model.save_pretrained(options.output, safe_serialization=True, max_shard_size="10GB")
    tokenizer.save_pretrained(options.output)
    if not (options.output / "model.safetensors").is_file():
        raise RuntimeError("Expected a single merged model.safetensors file.")
    print(f"Saved merged checkpoint to {options.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/base")
    parser.add_argument("--data", type=Path, default=Path("/task/train.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("/workspace/submission"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=4096)
    options = parser.parse_args()
    if options.steps < 1 or options.learning_rate <= 0:
        parser.error("steps and learning rate must be positive")
    train(options)


if __name__ == "__main__":
    main()
