"""Evaluate tool choice, arguments, and abstention on hand-written chat tasks.

    python eval.py --model /path/to/model --tasks tasks.json

Only text after </think> is graded. Each sample must contain exactly one JSON
call for a ``call`` task, or no calls and the required text for a ``no_call``
task. Task scores estimate pass@1 by averaging independent samples.
"""

import argparse
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterator


@dataclass(frozen=True)
class ToolCall:
    name: Any
    arguments: Any


@dataclass(frozen=True)
class SampleResult:
    passed: bool
    reason: str
    output: str = ""


@dataclass(frozen=True)
class TaskResult:
    task_id: str
    category: str
    samples: tuple[SampleResult, ...]

    @property
    def score(self) -> float:
        return mean(sample.passed for sample in self.samples)


def final_answer(output: str) -> str:
    """Discard reasoning before the last closing think tag."""
    return output.rsplit("</think>", 1)[-1].strip()


def json_values_in(answer: str) -> Iterator[Any]:
    """Read JSON objects and arrays embedded in prose or Markdown fences.

    Let the JSON decoder handle braces inside strings and escaped quotes.
    Consume each complete value once so argument objects aren't counted again.
    """
    decoder = json.JSONDecoder()
    position = 0
    while position < len(answer):
        if answer[position] not in "{[":
            position += 1
            continue
        try:
            value, end = decoder.raw_decode(answer, position)
        except json.JSONDecodeError:
            position += 1
            continue
        yield value
        position = end


def calls_in(value: Any) -> Iterator[ToolCall]:
    """Recognize a call object or a list of calls, without inspecting arguments."""
    if isinstance(value, list):
        for item in value:
            yield from calls_in(item)
    elif isinstance(value, dict) and "name" in value:
        yield ToolCall(value["name"], value.get("parameters", value.get("arguments", {})))


def parse_calls(answer: str) -> list[ToolCall]:
    return [call for value in json_values_in(answer) for call in calls_in(value)]


def normalized(value: Any) -> str:
    return str(value).strip().lower()


def argument_matches(value: Any, matcher: dict[str, Any]) -> bool:
    if value is None:
        return False
    if "eq" in matcher:
        return normalized(value) == normalized(matcher["eq"])
    if "contains" in matcher:
        return all(text.lower() in normalized(value) for text in matcher["contains"])
    if "set" in matcher:
        items = value if isinstance(value, list) else re.split(r",\s*|\s+and\s+", str(value))
        return {normalized(item) for item in items} == {
            normalized(item) for item in matcher["set"]
        }
    raise ValueError(f"Unknown argument matcher: {matcher!r}")


def grade_expected_call(expectation: dict[str, Any], calls: list[ToolCall]) -> SampleResult:
    if len(calls) != 1:
        return SampleResult(False, f"expected exactly one tool call; found {len(calls)}")

    call = calls[0]
    if call.name != expectation["name"]:
        return SampleResult(False, f"wrong tool {call.name!r}; expected {expectation['name']!r}")
    if not isinstance(call.arguments, dict):
        return SampleResult(False, "tool arguments must be a JSON object")
    for name, matcher in expectation.get("args", {}).items():
        value = call.arguments.get(name)
        if not argument_matches(value, matcher):
            return SampleResult(False, f"bad argument {name}={value!r}; expected {matcher!r}")
    return SampleResult(True, "ok")


def grade_expected_answer(
    expectation: dict[str, Any], answer: str, calls: list[ToolCall]
) -> SampleResult:
    if calls:
        return SampleResult(False, f"expected no tool calls; found {len(calls)}")
    missing = [
        text for text in expectation.get("answer_contains", [])
        if text.lower() not in answer.lower()
    ]
    if missing:
        return SampleResult(False, f"answer missing {missing!r}")
    return SampleResult(True, "ok")


def grade_sample(
    task: dict[str, Any], output: str, finish_reason: str | None = None
) -> SampleResult:
    if finish_reason == "length":
        return SampleResult(False, "truncated")

    expectation = task["expect"]
    answer = final_answer(output)
    calls = parse_calls(answer)
    if expectation["type"] == "call":
        return grade_expected_call(expectation, calls)
    if expectation["type"] == "no_call":
        return grade_expected_answer(expectation, answer, calls)
    raise ValueError(f"Unknown expectation type: {expectation['type']!r}")


def load_tasks(paths: list[str]) -> list[dict[str, Any]]:
    tasks = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as source:
            tasks.extend(json.load(source))
    if not tasks:
        raise ValueError("Provide at least one task to evaluate.")
    return tasks


def render_prompt(tokenizer, task) -> str:
    """Ensure tools are visible even when a model's chat template ignores tools."""
    messages = [dict(message) for message in task["messages"]]
    template_options = dict(tools=task["tools"], tokenize=False, add_generation_prompt=True)
    prompt = tokenizer.apply_chat_template(messages, **template_options)
    names = [tool["function"]["name"] for tool in task["tools"]]
    if not all(name in prompt for name in names):
        schemas = "Available tools (JSON schemas):\n" + json.dumps(task["tools"])
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] += "\n\n" + schemas
        else:
            messages.insert(0, {"role": "system", "content": schemas})
        prompt = tokenizer.apply_chat_template(messages, **template_options)
    return prompt


def evaluate(tasks: list[dict[str, Any]], options: argparse.Namespace, model=None) -> list[TaskResult]:
    # Keep grading and reporting usable without installing the inference runtime.
    from vllm import LLM, SamplingParams

    if model is None:
        model = LLM(options.model, dtype="bfloat16", max_model_len=8192, seed=0)
    tokenizer = model.get_tokenizer()
    prompts = [render_prompt(tokenizer, task) for task in tasks]
    sampling = SamplingParams(
        n=options.n, temperature=0.6, top_p=0.95, max_tokens=options.max_tokens, seed=0
    )
    outputs = model.generate(prompts, sampling)
    results = []
    for task, output in zip(tasks, outputs, strict=True):
        if len(output.outputs) != options.n:
            raise ValueError(f"Task {task['id']}: expected {options.n} samples, got {len(output.outputs)}")
        results.append(TaskResult(
            task_id=str(task["id"]),
            category=task["category"],
            samples=tuple(
                sample_result(task, sample)
                for sample in output.outputs
            ),
        ))
    return results


def sample_result(task, sample) -> SampleResult:
    result = grade_sample(task, sample.text, sample.finish_reason)
    return SampleResult(result.passed, result.reason, sample.text)


def result_report(results) -> dict:
    return {"score": mean(result.score for result in results),
            "tasks": [asdict(result) for result in results]}


def print_report(results: list[TaskResult]) -> None:
    scores_by_category = defaultdict(list)
    for result in results:
        print(f"{result.task_id:>6}  {result.category:<20} {result.score:.2f}")
        for number, sample in enumerate(result.samples, start=1):
            status = "PASS" if sample.passed else "FAIL"
            print(f"        sample {number}: {status} — {sample.reason}")
        scores_by_category[result.category].append(result.score)

    print(f"\nTOTAL {sum(result.score for result in results):.1f}/{len(results)}")
    for category, scores in scores_by_category.items():
        print(f"  {category:<20} {100 * mean(scores):5.1f}%  (n={len(scores)})")


def positive_integer(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def parse_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--report", help="Write scores and sample results as JSON")
    parser.add_argument("--tasks", nargs="+", default=["tasks.json"])
    parser.add_argument(
        "--n", type=positive_integer, default=4,
        help="samples per task (pass@1 = mean over samples)",
    )
    parser.add_argument("--max-tokens", type=positive_integer, default=3072)
    return parser.parse_args()


def main() -> None:
    options = parse_options()
    tasks = load_tasks(options.tasks)
    results = evaluate(tasks, options)
    print_report(results)
    if options.report:
        report = result_report(results)
        Path(options.report).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
