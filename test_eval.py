"""Regression coverage for call counting and per-sample reporting."""

import io
import json
import unittest
from contextlib import redirect_stdout

from eval import SampleResult, TaskResult, grade_sample, parse_calls, print_report


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.task = {"expect": {"type": "call", "name": "search", "args": {"query": {"eq": "hello"}}}}
        self.call = json.dumps({"name": "search", "arguments": {"query": "hello"}})

    def test_one_fenced_call_passes(self):
        self.assertTrue(grade_sample(self.task, f"```json\n{self.call}\n```").passed)

    def test_multiple_calls_fail_in_prose_and_arrays(self):
        for answer in [f"{self.call}\n{self.call}", f"[{self.call}, {self.call}]"]:
            with self.subTest(answer=answer):
                result = grade_sample(self.task, answer)
                self.assertFalse(result.passed)
                self.assertIn("found 2", result.reason)

    def test_json_strings_and_nested_arguments_do_not_create_calls(self):
        arguments = {"query": 'brace } and escaped " quote', "nested": {"name": "data"}}
        answer = json.dumps({"name": "search", "parameters": arguments})
        calls = parse_calls(answer)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].arguments, arguments)

    def test_reasoning_is_excluded(self):
        self.assertTrue(grade_sample(self.task, f"{self.call}</think>{self.call}").passed)

    def test_no_call_requires_text_and_rejects_calls(self):
        task = {"expect": {"type": "no_call", "answer_contains": ["hello"]}}
        self.assertTrue(grade_sample(task, "Hello there").passed)
        self.assertFalse(grade_sample(task, "goodbye").passed)
        self.assertFalse(grade_sample(task, self.call).passed)

    def test_failures_explain_the_sample(self):
        cases = [
            ("plain text", None, "found 0"),
            ('{"name":"other"}', None, "wrong tool"),
            ('{"name":"search","arguments":[]}', None, "JSON object"),
            ('{"name":"search","arguments":{"query":"wrong"}}', None, "bad argument"),
            (self.call, "length", "truncated"),
        ]
        for answer, finish_reason, reason in cases:
            with self.subTest(reason=reason):
                result = grade_sample(self.task, answer, finish_reason)
                self.assertFalse(result.passed)
                self.assertIn(reason, result.reason)

    def test_report_preserves_each_result_and_averages_scores(self):
        result = TaskResult("1", "search", (
            SampleResult(True, "ok"),
            SampleResult(False, "wrong tool 'other'"),
            SampleResult(False, "truncated"),
            SampleResult(True, "ok"),
        ))
        output = io.StringIO()
        with redirect_stdout(output):
            print_report([result])
        report = output.getvalue()
        for line in ["sample 1: PASS — ok", "sample 2: FAIL — wrong tool 'other'",
                     "sample 3: FAIL — truncated", "sample 4: PASS — ok",
                     "TOTAL 0.5/1", "50.0%"]:
            self.assertIn(line, report)


if __name__ == "__main__":
    unittest.main()
