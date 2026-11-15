"""Data and masking checks; no training libraries or weights needed locally."""
import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from eval import grade_sample, render_prompt
from train import encode_example


class TrainingTests(unittest.TestCase):
    def test_starter_answers_pass_development_expectations(self):
        tasks = json.loads(Path('tasks.json').read_text())
        examples = [json.loads(line) for line in Path('train.jsonl').read_text().splitlines()]
        self.assertEqual(len(tasks), len(examples))
        for task, example in zip(tasks, examples):
            self.assertEqual(task['messages'], example['messages'])
            self.assertTrue(grade_sample(task, example['answer']).passed, task['id'])

    def test_labels_mask_prompt_and_include_eos(self):
        tokenizer = MagicMock(eos_token_id=99)
        tokenizer.encode.side_effect = [[1,2,3], [4,5]]
        with unittest.mock.patch('eval.render_prompt', return_value='prompt'):
            encoded = encode_example(tokenizer, {'answer':'answer'}, 10)
        self.assertEqual(encoded['labels'], [-100,-100,-100,4,5,99])
        self.assertEqual(encoded['input_ids'], [1,2,3,4,5,99])

    def test_overlong_examples_fail_instead_of_losing_answer(self):
        tokenizer = MagicMock(eos_token_id=99)
        tokenizer.encode.side_effect = [[1,2,3], [4,5]]
        with unittest.mock.patch('eval.render_prompt', return_value='prompt'), self.assertRaises(ValueError):
            encode_example(tokenizer, {'answer':'answer'}, 4)

    def test_ignored_tools_are_added_to_system_message(self):
        task = json.loads(Path('tasks.json').read_text())[0]
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.side_effect = lambda messages, **kwargs: json.dumps(messages)
        prompt = render_prompt(tokenizer, task)
        for tool in task['tools']:
            self.assertIn(tool['function']['name'], prompt)
        self.assertNotIn('Available tools', task['messages'][0]['content'])
