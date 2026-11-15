"""Verify report values and safe rendering of model/tool output."""

import json
import tempfile
import unittest
from pathlib import Path

from report import render_report, trace_section


class ReportTests(unittest.TestCase):
    def test_scores_samples_and_untrusted_text(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / 'reports').mkdir()
            (run / 'state.json').write_text('{"gpu":"H100","image_id":"image","agent":{"model":"gpt-6.1-sol","reasoning_effort":"high"}}')
            task = {"task_id":"H01", "category":"tool_choice", "samples":[
                {"passed":False,"reason":"wrong tool", "output":"<script>alert(1)</script>"}]}
            for kind, score in [('baseline',0), ('submission',1)]:
                task['samples'][0]['passed'] = bool(score)
                (run / f'reports/{kind}.json').write_text(json.dumps({'score':score,'tasks':[task]}))
            output = render_report(run).read_text()
            self.assertIn('+100 pp', output)
            self.assertIn('gpt-6.1-sol', output)
            self.assertIn('Reasoning: high', output)
            self.assertIn('Sample 1', output)
            self.assertIn('&lt;script&gt;', output)
            self.assertNotIn('<script>', output)
            self.assertIn('No trace recorded', output)

    def test_overlapping_commands_match_by_action_id(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.jsonl'
            events = [
                {'event':'action','time':1,'action_id':'a','note':'First','command':['first']},
                {'event':'action','time':2,'action_id':'b','note':'Second','command':['second']},
                {'event':'agent_message','time':1.5,'message':'Public update <safe>'},
                {'event':'result','time':3,'action_id':'b','stdout':'SECOND OUTPUT','exit_code':0},
                {'event':'result','time':4,'action_id':'a','stdout':'FIRST OUTPUT','exit_code':0},
            ]
            path.write_text('\n'.join(json.dumps(event) for event in events))
            output = trace_section(path, 0)
            self.assertLess(output.index('FIRST OUTPUT'), output.index('Second'))
            self.assertGreater(output.index('SECOND OUTPUT'), output.index('Second'))
            self.assertIn('Public update &lt;safe&gt;', output)
            self.assertLess(output.index('Public update'), output.index('Second'))
            self.assertIn('Agent messages · 1', output)
            self.assertLess(output.index('Public update'), output.index('Commands and outputs'))

    def test_recorded_smoke_totals(self):
        run = Path('runs/smoke-10m')
        if not run.exists():
            self.skipTest('Smoke artifacts are not included in source control')
        output = render_report(run, Path('private/tasks_holdout.json')).read_text()
        self.assertIn('82.5%', output)
        self.assertIn('22.5%', output)
        self.assertIn('+60 pp', output)
        self.assertIn('17 actions', output)
        self.assertEqual(output.count('class="compare"'), 40)


if __name__ == '__main__':
    unittest.main()
