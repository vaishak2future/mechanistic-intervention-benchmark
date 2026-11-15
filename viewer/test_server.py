import unittest
from viewer.server import leaderboard, normalize_trace, score_for, redact, snapshot_messages
from tempfile import TemporaryDirectory
from pathlib import Path
import json

class ViewerTests(unittest.TestCase):
    def test_pending_is_not_zero(self):
        self.assertIsNone(score_for('IN_PROGRESS', {'reward': 0}))
        self.assertEqual(score_for('FINISHED', {'reward': 0}), 0)
        self.assertIsNone(score_for('ERROR', {'reward': 1}))

    def test_separate_budgets_and_exclude_unscored(self):
        base = dict(model='sol', agent='codex', name='sol', label='Sol', scores={'repair': 0.5})
        rows = [{**base, 'run_id':'a', 'cohort':'comparison', 'budget_seconds':2700,'score':20},
                {**base, 'run_id':'b', 'cohort':'comparison', 'budget_seconds':2700,'score':None},
                {**base, 'run_id':'c', 'cohort':'smoke', 'budget_seconds':600,'score':80}]
        groups = leaderboard(rows)
        self.assertEqual(len(groups), 2)
        comparison = next(g for g in groups if g['cohort']=='comparison')
        self.assertEqual(comparison['mean'],20)
        self.assertEqual(comparison['completed'],1)

    def test_codex_interleaves_messages_and_tools(self):
        events = [{'type':'item.completed','item':{'type':'agent_message','text':'Plan'}},
                  {'type':'item.started','item':{'type':'command_execution','id':'x','command':'ls'}},
                  {'type':'item.completed','item':{'type':'command_execution','id':'x','command':'ls','aggregated_output':'a','exit_code':0}},
                  {'type':'item.completed','item':{'type':'agent_message','text':'Done'}}]
        messages = normalize_trace(events)
        self.assertEqual([m['kind'] for m in messages],['message','tool','result','message'])
        self.assertEqual(messages[1]['call_id'],messages[2]['call_id'])

    def test_claude_omits_reasoning_and_preserves_public_sequence(self):
        events = [{'type':'assistant','message':{'id':'a','content':[{'type':'thinking','thinking':'secret'},{'type':'text','text':'Hello'},{'type':'tool_use','id':'t','name':'Bash','input':{'command':'ls'}}]}},
                  {'type':'user','message':{'content':[{'type':'tool_result','tool_use_id':'t','content':'files'}]}}]
        messages = normalize_trace(events)
        self.assertEqual([m['kind'] for m in messages],['message','tool','result'])
        self.assertNotIn('secret',str(messages))

    def test_snapshot_groups_messages_and_calls(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/'trace.jsonl'
            path.write_text('\n'.join(json.dumps({'message':line}) for line in ['Setup', 'Agent', 'My plan', 'Tool · Bash', '$ ls', 'Result · x', 'files']))
            messages=snapshot_messages(path)
        self.assertEqual([m['kind'] for m in messages],['message','tool','result'])
        self.assertEqual(messages[0]['text'],'My plan')

    def test_secret_redaction(self):
        self.assertNotIn('sk-abcdefghijklmnopqrstuv',redact('token sk-abcdefghijklmnopqrstuv'))

if __name__ == '__main__': unittest.main()
