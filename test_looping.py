import unittest
from looping.patch import validate_patch

class PatchTests(unittest.TestCase):
    def test_allowed_edit_and_identity(self):
        for patch in [{'edits':[]},{'edits':[{'operation':'attention_heads','layer':3,'indices':[2,4],'gain':0.5}]}]:
            self.assertEqual(validate_patch(patch),patch)
    def test_reject_arbitrary_payload_nonfinite_and_overlap(self):
        bad=[{'edits':[],'weights':'evil'}, {'edits':[{'operation':'add_delta','layer':1,'gain':1}]},
             {'edits':[{'operation':'mlp_branch','layer':1,'gain':float('nan')}]},
             {'edits':[{'operation':'attention_branch','layer':1,'gain':0.5},
                       {'operation':'attention_heads','layer':1,'indices':[2],'gain':0.5}]},
             {'edits':[{'operation':'mlp_neurons','layer':1,'indices':list(range(129)),'gain':0}]},
             {'edits':[{'operation':'mlp_branch','layer':n,'gain':0.5} for n in range(5)]}]
        for patch in bad:
            with self.subTest(patch=patch),self.assertRaises(ValueError):validate_patch(patch)

if __name__=='__main__':unittest.main()

class PromptTests(unittest.TestCase):
    def test_tools_are_optional_and_messages_are_not_mutated(self):
        from looping.prompts import render_prompt
        class Tokenizer:
            def apply_chat_template(self, messages, **options):
                self.options = options
                return str(messages)
        tokenizer = Tokenizer()
        task = {'messages': [{'role': 'system', 'content': 'Use tools only when needed.'},
                             {'role': 'user', 'content': 'What is 2 + 2?'}],
                'tools': [{'function': {'name': 'weather'}}]}
        rendered = render_prompt(tokenizer, task)
        self.assertNotIn('tools', tokenizer.options)
        self.assertIn('weather', rendered)
        self.assertEqual(task['messages'][0]['content'], 'Use tools only when needed.')
        task['tools'] = []
        self.assertNotIn('Available tools', render_prompt(tokenizer, task))

    def test_looping_requires_repeated_text_and_nontermination(self):
        from looping.prompts import loop_evidence
        repeated = ('I need to reconsider the same question because I have not yet found the answer. ' * 40)
        self.assertTrue(loop_evidence(repeated, 'length')['detected'])
        self.assertFalse(loop_evidence(repeated, 'stop')['detected'])
        self.assertFalse(loop_evidence(' '.join(str(i) for i in range(500)), 'length')['detected'])
