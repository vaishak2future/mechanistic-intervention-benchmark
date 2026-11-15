"""Fixed-decoding evaluation of bounded weight interventions, with a warm dev server."""
import argparse
from collections import Counter
from functools import partial
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import re
from pathlib import Path
import time
from urllib.request import Request, urlopen

from eval import final_answer, parse_calls
from prompts import render_prompt, loop_evidence
from patch import apply_patch_to_model, validate_patch, LIMITS

MAX_TOKENS = 1024


def grade(task, text, finish_reason):
    if finish_reason == 'length':
        return False, 'budget_exhausted'
    answer = final_answer(text)
    calls = parse_calls(answer)
    expected = task['expect']
    if expected['type'] == 'boxed':
        boxes = re.findall(r'\\boxed\{([^{}]+)\}', answer)
        return (not calls and bool(boxes) and boxes[-1].strip() == expected['text']), 'final_boxed_answer'
    if expected['type'] == 'text':
        return (not calls and answer == expected['text']), 'exact_answer'
    return (len(calls) == 1 and calls[0].name == expected['name'] and
            calls[0].arguments == expected['arguments']), 'tool_call'


def repetition(text):
    words = text.lower().split()
    grams = [tuple(words[i:i+5]) for i in range(max(0, len(words)-4))]
    return 0.0 if not grams else 1 - len(set(grams)) / len(grams)


class Evaluator:
    def __init__(self):
        from vllm import LLM, SamplingParams
        self.engine = LLM('/base', dtype='bfloat16', max_model_len=4096, seed=0,
                          enable_prefix_caching=False, gpu_memory_utilization=0.35,
                          max_num_seqs=32)
        self.tokenizer = self.engine.get_tokenizer()
        self.sampling = SamplingParams(temperature=0, max_tokens=MAX_TOKENS, seed=0)
        self.engine.apply_model(partial(apply_patch_to_model, patch={'edits': []}))

    def evaluate(self, tasks, patch):
        started = time.monotonic()
        validate_patch(patch)
        audit = self.engine.apply_model(partial(apply_patch_to_model, patch=patch))
        prompts = [render_prompt(self.tokenizer, task) for task in tasks]
        # Keep each case independent of cohort membership and batch composition.
        outputs = [self.engine.generate([prompt], self.sampling, use_tqdm=False)[0]
                   for prompt in prompts]
        rows = []
        for task, output in zip(tasks, outputs, strict=True):
            sample = output.outputs[0]
            correct, reason = grade(task, sample.text, sample.finish_reason)
            rows.append({'id': task['id'], 'group': task['group'], 'correct': correct,
                         'reason': reason if not correct else 'correct', 'tokens': len(sample.token_ids),
                         'finish_reason': sample.finish_reason, 'repetition': repetition(sample.text),
                         'output': sample.text, 'loop': loop_evidence(sample.text, sample.finish_reason)})
        return {'patch': patch, 'audit': audit, 'seconds': time.monotonic()-started,
                'accuracy': sum(r['correct'] for r in rows)/len(rows), 'tasks': rows,
                'decoding': {'temperature': 0, 'max_tokens': MAX_TOKENS, 'seed': 0}}


def serve():
    evaluator = Evaluator()
    tasks = json.loads(Path('/pilot/dev.json').read_text())
    baseline = evaluator.evaluate(tasks, {'edits': []})
    Path('/prepared/looping-dev-baseline.json').write_text(json.dumps(baseline, indent=2)+'\n')
    # Exercise a nonidentity edit and ensure resetting restores exact deterministic outputs.
    evaluator.evaluate(tasks[:1], {'edits':[{'operation':'attention_heads','layer':0,'indices':[0],'gain':0.5}]})
    restored = evaluator.evaluate(tasks, {'edits': []})
    if [r['output'] for r in baseline['tasks']] != [r['output'] for r in restored['tasks']]:
        raise RuntimeError('Identity reset did not reproduce the baseline outputs.')
    Path('/prepared/looping-ready.json').write_text(json.dumps({'ready':True,'reset_verified':True,'limits':LIMITS})+'\n')

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(b'{"ready":true}')
        def do_POST(self):
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 16384: raise ValueError('Patch request too large.')
                patch = json.loads(self.rfile.read(size))
                result = evaluator.evaluate(tasks, patch)
                body = json.dumps(result).encode(); self.send_response(200)
            except Exception as error:
                body = json.dumps({'error':str(error)}).encode(); self.send_response(400)
            self.end_headers(); self.wfile.write(body)
    HTTPServer(('127.0.0.1',8766),Handler).serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serve',action='store_true')
    parser.add_argument('--patch',type=Path)
    parser.add_argument('--report',type=Path,default=Path('/workspace/dev-report.json'))
    parser.add_argument('--grade',type=Path,help='Operator only: task set for a clean grader.')
    options=parser.parse_args()
    if options.serve: return serve()
    patch=json.loads(options.patch.read_text()) if options.patch else {'edits':[]}
    validate_patch(patch)
    if options.grade:
        evaluator=Evaluator()
        tasks=json.loads(options.grade.read_text())
        result={'baseline':evaluator.evaluate(tasks,{'edits':[]}),
                'submission':evaluator.evaluate(tasks,patch)}
    else:
        request=Request('http://127.0.0.1:8766',data=json.dumps(patch).encode(),headers={'Content-Type':'application/json'})
        with urlopen(request,timeout=600) as response: result=json.load(response)
    options.report.write_text(json.dumps(result,indent=2)+'\n')
    if 'tasks' in result:
        print(json.dumps({k:v for k,v in result.items() if k!='tasks'},indent=2))
        for row in result['tasks']:print(row['id'],row['correct'],row['reason'],row['tokens'])
    else: print('Baseline:',result['baseline']['accuracy'],'Submission:',result['submission']['accuracy'])


if __name__=='__main__':main()
