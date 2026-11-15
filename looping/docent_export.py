"""Publish the recorded messages and tool events as one ordered Docent trace."""
import argparse
import json
from pathlib import Path
import re
from docent import Docent
from docent.data_models import AgentRun, Transcript
from docent.data_models.chat import AssistantMessage, SystemMessage, ToolCall, ToolMessage, UserMessage

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/looping-sol-clean-10m'


def main():
    global RUN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=RUN)
    RUN = parser.parse_args().run.resolve()
    summary=json.loads((RUN/'summary.json').read_text())
    key=(Path.home()/'Projects/creds/docent.token').read_text().strip()
    def clean(text):
        return re.sub(r'\b(?:sk-|as-)[A-Za-z0-9_-]{20,}','[REDACTED]',text.replace(key,'[REDACTED]'))
    messages=[UserMessage(content=(RUN/'task.txt').read_text())]
    calls=set()
    for line in (RUN/'trace.jsonl').read_text().splitlines():
        event=json.loads(line)
        metadata={'timestamp':event['time'],'source':'live_custom_runner'}
        if event['event']=='agent_message':
            messages.append(AssistantMessage(content=clean(event['message']),metadata=metadata))
        elif event['event']=='action':
            calls.add(event['action_id'])
            messages.append(AssistantMessage(content=event.get('note') or '',tool_calls=[ToolCall(
                id=event['action_id'],function='command',arguments={'command':event['command']})],metadata=metadata))
        elif event['event'] in {'result','error'} and event.get('action_id') in calls:
            text=event.get('stdout','')+event.get('stderr','')+event.get('message','')
            messages.append(ToolMessage(content=clean(text),tool_call_id=event['action_id'],function='command',
                metadata={**metadata,'exit_code':event.get('exit_code')}))
        else:
            messages.append(SystemMessage(content=clean(event.get('message',event['event'])),metadata=metadata))
    rows = ['| Measure | Baseline | Submitted |', '|---|---:|---:|']
    for group in ['repair', 'specificity', 'transfer']:
        score = summary['scores'][group]
        rows.append(f"| {group.title()} (n={score['n']}) | {score['baseline']:.1%} | {score['submission']:.1%} |")
    retention = summary['scores']['specificity_retention']
    loops = summary['scores']['loop_recovery']
    diagnostics = summary['scores']['diagnostics']
    report = ('Isolated grading results (operator, not an agent message)\n\n' + '\n'.join(rows)
        + f"\n\nBaseline-correct controls retained: {retention['retained']}/{retention['n']}."
        + f"\nConfirmed baseline looping cases correctly resolved: {loops['recovered']}/{loops['n']}."
        + f"\nRepetitive nontermination: {diagnostics['baseline']['confirmed_loops']} → {diagnostics['submission']['confirmed_loops']} cases."
        + f"\nAgent: gpt-6.1-sol, high reasoning; {summary['agent_seconds']:.0f} seconds used."
        + '\nAll run compute stopped; temporary workspace deleted.'
        + '\n\nThe mandatory-call template preamble was removed before calibration. '
          'Repair cases exhibit repeated-text nontermination at baseline. '
          'Specificity cases are baseline-correct. All cohorts were frozen before the agent started.'
        + '\n\nValidated submission:\n```json\n' + json.dumps(summary['patch'], indent=2) + '\n```')
    messages.append(SystemMessage(content=report))
    scores={f'{group}_{kind}':value[kind] for group,value in summary['scores'].items()
            if group in {'repair','specificity','transfer'} for kind in ['baseline','submission']}
    run=AgentRun(name='Looping repair · Sol · 10-minute pilot',transcripts=[Transcript(name='Unified trace',messages=messages)],
        metadata={**summary,'detailed_scores':summary['scores'],'scores':scores})
    client=Docent(api_key=key)
    collection=client.create_collection(name='MobileLLM looping repair — constrained interventions',
        description='Sol, 10-minute budget. One chronological trace and isolated repair, specificity and transfer results.')
    receipt={'collection_id':collection,'agent_run_id':run.id,'url':f'{client.frontend_url}/dashboard/{collection}'}
    (RUN/'docent.json').write_text(json.dumps(receipt,indent=2))
    client.add_agent_runs(collection,[run])
    uploaded=client.get_agent_run(collection,run.id)
    assert uploaded and len(uploaded.transcripts[0].messages)==len(messages)
    receipt.update(verified=True,messages=len(messages))
    (RUN/'docent.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
