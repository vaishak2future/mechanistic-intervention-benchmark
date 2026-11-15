"""Publish each completed hosted run as one chronological Docent transcript."""
import hashlib
import argparse
import json
from pathlib import Path
import re

from docent import Docent
from docent.data_models import AgentRun, Transcript
from docent.data_models.chat import AssistantMessage, SystemMessage, ToolCall, ToolMessage, UserMessage
from scoring import net_improvement

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'runs/looping-hour-comparison'


def clean(value):
    text=value if isinstance(value,str) else json.dumps(value)
    for filename in ['openai.token','anthropic.token','docent.token','thinky.token','tinker.token']:
        path=Path.home()/'Projects/creds'/filename
        if path.exists():
            for secret in re.findall(r'(?:sk-[A-Za-z0-9_-]+|[A-Za-z0-9_-]{40,})',path.read_text()):
                text=text.replace(secret,'[REDACTED]')
    return re.sub(r'\b(?:sk-|as-)[A-Za-z0-9_-]{20,}','[REDACTED]',text)


def codex_messages(events):
    messages=[]
    started=set()
    for index,event in enumerate(events):
        item=event.get('item',{})
        kind=item.get('type')
        metadata={'event_index':index,'source':'original_agent_trajectory'}
        if event.get('type')=='item.completed' and kind=='agent_message':
            messages.append(AssistantMessage(content=clean(item['text']),metadata=metadata))
        elif kind=='command_execution':
            identifier=item['id']
            if event['type']=='item.started':
                started.add(identifier)
                messages.append(AssistantMessage(content='',tool_calls=[ToolCall(
                    id=identifier,function='bash',arguments={'command':clean(item['command'])})],metadata=metadata))
            elif event['type']=='item.completed':
                if identifier not in started:
                    messages.append(AssistantMessage(content='',tool_calls=[ToolCall(
                        id=identifier,function='bash',arguments={'command':clean(item.get('command',''))})],metadata=metadata))
                messages.append(ToolMessage(content=clean(item.get('aggregated_output','')),
                    tool_call_id=identifier,function='bash',metadata={**metadata,'exit_code':item.get('exit_code')}))
        elif event.get('type')=='item.completed' and kind=='file_change':
            identifier=item['id']
            messages.append(AssistantMessage(content='',tool_calls=[ToolCall(id=identifier,function='file_change',
                arguments=json.loads(clean({'changes':item.get('changes',[])})))],metadata=metadata))
            messages.append(ToolMessage(content=clean({'status':item.get('status')}),tool_call_id=identifier,
                function='file_change',metadata=metadata))
        elif event.get('type') in {'error','turn.failed'}:
            messages.append(SystemMessage(content=clean(event),metadata=metadata))
    return messages


def claude_messages(events):
    messages=[]
    tools={}
    seen=set()
    for index,event in enumerate(events):
        message=event.get('message',{})
        metadata={'event_index':index,'timestamp':event.get('timestamp'),'source':'original_agent_trajectory'}
        for block in message.get('content',[]) if isinstance(message,dict) else []:
            if not isinstance(block,dict):
                continue
            identity=(message.get('id'),json.dumps(block,sort_keys=True))
            if message.get('id') and identity in seen:
                continue
            seen.add(identity)
            if event['type']=='assistant' and block['type']=='text':
                messages.append(AssistantMessage(content=clean(block['text']),metadata=metadata))
            elif event['type']=='assistant' and block['type']=='tool_use':
                tools[block['id']]=block['name']
                messages.append(AssistantMessage(content='',tool_calls=[ToolCall(id=block['id'],function=block['name'],
                    arguments=json.loads(clean(block.get('input',{}))))],metadata=metadata))
            elif event['type']=='user' and block['type']=='tool_result':
                identifier=block['tool_use_id']
                messages.append(ToolMessage(content=clean(block.get('content','')),tool_call_id=identifier,
                    function=tools.get(identifier,'tool'),metadata={**metadata,'is_error':block.get('is_error',False)}))
        if event.get('type')=='result' and event.get('is_error'):
            messages.append(SystemMessage(content=clean(event.get('result',event)),metadata=metadata))
    return messages


def build_run(receipt):
    directory=OUTPUT/f"{receipt['name']}-{receipt['repeat']}"
    collected=json.loads((directory/'collected.json').read_text())
    paths=list((directory/'outputs').rglob('trajectory.jsonl'))
    if len(paths)!=1:
        raise ValueError(f'Expected exactly one agent trajectory in {directory}')
    events=[json.loads(line) for line in paths[0].read_text().splitlines() if line.strip()]
    summaries=list((directory/'outputs').rglob('summary.json'))
    agent_summary=json.loads(summaries[0].read_text()) if summaries else {}
    task_prompt=agent_summary.get('task_prompt') or (ROOT/'build/looping-hour-v1/repair-looping/instruction.md').read_text()
    messages=[UserMessage(content=clean(task_prompt))]
    messages += (codex_messages(events) if receipt['agent'].startswith('codex') else claude_messages(events))
    scores=collected['scores'] or {}
    table=['| Measure | Score |','|---|---:|']
    if scores:
        table.append(f"| **Net improvement (primary)** | **{net_improvement(scores)['score']:.2f}** |")
    table += [f'| {name.title()} | {scores[name]:.1%} |' for name in ['repair','specificity','transfer'] if name in scores]
    messages.append(SystemMessage(content='Isolated verifier result (operator):\n\n'+'\n'.join(table)+
        '\n\n'+clean({'status':collected['status'],'error':collected['error'],'timings':collected['timings']})))
    metadata={**receipt,'scores':scores,'timings':collected['timings'],
              'agent_metadata':agent_summary.get('metadata',{}),
              'trajectory_sha256':hashlib.sha256(paths[0].read_bytes()).hexdigest(),
              'note':'Only recorded public messages and tool events are shown; no hidden reasoning is reconstructed.'}
    if scores:
        metadata.update(primary_score=net_improvement(scores)['score'],
                        primary_score_name='net_improvement', scoring=net_improvement(scores))
    minutes=receipt.get('budget_seconds',3600)//60
    return AgentRun(name=f"{receipt['name'].title()} · trial {receipt['repeat']} · {minutes}-minute budget",
                    transcripts=[Transcript(name='Unified trace',messages=messages)],metadata=metadata)


def main():
    global OUTPUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    OUTPUT=parser.parse_args().output.resolve()
    runs=json.loads((OUTPUT/'runs.json').read_text())
    client=Docent(api_key=(Path.home()/'Projects/creds/docent.token').read_text().strip())
    receipt_path=OUTPUT/'docent.json'
    if receipt_path.exists():
        collection=json.loads(receipt_path.read_text())
    else:
        identifier=client.create_collection(name='Mechanistic Intervention Benchmark — '+OUTPUT.name,
            description='Frozen looping task; H100 budgets; net improvement with repair, retention and transfer diagnostics.')
        collection={'collection_id':identifier,'url':f'{client.frontend_url}/dashboard/{identifier}'}
        receipt_path.write_text(json.dumps(collection,indent=2)+'\n')
    client.update_collection(collection['collection_id'], description=
        'Primary score: net improvement = 100 × max(0, newly solved failures − broken controls) / 11. '
        'No improvement earns zero; regression counts remain visible. Repair, specificity and transfer '
        'are diagnostics, not separate rewards. Scores recomputed from saved verifier results; no reruns.')
    for receipt in runs:
        directory=OUTPUT/f"{receipt['name']}-{receipt['repeat']}"
        if not (directory/'collected.json').exists():
            continue
        uploaded_path=directory/'docent.json'
        if uploaded_path.exists():
            saved=json.loads(uploaded_path.read_text())
            if saved.get('verified'):
                collected=json.loads((directory/'collected.json').read_text())
                if collected.get('scores'):
                    score=net_improvement(collected['scores'])
                    metadata={'primary_score':score['score'],'primary_score_name':'net_improvement',
                              'scoring':score,'scores':collected['scores']}
                    client.update_agent_run_metadata(collection['collection_id'],saved['agent_run_id'],metadata)
                    uploaded=client.get_agent_run(collection['collection_id'],saved['agent_run_id'])
                    assert uploaded.metadata['primary_score']==score['score']
                    print('Updated primary score',receipt['name'],receipt['repeat'],score['score'],flush=True)
                continue
            uploaded=client.get_agent_run(collection['collection_id'],saved['agent_run_id'])
            if uploaded is None:
                raise RuntimeError('Previous upload is unconfirmed; inspect before retrying.')
        else:
            run=build_run(receipt)
            saved={'agent_run_id':run.id,'messages':len(run.transcripts[0].messages)}
            uploaded_path.write_text(json.dumps(saved,indent=2)+'\n')
            client.add_agent_runs(collection['collection_id'],[run])
            uploaded=client.get_agent_run(collection['collection_id'],run.id)
        assert uploaded and len(uploaded.transcripts[0].messages)==saved['messages']
        saved['verified']=True
        uploaded_path.write_text(json.dumps(saved,indent=2)+'\n')
        print('Uploaded',receipt['name'],receipt['repeat'],flush=True)
    print(collection['url'])


if __name__=='__main__':main()
