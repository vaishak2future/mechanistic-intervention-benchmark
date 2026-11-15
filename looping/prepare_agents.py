"""Pin the official hosted harnesses and give each a single-model, high-effort run."""
import shutil
import argparse
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def prepare(source_name, name):
    source=ROOT/'build/agents'/source_name
    destination=ROOT/'build/agents'/name
    if destination.exists():
        raise ValueError('Choose a fresh agent release.')
    shutil.copytree(source,destination,ignore=shutil.ignore_patterns('__pycache__','agent_output_example'))
    for path in destination.glob('*'):
        if path.suffix in {'.py','.sh','.yaml','.txt'}:
            path.write_text(path.read_text().replace(source_name,name))
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--readable', action='store_true', help='Build self-contained v3 from the pinned v1 harnesses.')
    if parser.parse_args().readable:
        for family in ['codex', 'claude']:
            destination = prepare(f'{family}-looping-v1', f'{family}-looping-v3')
            runner = destination/'collect_outputs.py'
            text = runner.read_text()
            old = '            sys.stdout.write(line)\n            sys.stdout.flush()'
            if text.count(old) != 1:
                raise ValueError('Expected exactly one stdout stream to format.')
            text = text.replace('        for line in process.stdout:',
                                '        display = LiveMessages()\n        for line in process.stdout:')
            # Valkyrie sets PYTHONSAFEPATH=1: sibling modules are not importable.
            formatter = (ROOT/'looping/live_messages.py').read_text()
            text = text.replace('def main()', formatter + '\n\ndef main()', 1)
            runner.write_text(text.replace(old, '            display.emit(line)'))
            print(destination)
        return
    codex=prepare('codex-mobilellm-v1','codex-looping-v1')
    contract=codex/'contract.yaml'
    contract.write_text(contract.read_text().replace('      - gpt-6.1-sol','      - gpt-6-astra\n      - gpt-6.1-sol'))
    setup=codex/'setup.sh'
    lines=setup.read_text().splitlines()
    setup.write_text('\n'.join('npm i -g @openai/codex@0.169.0' if line.startswith('npm i -g @openai/codex') else line for line in lines)+'\n')
    runner=codex/'collect_outputs.py'
    text=runner.read_text().replace('codex-v1.0.0','codex-looping-v1')
    text=text.replace('        "--skip-git-repo-check",','        "-c",\n        "features.multi_agent=false",\n        "-c",\n        \'web_search="disabled"\',\n        "--skip-git-repo-check",')
    text=text.replace('"model": model,','"model": model,\n            "reasoning_effort": "high",')
    runner.write_text(text)
    (codex/'MODEL.txt').write_text('Models: gpt-6.1-sol, gpt-6-astra\nReasoning: high\nCodex: 0.169.0\n')

    claude=prepare('claude_code-v1.0.0','claude-looping-v1')
    setup=claude/'setup.sh'
    setup.write_text(setup.read_text().replace('curl -fsSL https://claude.ai/install.sh | bash','curl -fsSL https://claude.ai/install.sh | bash -s -- 2.1.325'))
    runner=claude/'collect_outputs.py'
    text=runner.read_text().replace('        "--verbose",','        "--verbose",\n        "--effort",\n        "high",\n        "--tools",\n        "Bash,Edit,Write,Read,Glob,Grep",')
    text=text.replace('"Bash Edit Write Read Glob Grep LS WebFetch NotebookEdit NotebookRead "\n        "TodoRead TodoWrite Agent Skill SlashCommand Task WebSearch",','"Bash Edit Write Read Glob Grep",')
    text=text.replace('"model": model,','"model": model,\n            "reasoning_effort": "high",')
    runner.write_text(text)
    (claude/'MODEL.txt').write_text('Models: claude-opus-5-5, claude-sonnet-5-5\nReasoning: high\nClaude Code: 2.1.325\n')
    print(codex,claude,sep='\n')


if __name__=='__main__':main()
