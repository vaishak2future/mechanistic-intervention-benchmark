from pathlib import Path
Path('/logs').mkdir(exist_ok=True)
Path('/workspace/patch.json').write_text('{"edits": []}\n')
Path('/logs/submission.json').write_text('{"edits": []}\n')
