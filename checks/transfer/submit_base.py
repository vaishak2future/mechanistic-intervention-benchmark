"""Submit unchanged base weights; no training or agent API calls."""
import json
import sys
from pathlib import Path
sys.path.insert(0, '/task')
from submit import submit

weights = list(Path('/base').glob('*.safetensors'))
if len(weights) != 1:
    raise RuntimeError('Expected one pinned base safetensors file.')
print(json.dumps(submit(weights[0], Path('/workspace/submission')), indent=2))
