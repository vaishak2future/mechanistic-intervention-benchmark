"""A small, data-only language for editing existing output projections."""
import math

LIMITS = {'layers': 22, 'heads': 24, 'head_dim': 64, 'neurons': 6144,
          'max_edits': 8, 'max_layers': 4, 'max_heads': 8, 'max_neurons': 128,
          'max_branches': 4, 'min_gain': -1.0, 'max_gain': 2.0}


def validate_patch(patch):
    if not isinstance(patch, dict) or set(patch) != {'edits'} or not isinstance(patch['edits'], list):
        raise ValueError('Expected exactly {"edits": [...]}')
    edits = patch['edits']
    if len(edits) > LIMITS['max_edits']:
        raise ValueError('At most eight edits are allowed.')
    layers, targets = set(), set()
    heads = neurons = branches = 0
    for edit in edits:
        if not isinstance(edit, dict):
            raise ValueError('Each edit must be an object.')
        operation = edit.get('operation')
        indexed = operation in {'attention_heads', 'mlp_neurons'}
        if operation not in {'attention_branch', 'mlp_branch', 'attention_heads', 'mlp_neurons'}:
            raise ValueError('Unknown operation.')
        if set(edit) != ({'operation', 'layer', 'gain', 'indices'} if indexed else {'operation', 'layer', 'gain'}):
            raise ValueError('Unexpected or missing edit fields.')
        layer, gain = edit['layer'], edit['gain']
        if type(layer) is not int or not 0 <= layer < LIMITS['layers']:
            raise ValueError('Layer must be an integer from 0 through 21.')
        if type(gain) not in (int, float) or not math.isfinite(gain) or not -1 <= gain <= 2:
            raise ValueError('Gain must be finite and between -1 and 2.')
        layers.add(layer)
        branch = 'attention' if operation.startswith('attention') else 'mlp'
        if indexed:
            indices = edit['indices']
            upper = LIMITS['heads'] if branch == 'attention' else LIMITS['neurons']
            if not isinstance(indices, list) or not indices or any(type(i) is not int or not 0 <= i < upper for i in indices):
                raise ValueError('Invalid component indices.')
            if len(set(indices)) != len(indices):
                raise ValueError('Duplicate component indices.')
            for index in indices:
                key = (layer, branch, index)
                if key in targets or (layer, branch, 'all') in targets:
                    raise ValueError('Overlapping edits are not allowed.')
                targets.add(key)
            if branch == 'attention': heads += len(indices)
            else: neurons += len(indices)
        else:
            if any(t[:2] == (layer, branch) for t in targets):
                raise ValueError('Overlapping edits are not allowed.')
            targets.add((layer, branch, 'all'))
            branches += 1
    if len(layers) > 4 or heads > 8 or neurons > 128 or branches > 4:
        raise ValueError('Edit budget exceeded: four layers, four branches, eight heads, 128 neurons.')
    return patch


def apply_patch_to_model(model, patch):
    """Reset to pristine projections, then compile approved edits into weights."""
    import re
    import torch
    validate_patch(patch)
    parameters = dict(model.named_parameters())
    if not hasattr(model, '_looping_originals'):
        model._looping_originals = {
            name: weight.detach().clone() for name, weight in parameters.items()
            if re.search(r'layers\.\d+\.', name) and name.endswith(('o_proj.weight', 'down_proj.weight'))
        }
        if len(model._looping_originals) != 44:
            raise RuntimeError('Unexpected model architecture: expected 22 attention and 22 MLP projections.')
    originals = model._looping_originals
    with torch.inference_mode():
        for name, original in originals.items():
            parameters[name].copy_(original)
        changed = []
        for edit in patch['edits']:
            attention = edit['operation'].startswith('attention')
            suffix = 'o_proj.weight' if attention else 'down_proj.weight'
            names = [n for n in originals if f"layers.{edit['layer']}." in n and n.endswith(suffix)]
            if len(names) != 1:
                raise RuntimeError('Projection not uniquely identified.')
            name = names[0]
            weight = parameters[name]
            expected = (1536, 1536 if attention else 6144)
            if tuple(weight.shape) != expected:
                raise RuntimeError(f'Unexpected projection shape: {name} {tuple(weight.shape)}')
            if edit['operation'].endswith('branch'):
                weight.mul_(edit['gain'])
            else:
                for index in edit['indices']:
                    start = index * 64 if attention else index
                    width = 64 if attention else 1
                    weight[:, start:start + width].mul_(edit['gain'])
            changed.append(name)
    return {'changed_projections': changed, 'operations': len(patch['edits']), 'reset_to_base': True}
