"""Net improvement for the frozen 7 repair / 4 transfer / 6 control pilot."""
import math

SCORING_VERSION = 'bounded-net-improvement-v2'
COUNTS = {'repair': 7, 'transfer': 4, 'specificity': 6}


def net_improvement(diagnostics):
    """Recover exact case counts from the saved cohort accuracies."""
    correct = {}
    for name, count in COUNTS.items():
        value = diagnostics[name]
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'Invalid {name} accuracy: {value}')
        successes = value * count
        if not math.isclose(successes, round(successes), abs_tol=1e-8):
            raise ValueError(f'{name} accuracy does not match the frozen cohort size.')
        correct[name] = round(successes)
    repaired = correct['repair'] + correct['transfer']
    regressions = COUNTS['specificity'] - correct['specificity']
    return {
        'scoring_version': SCORING_VERSION,
        'score': 100 * max(0, repaired - regressions) / 11,
        'signed_net_improvement': 100 * (repaired - regressions) / 11,
        'newly_solved': repaired,
        'regressions': regressions,
        'baseline_failures': 11,
    }


def rewards(diagnostics):
    """Valkyrie uses `reward` as its primary metric and multiplies it by 100."""
    result = net_improvement(diagnostics)
    return {'reward': result['score'] / 100, **{name: diagnostics[name] for name in COUNTS},
            'regressions': result['regressions'], 'valid_submission': 1}
