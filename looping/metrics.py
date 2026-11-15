"""Report repair, preservation and transfer without hiding tradeoffs."""

def scores(report):
    before, after = report['baseline']['tasks'], report['submission']['tasks']
    if [row['id'] for row in before] != [row['id'] for row in after]:
        raise ValueError('Baseline and submitted task IDs differ.')
    result = {}
    for group in ['repair', 'specificity', 'transfer']:
        pairs = [(a,b) for a,b in zip(before,after,strict=True) if a['group']==group]
        result[group] = {'baseline':sum(a['correct'] for a,b in pairs)/len(pairs),
                         'submission':sum(b['correct'] for a,b in pairs)/len(pairs), 'n':len(pairs)}
    controls = [(a,b) for a,b in zip(before,after,strict=True) if a['group']=='specificity' and a['correct']]
    result['specificity_retention'] = {'retained':sum(b['correct'] for a,b in controls), 'n':len(controls)}
    loops = [(a,b) for a,b in zip(before,after,strict=True) if a['finish_reason']=='length']
    confirmed = [(a,b) for a,b in zip(before,after,strict=True) if a['loop']['detected']]
    result['loop_recovery'] = {'recovered':sum(b['correct'] for a,b in confirmed), 'n':len(confirmed)}
    result['budget_exhaustion_recovery'] = {'recovered':sum(b['correct'] for a,b in loops), 'n':len(loops)}
    failures = [(a,b) for a,b in zip(before,after,strict=True) if a['group']=='repair' and not a['correct']]
    result['repair_recovery'] = {'repaired':sum(b['correct'] for a,b in failures), 'n':len(failures)}
    result['diagnostics'] = {
        name: {'confirmed_loops':sum(r['loop']['detected'] for r in rows),
               'budget_exhausted':sum(r['finish_reason']=='length' for r in rows),
               'mean_tokens':sum(r['tokens'] for r in rows)/len(rows),
               'mean_repetition':sum(r['repetition'] for r in rows)/len(rows)}
        for name,rows in [('baseline',before),('submission',after)]}
    return result

