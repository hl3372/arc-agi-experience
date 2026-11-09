"""Descriptive profile cells; no inference of difficulty or judgment from origin."""
from collections import Counter
from .allocation import COMPARISONS


def summarize(kernels, tasks):
    cells = []
    key = lambda row: (row['arm'], row['invitation'] or '', row['bank_setting'], row['context'],row.get('cohort','unknown'),row.get('generation_model','unknown'))
    for cell in sorted({key(row) for row in kernels}):
        ks = [row for row in kernels if key(row) == cell]
        ts = [row for row in tasks if key(row) == cell]
        natural = [row for row in ts if row['natural_eligible']]
        merit = [row for row in ts if row.get('merit_admitted', row['admission'] == 'natural')]
        cells.append({'arm': cell[0], 'invitation': cell[1] or None,
            'bank_setting': cell[2], 'context': cell[3], 'cohort':cell[4], 'generation_model':cell[5], 'input_slots': len(ks),
            'generated': sum(row['generated'] for row in ks),
            'selector_eligibility': dict(Counter((row.get('assessment') or {}).get('eligibility', 'unassessed') for row in ks)),
            'exact_conflicts': sum(bool(row.get('exact_conflict')) for row in ks),
            'developments': len(ts), 'grids': sum(row['realized'] for row in ts),
            'evidence_clear': sum(row['evidence_clear'] for row in ts),
            'usable': sum(row['usable_after_exact_palette_union'] for row in ts),
            'admission': dict(Counter(row['admission'] for row in ts)),
            'hold_kinds': dict(Counter(row['hold_kind'] for row in ts if row.get('hold_kind'))),
            'primary_misses': sum(row.get('original_primary_result') == 'mismatched' for row in ts),
            'verified_unsolved': sum(row.get('blind_performance', {}).get('label') == 'verified; unsolved at tested budget' for row in ts),
            'naturally_eligible_developed': {'attempts': len(natural), 'usable': sum(row['usable_after_exact_palette_union'] for row in natural)},
            'merit_only': {'attempts': len(merit), 'usable': sum(row['usable_after_exact_palette_union'] for row in merit)}})
    return {'comparisons': [{'profile': p, 'comparator': c} for p, c, _ in COMPARISONS],
        'cells': cells, 'profile_fulfillment': 'requires_completed_grid_and_independent_research_review',
        'difficulty_interpretation': 'solver performance at tested budget, not intrinsic hardness or training value',
        'private_author_accounts_are_evidence': False,
        'natural_view': 'Fresh tasks have quality-based eligibility without a capacity quota. Saved recovery retains the original natural-only attribution separately.'}
