"""Deterministic layer: typed routing rules and evidence requirements.

This is the half of the knowledge base that executes. It runs before the LLM and its
result cannot be argued away by one: the guardrail stage re-applies it afterwards. The
comparison semantics mirror demo/governance.py deliberately, including the important
one — a blocking rule whose field is missing produces an evidence request rather than
being quietly skipped, because unknown data is not a pass.
"""
import math

NUMERIC_OPERATORS = ('gt', 'gte', 'lt', 'lte', 'eq', 'ne')
COMPARISONS = {
    'gt': lambda a, b: a > b, 'gte': lambda a, b: a >= b,
    'lt': lambda a, b: a < b, 'lte': lambda a, b: a <= b,
    'eq': lambda a, b: a == b, 'ne': lambda a, b: a != b,
}
# Conservative ordering. An evidence request outranks a referral because the underwriter
# cannot resolve what has not been supplied; both outrank proposed terms.
OUTCOME_PRIORITY = {'pass': 0, 'note': 1, 'refer': 2, 'request_evidence': 3}


def compare(actual, operator, value):
    if operator not in COMPARISONS:
        raise ValueError(f'Unsupported operator: {operator}')
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isfinite(actual):
            return None
    elif isinstance(value, bool) and not isinstance(actual, bool):
        return None
    elif isinstance(value, str) and not isinstance(actual, str):
        return None
    return COMPARISONS[operator](actual, value)


def evaluate(profile, rules):
    """Return one result per rule, with its status and whether it blocks."""
    results = []
    for rule in rules:
        scope = rule.get('scope', 'Both')
        if scope not in ('Both', profile.get('product')):
            results.append({**rule, 'status': 'out_of_scope', 'actual': None, 'blocks': False})
            continue
        actual = profile.get(rule['field'])
        matched = compare(actual, rule['operator'], rule['value'])
        if matched is None:
            status = 'missing'
        else:
            status = 'matched' if matched else 'not_matched'
        blocking = rule.get('action') != 'note' and status in ('matched', 'missing')
        results.append({**rule, 'status': status, 'actual': actual, 'blocks': blocking})
    return results


def outcome(results):
    """Collapse rule results into one conservative routing outcome."""
    worst = 'pass'
    for result in results:
        if not result['blocks']:
            continue
        # A blocking rule with missing data cannot be resolved by a referral alone.
        action = 'request_evidence' if result['status'] == 'missing' else result['action']
        if OUTCOME_PRIORITY[action] > OUTCOME_PRIORITY[worst]:
            worst = action
    return worst


def fired(results):
    return [r for r in results if r['status'] in ('matched', 'missing')]


def evidence_requirements(profile, grid):
    """Routine requirements for this age and total exposure, from the reviewed grid."""
    if not grid.get('bands'):
        return {'requirements': [], 'band': None, 'cites': grid.get('cites', [])}
    age = profile.get('age')
    cover = profile.get('cover') or 0
    existing = profile.get('existingCover') or 0
    exposure = cover + existing
    chosen = grid['bands'][-1]
    for band in grid['bands']:
        if isinstance(age, (int, float)) and age <= band['max_age'] and exposure <= band['max_exposure']:
            chosen = band
            break
    return {'requirements': list(chosen['requirements']),
            'band': {'max_age': chosen['max_age'], 'max_exposure': chosen['max_exposure']},
            'total_exposure': exposure,
            'exposure_note': ('Existing cover not declared; exposure counts the requested amount only.'
                              if profile.get('existingCover') is None else None),
            'cites': grid.get('cites', [])}


def condition_requirements(cards):
    """Impairment-specific evidence named by the cards that were retrieved."""
    out = []
    for card in cards:
        for item in (card.get('guidance') or {}).get('required_evidence', []):
            out.append({'requirement': item, 'source': card['id']})
    return out


def missing_attributes(profile, catalogue):
    """Optional attributes the case has not supplied, so the LLM can name them precisely."""
    return sorted(field for field in catalogue if profile.get(field) is None)
