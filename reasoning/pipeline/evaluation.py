"""Score the pipeline against recorded human decisions.

One number would hide the thing that matters. Disagreeing with an underwriter by being
more cautious costs handling time; disagreeing by being less cautious is how a case gets
accepted that should not have been. These are counted separately and never averaged into
each other, and the tuner treats the second as a hard constraint rather than a term in a
sum it can trade away.

Agreement is reported with a bootstrap interval, because a bank of thirty cases cannot
support a figure quoted to the percentage point.
"""
import random
import re
from statistics import mean

from . import validators
from .casebank import HUMAN_OUTCOMES

STOPWORDS = {'the', 'and', 'for', 'with', 'from', 'that', 'this', 'was', 'were', 'has', 'have', 'not', 'are',
             'case', 'report', 'declared', 'recorded', 'requested', 'evidence', 'required', 'rule', 'rules',
             'outcome', 'after', 'before', 'over', 'under', 'into', 'than', 'then', 'been', 'will', 'may',
             'which', 'where', 'what', 'when', 'about', 'more', 'less', 'also', 'only', 'per', 'its'}
QUALITY_WEIGHT = 0.25   # how much quality counts next to routing when the tuner is asked to combine them


def tokens(text):
    return {t for t in re.findall(r'[a-z0-9]+', (text or '').lower()) if len(t) >= 3 and t not in STOPWORDS}


def _item_match(wanted, offered):
    a, b = tokens(wanted), tokens(offered)
    if not a or not b:
        return False
    overlap = len(a & b)
    return overlap / len(a | b) >= 0.3 or overlap >= max(1, min(len(a), len(b)) // 2 + (min(len(a), len(b)) == 1))


def evidence_match(requested_by_human, offered_by_pipeline):
    """Recall and precision of the evidence request against what the underwriter asked for."""
    if not requested_by_human:
        return None
    offered = [o for o in offered_by_pipeline or [] if isinstance(o, str)]
    hit_wanted = [w for w in requested_by_human if any(_item_match(w, o) for o in offered)]
    hit_offered = [o for o in offered if any(_item_match(w, o) for w in requested_by_human)]
    recall = len(hit_wanted) / len(requested_by_human)
    precision = (len(hit_offered) / len(offered)) if offered else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {'recall': round(recall, 3), 'precision': round(precision, 3), 'f1': round(f1, 3),
            'matched': hit_wanted, 'unmatched': [w for w in requested_by_human if w not in hit_wanted]}


def quality(result, case):
    """How good the reasoning is, beyond whether the routing class was right.

    Every component here is deterministic. A model-graded rubric can be attached under
    `judge` by the caller; it is reported next to these, never blended into safety.
    """
    output = result.get('output') or {}
    text = ' '.join([output.get('explanation') or ''] + [r for r in output.get('reasons', []) if isinstance(r, str)])
    citations = set(output.get('citations') or [])
    fired = result.get('rules', {}).get('fired', [])
    human = case.human if hasattr(case, 'human') else (case.get('human') or {})

    named = [r['id'] for r in fired if r['id'] in text or r['id'] in citations
             or (r.get('title') and r['title'].lower() in text.lower())]
    covered = [r['id'] for r in fired if set(r.get('cites', [])) & citations]
    match = evidence_match(human.get('evidence_requested') or [], output.get('missing_information') or [])
    rationale = tokens(human.get('rationale') or '')
    overlap = (len(rationale & tokens(text)) / len(rationale)) if rationale else None
    allowed = validators.allowed_figures(result.get('profile'), result.get('routed_profile'), result.get('evidence'),
                                         result.get('retrieval', {}).get('sources'), result.get('instructions'),
                                         [r.get('value') for r in fired], result.get('evidence_requirements'))
    unsupported = validators.check_figures(output, allowed)

    components = {'rules_named': round(len(named) / len(fired), 3) if fired else None,
                  'citations_cover_rules': round(len(covered) / len(fired), 3) if fired else None,
                  'evidence_request_f1': match['f1'] if match else None,
                  'rationale_overlap': round(overlap, 3) if overlap is not None else None,
                  'figures_supported': 1.0 if not unsupported else 0.0}
    usable = [v for v in components.values() if v is not None]
    return {**components, 'evidence_match': match, 'unsupported_figures': unsupported,
            'score': round(mean(usable), 3) if usable else None}


def objective(summary, mode='severity'):
    """The number the tuner compares. Severity always; quality only when asked for."""
    if mode == 'combined' and summary.get('quality', {}).get('score') is not None:
        return round(summary['severity_score'] + QUALITY_WEIGHT * summary['quality']['score'], 4)
    return summary['severity_score']

# How cautious each pipeline output is. Ordering only; the gaps are not distances.
CAUTION = {'straight_through': 0, 'propose_terms': 1, 'refer': 2, 'request_evidence': 3}
VERDICT_CREDIT = {'match': 1.0, 'acceptable': 0.8, 'conservative': 0.5, 'unsafe': 0.0}


def classify(predicted, outcome):
    """Compare one recommendation with what the underwriter recorded."""
    expected = HUMAN_OUTCOMES[outcome]
    ideal = expected['ideal']
    if predicted == ideal:
        return 'match'
    if predicted in expected['also_acceptable']:
        return 'acceptable'
    if CAUTION.get(predicted, 0) < CAUTION.get(ideal, 0):
        return 'unsafe'
    return 'conservative'


def bootstrap_interval(values, samples=1000, seed=13, level=0.9):
    if not values:
        return None
    rng = random.Random(seed)
    means = sorted(mean(rng.choices(values, k=len(values))) for _ in range(samples))
    low = means[int((1 - level) / 2 * samples)]
    high = means[min(samples - 1, int((1 + level) / 2 * samples))]
    return [round(low, 3), round(high, 3)]


def evaluate(pipeline, cases, offline=True, on_error='record', judge=None):
    """Run every case and summarise how the pipeline compares with the humans."""
    rows, errors = [], []
    for case in cases:
        try:
            result = pipeline.assess(case.to_pipeline_case(), offline=offline)
            predicted = result['recommendation']
        except Exception as error:
            if on_error == 'raise':
                raise
            errors.append({'case_id': case.case_id, 'error': f'{type(error).__name__}: {error}'})
            continue
        outcome = case.human['outcome']
        verdict = classify(predicted, outcome)
        graded = quality(result, case)
        if judge is not None and predicted != 'straight_through':
            graded['judge'] = judge(result, case)
        rows.append({'case_id': case.case_id, 'product': case.profile.get('product'),
                     'human_outcome': outcome, 'ideal': HUMAN_OUTCOMES[outcome]['ideal'],
                     'predicted': predicted, 'verdict': verdict,
                     'credit': VERDICT_CREDIT[verdict],
                     'cohort': case.human.get('cohort'),
                     'rules_fired': [r['id'] for r in result['rules']['fired']],
                     'evidence_source': case.evidence_source,
                     'quality': graded})
    return summarise(rows, errors)


def summarise(rows, errors=None):
    errors = errors or []
    n = len(rows)
    if not n:
        return {'cases': 0, 'errors': errors, 'note': 'No cases were scored.'}
    counts = {verdict: sum(1 for r in rows if r['verdict'] == verdict) for verdict in VERDICT_CREDIT}
    agreement = [1.0 if r['verdict'] in ('match', 'acceptable') else 0.0 for r in rows]
    requested = [r for r in rows if r['predicted'] == 'request_evidence']
    wanted = [r for r in rows if r['human_outcome'] == 'more_evidence']
    hit = [r for r in requested if r['human_outcome'] == 'more_evidence']

    by_outcome = {}
    for outcome in sorted({r['human_outcome'] for r in rows}):
        group = [r for r in rows if r['human_outcome'] == outcome]
        by_outcome[outcome] = {'n': len(group),
                               'severity_score': round(mean(r['credit'] for r in group), 3),
                               'predictions': {p: sum(1 for r in group if r['predicted'] == p)
                                               for p in sorted({r['predicted'] for r in group})}}
    by_product = {}
    for product in sorted({r['product'] for r in rows if r['product']}):
        group = [r for r in rows if r['product'] == product]
        by_product[product] = {'n': len(group),
                               'severity_score': round(mean(r['credit'] for r in group), 3),
                               'unsafe': sum(1 for r in group if r['verdict'] == 'unsafe')}

    quality_rows = [r['quality'] for r in rows if r.get('quality')]
    quality_summary = None
    if quality_rows:
        def average(key):
            values = [q[key] for q in quality_rows if q.get(key) is not None]
            return round(mean(values), 3) if values else None
        judged = [q['judge']['score'] for q in quality_rows if (q.get('judge') or {}).get('score') is not None]
        quality_summary = {'score': average('score'), 'rules_named': average('rules_named'),
                           'citations_cover_rules': average('citations_cover_rules'),
                           'evidence_request_f1': average('evidence_request_f1'),
                           'rationale_overlap': average('rationale_overlap'),
                           'figures_supported': average('figures_supported'),
                           'cases_with_unsupported_figures': sum(1 for q in quality_rows if q['unsupported_figures']),
                           'judge': {'graded': len(judged), 'score': round(mean(judged), 3)} if judged else None,
                           'note': 'Deterministic reasoning-quality checks, averaged over cases where each applies. '
                                   'Reported next to routing agreement, never blended into the safety count.'}

    return {
        'cases': n,
        'quality': quality_summary,
        'severity_score': round(mean(r['credit'] for r in rows), 4),
        'agreement': round(mean(agreement), 4),
        'agreement_interval_90pct': bootstrap_interval(agreement),
        'exact_match': round(counts['match'] / n, 4),
        'verdicts': counts,
        'unsafe_disagreements': counts['unsafe'],
        'conservative_disagreements': counts['conservative'],
        'evidence_requests': {
            'precision': round(len(hit) / len(requested), 3) if requested else None,
            'recall': round(len(hit) / len(wanted), 3) if wanted else None,
            'requested': len(requested), 'humans_requested': len(wanted)},
        'by_human_outcome': by_outcome,
        'by_product': by_product,
        'errors': errors,
        'rows': rows,
        'scoring_note': ('severity_score credits an exact match 1.0, an acceptable alternative 0.8, a more '
                         'cautious disagreement 0.5 and a less cautious one 0.0. Unsafe disagreements are '
                         'also counted separately and are never traded against the score.'),
    }


def rule_evidence(rows):
    """For each rule that fired, what the humans decided on those cases.

    A proposal to move a threshold is only reviewable next to this: how often the rule
    fired, on cases the underwriter then accepted at terms, referred, or asked evidence
    for. A rule that fires on four cases the humans all rated is doing its job; one that
    fires only on cases they accepted is the one worth looking at.
    """
    evidence = {}
    for row in rows:
        for rule_id in row.get('rules_fired', []):
            entry = evidence.setdefault(rule_id, {'fired': 0, 'human_outcomes': {}, 'verdicts': {}, 'cases': []})
            entry['fired'] += 1
            entry['human_outcomes'][row['human_outcome']] = entry['human_outcomes'].get(row['human_outcome'], 0) + 1
            entry['verdicts'][row['verdict']] = entry['verdicts'].get(row['verdict'], 0) + 1
            entry['cases'].append(row['case_id'])
    return dict(sorted(evidence.items()))


def support(baseline, candidate):
    """Which cases a change helped and which it hurt, by credit.

    A change that raises the average because one case moved is one case's opinion. The
    tuner asks for a minimum number of improved cases before it will carry a change.
    """
    before = {r['case_id']: r for r in baseline.get('rows', [])}
    improved, worsened = [], []
    for row in candidate.get('rows', []):
        was = before.get(row['case_id'])
        if not was:
            continue
        if row['credit'] > was['credit']:
            improved.append(row['case_id'])
        elif row['credit'] < was['credit']:
            worsened.append(row['case_id'])
    return {'improved': improved, 'worsened': worsened}


def compare(baseline, candidate):
    """Did a configuration change help, and did it cost safety to do it?"""
    def q(summary):
        return ((summary.get('quality') or {}).get('score'))
    return {
        'severity_score': {'from': baseline['severity_score'], 'to': candidate['severity_score'],
                           'delta': round(candidate['severity_score'] - baseline['severity_score'], 4)},
        'quality_score': {'from': q(baseline), 'to': q(candidate),
                          'delta': round(q(candidate) - q(baseline), 4) if q(baseline) is not None and q(candidate) is not None else None},
        'agreement': {'from': baseline['agreement'], 'to': candidate['agreement'],
                      'delta': round(candidate['agreement'] - baseline['agreement'], 4)},
        'unsafe_disagreements': {'from': baseline['unsafe_disagreements'],
                                 'to': candidate['unsafe_disagreements'],
                                 'delta': candidate['unsafe_disagreements'] - baseline['unsafe_disagreements']},
        'conservative_disagreements': {'from': baseline['conservative_disagreements'],
                                       'to': candidate['conservative_disagreements'],
                                       'delta': (candidate['conservative_disagreements']
                                                 - baseline['conservative_disagreements'])},
        'changed_cases': [{'case_id': b['case_id'], 'human': b['human_outcome'],
                           'from': b['predicted'], 'to': c['predicted'],
                           'verdict': f"{b['verdict']} -> {c['verdict']}"}
                          for b, c in zip(baseline.get('rows', []), candidate.get('rows', []))
                          if b['case_id'] == c['case_id'] and b['predicted'] != c['predicted']],
    }


def render(summary, title='Evaluation'):
    lines = [title, '=' * len(title),
             f"cases {summary['cases']}   severity score {summary['severity_score']}   "
             f"agreement {summary['agreement']} "
             f"(90% interval {summary.get('agreement_interval_90pct')})",
             f"exact matches {summary['verdicts']['match']}, acceptable {summary['verdicts']['acceptable']}, "
             f"more cautious than the underwriter {summary['verdicts']['conservative']}, "
             f"LESS cautious {summary['verdicts']['unsafe']}", '']
    requests = summary['evidence_requests']
    lines.append(f"evidence requests: {requests['requested']} made, {requests['humans_requested']} recorded "
                 f"by humans, precision {requests['precision']}, recall {requests['recall']}")
    quality_summary = summary.get('quality')
    if quality_summary:
        lines.append(f"reasoning quality {quality_summary['score']}: rules named {quality_summary['rules_named']}, "
                     f"citations cover rules {quality_summary['citations_cover_rules']}, evidence-request F1 "
                     f"{quality_summary['evidence_request_f1']}, rationale overlap {quality_summary['rationale_overlap']}, "
                     f"cases with unsupported figures {quality_summary['cases_with_unsupported_figures']}"
                     + (f", judge {quality_summary['judge']['score']} over {quality_summary['judge']['graded']}"
                        if quality_summary.get('judge') else ''))
    lines.append('')
    lines.append(f"{'human outcome':<16} {'n':>3}  {'score':>6}  predictions")
    for outcome, entry in summary['by_human_outcome'].items():
        predictions = ', '.join(f'{k} {v}' for k, v in entry['predictions'].items())
        lines.append(f"{outcome:<16} {entry['n']:>3}  {entry['severity_score']:>6}  {predictions}")
    if summary['errors']:
        lines += ['', f"{len(summary['errors'])} case(s) could not be scored:"]
        lines += [f"  {e['case_id']}: {e['error']}" for e in summary['errors'][:5]]
    if summary['cases'] < 50:
        lines += ['', f"Only {summary['cases']} cases. Treat every figure here as indicative: the interval "
                      'above is wide for a reason, and a tuning gain measured on this few cases is as likely '
                      'to be noise as signal.']
    return '\n'.join(lines)
