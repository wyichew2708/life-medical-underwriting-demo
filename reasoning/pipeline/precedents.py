"""Prior decisions as context: the nearest recorded cases, de-identified, as sources.

Underwriters reason by precedent, and a bank of recorded decisions is the one source of
guidance that says what this book actually did rather than what the manual says. So the
nearest prior cases are retrieved for the model to read — as context for consistency,
never as rules. They are typed `rag`, which means the validators will not let them stand
as the internal citation that proposed terms require, and they carry no name, sex,
occupation or decider.

Nearest is a fixed, inspectable distance over the catalogue's own fields, scaled by each
field's declared range, with a penalty when a field is present on one side only. The case
being assessed is never its own precedent: an evaluation that let a case see its recorded
outcome would be measuring leakage.
"""
from . import casebank

DISTANCE_FIELDS = ('age', 'bmi', 'cover', 'hba1c', 'systolic', 'egfr', 'ldl', 'hospitalisations',
                   'diagnosisYears', 'existingCover')
CATEGORICAL_PENALTY = {'smoker': 1.0, 'condition': 1.0}
MISSING_PENALTY = 0.25
LOG_SCALED = ('cover', 'existingCover')
AGE_BANDS = [(0, 35, 'under 36'), (36, 50, '36-50'), (51, 65, '51-65'), (66, 200, '66 and over')]
COVER_BANDS = [(0, 250000, 'up to 250k'), (250001, 1000000, '250k-1m'), (1000001, 2000000, '1m-2m'),
               (2000001, 10 ** 12, 'over 2m')]


def _band(value, bands):
    for low, high, label in bands:
        if isinstance(value, (int, float)) and low <= value <= high:
            return label
    return 'not stated'


def _scale(field, value, spec):
    import math
    if field in LOG_SCALED:
        return math.log10(max(float(value), 1.0)) / math.log10(max(spec['max'], 10))
    span = max(spec['max'] - spec['min'], 1e-9)
    return (float(value) - spec['min']) / span


class PrecedentIndex:
    def __init__(self, cases, core, extra):
        self.catalogue = {**core, **extra}
        self.entries = [self._entry(case) for case in cases]

    @classmethod
    def from_bank(cls, core, extra, directory=None):
        try:
            cases, _ = casebank.load_bank(directory or casebank.BANK_DIR, strict=False)
        except casebank.CaseBankError:
            cases = []
        return cls(cases, core, extra)

    def __len__(self):
        return len(self.entries)

    def _entry(self, case):
        profile = case.profile
        human = case.human
        return {'case_id': case.case_id, 'product': profile.get('product'), 'profile': profile,
                'outcome': human.get('outcome'), 'decided_at': human.get('decided_at'),
                'rationale': human.get('rationale') or human.get('notes') or '',
                'evidence_requested': list(human.get('evidence_requested') or []),
                'rating': human.get('rating'), 'later_outcome': human.get('later_outcome'),
                'cohort': human.get('cohort')}

    def distance(self, a, b):
        total = 0.0
        for field in DISTANCE_FIELDS:
            spec = self.catalogue.get(field)
            if not spec:
                continue
            va, vb = a.get(field), b.get(field)
            present = [isinstance(v, (int, float)) and not isinstance(v, bool) for v in (va, vb)]
            if all(present):
                total += (_scale(field, va, spec) - _scale(field, vb, spec)) ** 2
            elif any(present):
                total += MISSING_PENALTY
        for field, penalty in CATEGORICAL_PENALTY.items():
            if a.get(field) != b.get(field):
                total += penalty
        return round(total ** 0.5, 4)

    def nearest(self, profile, k=3, exclude=None):
        """The k closest prior cases on the same product line, closest first, ties by id."""
        if k <= 0:
            return []
        product = profile.get('product')
        scored = [(self.distance(profile, e['profile']), e['case_id'], e) for e in self.entries
                  if e['product'] == product and e['case_id'] != exclude]
        scored.sort(key=lambda item: (item[0], item[1]))
        return [{'entry': e, 'distance': d, 'identical_profile': d == 0.0} for d, _, e in scored[:k]]

    @staticmethod
    def as_source(match):
        e = match['entry']
        p = e['profile']
        described = HUMAN_OUTCOME_TEXT.get(e['outcome'], e['outcome'])
        facts = [f"{p.get('product')} application", f"age {_band(p.get('age'), AGE_BANDS)}",
                 f"cover {_band(p.get('cover'), COVER_BANDS)}",
                 f"BMI {p['bmi']}" if isinstance(p.get('bmi'), (int, float)) else None,
                 'smoker' if p.get('smoker') else 'non-smoker',
                 f"condition {p.get('condition')}"]
        for field, label in (('hba1c', 'HbA1c'), ('systolic', 'systolic BP'), ('egfr', 'eGFR'), ('ldl', 'LDL')):
            value = p.get(field)
            facts.append(f'{label} {value}' if isinstance(value, (int, float)) else f'{label} not recorded')
        text = 'Prior decision (de-identified): ' + ', '.join(f for f in facts if f) + '. '
        text += f"Recorded outcome: {described}."
        if e.get('rating'):
            text += f" Rating: {e['rating']}."
        if e.get('rationale'):
            text += f" Rationale: {e['rationale']}"
        if e.get('evidence_requested'):
            text += ' Evidence requested: ' + '; '.join(e['evidence_requested']) + '.'
        if e.get('later_outcome'):
            later = e['later_outcome']
            text += f" Later outcome once evidence arrived: {later.get('outcome')}."
        if match['identical_profile']:
            text += ' Note: identical declared profile to this case.'
        return {'id': f"PREC-{e['case_id']}", 'type': 'rag',
                'title': f"Prior decision {e['case_id']}: {e['outcome']}"
                         + (f" ({e['decided_at']})" if e.get('decided_at') else ''),
                'excerpt': text[:1500], 'distance': match['distance'],
                'note': 'Context for consistency, not a rule. Cannot support proposed terms on its own.'}


HUMAN_OUTCOME_TEXT = {k: v['description'] for k, v in casebank.HUMAN_OUTCOMES.items()}
