"""Reconcile what the applicant declared with what the documents say, before any rule runs.

The declared profile is the applicant's account. The documents are the evidence. When
they agree, the value is confirmed. When the documents supply a value the applicant left
blank, it fills the gap — if the quote was verified against the page. When they disagree,
the more adverse value is used for routing and the disagreement itself becomes an evidence
warning, because a declaration that contradicts its own supporting document is a finding
an underwriter has to see whichever value turns out to be right.

Two asymmetries are deliberate. An unverified value — one whose quote could not be found
in the page text, or came from an image with no text layer — may make the case more
cautious and never less: it can trigger a rule, it cannot satisfy an evidence requirement.
And a discrepancy is never resolved in the applicant's favour by this code; the routed
value is the worse one, and the human decides.
"""
from . import rules as rules_module

# Which way is worse, per field. A field not listed has no direction and a discrepancy on
# it is flagged without changing the routed value.
ADVERSE = {'age': 'high', 'cover': 'high', 'bmi': 'high', 'annualIncome': 'low', 'existingCover': 'high',
           'termYears': 'high', 'systolic': 'high', 'diastolic': 'high', 'hba1c': 'high', 'egfr': 'low',
           'ldl': 'high', 'diagnosisYears': 'high', 'hospitalisations': 'high', 'reportAgeDays': 'high'}
# Below this a difference is measurement noise, not a discrepancy. Absolute where listed,
# otherwise two percent of the declared value.
TOLERANCE = {'age': 0, 'bmi': 0.6, 'systolic': 5, 'diastolic': 5, 'hba1c': 0.15, 'egfr': 3, 'ldl': 0.15,
             'diagnosisYears': 1, 'hospitalisations': 0, 'reportAgeDays': 30, 'termYears': 0}
RELATIVE_TOLERANCE = 0.02
CONDITION_SEVERITY = {'none': 0, 'controlled': 1, 'complex': 2}


def _consistent(field, declared, extracted, spec):
    if spec['type'] == 'number':
        tolerance = TOLERANCE.get(field, abs(declared) * RELATIVE_TOLERANCE)
        return abs(declared - extracted) <= tolerance
    return declared == extracted


def _more_adverse(field, declared, extracted, spec):
    if spec['type'] == 'number':
        direction = ADVERSE.get(field)
        if direction == 'high':
            return max(declared, extracted)
        if direction == 'low':
            return min(declared, extracted)
        return declared
    if spec['type'] == 'boolean':
        return declared or extracted
    if field == 'condition':
        return max(declared, extracted, key=lambda v: CONDITION_SEVERITY.get(v, -1))
    return declared


def _plausible(spec, value):
    if spec['type'] == 'number':
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and spec['min'] <= value <= spec['max'])
    if spec['type'] == 'boolean':
        return isinstance(value, bool)
    return value in spec.get('values', [])


def would_fire(field, value, rules):
    """Would this value, on its own, match a routing rule on the field?"""
    for rule in rules or []:
        if rule.get('field') == field and rule.get('enabled', True) and rule.get('action') != 'note':
            if rules_module.compare(value, rule['operator'], rule['value']):
                return True
    return False


def _cite(value):
    return f"{value.get('id')} p.{value.get('page')} \"{value.get('quote')}\""


def reconcile(profile, evidence, core, extra, rules=None):
    """Return the routed profile, one record per extracted value, and the warnings raised."""
    catalogue = {**core, **extra}
    routed = dict(profile)
    records, warnings = [], []
    for value in (evidence or {}).get('values') or []:
        field = value.get('field')
        spec = catalogue.get(field)
        if spec is None or field == 'product':
            records.append({'field': field, 'status': 'unknown_field', 'source': _cite(value)})
            continue
        extracted = value.get('value')
        verified = value.get('verified')
        record = {'field': field, 'declared': profile.get(field), 'extracted': extracted,
                  'verified': verified, 'source': _cite(value)}
        if not _plausible(spec, extracted):
            record['status'] = 'implausible'
            warnings.append(f'Document value for {field} ({extracted!r}) is outside the catalogue range; ignored.')
            records.append(record)
            continue
        declared = profile.get(field)
        if declared is None:
            if verified is True:
                routed[field] = extracted
                record['status'] = 'filled'
                record['routed'] = extracted
            elif would_fire(field, extracted, rules):
                # Unverified, but adverse: it can only make the case more cautious.
                routed[field] = extracted
                record['status'] = 'filled_unverified_adverse'
                record['routed'] = extracted
                warnings.append(f'{field} was not declared; the document value {extracted} could not be '
                                'verified against the page text and was used only because it triggers a rule.')
            else:
                record['status'] = 'unverified_not_applied'
                record['note'] = ('Not declared and not verifiable against the page text. An unverified '
                                  'value cannot satisfy an evidence requirement.')
        elif _consistent(field, declared, extracted, spec):
            record['status'] = 'confirmed'
            record['routed'] = routed[field]
        else:
            adverse = _more_adverse(field, declared, extracted, spec)
            # Several documents may disagree; the routed value only ever gets worse.
            routed[field] = _more_adverse(field, routed[field], adverse, spec)
            record['status'] = 'discrepancy'
            record['routed'] = routed[field]
            warnings.append(f'Declared {field} {declared} but {value.get("id")} p.{value.get("page")} records '
                            f'{extracted}; routed on {routed[field]}. The declaration and the document disagree.')
        records.append(record)

    discrepancies = [r for r in records if r['status'] == 'discrepancy']
    filled = [r for r in records if r['status'] in ('filled', 'filled_unverified_adverse')]
    confirmed = [r for r in records if r['status'] == 'confirmed']
    if not records:
        summary = 'No typed values were extracted from the documents; the declared profile was routed as supplied.'
    else:
        summary = (f'{len(confirmed)} value(s) confirmed by the documents, {len(filled)} filled from them, '
                   f'{len(discrepancies)} discrepancy(ies).')
        if discrepancies:
            summary += ' Routing used the more adverse value wherever the declaration and a document disagreed.'
    return {'profile': routed, 'records': records, 'warnings': warnings, 'discrepancies': discrepancies,
            'filled': filled, 'summary': summary}
