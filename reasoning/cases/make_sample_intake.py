"""Generate a synthetic intake folder: case documents, outcomes and an import manifest.

This exists so the case-bank and tuning tools can be run end to end without real customer
files. Every applicant, report and decision here is invented. The "human outcomes" were
produced by the rule of thumb documented below plus a handful of deliberate
disagreements, because a bank in which the underwriter is a clean function of the inputs
would make the tuner look far better than it is.

The ML screen attached to each case is produced by the real trained model in
ml_training/ when it is available. `evidence_complete` is asserted by this generator, not
certified by anyone: in a real bank it comes from the evidence registry.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
INTAKE = HERE / 'sample_intake'
DOCUMENTS = INTAKE / 'documents'
ML_TRAINING = HERE.parent.parent / 'ml_training'


def pymupdf():
    try:
        import pymupdf
        return pymupdf
    except ImportError:
        import fitz
        return fitz


# Outcome archetypes: what kind of case tends to end in each recorded decision.
ARCHETYPES = [
    ('standard', 8, dict(age=(26, 45), cover=(150000, 700000), bmi=(20, 27), smoker=False,
                         condition='none', hba1c=(4.9, 5.6), systolic=(105, 128), egfr=(88, 115),
                         ldl=(1.9, 3.2), hospitalisations=0, report_age=(10, 90))),
    ('terms', 5, dict(age=(44, 62), cover=(200000, 800000), bmi=(27, 34), smoker=False,
                      condition='controlled', hba1c=(6.1, 7.4), systolic=(132, 152), egfr=(62, 88),
                      ldl=(3.0, 4.3), hospitalisations=0, report_age=(20, 150))),
    ('more_evidence', 5, dict(age=(33, 58), cover=(400000, 1500000), bmi=(23, 33), smoker=False,
                              condition='controlled', hba1c=None, systolic=None, egfr=None,
                              ldl=None, hospitalisations=0, report_age=None)),
    ('refer', 5, dict(age=(50, 71), cover=(900000, 2600000), bmi=(26, 35), smoker=True,
                      condition='controlled', hba1c=(6.0, 7.8), systolic=(138, 158), egfr=(58, 85),
                      ldl=(3.1, 4.6), hospitalisations=(0, 1), report_age=(30, 200))),
    ('postpone', 2, dict(age=(38, 60), cover=(300000, 900000), bmi=(24, 32), smoker=False,
                         condition='complex', hba1c=(7.0, 8.4), systolic=(140, 160), egfr=(55, 78),
                         ldl=(3.2, 4.4), hospitalisations=(1, 2), report_age=(200, 420))),
    ('decline', 3, dict(age=(55, 74), cover=(500000, 2000000), bmi=(31, 44), smoker=True,
                        condition='complex', hba1c=(8.6, 11.5), systolic=(158, 185), egfr=(24, 48),
                        ldl=(4.0, 5.6), hospitalisations=(2, 4), report_age=(60, 300))),
]
OCCUPATIONS = ['Office professional', 'Teacher', 'Accountant', 'Engineer', 'Business owner',
               'Sales executive', 'Nurse', 'Designer', 'Delivery rider', 'Construction supervisor']
NAMES = ['Wei Ming Tan', 'Aisha Rahman', 'Daniel Ong', 'Priya Nair', 'Marcus Lee', 'Siti Zubaidah',
         'Kenneth Goh', 'Ravi Kumar', 'Joanne Lim', 'Farid Hassan', 'Chloe Teo', 'Arun Pillai',
         'Grace Yeo', 'Hakim Yusof', 'Ivy Chan', 'Nurul Aina', 'Benjamin Koh', 'Sharmila Devi',
         'Adrian Sim', 'Mei Ling Ho', 'Zainab Omar', 'Terence Ng', 'Lydia Wong', 'Iqbal Salleh',
         'Samuel Chia', 'Rachel Phua', 'Vikram Raj', 'Natalie Seah']


def spread(rng, bounds, digits=0):
    if bounds is None:
        return None
    if not isinstance(bounds, tuple):
        return bounds
    low, high = bounds
    value = rng.uniform(low, high)
    return round(value, digits) if digits else int(round(value))


def build_cases(seed=4242):
    import random
    rng = random.Random(seed)
    cases, index = [], 0
    for outcome, count, shape in ARCHETYPES:
        for _ in range(count):
            index += 1
            product = 'Life' if rng.random() < 0.6 else 'Medical'
            income = rng.choice([60000, 85000, 110000, 150000, 240000, 320000])
            profile = {
                'name': NAMES[(index - 1) % len(NAMES)],
                'age': spread(rng, shape['age']),
                'sex': rng.choice(['Female', 'Male']),
                'occupation': rng.choice(OCCUPATIONS),
                'product': product,
                'cover': int(round(spread(rng, shape['cover']), -3)),
                'bmi': spread(rng, shape['bmi'], 1),
                'smoker': shape['smoker'] if isinstance(shape['smoker'], bool) else rng.random() < 0.5,
                'condition': shape['condition'],
                'annualIncome': income,
                'existingCover': rng.choice([0, 100000, 250000, 500000]),
                'termYears': rng.choice([10, 15, 20, 25, 30]),
                'systolic': spread(rng, shape['systolic']),
                'diastolic': None,
                'hba1c': spread(rng, shape['hba1c'], 1),
                'egfr': spread(rng, shape['egfr']),
                'ldl': spread(rng, shape['ldl'], 1),
                'diagnosisYears': 0 if shape['condition'] == 'none' else rng.randint(1, 12),
                'hospitalisations': spread(rng, shape['hospitalisations']),
                'reportAgeDays': spread(rng, shape['report_age']),
            }
            if profile['systolic']:
                profile['diastolic'] = int(profile['systolic'] * rng.uniform(0.60, 0.68))
            cases.append({'case_id': f'CASE-{index:03d}', 'profile': profile, 'outcome': outcome,
                          'cohort': f"{product.lower()}-{outcome}"})

    # Three deliberate disagreements. Real underwriters are not a function of the inputs:
    # one clean case escalated on a file note, one rated case accepted at standard rates,
    # one large case where the underwriter wanted evidence the rules do not ask for.
    cases[0]['outcome'] = 'refer'
    cases[0]['note'] = 'Escalated on a prior-claim file note not visible in the structured data.'
    cases[8]['outcome'] = 'standard'
    cases[8]['note'] = 'Long-standing well-controlled condition; senior underwriter accepted at standard rates.'
    cases[-1]['outcome'] = 'more_evidence'
    cases[-1]['note'] = 'Declined only after a specialist report was obtained; recorded as an evidence request.'

    # Two cases where the report does not say what the applicant declared. The profile carries
    # the declaration; the PDF and the evidence carry what the document actually shows.
    clean = dict(cases[2]['profile'])
    cases.append({'case_id': f'CASE-{len(cases) + 1:03d}', 'outcome': 'more_evidence', 'cohort': 'life-discrepancy',
                  'profile': {**clean, 'name': 'Ethan Lau', 'product': 'Life', 'hba1c': 5.4, 'condition': 'none'},
                  'document_values': {'hba1c': 7.2},
                  'note': 'Declared HbA1c 5.4; the attached report shows 7.2. Repeat test and GP report requested.'})
    cases.append({'case_id': f'CASE-{len(cases) + 1:03d}', 'outcome': 'refer', 'cohort': 'medical-discrepancy',
                  'profile': {**clean, 'name': 'Hannah Quek', 'product': 'Medical', 'smoker': False},
                  'document_values': {'smoker': True},
                  'note': 'Declared non-smoker; the report records nicotine use. Referred for non-disclosure review.'})
    return cases


def document_profile(case):
    """What the document shows: the declared profile with any document-only values on top."""
    return {**case['profile'], **case.get('document_values', {})}


def report_pdf(case, path):
    """A one-page synthetic clinical summary carrying the case's own facts."""
    fitz = pymupdf()
    profile = document_profile(case)
    lines = [
        'SYNTHETIC MEDICAL REPORT — DEMONSTRATION ONLY',
        f"Case reference: {case['case_id']}",
        f"Patient: {profile['name']}    Age: {profile['age']}    Sex: {profile['sex']}",
        f"Occupation: {profile['occupation']}",
        '',
        'Clinical summary',
        f"Declared condition status: {profile['condition']}"
        + (f" (diagnosed {profile['diagnosisYears']} years ago)" if profile['diagnosisYears'] else ''),
        f"Tobacco or nicotine use: {'yes' if profile['smoker'] else 'no'}",
        f"Hospital admissions in the past 12 months: {profile['hospitalisations']}",
        '',
        'Measurements',
        f"Body mass index: {profile['bmi']}",
        f"Blood pressure: {profile['systolic']}/{profile['diastolic']} mmHg"
        if profile['systolic'] else 'Blood pressure: not recorded in this report',
        f"HbA1c: {profile['hba1c']}%" if profile['hba1c'] else 'HbA1c: not performed',
        f"eGFR: {profile['egfr']} mL/min/1.73m2" if profile['egfr'] else 'eGFR: not performed',
        f"LDL cholesterol: {profile['ldl']} mmol/L" if profile['ldl'] else 'LDL cholesterol: not performed',
        '',
        f"Report age at application: "
        + (f"{profile['reportAgeDays']} days" if profile['reportAgeDays'] else 'not stated'),
        '',
        'This document is fictional, generated for software testing. It is not a clinical record',
        'and describes no real person.',
    ]
    doc = fitz.open()
    page = doc.new_page()
    y = 60
    for line in lines:
        page.insert_text((56, y), line, fontsize=11 if not line.isupper() else 12, fontname='helv')
        y += 18
    doc.save(path)
    doc.close()


def synthetic_values(profile):
    """Typed values with the exact quote the report prints, as an extractor would return them."""
    values = [{'field': 'condition', 'value': profile['condition'],
               'quote': f"Declared condition status: {profile['condition']}"},
              {'field': 'smoker', 'value': profile['smoker'],
               'quote': f"Tobacco or nicotine use: {'yes' if profile['smoker'] else 'no'}"},
              {'field': 'bmi', 'value': profile['bmi'], 'quote': f"Body mass index: {profile['bmi']}"},
              {'field': 'hospitalisations', 'value': profile['hospitalisations'],
               'quote': f"Hospital admissions in the past 12 months: {profile['hospitalisations']}"}]
    if profile['systolic']:
        quote = f"Blood pressure: {profile['systolic']}/{profile['diastolic']} mmHg"
        values += [{'field': 'systolic', 'value': profile['systolic'], 'quote': quote},
                   {'field': 'diastolic', 'value': profile['diastolic'], 'quote': quote}]
    if profile['hba1c']:
        values.append({'field': 'hba1c', 'value': profile['hba1c'], 'quote': f"HbA1c: {profile['hba1c']}%"})
    if profile['egfr']:
        values.append({'field': 'egfr', 'value': profile['egfr'], 'quote': f"eGFR: {profile['egfr']} mL/min/1.73m2"})
    if profile['ldl']:
        values.append({'field': 'ldl', 'value': profile['ldl'], 'quote': f"LDL cholesterol: {profile['ldl']} mmol/L"})
    return [{**v, 'id': 'DOC-1', 'page': 1} for v in values]


def synthetic_evidence(case):
    """Evidence consistent with the generated report. Written here, not extracted."""
    profile = document_profile(case)
    findings = [{'id': 'DOC-1', 'source': 'Synthetic medical report', 'page': 1,
                 'quote': f"Declared condition status: {profile['condition']}",
                 'text': f"Declared condition status recorded as {profile['condition']}"
                         + (f", diagnosed {profile['diagnosisYears']} years ago" if profile['diagnosisYears'] else '')
                         + f". Tobacco use: {'yes' if profile['smoker'] else 'no'}."}]
    measured = [label for label, value in [('HbA1c', profile['hba1c']), ('eGFR', profile['egfr']),
                                           ('LDL', profile['ldl']), ('blood pressure', profile['systolic'])]
                if value is not None]
    if measured:
        findings.append({'id': 'DOC-1', 'source': 'Synthetic medical report', 'page': 1,
                         'quote': f"HbA1c: {profile['hba1c']}%" if profile['hba1c'] else
                                  f"Blood pressure: {profile['systolic']}/{profile['diastolic']} mmHg",
                         'text': 'Report records ' + ', '.join(measured)
                                 + f". BMI {profile['bmi']}."})
    warnings = []
    complete = bool(measured)
    if not measured:
        warnings.append('The report contains no laboratory results; the metabolic and renal screen is absent.')
    if profile['reportAgeDays'] and profile['reportAgeDays'] > 365:
        warnings.append('The report is more than a year old at the date of application.')
    return {'complete': complete, 'findings': findings, 'warnings': warnings,
            'values': synthetic_values(profile)}


def decision_detail(case):
    """What a recorded decision carries beyond its label, written from the archetype."""
    profile = document_profile(case)
    outcome = case['outcome']
    drivers = []
    if profile['smoker']:
        drivers.append('tobacco use')
    if profile['condition'] != 'none':
        drivers.append(f"{profile['condition']} declared condition")
    if profile['hba1c'] and profile['hba1c'] >= 6.5:
        drivers.append(f"HbA1c {profile['hba1c']}")
    if profile['egfr'] and profile['egfr'] < 60:
        drivers.append(f"eGFR {profile['egfr']}")
    if profile['systolic'] and profile['systolic'] >= 160:
        drivers.append(f"systolic {profile['systolic']}")
    if profile['cover'] > 1000000:
        drivers.append('cover above 1m')
    detail = {'rationale': (f"{outcome}: " + (', '.join(drivers) if drivers else 'no adverse findings')
                            + '. ' + case.get('note', ''))[:400].strip()}
    if outcome == 'more_evidence':
        wanted = ['GP report covering the declared condition'] if profile['condition'] != 'none' else []
        if profile['hba1c'] is None:
            wanted += ['HbA1c', 'eGFR', 'lipid profile']
        elif profile['hba1c'] >= 6.5:
            wanted += ['repeat HbA1c', 'attending physician statement']
        if profile['systolic'] is None:
            wanted.append('blood pressure readings')
        detail['evidence_requested'] = wanted or ['attending physician statement']
        detail['later_outcome'] = {'outcome': 'terms' if profile['condition'] != 'none' else 'standard',
                                   'decided_at': '2026-08-20'}
    if outcome == 'terms':
        detail['rating'] = '+50%' if profile['hba1c'] and profile['hba1c'] >= 7.0 else '+25%'
    if case.get('document_values'):
        detail['evidence_requested'] = detail.get('evidence_requested') or ['repeat test', 'GP report']
        detail['rules_cited'] = ['non-disclosure review']
    return detail


def ml_screen(profile):
    """Score with the real trained model when it is there; say so plainly when it is not."""
    try:
        sys.path.insert(0, str(ML_TRAINING))
        from predict import Scorer
        result = Scorer().score(profile, [])
        return {**{k: result[k] for k in ('model', 'standard', 'confidence', 'calibrated', 'ood')},
                'available': True,
                # Asserted by this generator so the sample bank can exercise the STP path.
                # In a real bank this comes from the evidence registry, not from a script.
                'evidence_complete': True,
                'source': 'ml_training model; evidence_complete asserted by the sample generator'}
    except Exception as error:
        return {'available': False, 'error': f'No trained model available ({type(error).__name__}).',
                'source': 'sample generator'}


def main():
    DOCUMENTS.mkdir(parents=True, exist_ok=True)
    cases = build_cases()
    manifest = []
    for case in cases:
        pdf = DOCUMENTS / f"{case['case_id']}-report.pdf"
        report_pdf(case, pdf)
        evidence = synthetic_evidence(case)
        manifest.append({
            'case_id': case['case_id'],
            'profile': case['profile'],
            'documents': [pdf.name],
            'evidence': evidence,
            'evidence_source': 'synthetic: written by the sample generator, not extracted from the PDF',
            'ml': ml_screen(case['profile']),
            'human': {'outcome': case['outcome'], 'decided_by': 'sample generator (fictional underwriter)',
                      'decided_at': '2026-08-01', 'cohort': case['cohort'],
                      'notes': case.get('note', 'Synthetic outcome generated from the case archetype.'),
                      **decision_detail(case)},
        })
    (INTAKE / 'cases.json').write_text(json.dumps(
        {'note': 'Fictional cases for testing the case bank and tuner. No real people or decisions.',
         'cases': manifest}, indent=2) + '\n')
    counts = {}
    for entry in manifest:
        counts[entry['human']['outcome']] = counts.get(entry['human']['outcome'], 0) + 1
    print(f"Wrote {len(manifest)} cases and {len(list(DOCUMENTS.glob('*.pdf')))} PDFs to {INTAKE}")
    print('  outcomes: ' + ', '.join(f'{k} {v}' for k, v in sorted(counts.items())))
    print(f"\nImport them with:\n  python import_cases.py --manifest {INTAKE / 'cases.json'}")


if __name__ == '__main__':
    main()
