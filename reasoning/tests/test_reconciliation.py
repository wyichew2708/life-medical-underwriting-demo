"""Declared versus documented: extraction values, reconciliation, and the figure check.

The declared profile is the applicant's account and the documents are the evidence. These
tests pin the asymmetries: a verified document value fills a gap, an unverified one may
only make the case more cautious, a disagreement routes on the worse value and is never
resolved in the applicant's favour, and the model may not write a number it was not given.
"""
import json, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import casebank, extraction, reconcile, validators  # noqa: E402
from pipeline.config import PipelineConfig  # noqa: E402
from pipeline.engine import Pipeline  # noqa: E402
from pipeline.knowledge import load as load_knowledge  # noqa: E402

CORE, EXTRA = extraction.catalogue()
RULES = load_knowledge().rules

PROFILE = {'name': 'Test Person', 'age': 44, 'sex': 'Female', 'occupation': 'Teacher', 'product': 'Life',
           'cover': 400000, 'bmi': 26.0, 'smoker': False, 'condition': 'none',
           'annualIncome': 90000, 'existingCover': 0, 'termYears': 20,
           'systolic': 124, 'diastolic': 80, 'hba1c': 5.4, 'egfr': 98, 'ldl': 2.4,
           'diagnosisYears': 0, 'hospitalisations': 0, 'reportAgeDays': 40}


def value(field, val, verified=None, page=1):
    entry = {'field': field, 'value': val, 'id': 'DOC-1', 'page': page, 'quote': f'{field} {val}'}
    if verified is not None:
        entry['verified'] = verified
    return entry


def evidence(*values, complete=True, warnings=()):
    return {'complete': complete, 'warnings': list(warnings), 'values': list(values),
            'findings': [{'id': 'DOC-1', 'page': 1, 'quote': 'report', 'text': 'A synthetic report.'}]}


class Reconciliation(unittest.TestCase):
    def test_a_value_inside_tolerance_is_confirmed_and_routing_is_unchanged(self):
        out = reconcile.reconcile(PROFILE, evidence(value('hba1c', 5.5)), CORE, EXTRA, RULES)
        self.assertEqual(out['records'][0]['status'], 'confirmed')
        self.assertEqual(out['profile']['hba1c'], 5.4)
        self.assertEqual(out['warnings'], [])

    def test_a_disagreement_routes_on_the_worse_value_and_warns(self):
        out = reconcile.reconcile(PROFILE, evidence(value('hba1c', 7.2)), CORE, EXTRA, RULES)
        self.assertEqual(out['records'][0]['status'], 'discrepancy')
        self.assertEqual(out['profile']['hba1c'], 7.2)
        self.assertEqual(len(out['discrepancies']), 1)
        self.assertIn('disagree', out['warnings'][0])

    def test_a_disagreement_in_the_applicants_favour_still_routes_on_the_declared_worse_value(self):
        declared = {**PROFILE, 'hba1c': 7.2}
        out = reconcile.reconcile(declared, evidence(value('hba1c', 5.4)), CORE, EXTRA, RULES)
        self.assertEqual(out['records'][0]['status'], 'discrepancy')
        self.assertEqual(out['profile']['hba1c'], 7.2)

    def test_lower_is_worse_for_renal_function(self):
        out = reconcile.reconcile(PROFILE, evidence(value('egfr', 52)), CORE, EXTRA, RULES)
        self.assertEqual(out['profile']['egfr'], 52)
        out = reconcile.reconcile({**PROFILE, 'egfr': 52}, evidence(value('egfr', 98)), CORE, EXTRA, RULES)
        self.assertEqual(out['profile']['egfr'], 52)

    def test_a_declared_non_smoker_documented_as_smoking_is_routed_as_a_smoker(self):
        out = reconcile.reconcile(PROFILE, evidence(value('smoker', True)), CORE, EXTRA, RULES)
        self.assertTrue(out['profile']['smoker'])
        self.assertEqual(out['records'][0]['status'], 'discrepancy')

    def test_condition_severity_only_rises(self):
        out = reconcile.reconcile(PROFILE, evidence(value('condition', 'controlled')), CORE, EXTRA, RULES)
        self.assertEqual(out['profile']['condition'], 'controlled')
        out = reconcile.reconcile({**PROFILE, 'condition': 'complex'}, evidence(value('condition', 'none')),
                                  CORE, EXTRA, RULES)
        self.assertEqual(out['profile']['condition'], 'complex')

    def test_a_verified_value_fills_a_gap(self):
        blank = {**PROFILE, 'hba1c': None}
        out = reconcile.reconcile(blank, evidence(value('hba1c', 5.3, verified=True)), CORE, EXTRA, RULES)
        self.assertEqual(out['records'][0]['status'], 'filled')
        self.assertEqual(out['profile']['hba1c'], 5.3)

    def test_an_unverified_value_fills_a_gap_only_when_it_makes_the_case_more_cautious(self):
        blank = {**PROFILE, 'hba1c': None}
        benign = reconcile.reconcile(blank, evidence(value('hba1c', 5.3, verified=False)), CORE, EXTRA, RULES)
        self.assertEqual(benign['records'][0]['status'], 'unverified_not_applied')
        self.assertIsNone(benign['profile']['hba1c'])
        adverse = reconcile.reconcile(blank, evidence(value('hba1c', 8.9, verified=False)), CORE, EXTRA, RULES)
        self.assertEqual(adverse['records'][0]['status'], 'filled_unverified_adverse')
        self.assertEqual(adverse['profile']['hba1c'], 8.9)
        # No verification flag at all is treated exactly like a failed one.
        untagged = reconcile.reconcile(blank, evidence(value('hba1c', 5.3)), CORE, EXTRA, RULES)
        self.assertEqual(untagged['records'][0]['status'], 'unverified_not_applied')

    def test_an_implausible_value_is_ignored_with_a_warning(self):
        out = reconcile.reconcile(PROFILE, evidence(value('hba1c', 61.0)), CORE, EXTRA, RULES)
        self.assertEqual(out['records'][0]['status'], 'implausible')
        self.assertEqual(out['profile']['hba1c'], 5.4)
        self.assertIn('outside the catalogue range', out['warnings'][0])

    def test_two_documents_that_disagree_route_on_the_worst_of_all(self):
        out = reconcile.reconcile(PROFILE, evidence(value('systolic', 150), value('systolic', 165)),
                                  CORE, EXTRA, RULES)
        self.assertEqual(out['profile']['systolic'], 165)
        self.assertEqual(len(out['discrepancies']), 2)

    def test_no_values_means_the_declared_profile_is_routed_as_supplied(self):
        out = reconcile.reconcile(PROFILE, {'complete': True, 'findings': [], 'warnings': []}, CORE, EXTRA, RULES)
        self.assertEqual(out['profile'], PROFILE)
        self.assertIn('routed as supplied', out['summary'])


class EngineReconciliation(unittest.TestCase):
    def pipeline(self):
        return Pipeline(client=None, config=PipelineConfig())

    def ml(self):
        return {'available': True, 'model': 'stub', 'standard': True, 'confidence': 0.99, 'calibrated': True,
                'ood': False, 'evidence_complete': True}

    def test_rules_run_on_the_routed_profile(self):
        case = {'profile': PROFILE, 'evidence': evidence(value('hba1c', 7.2)), 'ml': self.ml()}
        result = self.pipeline().assess(case, offline=True)
        fired = {r['id'] for r in result['rules']['fired']}
        self.assertIn('RTE-HBA1C-65', fired)
        self.assertEqual(result['routed_profile']['hba1c'], 7.2)
        self.assertEqual(result['profile']['hba1c'], 5.4)
        self.assertNotEqual(result['recommendation'], 'straight_through')
        self.assertIn('Evidence warnings are unresolved.', result['straight_through']['blocked_by'])
        self.assertTrue(any('contradicted by' in r for r in result['output']['reasons']))

    def test_a_disagreement_alone_is_at_least_a_referral(self):
        # A discrepancy on a field with no rule to trigger still cannot pass through.
        case = {'profile': PROFILE, 'evidence': evidence(value('bmi', 29.0)), 'ml': self.ml()}
        result = self.pipeline().assess(case, offline=True)
        self.assertEqual(result['rules']['outcome'], 'refer')
        self.assertEqual(result['recommendation'], 'refer')

    def test_a_confirmed_profile_still_passes_straight_through(self):
        case = {'profile': PROFILE, 'evidence': evidence(value('hba1c', 5.4), value('smoker', False)),
                'ml': self.ml()}
        result = self.pipeline().assess(case, offline=True)
        self.assertEqual(result['recommendation'], 'straight_through')
        self.assertEqual(result['reconciliation']['discrepancies'], [])

    def test_a_verified_lab_value_satisfies_a_missing_field(self):
        blank = {**PROFILE, 'hba1c': None}
        case = {'profile': blank, 'evidence': evidence(value('hba1c', 5.3, verified=True)), 'ml': self.ml()}
        result = self.pipeline().assess(case, offline=True)
        self.assertNotIn('hba1c', result['evidence_requirements']['unsupplied_optional_attributes'])
        self.assertEqual(result['routed_profile']['hba1c'], 5.3)


class Figures(unittest.TestCase):
    def test_numbers_are_read_from_prose_and_identifiers_are_skipped(self):
        found = validators.figures_in('HbA1c 7.2% under RTE-HBA1C-65 (DOC-1 p.2), cover 1,500,000 and 95%')
        self.assertIn(7.2, found)
        self.assertIn(0.072, found)
        self.assertIn(1500000.0, found)
        self.assertIn(95.0, found)
        self.assertIn(0.95, found)
        self.assertNotIn(65.0, found)
        self.assertNotIn(1.0, found)
        self.assertNotIn(2.0, found)

    def test_a_figure_absent_from_the_case_data_fails_validation(self):
        allowed = validators.allowed_figures(PROFILE, [{'excerpt': 'HbA1c at or above 6.5% requires evidence.'}])
        output = {'recommendation': 'refer', 'explanation': 'HbA1c of 7.9 exceeds the 6.5 threshold.',
                  'reasons': ['HbA1c 7.9'], 'citations': ['UW-MED-01'], 'missing_information': []}
        self.assertEqual(validators.check_figures(output, allowed), [7.9])
        output['explanation'] = 'HbA1c of 5.4 is under the 6.5 threshold; two rules were checked.'
        output['reasons'] = ['HbA1c 5.4, BMI 26']
        self.assertEqual(validators.check_figures(output, allowed), [])

    def test_the_figure_check_is_part_of_output_validation_when_a_set_is_given(self):
        sources = [{'id': 'UW-MED-01', 'type': 'internal', 'excerpt': 'Threshold 6.5%'}]
        good = {'recommendation': 'refer', 'explanation': 'HbA1c 5.4 recorded.', 'reasons': ['ok'],
                'citations': ['UW-MED-01'], 'missing_information': []}
        bad = {**good, 'explanation': 'HbA1c 8.8 recorded.'}
        allowed = validators.allowed_figures(PROFILE, sources)
        self.assertEqual(validators.validate_output(good, {'complete': True}, sources, 'refer', figures=allowed), [])
        errors = validators.validate_output(bad, {'complete': True}, sources, 'refer', figures=allowed)
        self.assertTrue(any('8.8' in e for e in errors))
        # Without a figure set the check is off, so older callers are unaffected.
        self.assertEqual(validators.validate_output(bad, {'complete': True}, sources, 'refer'), [])


class ExtractionValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            extraction._pymupdf()
        except extraction.ExtractionError as error:
            raise unittest.SkipTest(str(error))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        fitz = extraction._pymupdf()
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((60, 80), 'SYNTHETIC REPORT', fontsize=12, fontname='helv')
        page.insert_text((60, 110), 'HbA1c: 6.1%', fontsize=11, fontname='helv')
        page.insert_text((60, 130), 'Tobacco or nicotine use: yes', fontsize=11, fontname='helv')
        path = Path(self.tmp.name) / 'report.pdf'
        doc.save(path)
        doc.close()
        self.documents = [{'id': 'DOC-1', 'name': 'report.pdf', 'type': 'application/pdf', 'path': path}]

    def tearDown(self):
        self.tmp.cleanup()

    def client(self, response):
        return type('Stub', (), {'configured': True, 'model': 'stub-vision',
                                 'complete': lambda self, messages: response})()

    def response(self, **overrides):
        return {'complete': True, 'warnings': [],
                'findings': [{'id': 'DOC-1', 'page': 1, 'quote': 'HbA1c: 6.1%', 'text': 'HbA1c 6.1 percent.'}],
                'values': [{'field': 'hba1c', 'value': 6.1, 'id': 'DOC-1', 'page': 1, 'quote': 'HbA1c: 6.1%'},
                           {'field': 'smoker', 'value': True, 'id': 'DOC-1', 'page': 1,
                            'quote': 'Tobacco or nicotine use: yes'}], **overrides}

    def test_values_are_kept_and_verified_against_the_page_text(self):
        out = extraction.extract(self.documents, PROFILE, self.client(self.response()))
        self.assertEqual([v['field'] for v in out['values']], ['hba1c', 'smoker'])
        self.assertTrue(all(v['verified'] for v in out['values']))
        self.assertTrue(out['findings'][0]['verified'])

    def test_a_quote_not_on_the_page_is_marked_unverified_and_warned(self):
        response = self.response()
        response['values'][0]['quote'] = 'HbA1c: 9.9%'
        out = extraction.extract(self.documents, PROFILE, self.client(response))
        self.assertFalse(out['values'][0]['verified'])
        self.assertTrue(out['values'][1]['verified'])
        self.assertTrue(any('not found in the page text' in w for w in out['warnings']))

    def test_values_for_unknown_fields_or_wrong_types_are_dropped_not_fatal(self):
        response = self.response()
        response['values'] += [{'field': 'blood_type', 'value': 'O', 'id': 'DOC-1', 'page': 1, 'quote': 'x'},
                               {'field': 'hba1c', 'value': 'six', 'id': 'DOC-1', 'page': 1, 'quote': 'x'},
                               {'field': 'condition', 'value': 'terminal', 'id': 'DOC-1', 'page': 1, 'quote': 'x'}]
        out = extraction.extract(self.documents, PROFILE, self.client(response))
        self.assertEqual(len(out['values']), 2)
        self.assertEqual(len([w for w in out['warnings'] if w.startswith('Dropped')]), 3)

    def test_a_value_citing_a_missing_page_is_rejected(self):
        response = self.response()
        response['values'][0]['page'] = 4
        with self.assertRaises(extraction.ExtractionError):
            extraction.extract(self.documents, PROFILE, self.client(response))

    def test_an_image_has_no_text_layer_so_nothing_can_be_verified(self):
        fitz = extraction._pymupdf()
        with fitz.open(self.documents[0]['path']) as doc:
            png = doc[0].get_pixmap().tobytes('png')
        path = Path(self.tmp.name) / 'scan.png'
        path.write_bytes(png)
        documents = [{'id': 'DOC-1', 'name': 'scan.png', 'type': 'image/png', 'path': path}]
        out = extraction.extract(documents, PROFILE, self.client(self.response()))
        self.assertIsNone(out['values'][0]['verified'])
        routed = reconcile.reconcile({**PROFILE, 'hba1c': None}, out, CORE, EXTRA, RULES)
        self.assertEqual(routed['records'][0]['status'], 'unverified_not_applied')

    def test_the_case_bank_extractor_delegates_and_keeps_its_error_type(self):
        bank = Path(self.tmp.name) / 'bank' / 'CASE-1'
        (bank / 'documents').mkdir(parents=True)
        (bank / 'documents' / 'report.pdf').write_bytes(self.documents[0]['path'].read_bytes())
        (bank / 'case.json').write_text(json.dumps(
            {'profile': PROFILE, 'human': {'outcome': 'standard', 'decided_by': 'an underwriter'}}))
        cases, _ = casebank.load_bank(bank.parent)
        out = casebank.extract_evidence(cases[0], self.client(self.response()))
        self.assertEqual(out['values'][0]['field'], 'hba1c')
        bad = self.response()
        bad['findings'][0]['page'] = 9
        with self.assertRaises(casebank.CaseBankError):
            casebank.extract_evidence(cases[0], self.client(bad))


if __name__ == '__main__':
    unittest.main()
