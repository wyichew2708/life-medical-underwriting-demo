"""Recording an underwriter's decision on an assessed case."""
import json, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import casebank  # noqa: E402
from pipeline.config import PipelineConfig  # noqa: E402
from pipeline.engine import Pipeline  # noqa: E402

PROFILE = {'name': 'Test Person', 'age': 52, 'sex': 'Male', 'occupation': 'Engineer', 'product': 'Life',
           'cover': 600000, 'bmi': 29.0, 'smoker': False, 'condition': 'controlled', 'annualIncome': 120000,
           'existingCover': 0, 'termYears': 20, 'systolic': 138, 'diastolic': 88, 'hba1c': 6.9, 'egfr': 80,
           'ldl': 3.4, 'diagnosisYears': 4, 'hospitalisations': 0, 'reportAgeDays': 30}


class Feedback(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.bank = Path(self.tmp.name) / 'bank'
        self.pipeline = Pipeline(client=None, config=PipelineConfig(), precedents=False)
        self.assessment = self.pipeline.assess({'profile': PROFILE, 'evidence': {'complete': True, 'findings': [],
                                                                                 'warnings': [], 'values': []}},
                                               offline=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_decision_is_appended_with_a_snapshot_of_what_the_pipeline_said(self):
        case = casebank.record_feedback(
            {'profile': PROFILE, 'assessment': self.assessment,
             'human': {'outcome': 'terms', 'decided_by': 'A. Underwriter', 'rating': '+25%',
                       'rationale': 'Controlled for four years.', 'evidence_requested': 'none'}},
            directory=self.bank, validate_profile=self.pipeline.validate_profile)
        self.assertTrue(case.case_id.startswith('FB-'))
        stored = json.loads((self.bank / case.case_id / 'case.json').read_text())
        self.assertEqual(stored['human']['outcome'], 'terms')
        self.assertEqual(stored['human']['source'], 'feedback')
        self.assertEqual(stored['assessment']['recommendation'], self.assessment['recommendation'])
        self.assertEqual(stored['assessment']['provenance']['config_hash'], PipelineConfig().fingerprint())
        self.assertIn('RTE-HBA1C-65', stored['assessment']['rules_fired'])
        self.assertEqual(stored['evidence_source'], 'assessment snapshot at feedback time')
        cases, problems = casebank.load_bank(self.bank)
        self.assertEqual(problems, [])
        self.assertEqual(cases[0].human['evidence_requested'], ['none'])
        self.assertIn('case_id', cases[0].to_pipeline_case())

    def test_feedback_never_overwrites_and_needs_a_valid_decision(self):
        payload = {'case_id': 'CASE-X', 'profile': PROFILE,
                   'human': {'outcome': 'refer', 'decided_by': 'someone'}}
        casebank.record_feedback(payload, directory=self.bank)
        with self.assertRaises(casebank.CaseBankError):
            casebank.record_feedback(payload, directory=self.bank)
        with self.assertRaises(casebank.CaseBankError):
            casebank.record_feedback({'profile': PROFILE, 'human': {'outcome': 'approve', 'decided_by': 'x'}},
                                     directory=self.bank)
        with self.assertRaises(casebank.CaseBankError):
            casebank.record_feedback({'profile': PROFILE, 'human': {'outcome': 'refer'}}, directory=self.bank)
        with self.assertRaises(Exception):
            casebank.record_feedback({'profile': {**PROFILE, 'age': 300},
                                      'human': {'outcome': 'refer', 'decided_by': 'x'}},
                                     directory=self.bank, validate_profile=self.pipeline.validate_profile)

    def test_a_recorded_case_is_scored_by_the_evaluator_like_any_other(self):
        from pipeline import evaluation
        casebank.record_feedback({'case_id': 'CASE-FB', 'profile': PROFILE, 'assessment': self.assessment,
                                  'human': {'outcome': 'terms', 'decided_by': 'x'}}, directory=self.bank)
        cases, _ = casebank.load_bank(self.bank)
        summary = evaluation.evaluate(self.pipeline, cases, offline=True)
        self.assertEqual(summary['cases'], 1)
        self.assertIn(summary['rows'][0]['verdict'], ('match', 'acceptable', 'conservative', 'unsafe'))


if __name__ == '__main__':
    unittest.main()
