"""Contract and behaviour tests for the mock ML adapter.

These check that the adapter cannot accidentally enable an acceptance it has no basis
for: unknown documents stay uncertified, unfamiliar inputs are flagged, excluded
attributes stay out of the feature row, and the response satisfies every field the demo
runner re-validates in demo/server.py.
"""
import json, math, sys, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'data'))

import generate  # noqa: E402
import schema  # noqa: E402

STANDARD_PROFILE = {'name': 'Fictional Applicant', 'age': 32, 'sex': 'Male',
                    'occupation': 'Office professional', 'product': 'Life', 'cover': 500000,
                    'bmi': 23.1, 'smoker': False, 'condition': 'none', 'annualIncome': 90000,
                    'existingCover': 100000, 'termYears': 25, 'systolic': 118, 'diastolic': 76,
                    'hba1c': 5.2, 'egfr': 100, 'ldl': 2.5, 'diagnosisYears': 0,
                    'hospitalisations': 0, 'reportAgeDays': 30}


class FeatureSchema(unittest.TestCase):
    def test_field_catalogue_matches_the_demo(self):
        demo = json.loads((ROOT.parent / 'demo' / 'dist' / 'attributes.json').read_text())
        self.assertEqual(set(demo['extra']), set(schema.EXTRA))
        self.assertEqual(set(demo['core']), set(schema.CORE))

    def test_name_and_sex_never_become_features(self):
        row = schema.features_from_profile({**STANDARD_PROFILE, 'sex': 'Female'})
        self.assertNotIn('sex', row)
        self.assertNotIn('name', row)
        self.assertEqual(set(row), set(schema.FEATURE_ORDER))

    def test_unknown_optional_values_stay_unknown(self):
        row = schema.features_from_profile({**STANDARD_PROFILE, 'hba1c': None, 'annualIncome': None})
        self.assertIsNone(row['hba1c'])
        self.assertIsNone(row['cover_to_income'], 'a ratio over an unknown income must not be invented')

    def test_out_of_range_declarations_are_treated_as_unknown_not_clipped(self):
        row = schema.features_from_profile({**STANDARD_PROFILE, 'hba1c': 99, 'bmi': 3})
        self.assertIsNone(row['hba1c'])
        self.assertIsNone(row['bmi'])

    def test_unseen_occupation_gets_a_default_class(self):
        row = schema.features_from_profile({**STANDARD_PROFILE, 'occupation': 'Astronaut'})
        self.assertEqual(row['occupation_class'], schema.DEFAULT_OCCUPATION[0])

    def test_derived_features_use_declared_values(self):
        row = schema.features_from_profile(STANDARD_PROFILE)
        self.assertEqual(row['pulse_pressure'], 42)
        self.assertEqual(row['total_exposure'], 600000)
        self.assertAlmostEqual(row['cover_to_income'], 500000 / 90000)


class Generator(unittest.TestCase):
    def test_generation_is_deterministic_for_a_seed(self):
        a, b = generate.build(300, 11), generate.build(300, 11)
        self.assertTrue(a.equals(b))

    def test_latent_columns_are_marked_for_exclusion(self):
        frame = generate.build(200, 5)
        self.assertIn('_latent_risk', frame.columns)
        self.assertTrue(all(c.startswith('_') for c in frame.columns if 'latent' in c))

    def test_shifted_probe_keeps_complete_values(self):
        probe = generate.build(120, 5, shift=True)
        for field in ('hba1c', 'egfr', 'systolic'):
            self.assertFalse(probe[field].isna().any())

    def test_clean_declarations_record_zero_years_not_blank(self):
        frame = generate.build(500, 3)
        clean = frame[frame['condition'] == 'none']
        self.assertTrue((clean['diagnosisYears'] == 0).all())


class AdapterContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from predict import Scorer
            cls.scorer = Scorer()
        except (FileNotFoundError, ImportError) as error:
            raise unittest.SkipTest(f'No trained artifact: {error}')

    def result(self, profile=None, documents=()):
        return self.scorer.score(profile or STANDARD_PROFILE, documents)

    def test_response_satisfies_every_field_the_demo_revalidates(self):
        r = self.result()
        for key in ('standard', 'calibrated', 'ood', 'evidence_complete'):
            self.assertIsInstance(r[key], bool, key)
        self.assertIsInstance(r['confidence'], float)
        self.assertTrue(0 <= r['confidence'] <= 1 and math.isfinite(r['confidence']))
        self.assertTrue(isinstance(r['model'], str) and r['model'])
        self.assertTrue(all(isinstance(h, str) for h in r['verified_document_hashes']))

    def test_confidence_never_asserts_certainty(self):
        for profile in (STANDARD_PROFILE, {**STANDARD_PROFILE, 'age': 74, 'smoker': True,
                                           'condition': 'complex', 'hba1c': 11.0}):
            self.assertLess(self.result(profile)['confidence'], 1.0)

    def test_unverified_upload_blocks_evidence_completeness(self):
        r = self.result(documents=[{'id': 'DOC-1', 'sha256': 'f' * 64}])
        self.assertFalse(r['evidence_complete'])
        self.assertEqual(r['verified_document_hashes'], [])

    def test_partially_verified_upload_still_blocks(self):
        self.scorer.registry_path = Path('/nonexistent')
        r = self.result(documents=[{'sha256': 'a' * 64}, {'sha256': 'b' * 64}])
        self.assertFalse(r['evidence_complete'])

    def test_a_case_with_no_documents_is_not_complete(self):
        self.assertFalse(self.result(documents=[])['evidence_complete'])

    def test_shifted_input_is_flagged_out_of_distribution(self):
        r = self.result({**STANDARD_PROFILE, 'age': 98, 'bmi': 61, 'hba1c': 24, 'egfr': 11,
                         'cover': 10000000, 'occupation': 'Blast technician'})
        self.assertTrue(r['ood'])
        self.assertTrue(r['diagnostics']['ood_reasons'])

    def test_an_ordinary_case_is_not_flagged_out_of_distribution(self):
        self.assertFalse(self.result()['ood'])

    def test_demo_profiles_route_the_way_the_demo_narrates(self):
        payload = json.loads((ROOT / 'data' / 'demo_profiles.json').read_text())
        by_id = {p['id']: self.result(p) for p in payload['profiles']}
        gate = lambda r: r['standard'] and r['confidence'] >= 0.95 and r['calibrated'] and not r['ood']
        for case in ('P01', 'P02'):
            self.assertTrue(gate(by_id[case]), f'{case} should clear the model gate')
        for case in ('P03', 'P04', 'P05'):
            self.assertFalse(gate(by_id[case]), f'{case} should be held back for reasoning')
        self.assertTrue(by_id['P05']['ood'])

    def test_scoring_does_not_depend_on_sex(self):
        male = self.result({**STANDARD_PROFILE, 'sex': 'Male'})
        female = self.result({**STANDARD_PROFILE, 'sex': 'Female'})
        self.assertEqual(male['confidence'], female['confidence'])


if __name__ == '__main__':
    unittest.main()
