"""Drift reporting, the shadow model and the opt-in scoring log."""
import json, sys, tempfile, unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'data'))
import drift  # noqa: E402
import experiments  # noqa: E402
import generate  # noqa: E402
import modeling  # noqa: E402
import predict  # noqa: E402


class Drift(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = generate.build(3000, 23)
        cls.reference = modeling.frame_to_features(modeling.temporal_split(modeling.strip_latent(cls.frame))['train'])

    def test_a_later_slice_of_the_same_population_is_stable(self):
        later = modeling.frame_to_features(modeling.temporal_split(modeling.strip_latent(self.frame))['test'])
        summary = drift.report(self.reference, later)
        self.assertEqual(summary['overall'], 'stable', summary['worst'])
        self.assertLess(max(e['psi'] for e in summary['features'].values()), drift.WATCH)

    def test_the_shifted_probe_is_flagged_on_the_fields_that_moved(self):
        probe = modeling.frame_to_features(modeling.strip_latent(generate.build(400, 24, shift=True)))
        summary = drift.report(self.reference, probe)
        self.assertEqual(summary['overall'], 'act')
        for field in ('age', 'bmi', 'hba1c', 'egfr', 'cover'):
            self.assertEqual(summary['features'][field]['verdict'], 'act', field)

    def test_psi_is_zero_for_identical_distributions_and_grows_with_separation(self):
        self.assertAlmostEqual(drift.psi(np.array([.5, .5]), np.array([.5, .5])), 0.0, places=6)
        self.assertGreater(drift.psi(np.array([.5, .5]), np.array([.9, .1])), drift.ACT)

    def test_a_scoring_log_reports_the_gate_and_the_shadow(self):
        extras = pd.DataFrame([{'p_standard': 0.97, 'standard': True, 'confidence': 0.97, 'ood': False,
                                'shadow': {'candidate': 'x', 'agrees': i % 4 != 0}} for i in range(80)])
        gate = drift.gate_drift({'ood': {'train_flag_rate_target': 0.01},
                                 'stp_operating_point': {'model_pass_rate': 0.3}}, extras)
        self.assertEqual(gate['scored'], 80)
        self.assertEqual(gate['ood_flag_rate']['verdict'], 'stable')
        self.assertEqual(gate['model_pass_rate']['verdict'], 'act')
        self.assertEqual(gate['shadow']['disagreements'], 20)

    def test_the_render_is_markdown_with_every_feature(self):
        summary = drift.report(self.reference, self.reference.iloc[:200])
        text = drift.render(summary)
        self.assertIn('# Drift report', text)
        for field in ('age', 'hba1c', 'condition'):
            self.assertIn(f'| {field} |', text)


class ShadowAndLog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        generate.build(1500, 31).to_csv(root / 'applications.csv', index=False)
        data = modeling.prepare(root / 'applications.csv', seed=31)
        catalogue = experiments.catalogue(31)
        for name in ('logistic-l2', 'hist-gbm-shallow'):
            bundle, metrics = experiments.run_one(name, catalogue[name], data, 31)
            joblib.dump({'model': bundle['model'], 'preprocessor': bundle['preprocessor'],
                         'ood_detector': bundle['ood_detector'], 'feature_order': bundle['feature_order'],
                         'metadata': modeling.metadata(bundle, metrics, candidate=name)},
                        root / f'{name}.joblib')
        cls.root = root
        cls.profile = {'age': 40, 'product': 'Life', 'cover': 300000, 'bmi': 24.0, 'smoker': False,
                       'condition': 'none', 'annualIncome': 90000, 'systolic': 120, 'diastolic': 78,
                       'hba1c': 5.3, 'egfr': 100, 'ldl': 2.5}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_without_a_shadow_the_response_is_unchanged_in_shape(self):
        scorer = predict.Scorer(self.root / 'logistic-l2.joblib', self.root / 'registry.json',
                                shadow_path=self.root / 'absent.joblib')
        result = scorer.score(self.profile, [])
        self.assertIsNone(result['diagnostics']['shadow'])
        self.assertEqual(scorer.shadow_stats, {'compared': 0, 'disagreements': 0})
        for key in ('model', 'standard', 'confidence', 'calibrated', 'ood', 'evidence_complete',
                    'verified_document_hashes'):
            self.assertIn(key, result)

    def test_the_shadow_scores_alongside_and_disagreements_are_counted(self):
        scorer = predict.Scorer(self.root / 'logistic-l2.joblib', self.root / 'registry.json',
                                shadow_path=self.root / 'hist-gbm-shallow.joblib')
        result = scorer.score(self.profile, [])
        shadow = result['diagnostics']['shadow']
        self.assertEqual(shadow['candidate'], 'hist-gbm-shallow')
        self.assertIn('agrees', shadow)
        self.assertEqual(scorer.shadow_stats['compared'], 1)
        # The contract fields the demo checks are untouched by the shadow.
        self.assertEqual(result['model'], scorer.metadata['model_version'])

    def test_the_scoring_log_is_opt_in_and_holds_no_name(self):
        log = self.root / 'log' / 'scoring.jsonl'
        scorer = predict.Scorer(self.root / 'logistic-l2.joblib', self.root / 'registry.json',
                                shadow_path=self.root / 'hist-gbm-shallow.joblib', log_path=log)
        scorer.score({**self.profile, 'name': 'Secret Person', 'sex': 'Female'}, [])
        scorer.score(self.profile, [])
        lines = [json.loads(l) for l in log.read_text().splitlines()]
        self.assertEqual(len(lines), 2)
        self.assertNotIn('name', lines[0]['features'])
        self.assertNotIn('Secret', log.read_text())
        self.assertIn('p_standard', lines[0])
        self.assertEqual(lines[0]['shadow']['candidate'], 'hist-gbm-shallow')
        window, extras = drift.load_window(log)
        self.assertEqual(len(window), 2)
        self.assertEqual(int(extras['shadow'].notna().sum()), 2)

    def test_promotion_keeps_the_outgoing_champion_as_the_shadow(self):
        import subprocess
        root = self.root / 'promo'
        (root / 'experiments' / 'runs' / 'hist-gbm-shallow').mkdir(parents=True)
        (root / 'artifacts').mkdir()
        # A leaderboard naming both candidates, and a kept bundle for the one being promoted.
        board = {'meta': {'seed': 31, 'rows': 1500, 'split_rule': modeling.SPLIT_RULE},
                 'candidates': [{'name': n, 'rank': i + 1, 'score': 0.4, 'components': {}, 'viable': True,
                                 'viability': 'provisional', 'not_viable_because': [],
                                 'metrics': {'discrimination': {'auc': 0.8}, 'calibration': {'ece': 0.01}}}
                                for i, n in enumerate(('logistic-l2', 'hist-gbm-shallow'))]}
        (root / 'experiments' / 'leaderboard.json').write_text(json.dumps(board))
        kept = joblib.load(self.root / 'hist-gbm-shallow.joblib')
        joblib.dump(kept, root / 'experiments' / 'runs' / 'hist-gbm-shallow' / 'model.joblib')
        metrics = json.loads(json.dumps({**modeling.evaluate(
            {'model': kept['model'], 'uncalibrated': kept['model'], 'preprocessor': kept['preprocessor'],
             'ood_detector': kept['ood_detector']},
            modeling.prepare(self.root / 'applications.csv', seed=31)), 'cost': {'fit_seconds': 1,
                                                                                    'predict_ms_per_1k_cases': 1,
                                                                                    'serialised_kb': 1}},
            default=str))
        (root / 'experiments' / 'runs' / 'hist-gbm-shallow' / 'metrics.json').write_text(json.dumps(metrics))
        joblib.dump(joblib.load(self.root / 'logistic-l2.joblib'), root / 'artifacts' / 'model.joblib')
        proc = subprocess.run([sys.executable, str(ROOT / 'promote.py'), '--candidate', 'hist-gbm-shallow',
                               '--by', 'tester', '--reason', 'testing the shadow',
                               '--experiments', str(root / 'experiments'), '--artifacts', str(root / 'artifacts'),
                               '--data', str(self.root / 'applications.csv')],
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue((root / 'artifacts' / 'shadow.joblib').exists())
        selection = json.loads((root / 'artifacts' / 'selection.json').read_text())
        self.assertEqual(selection['shadow']['candidate'], 'logistic-l2')
        self.assertEqual(joblib.load(root / 'artifacts' / 'model.joblib')['metadata']['candidate'], 'hist-gbm-shallow')


if __name__ == '__main__':
    unittest.main()
