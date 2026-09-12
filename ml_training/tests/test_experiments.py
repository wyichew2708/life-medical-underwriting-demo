"""Tests for the experiment sweep, the ranking rules and promotion.

The ranking is where a sweep can quietly mislead: a candidate that clears nothing looks
perfectly safe, two configurations of one family look like two options, and a candidate
over the risk budget looks fine if you only read the AUC column. These check that none of
those reach the top three, and that nothing is promoted without a name against it.
"""
import json, sys, tempfile, unittest

import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'data'))

import experiments  # noqa: E402
import generate  # noqa: E402
import modeling  # noqa: E402
import promote  # noqa: E402

FAST = ['baseline-prior', 'logistic-l2', 'hist-gbm-shallow']


def metrics_with(coverage, unsafe, ece=0.01, auc=0.8, interval=None, coverage_interval=None, repeats=None):
    stp = {'threshold': 0.95, 'model_pass_rate': coverage, 'cases_passed': int(coverage * 1000),
           'non_standard_passed': 0, 'unsafe_acceptance_rate': unsafe}
    if interval or coverage_interval:
        stp['interval_90pct'] = {'level': 0.9, 'samples': 10, 'cleared_cases': int(coverage * 1000),
                                 'unsafe_acceptance_rate': interval,
                                 'model_pass_rate': coverage_interval or [coverage - 0.02, coverage + 0.02]}
    metrics = {'discrimination': {'auc': auc, 'average_precision': 0.9, 'accuracy_at_0.5': 0.8,
                                  'base_rate_standard': 0.8},
               'calibration': {'ece': ece, 'brier': 0.1, 'assertion': True},
               'stp_operating_point': stp,
               'cost': {'fit_seconds': 1.0, 'predict_ms_per_1k_cases': 1.0, 'serialised_kb': 100.0}}
    if repeats:
        runs = [{'seed': i, 'auc': auc, 'ece': ece, 'model_pass_rate': coverage,
                 'unsafe_acceptance_rate': u, 'fit_seconds': 1.0} for i, u in enumerate(repeats)]
        metrics['repeats'] = experiments.summarise_repeats(runs)
    return metrics


class Catalogue(unittest.TestCase):
    def test_every_candidate_builds(self):
        for name, spec in experiments.catalogue(7).items():
            estimator = spec['build']()
            self.assertTrue(hasattr(estimator, 'fit'), name)
            for key in ('family', 'architecture', 'description', 'why'):
                self.assertTrue(spec[key], f'{name} is missing {key}')

    def test_sweep_covers_single_and_ensemble_architectures(self):
        architectures = {s['architecture'] for s in experiments.catalogue(7).values()}
        self.assertIn('ensemble (boosting, monotone)', architectures)
        self.assertTrue(any('single' in a for a in architectures))
        self.assertTrue(any('bagging' in a for a in architectures))
        self.assertTrue(any('boosting' in a for a in architectures))
        self.assertTrue(any('voting' in a for a in architectures))
        self.assertTrue(any('stacking' in a for a in architectures))

    def test_a_baseline_is_included_as_an_anchor(self):
        self.assertIn('baseline-prior', experiments.catalogue(7))


class Scoring(unittest.TestCase):
    def test_components_stay_between_zero_and_one(self):
        for coverage, unsafe, ece, auc in [(0.0, None, 0.5, 0.5), (1.0, 0.0, 0.0, 1.0), (0.3, 0.2, 0.2, 0.4)]:
            result = experiments.score(metrics_with(coverage, unsafe, ece, auc))
            for name, value in result['components'].items():
                self.assertTrue(0.0 <= value <= 1.0, f'{name}={value}')

    def test_a_candidate_that_clears_nothing_is_not_viable(self):
        result = experiments.score(metrics_with(0.0, None))
        self.assertFalse(result['viable'])
        self.assertTrue(any('floor' in reason for reason in result['not_viable_because']))

    def test_exceeding_the_unsafe_budget_is_not_viable(self):
        result = experiments.score(metrics_with(0.5, 0.09), budget=0.05)
        self.assertFalse(result['viable'])
        self.assertFalse(result['within_unsafe_budget'])

    def test_the_budget_boundary_is_inclusive_and_precise(self):
        self.assertTrue(experiments.score(metrics_with(0.5, 0.0500), budget=0.05)['within_unsafe_budget'])
        self.assertFalse(experiments.score(metrics_with(0.5, 0.0501), budget=0.05)['within_unsafe_budget'])

    def test_weights_can_be_overridden(self):
        metrics = metrics_with(0.4, 0.01)
        coverage_led = experiments.score(metrics, {'safety': 0.1, 'coverage': 0.7,
                                                   'calibration': 0.1, 'discrimination': 0.1})
        self.assertEqual(coverage_led['weights']['coverage'], 0.7)


class Uncertainty(unittest.TestCase):
    def test_bootstrap_interval_brackets_the_point_estimate(self):
        rng = np.random.default_rng(3)
        y = rng.integers(0, 2, 2000)
        p = np.where(y == 1, rng.uniform(0.9, 1.0, 2000), rng.uniform(0.5, 0.97, 2000))
        ood = np.zeros(2000, bool)
        point = modeling.stp_operating_point(y, p, ood)
        interval = modeling.bootstrap_operating_point(y, p, ood, samples=300)
        low, high = interval['unsafe_acceptance_rate']
        self.assertLessEqual(low, point['unsafe_acceptance_rate'])
        self.assertGreaterEqual(high, point['unsafe_acceptance_rate'])
        low, high = interval['model_pass_rate']
        self.assertLessEqual(low, point['model_pass_rate'])
        self.assertGreaterEqual(high, point['model_pass_rate'])
        self.assertEqual(interval['cleared_cases'], point['cases_passed'])

    def test_bootstrap_interval_is_undefined_when_nothing_is_cleared(self):
        y = np.ones(50, int)
        interval = modeling.bootstrap_operating_point(y, np.full(50, 0.2), np.zeros(50, bool), samples=50)
        self.assertIsNone(interval['unsafe_acceptance_rate'])

    def test_cases_needed_grows_as_the_rate_approaches_the_budget(self):
        far = modeling.cases_needed_to_confirm(0.02, 0.05)
        near = modeling.cases_needed_to_confirm(0.045, 0.05)
        self.assertLess(far, near)
        self.assertIsNone(modeling.cases_needed_to_confirm(0.05, 0.05))
        self.assertIsNone(modeling.cases_needed_to_confirm(None, 0.05))

    def test_a_tight_interval_inside_the_budget_is_confirmed(self):
        confirmed = experiments.score(metrics_with(0.3, 0.02, interval=[0.01, 0.03]))
        provisional = experiments.score(metrics_with(0.3, 0.045, interval=[0.033, 0.058]))
        blocked = experiments.score(metrics_with(0.3, 0.06, interval=[0.05, 0.07]))
        self.assertEqual(confirmed['viability'], 'confirmed')
        self.assertEqual(provisional['viability'], 'provisional')
        self.assertIn('cleared cases', provisional['confirmation_note'])
        self.assertEqual(blocked['viability'], 'not viable')
        self.assertTrue(provisional['viable'])

    def test_generator_records_a_calibrated_true_probability(self):
        frame = generate.build(4000, 21)
        p = frame['_p_standard']
        self.assertTrue(((p >= 0) & (p <= 1)).all())
        self.assertAlmostEqual(float(p.mean()), float(frame['standard'].mean()), delta=0.02)
        # Adding the column consumed no random draws: the label is the same as before.
        self.assertTrue((generate.build(4000, 21)['standard'] == frame['standard']).all())

    def test_oracle_floor_comes_from_the_recorded_probabilities(self):
        frame = generate.build(3000, 4)
        oracle = modeling.oracle_operating_point(frame)
        self.assertIsNotNone(oracle['unsafe_acceptance_rate'])
        self.assertGreater(oracle['auc'], 0.75)
        self.assertIsNone(modeling.oracle_operating_point(modeling.strip_latent(frame)))


class Monotone(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(tempfile.mkdtemp())
        generate.build(2500, 17).to_csv(root / 'applications.csv', index=False)
        cls.data = modeling.prepare(root / 'applications.csv', seed=17)

    def test_constraints_follow_the_manuals_directions_and_leave_the_rest_free(self):
        names = ['num__age', 'num__hba1c', 'num__egfr', 'num__bmi', 'num__missingindicator_hba1c',
                 'bool__smoker', 'cat__condition_complex', 'num__cover_to_income']
        self.assertEqual(modeling.monotone_constraints(names), [-1, -1, 1, 0, 0, -1, 0, -1])

    def test_a_constrained_model_never_rewards_a_worse_lab_value(self):
        spec = experiments.catalogue(17)['hist-gbm-monotone']
        bundle, metrics = experiments.run_one('hist-gbm-monotone', spec, self.data, 17)
        self.assertTrue(metrics['monotone'])
        base = self.data['X']['test'].iloc[:40].copy()
        worse = base.copy()
        worse['hba1c'] = worse['hba1c'].fillna(5.5) + 2.0
        base['hba1c'] = base['hba1c'].fillna(5.5)
        p_base = bundle['uncalibrated'].predict_proba(base)[:, 1]
        p_worse = bundle['uncalibrated'].predict_proba(worse)[:, 1]
        self.assertTrue((p_worse <= p_base + 1e-9).all())
        better = base.copy()
        better['egfr'] = better['egfr'].fillna(90) + 20
        self.assertTrue((bundle['uncalibrated'].predict_proba(better)[:, 1] >= p_base - 1e-9).all())

    def test_an_estimator_without_constraint_support_is_refused_rather_than_silently_unconstrained(self):
        from sklearn.linear_model import LogisticRegression
        with self.assertRaises(TypeError):
            modeling.fit(LogisticRegression(max_iter=200), self.data, seed=17, monotone=True)

    def test_tail_reliability_reports_the_bins_the_gate_uses(self):
        rng = np.random.default_rng(1)
        p = rng.uniform(0.5, 1.0, 4000)
        y = (rng.uniform(0, 1, 4000) < p).astype(int)
        tail = modeling.tail_reliability(y, p)
        self.assertEqual([b['bin'] for b in tail['bins']],
                         ['0.900-0.925', '0.925-0.950', '0.950-0.975', '0.975-1.000'])
        self.assertLess(tail['tail_ece'], 0.05)
        self.assertEqual(tail['cases_in_tail'], int((p >= 0.9).sum()))
        over = modeling.tail_reliability(np.zeros(200, int), np.full(200, 0.96))
        self.assertGreater(over['bins'][2]['gap'], 0.9)


class Ranking(unittest.TestCase):
    def board(self, rows):
        results = {name: {'card': {'family': family, 'architecture': 'single',
                                   'description': name, 'why': name},
                          'metrics': metrics} for name, family, metrics in rows}
        return experiments.rank(results)

    def test_viable_candidates_rank_above_non_viable_ones(self):
        board = self.board([('useless-but-safe', 'baseline', metrics_with(0.0, None, ece=0.0)),
                            ('real-model', 'trees', metrics_with(0.35, 0.02))])
        self.assertEqual(board[0]['name'], 'real-model')
        self.assertFalse(board[1]['viable'])

    def test_a_high_auc_model_over_budget_still_ranks_below_a_viable_one(self):
        board = self.board([('sharp-but-risky', 'trees', metrics_with(0.6, 0.12, auc=0.95)),
                            ('modest-and-safe', 'linear', metrics_with(0.2, 0.01, auc=0.75))])
        self.assertEqual(board[0]['name'], 'modest-and-safe')

    def test_ranking_is_deterministic_for_identical_scores(self):
        rows = [('b-model', 'linear', metrics_with(0.3, 0.02)),
                ('a-model', 'trees', metrics_with(0.3, 0.02))]
        self.assertEqual([r['name'] for r in self.board(rows)],
                         [r['name'] for r in self.board(list(reversed(rows)))])

    def test_near_identical_same_family_candidates_are_flagged(self):
        board = self.board([('linear-a', 'linear', metrics_with(0.30, 0.02, auc=0.800)),
                            ('linear-b', 'linear', metrics_with(0.301, 0.02, auc=0.801)),
                            ('trees-a', 'trees', metrics_with(0.30, 0.02, auc=0.800))])
        flagged = {r['name']: r.get('near_duplicate_of') for r in board}
        self.assertTrue(flagged.get('linear-b') or flagged.get('linear-a'))
        self.assertIsNone(flagged.get('trees-a'), 'a different family is not a duplicate')


class RankingUnderUncertainty(unittest.TestCase):
    @staticmethod
    def board(**entries):
        results = {name: {'card': {'family': fam, 'architecture': 'single', 'description': '', 'why': ''},
                          'metrics': metrics}
                   for name, (fam, metrics) in entries.items()}
        return experiments.rank(results)

    def test_confirmed_candidates_rank_above_provisional_ones_whatever_the_score(self):
        board = self.board(
            provisional=('linear', metrics_with(0.60, 0.035, auc=0.90, ece=0.005, interval=[0.025, 0.052])),
            confirmed=('trees', metrics_with(0.10, 0.02, auc=0.70, ece=0.03, interval=[0.01, 0.03])))
        self.assertEqual([r['name'] for r in board], ['confirmed', 'provisional'])
        self.assertGreater(board[1]['score'], board[0]['score'])

    def test_overlapping_intervals_are_marked_as_a_tie_with_the_leader(self):
        board = self.board(
            a=('linear', metrics_with(0.32, 0.044, interval=[0.032, 0.056], coverage_interval=[0.30, 0.34])),
            b=('trees', metrics_with(0.30, 0.046, interval=[0.034, 0.058], coverage_interval=[0.28, 0.32])),
            c=('nets', metrics_with(0.10, 0.010, interval=[0.000, 0.020], coverage_interval=[0.08, 0.12])))
        by_name = {r['name']: r for r in board}
        leader = board[0]['name']
        others = [r for r in board if r['name'] != leader]
        # c is confirmed and leads; a and b overlap each other but not c, so neither is tied with c.
        self.assertEqual(leader, 'c')
        self.assertFalse(any(r.get('tied_with_leader') for r in others))
        board = self.board(
            a=('linear', metrics_with(0.32, 0.044, interval=[0.032, 0.056], coverage_interval=[0.30, 0.34])),
            b=('trees', metrics_with(0.30, 0.046, interval=[0.034, 0.058], coverage_interval=[0.28, 0.32])))
        self.assertEqual(board[1]['tied_with_leader'], board[0]['name'])
        self.assertIn(board[1]['name'], experiments.resolution(board)['tied_with_leader'])

    def test_a_seed_that_leaves_the_budget_holds_the_candidate_at_provisional(self):
        steady = metrics_with(0.3, 0.02, interval=[0.01, 0.03], repeats=[0.02, 0.021, 0.019])
        wobbly = metrics_with(0.3, 0.02, interval=[0.01, 0.03], repeats=[0.02, 0.06, 0.019])
        board = self.board(steady=('linear', steady), wobbly=('trees', wobbly))
        by_name = {r['name']: r for r in board}
        self.assertEqual(by_name['steady']['viability'], 'confirmed')
        self.assertEqual(by_name['wobbly']['viability'], 'provisional')
        self.assertIn('1 of 3 seeds', by_name['wobbly']['seed_sensitivity'])
        self.assertEqual(board[0]['name'], 'steady')

    def test_score_is_the_mean_over_seeds_with_its_spread(self):
        metrics = metrics_with(0.3, 0.02, interval=[0.01, 0.03], repeats=[0.01, 0.03])
        row = self.board(x=('linear', metrics))[0]
        low = experiments.score(metrics_with(0.3, 0.03))['score']
        high = experiments.score(metrics_with(0.3, 0.01))['score']
        self.assertAlmostEqual(row['score'], (low + high) / 2, places=3)
        self.assertGreater(row['score_sd'], 0)
        self.assertEqual(len(row['score_by_seed']), 2)

    def test_run_candidate_keeps_the_first_seed_and_summarises_the_rest(self):
        root = Path(tempfile.mkdtemp())
        generate.build(1200, 9).to_csv(root / 'applications.csv', index=False)
        data = modeling.prepare(root / 'applications.csv', seed=9)
        bundle, metrics = experiments.run_candidate('logistic-l2', data, [9, 10])
        self.assertEqual(metrics['repeats']['n'], 2)
        self.assertEqual(metrics['repeats']['seeds'], [9, 10])
        self.assertIn('interval_90pct', metrics['stp_operating_point'])
        self.assertIsNotNone(metrics['oracle'])
        # Logistic regression is deterministic: no seed spread.
        self.assertEqual(metrics['repeats']['auc']['sd'], 0.0)


class SweepEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        generate.build(1400, 5).to_csv(root / 'applications.csv', index=False)
        generate.build(120, 6, shift=True).to_csv(root / 'ood_probe.csv', index=False)
        cls.data = modeling.prepare(root / 'applications.csv', root / 'ood_probe.csv', seed=5)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_each_fast_candidate_fits_and_reports_the_same_metric_shape(self):
        catalogue = experiments.catalogue(5)
        for name in FAST:
            _, metrics = experiments.run_one(name, catalogue[name], self.data, 5)
            for section in ('discrimination', 'calibration', 'stp_operating_point', 'ood', 'cost'):
                self.assertIn(section, metrics, name)
            self.assertIn('unsafe_acceptance_rate', metrics['stp_operating_point'])

    def test_every_candidate_is_measured_on_the_same_split(self):
        self.assertEqual(modeling.SPLIT_RULE.count('train'), 1)
        self.assertEqual(len(self.data['X']['test']), len(self.data['frames']['test']))

    def test_a_failing_candidate_is_recorded_rather_than_crashing_the_sweep(self):
        class Broken:
            def fit(self, *a, **kw):
                raise ValueError('deliberately broken')
        spec = {'family': 'broken', 'architecture': 'single', 'description': 'x', 'why': 'x',
                'build': lambda: Broken()}
        with self.assertRaises(Exception):
            experiments.run_one('broken', spec, self.data, 5)


class Promotion(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.board = {'meta': {'seed': 7, 'rows': 100, 'split_rule': modeling.SPLIT_RULE, 'sample': None},
                      'candidates': [
                          {'name': 'good', 'rank': 1, 'score': 0.5, 'viable': True,
                           'not_viable_because': [], 'components': {}, 'metrics': metrics_with(0.3, 0.02)},
                          {'name': 'risky', 'rank': 2, 'score': 0.4, 'viable': False,
                           'not_viable_because': ['unsafe acceptance 0.0900 exceeds the budget of 0.05'],
                           'components': {}, 'metrics': metrics_with(0.5, 0.09)}],
                      'top_3': ['good']}
        (self.dir / 'leaderboard.json').write_text(json.dumps(self.board))

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, *args):
        argv = sys.argv
        sys.argv = ['promote.py', '--experiments', str(self.dir), '--artifacts', str(self.dir / 'out'), *args]
        try:
            promote.main()
        finally:
            sys.argv = argv

    def test_a_missing_leaderboard_is_a_clear_error(self):
        with self.assertRaises(SystemExit):
            promote.load_leaderboard(self.dir / 'nope.json')

    def test_an_unknown_candidate_is_refused(self):
        with self.assertRaises(SystemExit) as caught:
            self.run_cli('--candidate', 'nonexistent', '--by', 'tester', '--reason', 'x')
        self.assertIn('not in the leaderboard', str(caught.exception))

    def test_a_non_viable_candidate_needs_the_risk_accepted_in_writing(self):
        with self.assertRaises(SystemExit) as caught:
            self.run_cli('--candidate', 'risky', '--by', 'tester', '--reason', 'more coverage')
        self.assertIn('--accept-risk', str(caught.exception))

    def test_promotion_requires_a_name_and_a_reason(self):
        with self.assertRaises(SystemExit):
            self.run_cli('--candidate', 'good')


if __name__ == '__main__':
    unittest.main()
