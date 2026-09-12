"""Run every candidate model against the same data and rank them for a human to choose.

Fourteen candidates — a base-rate baseline, linear models, tree ensembles with and without
monotone constraints, a small neural network, and multi-model ensembles — all fitted
through the shared machinery in modeling.py so the only difference between them is the
classifier itself.

Ranking is not by AUC. What this system does with a model is clear cases straight through
at a confidence threshold, so the leaderboard scores what that costs: how many cases a
candidate clears, how many of those were wrong, whether its probabilities mean what they
say, and only then how well it separates the classes. A candidate that exceeds the unsafe
acceptance budget is ranked below every candidate that stays inside it, whatever its AUC.

The sweep never selects anything. It prints three options and the command to promote one.
"""
import argparse, json, pickle, time
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import modeling

ROOT = Path(__file__).parent
EXPERIMENTS = ROOT / 'artifacts' / 'experiments'

# How much of the cleared population may be wrong before a candidate is out of contention.
# A demonstration tolerance; a real programme sets this with the risk owner, per product.
UNSAFE_BUDGET = 0.05
# A candidate that clears almost nothing is perfectly safe and completely useless. Without
# this floor the base-rate baseline wins the sweep by never clearing a case.
MIN_COVERAGE = 0.05
DEFAULT_WEIGHTS = {'safety': 0.40, 'coverage': 0.25, 'calibration': 0.20, 'discrimination': 0.15}


def catalogue(seed=7):
    """Candidate classifiers. Each builder returns a fresh, unfitted estimator."""
    def logistic(**kw):
        return LogisticRegression(**{'max_iter': 2000, 'random_state': seed, **kw})

    def hgb(**kw):
        return HistGradientBoostingClassifier(**{'early_stopping': True, 'validation_fraction': 0.15,
                                                 'n_iter_no_change': 25, 'random_state': seed, **kw})

    def forest(**kw):
        return RandomForestClassifier(**{'min_samples_leaf': 5, 'n_jobs': -1, 'random_state': seed, **kw})

    def elasticnet():
        # sklearn 1.8 deprecated `penalty`; a float l1_ratio now selects elastic net on its own.
        import sklearn
        modern = tuple(int(part) for part in sklearn.__version__.split('.')[:2]) >= (1, 8)
        extra = {} if modern else {'penalty': 'elasticnet'}
        return logistic(solver='saga', l1_ratio=0.5, C=0.5, max_iter=1500, tol=1e-3, **extra)

    return {
        'baseline-prior': {
            'family': 'baseline', 'architecture': 'single',
            'description': 'Predicts the training base rate for every case.',
            'why': 'Anchors the leaderboard. Any candidate that cannot beat this is not a model.',
            'build': lambda: DummyClassifier(strategy='prior')},
        'logistic-l2': {
            'family': 'linear', 'architecture': 'single',
            'description': 'L2-regularised logistic regression on the standardised features.',
            'why': 'Cheap, stable and inspectable. Coefficients can be read by a reviewer.',
            'build': lambda: logistic(C=1.0)},
        'logistic-elasticnet': {
            'family': 'linear', 'architecture': 'single',
            'description': 'Elastic-net logistic regression (saga), which can zero out features.',
            'why': 'Shows whether a sparser feature set performs as well as the full catalogue.',
            'build': lambda: elasticnet()},
        'random-forest': {
            'family': 'bagged trees', 'architecture': 'ensemble (bagging)',
            'description': '400 bootstrapped decision trees, majority vote.',
            'why': 'Robust to feature scaling and interactions; tends to be conservative at the extremes.',
            'build': lambda: forest(n_estimators=400)},
        'extra-trees': {
            'family': 'bagged trees', 'architecture': 'ensemble (bagging)',
            'description': '400 extremely randomised trees.',
            'why': 'More variance reduction than a random forest, at the cost of sharper probabilities.',
            'build': lambda: ExtraTreesClassifier(n_estimators=400, min_samples_leaf=5, n_jobs=-1,
                                                  random_state=seed)},
        'hist-gbm-default': {
            'family': 'boosted trees', 'architecture': 'ensemble (boosting)',
            'description': 'Histogram gradient boosting, 31 leaves, learning rate 0.06.',
            'why': 'The usual strong default for tabular underwriting features.',
            'build': lambda: hgb(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                                 min_samples_leaf=40, l2_regularization=1.0)},
        'hist-gbm-monotone': {
            'family': 'boosted trees', 'architecture': 'ensemble (boosting, monotone)',
            'description': 'The default boosting configuration with the manual\'s directions as constraints.',
            'why': 'Higher HbA1c, lower eGFR or smoking can never raise P(standard). Tests whether that '
                   'repairs the over-confident tail that puts unconstrained boosting over budget.',
            'monotone': True,
            'build': lambda: hgb(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                                 min_samples_leaf=40, l2_regularization=1.0)},
        'hist-gbm-shallow-monotone': {
            'family': 'boosted trees', 'architecture': 'ensemble (boosting, monotone)',
            'description': 'Shallow boosting under the same monotone constraints.',
            'why': 'The low-variance boosting configuration, constrained. If constraints help, they should '
                   'help here too.',
            'monotone': True,
            'build': lambda: hgb(max_iter=250, learning_rate=0.1, max_leaf_nodes=15,
                                 min_samples_leaf=60, l2_regularization=2.0)},
        'hist-gbm-deep': {
            'family': 'boosted trees', 'architecture': 'ensemble (boosting)',
            'description': 'Deeper boosting: 63 leaves, learning rate 0.03, up to 600 rounds.',
            'why': 'Tests whether more capacity buys anything on this feature set.',
            'build': lambda: hgb(max_iter=600, learning_rate=0.03, max_leaf_nodes=63,
                                 min_samples_leaf=20, l2_regularization=0.5)},
        'hist-gbm-shallow': {
            'family': 'boosted trees', 'architecture': 'ensemble (boosting)',
            'description': 'Shallow boosting: 15 leaves, learning rate 0.1, up to 250 rounds.',
            'why': 'The fast, low-variance end of the boosting range.',
            'build': lambda: hgb(max_iter=250, learning_rate=0.1, max_leaf_nodes=15,
                                 min_samples_leaf=60, l2_regularization=2.0)},
        'mlp-64-32': {
            'family': 'neural network', 'architecture': 'single (2 hidden layers)',
            'description': 'Multi-layer perceptron, 64 and 32 units, early stopping.',
            'why': 'A non-tree function class. Often poorly calibrated before correction.',
            'build': lambda: MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True,
                                           max_iter=400, random_state=seed)},
        'voting-soft': {
            'family': 'model ensemble', 'architecture': 'ensemble (soft voting)',
            'description': 'Averages the probabilities of logistic regression, boosting and a forest.',
            'why': 'Blending different function classes usually helps calibration more than ranking.',
            'build': lambda: VotingClassifier([
                ('linear', logistic(C=1.0)),
                ('boosting', hgb(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                                 min_samples_leaf=40, l2_regularization=1.0)),
                ('forest', forest(n_estimators=200))], voting='soft')},
        'stacking-lr': {
            'family': 'model ensemble', 'architecture': 'ensemble (stacking)',
            'description': 'Boosting, forest and linear base models with a logistic meta-learner.',
            'why': 'Lets the meta-learner weight base models instead of averaging them equally.',
            'build': lambda: StackingClassifier(
                estimators=[('boosting', hgb(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                             min_samples_leaf=40, l2_regularization=1.0)),
                            ('forest', forest(n_estimators=200)),
                            ('linear', logistic(C=1.0))],
                final_estimator=LogisticRegression(max_iter=1000, random_state=seed), cv=3)},
        'stacking-trees': {
            'family': 'model ensemble', 'architecture': 'ensemble (stacking)',
            'description': 'Two boosting configurations and extra trees under a logistic meta-learner.',
            'why': 'Tests whether stacking similar learners adds anything over one good one.',
            'build': lambda: StackingClassifier(
                estimators=[('deep', hgb(max_iter=300, learning_rate=0.03, max_leaf_nodes=63,
                                         min_samples_leaf=20, l2_regularization=0.5)),
                            ('shallow', hgb(max_iter=250, learning_rate=0.1, max_leaf_nodes=15,
                                            min_samples_leaf=60, l2_regularization=2.0)),
                            ('extra', ExtraTreesClassifier(n_estimators=200, min_samples_leaf=5,
                                                           n_jobs=-1, random_state=seed))],
                final_estimator=LogisticRegression(max_iter=1000, random_state=seed), cv=3)},
    }


def score(metrics, weights=None, budget=UNSAFE_BUDGET, min_coverage=MIN_COVERAGE):
    """Turn holdout metrics into one comparable number, with the parts kept visible.

    Viability is decided before the score is read. A candidate outside the unsafe budget,
    or one that clears too little to be worth deploying, cannot be rescued by a high score
    on the other components.

    Viability has three tiers, because a point estimate inside the budget is not the same
    as evidence of it. `confirmed` means the upper end of the 90% interval is inside the
    budget too; `provisional` means only the point estimate is; `not viable` means neither.
    """
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    stp = metrics['stp_operating_point']
    coverage = stp['model_pass_rate']
    unsafe = stp['unsafe_acceptance_rate']
    safety = 1.0 if unsafe is None else max(0.0, 1.0 - unsafe / budget)
    calibration = max(0.0, 1.0 - metrics['calibration']['ece'] / modeling.MAX_ACCEPTABLE_ECE)
    discrimination = max(0.0, (metrics['discrimination']['auc'] - 0.5) * 2)
    components = {'safety': round(safety, 4), 'coverage': round(coverage, 4),
                  'calibration': round(calibration, 4), 'discrimination': round(discrimination, 4)}
    total = sum(weights[k] * components[k] for k in weights)
    within_budget = unsafe is None or unsafe <= budget
    blocked = []
    if not within_budget:
        blocked.append(f'unsafe acceptance {unsafe:.4f} exceeds the budget of {budget}')
    if coverage < min_coverage:
        blocked.append(f'clears {coverage:.1%} of cases, below the {min_coverage:.0%} floor'
                       + (' (safety is undefined when nothing is cleared)' if unsafe is None else ''))

    interval = stp.get('interval_90pct') or {}
    unsafe_interval = interval.get('unsafe_acceptance_rate')
    cleared = interval.get('cleared_cases', stp.get('cases_passed'))
    confirmation = None
    if blocked:
        viability = 'not viable'
    elif unsafe_interval and unsafe_interval[1] <= budget:
        viability = 'confirmed'
    else:
        viability = 'provisional'
        needed = modeling.cases_needed_to_confirm(unsafe, budget)
        if unsafe_interval:
            confirmation = (f'inside the budget at the point estimate, but the 90% interval reaches '
                            f'{unsafe_interval[1]:.4f}'
                            + (f'; confirming a rate of {unsafe:.4f} needs about {needed} cleared cases '
                               f'and this test set cleared {cleared}' if needed else ''))
        else:
            confirmation = 'no interval was measured, so the budget cannot be confirmed'
    return {'score': round(total, 4), 'components': components, 'weights': weights,
            'within_unsafe_budget': within_budget, 'viable': not blocked, 'viability': viability,
            'not_viable_because': blocked, 'confirmation_note': confirmation,
            'unsafe_acceptance_rate': unsafe, 'unsafe_interval_90pct': unsafe_interval,
            'coverage_interval_90pct': interval.get('model_pass_rate'), 'cleared_cases': cleared,
            'budget': budget, 'min_coverage': min_coverage}


VIABILITY_ORDER = {'confirmed': 0, 'provisional': 1, 'not viable': 2}


def run_one(name, spec, data, seed):
    started = time.perf_counter()
    bundle = modeling.fit(spec['build'](), data, seed=seed, monotone=spec.get('monotone', False))
    fit_seconds = round(time.perf_counter() - started, 2)

    started = time.perf_counter()
    metrics = modeling.evaluate(bundle, data)
    predict_ms = round((time.perf_counter() - started) * 1000 / max(len(data['X']['test']), 1) * 1000, 2)
    size_kb = round(len(pickle.dumps(bundle['model'])) / 1024, 1)
    return bundle, {**metrics, 'cost': {'fit_seconds': fit_seconds,
                                        'predict_ms_per_1k_cases': predict_ms,
                                        'serialised_kb': size_kb}}


def run_candidate(name, data, seeds, catalogue_for=None):
    """Fit one candidate once per seed.

    The first seed's bundle is the one kept and promoted. The other seeds exist to measure
    how much of the result is the algorithm and how much is the seed: a candidate whose
    unsafe rate crosses the budget on some seeds and not others has not shown it is safe.
    """
    catalogue_for = catalogue_for or catalogue
    runs, primary = [], None
    for seed in seeds:
        bundle, metrics = run_one(name, catalogue_for(seed)[name], data, seed)
        if primary is None:
            primary = (bundle, metrics)
        stp = metrics['stp_operating_point']
        runs.append({'seed': seed, 'auc': metrics['discrimination']['auc'],
                     'ece': metrics['calibration']['ece'], 'model_pass_rate': stp['model_pass_rate'],
                     'unsafe_acceptance_rate': stp['unsafe_acceptance_rate'],
                     'fit_seconds': metrics['cost']['fit_seconds']})
    bundle, metrics = primary
    if len(runs) > 1:
        metrics['repeats'] = summarise_repeats(runs)
    return bundle, metrics


def summarise_repeats(runs):
    def spread(key):
        values = [r[key] for r in runs if r[key] is not None]
        if not values:
            return None
        mean = sum(values) / len(values)
        sd = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5 if len(values) > 1 else 0.0
        return {'mean': round(mean, 4), 'sd': round(sd, 4), 'min': round(min(values), 4),
                'max': round(max(values), 4), 'n': len(values)}
    return {'n': len(runs), 'seeds': [r['seed'] for r in runs], 'runs': runs,
            'unsafe_acceptance_rate': spread('unsafe_acceptance_rate'),
            'model_pass_rate': spread('model_pass_rate'), 'auc': spread('auc'), 'ece': spread('ece'),
            'note': 'The first seed is the bundle kept. The rest measure seed sensitivity only; the data '
                    'split is temporal and does not change between seeds.'}


def _metrics_like(run, template):
    """A metrics dict for one repeat run, shaped so score() can read it."""
    return {'stp_operating_point': {'model_pass_rate': run['model_pass_rate'],
                                    'unsafe_acceptance_rate': run['unsafe_acceptance_rate'],
                                    'cases_passed': None},
            'calibration': {'ece': run['ece']}, 'discrimination': {'auc': run['auc']}}


def rank(results, weights=None, budget=UNSAFE_BUDGET, min_coverage=MIN_COVERAGE):
    """Confirmed candidates first, then provisional, then the rest; within a tier by score.

    With seed repeats the score is the mean over seeds, and a candidate that left the
    budget on any seed is held at provisional however good its first seed looked.
    """
    scored = []
    for name, entry in results.items():
        metrics = entry['metrics']
        assessment = score(metrics, weights, budget, min_coverage)
        row = {**entry['card'], 'name': name, **assessment, 'metrics': metrics}
        repeats = metrics.get('repeats')
        if repeats:
            by_seed = [score(_metrics_like(r, metrics), weights, budget, min_coverage) for r in repeats['runs']]
            scores = [b['score'] for b in by_seed]
            mean = sum(scores) / len(scores)
            row['score_first_seed'] = row['score']
            row['score'] = round(mean, 4)
            row['score_by_seed'] = scores
            row['score_sd'] = round((sum((v - mean) ** 2 for v in scores) / (len(scores) - 1)) ** 0.5, 4)
            over_budget = [r['seed'] for r, b in zip(repeats['runs'], by_seed) if not b['within_unsafe_budget']]
            under_floor = [r['seed'] for r, b in zip(repeats['runs'], by_seed) if b['coverage_interval_90pct'] is None
                           and b['components']['coverage'] < min_coverage]
            notes = []
            if over_budget and row['viable']:
                notes.append(f"exceeded the unsafe budget on {len(over_budget)} of {repeats['n']} seeds")
            if under_floor and row['viable']:
                notes.append(f"fell below the coverage floor on {len(under_floor)} of {repeats['n']} seeds")
            if notes:
                row['viability'] = 'provisional'
                row['seed_sensitivity'] = '; '.join(notes)
        scored.append(row)
    scored.sort(key=lambda r: (VIABILITY_ORDER[r['viability']], -r['score'], r['name']))
    for position, row in enumerate(scored, 1):
        row['rank'] = position
    return annotate_ties(annotate_duplicates(scored))


def _overlaps(a, b):
    return bool(a and b and a[0] <= b[1] and b[0] <= a[1])


def annotate_ties(board):
    """Mark every viable candidate the test set cannot separate from the leader.

    Two candidates whose unsafe-rate and coverage intervals both overlap the leader's are
    one result presented twice. The rank between them is an ordering of noise, and the
    board says so rather than letting position stand in for evidence.
    """
    leader = next((row for row in board if row['viable']), None)
    if not leader:
        return board
    for row in board:
        if row is leader or not row['viable']:
            continue
        if (_overlaps(row.get('unsafe_interval_90pct'), leader.get('unsafe_interval_90pct'))
                and _overlaps(row.get('coverage_interval_90pct'), leader.get('coverage_interval_90pct'))):
            row['tied_with_leader'] = leader['name']
    return board


def annotate_duplicates(board):
    """Flag candidates that a reviewer cannot tell apart from a higher-ranked one.

    Two configurations of the same family landing within noise of each other is a real
    result — it means the extra complexity bought nothing — but presenting both as
    separate options to choose between is not a choice.
    """
    for position, row in enumerate(board):
        for better in board[:position]:
            same_family = better['family'] == row['family']
            close = (abs(better['score'] - row['score']) < 0.01
                     and abs(better['metrics']['discrimination']['auc'] - row['metrics']['discrimination']['auc']) < 0.005
                     and abs(better['components']['coverage'] - row['components']['coverage']) < 0.01)
            if same_family and close:
                row['near_duplicate_of'] = better['name']
                break
    return board


def resolution(board, budget=UNSAFE_BUDGET):
    """What this test set can and cannot tell apart, in one record."""
    leader = next((row for row in board if row['viable']), None)
    if not leader:
        return None
    interval = leader.get('unsafe_interval_90pct')
    tied = [row['name'] for row in board if row.get('tied_with_leader')]
    confirmed = [row['name'] for row in board if row['viability'] == 'confirmed']
    needed = modeling.cases_needed_to_confirm(leader['unsafe_acceptance_rate'], budget)
    return {'leader': leader['name'], 'cleared_cases': leader.get('cleared_cases'),
            'unsafe_interval_90pct': interval,
            'half_width': round((interval[1] - interval[0]) / 2, 4) if interval else None,
            'confirmed': confirmed, 'tied_with_leader': tied,
            'cleared_cases_needed_to_confirm_leader': needed}


def resolution_text(res, budget=UNSAFE_BUDGET):
    if not res:
        return ['No viable candidate, so there is nothing to resolve.']
    lines = []
    interval = res['unsafe_interval_90pct']
    if interval:
        lines.append(f"The leader cleared {res['cleared_cases']} test cases, which resolves its unsafe rate to "
                     f"[{interval[0]:.4f}, {interval[1]:.4f}] at 90%, about ±{res['half_width']:.4f}.")
    if res['confirmed']:
        lines.append('Confirmed inside the budget at this test-set size: ' + ', '.join(res['confirmed']) + '.')
    else:
        needed = res['cleared_cases_needed_to_confirm_leader']
        lines.append(f"No candidate is confirmed inside the {budget} budget at this test-set size: every "
                     'viable one is inside it at the point estimate only.'
                     + (f" Confirming the leader's rate would take about {needed} cleared cases." if needed else ''))
    if res['tied_with_leader']:
        lines.append('Statistically tied with the leader on both unsafe rate and coverage: '
                     + ', '.join(res['tied_with_leader']) + '. The order among these is noise.')
    return lines


def oracle_text(oracle, board):
    if not oracle:
        return ['No oracle floor: the data did not record the generator\'s own probabilities.']
    rates = [r['unsafe_acceptance_rate'] for r in board if r['unsafe_acceptance_rate'] is not None]
    best = min(rates) if rates else None
    line = (f"Oracle floor on this data: a classifier that knew the generator's own probabilities, hidden "
            f"labs included, would clear {oracle['model_pass_rate']:.1%} with {oracle['unsafe_acceptance_rate']:.4f} "
            f"unsafe (AUC {oracle['auc']}). No model can beat that here.")
    lines = [line]
    floor_interval = (oracle.get('interval_90pct') or {}).get('unsafe_acceptance_rate')
    if best is not None and oracle['unsafe_acceptance_rate'] is not None:
        gap = best - oracle['unsafe_acceptance_rate']
        if floor_interval and best <= floor_interval[1]:
            lines.append(f'The lowest unsafe rate any candidate reached is {best:.4f}, inside the floor\'s own '
                         f'90% interval [{floor_interval[0]:.4f}, {floor_interval[1]:.4f}]. This test set cannot '
                         'measure any headroom between the best candidate and the ceiling.')
        else:
            lines.append(f'The lowest unsafe rate any candidate reached is {best:.4f}, {gap:+.4f} against the '
                         'floor. That gap is the headroom better modelling could still buy; the floor itself '
                         'is the label.')
    return lines


def _size(cost):
    return (f"{cost['serialised_kb'] / 1024:.0f} MB" if cost['serialised_kb'] >= 1024
            else f"{cost['serialised_kb']:.0f} KB")


def _with_interval(value, interval, pct=True):
    if value is None:
        return '—'
    fmt = (lambda v: f'{v:.1%}') if pct else (lambda v: f'{v:.4f}')
    if not interval:
        return fmt(value)
    return f'{fmt(value)} [{fmt(interval[0])}–{fmt(interval[1])}]'


def trade_off(row, leaders):
    """One sentence naming what this candidate wins and what it gives up."""
    best = {key: max(r['components'][key] for r in leaders) for key in row['components']}
    metrics, notes = row['metrics'], []
    wins = [k for k, v in row['components'].items() if v >= best[k] - 1e-9]
    loses = [k for k, v in row['components'].items() if v < best[k] - 0.02]
    stp = metrics['stp_operating_point']
    if wins:
        notes.append('best of the three on ' + ', '.join(wins))
    if loses:
        notes.append('weaker on ' + ', '.join(loses))
    if row.get('tied_with_leader'):
        notes.append(f"tied with {row['tied_with_leader']} within what this test set can resolve")
    if row.get('near_duplicate_of'):
        notes.append(f"indistinguishable from {row['near_duplicate_of']} on this data — prefer whichever "
                     f"is simpler or cheaper to run")
    if row.get('seed_sensitivity'):
        notes.append(row['seed_sensitivity'])
    cost = metrics['cost']
    notes.append(f"clears {_with_interval(stp['model_pass_rate'], row.get('coverage_interval_90pct'))} of cases with "
                 + (f"{_with_interval(stp['unsafe_acceptance_rate'], row.get('unsafe_interval_90pct'))} of those "
                    'labelled non-standard' if stp['unsafe_acceptance_rate'] is not None else 'no cases cleared')
                 + f", ECE {metrics['calibration']['ece']} (tail {metrics['calibration'].get('tail', {}).get('tail_ece')}), "
                 f"AUC {metrics['discrimination']['auc']}, "
                 f"{cost['fit_seconds']}s to fit, {_size(cost)} to serve")
    return '; '.join(notes) + '.'


def leaderboard_markdown(board, top, meta):
    lines = ['# Model experiment leaderboard', '',
             f"Generated {meta['generated_at']} over {meta['rows']} synthetic rows "
             f"({meta['split_rule']}), {meta.get('repeats', 1)} seed(s) per candidate.", '',
             'Ranked by viability first — confirmed inside the unsafe-acceptance budget, then inside it at '
             'the point estimate only, then outside it or below the coverage floor — and within a tier by '
             'the composite score. '
             f"Weights: {', '.join(f'{k} {v}' for k, v in meta['weights'].items())}; "
             f"unsafe budget {meta['budget']}; coverage floor {meta['min_coverage']}. "
             'Brackets are 90% bootstrap intervals over the test rows.', '',
             '| # | Candidate | Architecture | Score | Coverage [90%] | Unsafe [90%] | ECE | AUC | Fit (s) | Size | Viability |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for row in board:
        metrics = row['metrics']
        stp = metrics['stp_operating_point']
        score_text = f"{row['score']:.4f}" + (f" ±{row['score_sd']:.4f}" if row.get('score_sd') is not None else '')
        viability = row['viability']
        if row['viability'] == 'not viable':
            viability += ' — ' + '; '.join(row['not_viable_because'])
        elif row.get('seed_sensitivity'):
            viability += ' — ' + row['seed_sensitivity']
        if row.get('tied_with_leader'):
            viability += f"; tied with {row['tied_with_leader']}"
        lines.append(f"| {row['rank']} | `{row['name']}` | {row['architecture']} | {score_text} | "
                     f"{_with_interval(stp['model_pass_rate'], row.get('coverage_interval_90pct'))} | "
                     f"{_with_interval(stp['unsafe_acceptance_rate'], row.get('unsafe_interval_90pct'), pct=False)} | "
                     f"{metrics['calibration']['ece']} | {metrics['discrimination']['auc']} | "
                     f"{metrics['cost']['fit_seconds']} | {_size(metrics['cost'])} | {viability} |")
    lines += ['', '## What this test set can resolve', '']
    lines += resolution_text(meta.get('resolution'), meta['budget'])
    lines += ['', '## Oracle floor', '']
    lines += oracle_text(meta.get('oracle'), board)
    if meta.get('repeats', 1) > 1:
        lines += ['', '## Seed repeats', '',
                  'Score is the mean over seeds; the first seed is the bundle kept. A candidate whose unsafe '
                  'rate left the budget on any seed is held at provisional.', '',
                  '| Candidate | Score mean ± sd | Unsafe min–max | Coverage min–max | AUC min–max |',
                  '|---|---|---|---|---|']
        for row in board:
            rep = row['metrics'].get('repeats')
            if not rep:
                continue
            fmt = lambda spread, pct=False: ('—' if not spread else
                                             (f"{spread['min']:.1%}–{spread['max']:.1%}" if pct
                                              else f"{spread['min']:.4f}–{spread['max']:.4f}"))
            lines.append(f"| `{row['name']}` | {row['score']:.4f} ± {row.get('score_sd', 0):.4f} | "
                         f"{fmt(rep['unsafe_acceptance_rate'])} | {fmt(rep['model_pass_rate'], True)} | "
                         f"{fmt(rep['auc'])} |")
    lines += ['', '## Calibration in the tail', '',
              'Global ECE averages over bins the gate never looks at. This is reliability at and above 0.9, '
              'where straight-through decisions are made; the last column is the observed share of '
              'non-standard cases among those the model put at 0.95 or higher.', '',
              '| Candidate | Cases ≥0.9 | Tail ECE | 0.950–0.975 observed | 0.975–1.0 observed | Non-standard among ≥0.95 |',
              '|---|---|---|---|---|---|']
    for row in board:
        tail = row['metrics']['calibration'].get('tail') or {}
        bins = {b['bin']: b for b in tail.get('bins', [])}
        upper = [b for b in tail.get('bins', []) if b['bin'] >= '0.950' and b['n']]
        n_upper = sum(b['n'] for b in upper)
        wrong = (sum(b['n'] * (1 - b['observed_standard']) for b in upper) / n_upper) if n_upper else None
        cell = lambda key: (f"{bins[key]['observed_standard']:.3f} (n={bins[key]['n']})"
                            if bins.get(key) and bins[key]['n'] else '—')
        lines.append(f"| `{row['name']}` | {tail.get('cases_in_tail', '—')} | {tail.get('tail_ece', '—')} | "
                     f"{cell('0.950-0.975')} | {cell('0.975-1.000')} | "
                     + (f'{wrong:.4f}' if wrong is not None else '—') + ' |')
    lines += ['', '## The three to choose between', '']
    for row in top:
        lines += [f"### {row['rank']}. `{row['name']}` — {row['description']}", '',
                  f"- {row['why']}", f"- {trade_off(row, top)}"]
        if row.get('confirmation_note'):
            lines.append(f"- Viability: {row['viability']} — {row['confirmation_note']}.")
        lines += [f"- Promote with: `python promote.py --candidate {row['name']} --by \"your name\" "
                  f"--reason \"why you chose it\"`", '']
    lines += ['Scores compare candidates on one synthetic dataset. They are not evidence that any of these',
              'models is fit for underwriting, and the ranking will move with the data, the weights and the',
              'confidence threshold. Read the intervals and the trade-off, not only the rank.', '']
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description='Sweep candidate models and rank them.')
    ap.add_argument('--data', type=Path, default=ROOT / 'data' / 'applications.csv')
    ap.add_argument('--ood-data', type=Path, default=ROOT / 'data' / 'ood_probe.csv')
    ap.add_argument('--artifacts', type=Path, default=EXPERIMENTS)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--repeats', type=int, default=3,
                    help='fit each candidate this many times on consecutive seeds; 1 disables')
    ap.add_argument('--sample', type=int, help='subsample the dataset for a quick sweep')
    ap.add_argument('--only', help='comma-separated candidate names to run')
    ap.add_argument('--exclude', help='comma-separated candidate names to skip')
    ap.add_argument('--keep', type=int, default=3, help='how many fitted bundles to store for promotion')
    ap.add_argument('--budget', type=float, default=UNSAFE_BUDGET, help='unsafe acceptance tolerance')
    ap.add_argument('--min-coverage', type=float, default=MIN_COVERAGE,
                    help='least share of cases a candidate must clear to be worth promoting')
    ap.add_argument('--weight', action='append', default=[], metavar='NAME=VALUE',
                    help='override a score weight, e.g. --weight coverage=0.4')
    args = ap.parse_args()

    if not args.data.exists():
        raise SystemExit('No dataset. Run: python data/generate.py')
    weights = dict(DEFAULT_WEIGHTS)
    for item in args.weight:
        key, _, value = item.partition('=')
        if key not in weights:
            raise SystemExit(f'Unknown weight {key}. Valid: {", ".join(weights)}')
        weights[key] = float(value)

    candidates = catalogue(args.seed)
    if args.only:
        wanted = [n.strip() for n in args.only.split(',')]
        unknown = [n for n in wanted if n not in candidates]
        if unknown:
            raise SystemExit(f'Unknown candidate(s): {", ".join(unknown)}')
        candidates = {n: candidates[n] for n in wanted}
    for name in (args.exclude or '').split(','):
        candidates.pop(name.strip(), None)
    seeds = [args.seed + i for i in range(max(args.repeats, 1))]

    data = modeling.prepare(args.data, args.ood_data, sample=args.sample, seed=args.seed)
    print(f"{data['rows']} rows; split " + str({k: len(v) for k, v in data['frames'].items()}))
    print(f'Running {len(candidates)} candidate(s) on {len(seeds)} seed(s) each.\n')

    results, bundles = {}, {}
    for name, spec in candidates.items():
        print(f'  {name:<22} ', end='', flush=True)
        try:
            bundle, metrics = run_candidate(name, data, seeds)
        except Exception as error:  # a candidate that will not fit is a result, not a crash
            print(f'failed: {type(error).__name__}')
            results[name] = {'card': {k: spec[k] for k in ('family', 'architecture', 'description', 'why')},
                             'metrics': None, 'error': f'{type(error).__name__}: {error}'}
            continue
        bundles[name] = bundle
        results[name] = {'card': {k: spec[k] for k in ('family', 'architecture', 'description', 'why')},
                         'metrics': metrics}
        stp = metrics['stp_operating_point']
        rep = metrics.get('repeats')
        unsafe = stp['unsafe_acceptance_rate']
        print(f"auc {metrics['discrimination']['auc']:.3f}  ece {metrics['calibration']['ece']:.3f}  "
              f"clears {stp['model_pass_rate']:.1%}  unsafe "
              + (f"{unsafe:.4f}" if unsafe is not None else '—')
              + (f" (seeds {rep['unsafe_acceptance_rate']['min']:.4f}–{rep['unsafe_acceptance_rate']['max']:.4f})"
                 if rep and rep['unsafe_acceptance_rate'] else '')
              + f"  {metrics['cost']['fit_seconds']}s")

    failed = {n: r['error'] for n, r in results.items() if r['metrics'] is None}
    board = rank({n: r for n, r in results.items() if r['metrics']}, weights, args.budget,
                 args.min_coverage)
    if not board:
        raise SystemExit('Every candidate failed to fit. Nothing to rank.')
    viable = [r for r in board if r['viable']]
    top = viable[:3]
    if not top:
        raise SystemExit('No candidate was viable: every one either exceeded the unsafe budget or\n'
                         'cleared too few cases. Widen the budget deliberately, or improve the data.')
    res = resolution(board, args.budget)

    meta = {'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'rows': data['rows'], 'split_rule': modeling.SPLIT_RULE, 'seed': args.seed, 'seeds': seeds,
            'repeats': len(seeds), 'sample': args.sample, 'weights': weights, 'budget': args.budget,
            'min_coverage': args.min_coverage, 'data_file': str(args.data),
            'candidates_run': len(results), 'failed': failed,
            'oracle': data.get('oracle'), 'resolution': res,
            'score_definition': {
                'viability': 'confirmed = 90% interval of the unsafe rate inside the budget and coverage at or '
                             'above the floor; provisional = point estimate inside the budget only; '
                             'not viable = outside the budget or below the floor. Tiers rank in that order.',
                'safety': '1 - unsafe_acceptance_rate / budget, floored at 0',
                'coverage': 'share of holdout cases the model alone would clear at the STP threshold',
                'calibration': '1 - ECE / max acceptable ECE, floored at 0',
                'discrimination': '(AUC - 0.5) * 2, floored at 0',
                'repeats': 'with more than one seed the score is the mean over seeds and a candidate that '
                           'left the budget on any seed is held at provisional',
                'ties': 'a viable candidate whose unsafe and coverage intervals both overlap the leader\'s is '
                        'marked tied; the order among tied candidates is not a finding'}}

    args.artifacts.mkdir(parents=True, exist_ok=True)
    (args.artifacts / 'leaderboard.json').write_text(json.dumps(
        {'meta': meta, 'candidates': board, 'top_3': [r['name'] for r in top]}, indent=2, default=str) + '\n')
    (args.artifacts / 'leaderboard.md').write_text(leaderboard_markdown(board, top, meta))
    kept = []
    for row in viable[:max(args.keep, 0)]:
        run_dir = args.artifacts / 'runs' / row['name']
        run_dir.mkdir(parents=True, exist_ok=True)
        bundle = bundles[row['name']]
        joblib.dump({'model': bundle['model'], 'preprocessor': bundle['preprocessor'],
                     'ood_detector': bundle['ood_detector'], 'feature_order': bundle['feature_order'],
                     'metadata': modeling.metadata(bundle, row['metrics'], candidate=row['name'],
                                                   extra={'seed': args.seed, 'promoted': False})},
                    run_dir / 'model.joblib')
        (run_dir / 'metrics.json').write_text(json.dumps(row['metrics'], indent=2, default=str) + '\n')
        kept.append(row['name'])

    print('\n' + '=' * 78)
    print(f'TOP {len(top)} — choose one. Nothing has been promoted.')
    print('=' * 78)
    for row in top:
        print(f"\n{row['rank']}. {row['name']}  [{row['architecture']}]  score {row['score']:.4f}"
              + (f" ±{row['score_sd']:.4f} over {len(seeds)} seeds" if row.get('score_sd') is not None else '')
              + f"  [{row['viability']}]")
        print(f"   {row['description']}")
        print(f"   {row['why']}")
        print(f"   {trade_off(row, top)}")
        print(f"   components: " + ', '.join(f'{k} {v}' for k, v in row['components'].items()))
    print('\nWhat this test set can resolve:')
    for line in resolution_text(res, args.budget):
        print('   ' + line)
    print('\nOracle floor:')
    for line in oracle_text(data.get('oracle'), board):
        print('   ' + line)
    not_viable = [r for r in board if not r['viable']]
    if not_viable:
        print('\nRanked below the line (not promotable as they stand):')
        for row in not_viable:
            print(f"   {row['name']:<22} " + '; '.join(row['not_viable_because']))
    print(f"\nFull table: {args.artifacts / 'leaderboard.md'}")
    print(f"Bundles kept for promotion: {', '.join(kept) if kept else 'none'}")
    print('\nPromote your choice with:')
    print(f"  python promote.py --candidate {top[0]['name']} --by \"your name\" --reason \"why\"")
    if failed:
        print('\nFailed to fit: ' + ', '.join(f'{n} ({e})' for n, e in failed.items()))


if __name__ == '__main__':
    main()
