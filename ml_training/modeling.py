"""Shared modelling machinery: splits, preprocessing, calibration, metrics, artifacts.

Every candidate in an experiment sweep is fitted through exactly this code — the same
preprocessor, the same temporal splits, the same calibration step on the same held-out
period, the same novelty detector and the same metric definitions. Only the classifier
differs. A leaderboard is otherwise just a comparison of incidental differences in
preparation, and the algorithm choice it reports would not be the thing being measured.
"""
import platform
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import schema
from ood import NoveltyDetector

MODEL_VERSION = 'mock-underwriting-ml-2026-09'
STP_CONFIDENCE = 0.95          # The demo's gate. A real programme sets this per product.
MAX_ACCEPTABLE_ECE = 0.05      # Above this the model reports calibrated=false.
OOD_TRAIN_FLAG_RATE = 0.01     # Novelty threshold: flag the 1% least typical training rows.
SPLIT_RULE = 'temporal by application_month: train <=17, calibration 18-20, test >=21'


def frame_to_features(frame):
    rows = [schema.features_from_profile(record) for record in frame.to_dict('records')]
    return pd.DataFrame(rows, columns=schema.FEATURE_ORDER)


def load_frame(path):
    return strip_latent(pd.read_csv(path))


def strip_latent(frame):
    # Columns prefixed with _ are latent label drivers. They never reach a feature row.
    return frame[[c for c in frame.columns if not c.startswith('_')]]


def temporal_split(frame):
    return {'train': frame[frame['application_month'] <= 17],
            'calibration': frame[frame['application_month'].between(18, 20)],
            'test': frame[frame['application_month'] >= 21]}


def prepare(data_path, ood_path=None, sample=None, seed=7):
    raw = pd.read_csv(data_path)
    if sample and sample < len(raw):
        raw = raw.sample(sample, random_state=seed).sort_values('application_month')
    frame = strip_latent(raw)
    parts = temporal_split(frame)
    data = {'frames': parts,
            'X': {k: frame_to_features(v) for k, v in parts.items()},
            'y': {k: v[schema.TARGET].to_numpy() for k, v in parts.items()},
            'rows': int(len(frame))}
    if ood_path is not None:
        data['probe'] = frame_to_features(load_frame(ood_path))
    # The generator's own probabilities, when recorded, give the ceiling every candidate
    # is measured against. They are never features and never leave this dictionary.
    data['oracle'] = oracle_operating_point(temporal_split(raw)['test'])
    return data


def make_preprocessor():
    numeric = schema.NUMERIC_FIELDS + schema.DERIVED_FIELDS
    return ColumnTransformer([
        # Indicators keep "not measured" distinguishable from "measured and normal".
        ('num', Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=True)),
                          ('scale', StandardScaler())]), numeric),
        ('bool', SimpleImputer(strategy='most_frequent'), schema.BOOLEAN_FIELDS),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), schema.CATEGORICAL_FIELDS),
    ])


# Directions the manual already declares, as constraints a boosted model can be made to
# honour: +1 means a higher value may only raise P(standard), -1 may only lower it, 0 is
# unconstrained. BMI is left free because both ends of its range carry risk; indicators
# and one-hot columns are never constrained.
MONOTONE_DIRECTIONS = {'age': -1, 'cover': -1, 'smoker': -1, 'systolic': -1, 'diastolic': -1, 'hba1c': -1,
                       'egfr': +1, 'ldl': -1, 'hospitalisations': -1, 'diagnosisYears': -1,
                       'cover_to_income': -1, 'occupation_hazard': -1}


def monotone_constraints(feature_names):
    """One constraint per preprocessed column, matched on the column's field name."""
    constraints = []
    for name in feature_names:
        field = str(name).split('__')[-1]
        constraints.append(MONOTONE_DIRECTIONS.get(field, 0) if 'missingindicator' not in field else 0)
    return constraints


def calibrator(estimator):
    """Wrap an already-fitted estimator for calibration, across sklearn versions."""
    try:
        from sklearn.frozen import FrozenEstimator
        return CalibratedClassifierCV(FrozenEstimator(estimator), method='isotonic')
    except ImportError:  # sklearn < 1.6
        return CalibratedClassifierCV(estimator, method='isotonic', cv='prefit')


def fit(estimator, data, seed=7, monotone=False):
    """Fit one candidate end to end and return the servable bundle.

    With `monotone`, the classifier is given the manual's directions as constraints; the
    preprocessor is fitted first so the constraint vector lines up with its output columns.
    """
    pre = make_preprocessor()
    if monotone:
        pre.fit(data['X']['train'])
        if 'monotonic_cst' not in estimator.get_params():
            raise TypeError(f'{type(estimator).__name__} does not support monotone constraints.')
        estimator.set_params(monotonic_cst=monotone_constraints(pre.get_feature_names_out()))
    pipeline = Pipeline([('pre', pre), ('clf', estimator)])
    pipeline.fit(data['X']['train'], data['y']['train'])
    calibrated = calibrator(pipeline).fit(data['X']['calibration'], data['y']['calibration'])
    pre = pipeline.named_steps['pre']
    continuous = len(schema.NUMERIC_FIELDS) + len(schema.DERIVED_FIELDS)
    detector = NoveltyDetector(flag_rate=OOD_TRAIN_FLAG_RATE, seed=seed,
                               distance_columns=continuous).fit(pre.transform(data['X']['train']))
    return {'model': calibrated, 'uncalibrated': pipeline, 'preprocessor': pre,
            'ood_detector': detector, 'feature_order': schema.FEATURE_ORDER, 'monotone': bool(monotone)}


def reliability(y_true, p_standard, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    table, ece = [], 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (p_standard >= low) & (p_standard < high if high < 1 else p_standard <= 1)
        if not mask.any():
            continue
        predicted, observed = float(p_standard[mask].mean()), float(y_true[mask].mean())
        ece += mask.mean() * abs(predicted - observed)
        table.append({'bin': f'{low:.1f}-{high:.1f}', 'n': int(mask.sum()),
                      'predicted_standard': round(predicted, 4), 'observed_standard': round(observed, 4)})
    return round(float(ece), 4), table


TAIL_EDGES = [0.9, 0.925, 0.95, 0.975, 1.0]


def tail_reliability(y_true, p_standard, edges=TAIL_EDGES):
    """Calibration where the straight-through decision is made.

    A global ECE averages over bins the gate never looks at. Above 0.9 the bins are
    narrower and the question is sharper: among cases the model puts at 95-97.5%, how many
    were standard? The tail ECE weights only the cases in the tail.
    """
    in_tail = p_standard >= edges[0]
    table, ece = [], 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (p_standard >= low) & (p_standard < high if high < 1 else p_standard <= 1)
        if not mask.any():
            table.append({'bin': f'{low:.3f}-{high:.3f}', 'n': 0, 'predicted_standard': None,
                          'observed_standard': None})
            continue
        predicted, observed = float(p_standard[mask].mean()), float(y_true[mask].mean())
        ece += mask.sum() / max(in_tail.sum(), 1) * abs(predicted - observed)
        table.append({'bin': f'{low:.3f}-{high:.3f}', 'n': int(mask.sum()),
                      'predicted_standard': round(predicted, 4), 'observed_standard': round(observed, 4),
                      'gap': round(predicted - observed, 4)})
    return {'cases_in_tail': int(in_tail.sum()), 'share_in_tail': round(float(in_tail.mean()), 4),
            'tail_ece': round(float(ece), 4) if in_tail.any() else None, 'bins': table,
            'note': 'Reliability at and above 0.9, where straight-through decisions are made. '
                    'A positive gap means the model is over-confident in that bin.'}


def stp_operating_point(y_true, p_standard, ood_flag, threshold=STP_CONFIDENCE):
    """What the demo's gate would do: model-standard AND confidence >= threshold AND in-distribution.

    Business rules, evidence certification and human review sit on top of this in the demo;
    this measures the model's own contribution to an unsafe acceptance.
    """
    passes = (p_standard >= threshold) & ~ood_flag
    accepted = int(passes.sum())
    wrong = int(((y_true == 0) & passes).sum())
    return {'threshold': threshold, 'model_pass_rate': round(accepted / len(y_true), 4),
            'cases_passed': accepted, 'non_standard_passed': wrong,
            'unsafe_acceptance_rate': round(wrong / accepted, 4) if accepted else None,
            'note': 'Rate among cases the model alone would clear. Downstream gates are not modelled here.'}


def bootstrap_operating_point(y_true, p_standard, ood_flag, threshold=STP_CONFIDENCE,
                              samples=1000, seed=13, level=0.90):
    """How far the operating point can be trusted, given the size of this test set.

    Resamples test rows with replacement and recomputes coverage and unsafe acceptance
    each time. A leaderboard that ranks two candidates on a difference smaller than this
    interval is ranking them on noise, so the sweep reads the interval before the rank.
    """
    passes = (p_standard >= threshold) & ~ood_flag
    wrong = (y_true == 0) & passes
    n = len(y_true)
    rng = np.random.default_rng(seed)
    coverage, unsafe = [], []
    for _ in range(samples):
        index = rng.integers(0, n, n)
        accepted = int(passes[index].sum())
        coverage.append(accepted / n)
        if accepted:
            unsafe.append(int(wrong[index].sum()) / accepted)
    low, high = (1 - level) / 2, (1 + level) / 2
    interval = lambda values: ([round(float(np.quantile(values, low)), 4),
                                round(float(np.quantile(values, high)), 4)] if values else None)
    return {'level': level, 'samples': samples, 'cleared_cases': int(passes.sum()),
            'model_pass_rate': interval(coverage), 'unsafe_acceptance_rate': interval(unsafe),
            'note': 'Bootstrap over test rows. The unsafe interval is undefined when nothing is cleared.'}


def cases_needed_to_confirm(rate, budget, confidence=0.95):
    """Cleared cases needed before an unsafe rate this size can be shown to sit under the budget.

    Normal approximation to the one-sided binomial bound. None when the point estimate is
    already at or over the budget, because no sample size confirms that.
    """
    if rate is None or rate >= budget:
        return None
    z = {0.90: 1.2816, 0.95: 1.6449, 0.99: 2.3263}.get(confidence, 1.6449)
    return int(np.ceil(z * z * rate * (1 - rate) / (budget - rate) ** 2))


def oracle_operating_point(raw_frame, threshold=STP_CONFIDENCE):
    """What a classifier that knew the generator's own probabilities would achieve.

    Only possible on synthetic data that recorded `_p_standard`. The oracle sees every true
    value, including labs the row later hides, so no candidate can beat it; the gap between a
    candidate and this line is the headroom that better modelling could still buy.
    """
    if raw_frame is None or '_p_standard' not in raw_frame.columns:
        return None
    p = raw_frame['_p_standard'].to_numpy(float)
    y = raw_frame[schema.TARGET].to_numpy(int)
    point = stp_operating_point(y, p, np.zeros(len(y), bool), threshold)
    return {'threshold': threshold, 'model_pass_rate': point['model_pass_rate'],
            'unsafe_acceptance_rate': point['unsafe_acceptance_rate'],
            'cases_passed': point['cases_passed'],
            'auc': round(float(roc_auc_score(y, p)), 4) if len(np.unique(y)) > 1 else None,
            'interval_90pct': bootstrap_operating_point(y, p, np.zeros(len(y), bool), threshold),
            'note': "The generator's own probabilities, which see the hidden labs. This is the ceiling "
                    'on this data, not a model.'}


def subgroups(frame, y_true, p_standard):
    out = {}
    bands = pd.cut(frame['age'], [17, 35, 50, 65, 101], labels=['18-35', '36-50', '51-65', '66+'])
    for label, index in {**{f'age {b}': (bands == b).to_numpy() for b in bands.cat.categories},
                         'product Life': (frame['product'] == 'Life').to_numpy(),
                         'product Medical': (frame['product'] == 'Medical').to_numpy(),
                         'smoker': frame['smoker'].to_numpy(bool),
                         'non-smoker': ~frame['smoker'].to_numpy(bool),
                         'labs missing': frame['hba1c'].isna().to_numpy(),
                         'labs present': frame['hba1c'].notna().to_numpy()}.items():
        if index.sum() < 40 or len(np.unique(y_true[index])) < 2:
            out[label] = {'n': int(index.sum()), 'auc': None, 'note': 'Too few cases or one class only.'}
            continue
        out[label] = {'n': int(index.sum()),
                      'auc': round(float(roc_auc_score(y_true[index], p_standard[index])), 4),
                      'brier': round(float(brier_score_loss(y_true[index], p_standard[index])), 4),
                      'standard_rate': round(float(y_true[index].mean()), 4)}
    return out


def evaluate(bundle, data, with_subgroups=True):
    """Holdout metrics for one fitted bundle. Identical definitions for every candidate."""
    X_test, y_test = data['X']['test'], data['y']['test']
    p_test = bundle['model'].predict_proba(X_test)[:, 1]
    p_raw = bundle['uncalibrated'].predict_proba(X_test)[:, 1]
    matrix = bundle['preprocessor'].transform(X_test)
    test_ood = bundle['ood_detector'].flags(matrix)
    ece, table = reliability(y_test, p_test)
    ece_raw, _ = reliability(y_test, p_raw)

    probe_flag_rate = None
    if 'probe' in data:
        probe_flag_rate = round(float(bundle['ood_detector'].flags(
            bundle['preprocessor'].transform(data['probe'])).mean()), 4)

    metrics = {
        'discrimination': {
            'auc': round(float(roc_auc_score(y_test, p_test)), 4),
            'average_precision': round(float(average_precision_score(y_test, p_test)), 4),
            'accuracy_at_0.5': round(float(((p_test >= .5).astype(int) == y_test).mean()), 4),
            'base_rate_standard': round(float(y_test.mean()), 4)},
        'calibration': {
            'brier': round(float(brier_score_loss(y_test, p_test)), 4),
            'brier_uncalibrated': round(float(brier_score_loss(y_test, p_raw)), 4),
            'ece': ece, 'ece_uncalibrated': ece_raw, 'max_acceptable_ece': MAX_ACCEPTABLE_ECE,
            'method': 'isotonic on a held-out later period',
            'reliability': table,
            'tail': tail_reliability(y_test, p_test),
            'tail_uncalibrated': tail_reliability(y_test, p_raw),
            'assertion': ece <= MAX_ACCEPTABLE_ECE,
            'assertion_note': 'Holdout ECE on synthetic data only. Not evidence of calibration on a real portfolio.'},
        'stp_operating_point': {**stp_operating_point(y_test, p_test, test_ood),
                                'interval_90pct': bootstrap_operating_point(y_test, p_test, test_ood)},
        'monotone': bundle.get('monotone', False),
        'oracle': data.get('oracle'),
        'ood': {'detector': 'IsolationForest and Mahalanobis distance on preprocessed features',
                'train_flag_rate_target': OOD_TRAIN_FLAG_RATE,
                'isolation_threshold': round(bundle['ood_detector'].isolation_threshold_, 5),
                'distance_threshold': round(bundle['ood_detector'].distance_threshold_, 3),
                'test_flag_rate': round(float(test_ood.mean()), 4),
                'shifted_probe_flag_rate': probe_flag_rate,
                'note': 'A flag means unfamiliar input, not high risk. It routes the case away from STP.'},
    }
    if with_subgroups:
        metrics['subgroups'] = subgroups(data['frames']['test'], y_test, p_test)
    return metrics


LIMITATIONS = [
    'Synthetic generator, not portfolio experience. No mortality or claims validation.',
    'The label is a generated acceptance decision, not an adjudicated underwriter outcome.',
    'Labs are missing not at random; subgroup AUC for missing-lab cases reflects that design.',
    'No fairness evaluation. Sex and name are excluded from features but proxies are not audited.',
]


def metadata(bundle, metrics, candidate='hist-gbm-default', extra=None):
    return {'model_version': MODEL_VERSION,
            'candidate': candidate,
            'trained_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'calibrated': metrics['calibration']['assertion'],
            'ece': metrics['calibration']['ece'],
            'stp_confidence': STP_CONFIDENCE,
            'sklearn': __import__('sklearn').__version__,
            'python': platform.python_version(),
            'synthetic': True,
            **(extra or {})}


def write_model_card(path, m):
    stp = m['stp_operating_point']
    lines = [
        f"# Model card — {m['model_version']}", '',
        '**Synthetic demonstration model. Not approved for underwriting use.**', '',
        f"Candidate `{m.get('candidate', 'unknown')}`, trained {m['trained_at']} on generated data "
        f"({m['data']['rows']} rows, {m['data']['split_rule']}).", '',
        '## Intended use', '',
        'Supplies the `UW_ML_URL` adapter of the underwriting demo so the straight-through path can be',
        'exercised end to end with a real model object instead of scripted numbers. Any other use,',
        'including pricing, acceptance or medical inference, is out of scope.', '',
        '## Label', '', m['data']['target'], '',
        '## Measured on the latest-period holdout', '',
        '| Metric | Value |', '|---|---|',
        f"| AUC | {m['discrimination']['auc']} |",
        f"| Average precision | {m['discrimination']['average_precision']} |",
        f"| Accuracy @0.5 | {m['discrimination']['accuracy_at_0.5']} |",
        f"| Brier (calibrated) | {m['calibration']['brier']} |",
        f"| Brier (raw) | {m['calibration']['brier_uncalibrated']} |",
        f"| ECE (calibrated) | {m['calibration']['ece']} |",
        f"| ECE (raw) | {m['calibration']['ece_uncalibrated']} |",
        f"| Standard base rate | {m['discrimination']['base_rate_standard']} |", '',
        '## Calibration in the tail', '',
        'Where the gate decides. A positive gap is over-confidence.', '',
        '| Bin | n | Predicted standard | Observed standard | Gap |', '|---|---|---|---|---|',
    ] + [f"| {b['bin']} | {b['n']} | {b['predicted_standard']} | {b['observed_standard']} | {b.get('gap')} |"
         for b in m['calibration'].get('tail', {}).get('bins', [])] + [
        '', f"Tail ECE {m['calibration'].get('tail', {}).get('tail_ece')} over "
        f"{m['calibration'].get('tail', {}).get('cases_in_tail')} holdout cases at or above 0.9.", '',
        '## Straight-through operating point', '',
        f"At confidence >= {stp['threshold']} and no OOD flag the model alone clears "
        f"{stp['model_pass_rate']:.1%} of holdout cases, of which {stp['non_standard_passed']} were labelled "
        f"non-standard (unsafe acceptance {stp['unsafe_acceptance_rate']}).", '',
        '## Out-of-distribution behaviour', '',
        f"Flag rate {m['ood']['test_flag_rate']} on holdout, {m['ood']['shifted_probe_flag_rate']} on the "
        f"shifted probe cohort. {m['ood']['note']}", '',
        '## Limitations', '',
    ] + [f'- {x}' for x in m.get('limitations', LIMITATIONS)]
    if m.get('selection'):
        s = m['selection']
        lines += ['', '## Selection', '',
                  f"Chosen from {s.get('candidates_compared')} candidates by {s.get('selected_by')} on "
                  f"{s.get('selected_at')}. Reason: {s.get('reason')}", '']
    if m.get('subgroups'):
        lines += ['', '## Subgroups', '', '| Cohort | n | AUC | Brier |', '|---|---|---|---|']
        for name, s in m['subgroups'].items():
            lines.append(f"| {name} | {s['n']} | {s.get('auc')} | {s.get('brier')} |")
        lines += ['', 'Subgroup figures are synthetic-population diagnostics, not a fairness assessment.', '']
    path.write_text('\n'.join(lines))
