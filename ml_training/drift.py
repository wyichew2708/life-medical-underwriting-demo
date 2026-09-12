"""Has the population being scored moved away from the population the model was fitted on?

    python drift.py --window data/ood_probe.csv           # any CSV or JSON-lines of profiles
    python drift.py --window artifacts/scoring_log.jsonl  # what the adapter actually scored

A model's holdout numbers describe the period it was tested on. Once it is serving, the
only thing that keeps those numbers meaningful is evidence that the cases arriving still
look like that period. This compares the scoring window with the training period, feature
by feature, with the population stability index — the same arithmetic credit-risk model
monitoring has used for decades — plus the things the gate itself depends on: how often
the novelty detector fires, how often the gate would pass, and where a shadow model is
running, how often it disagrees with the champion.

The report never changes anything. It says what moved, by how much, and what the usual
thresholds call that; deciding to retrain, recalibrate or pull the model is a person's.
"""
import argparse, json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import modeling
import schema

ROOT = Path(__file__).parent
ARTIFACTS = ROOT / 'artifacts'
BINS = 10
EPSILON = 1e-4
# Conventional PSI bands. Demonstration thresholds; a programme sets its own with the risk owner.
WATCH, ACT = 0.10, 0.25
MIN_WINDOW = 50


def load_window(path):
    """Profiles to compare, from a CSV of applications or the adapter's JSON-lines scoring log."""
    path = Path(path)
    if path.suffix.lower() in ('.jsonl', '.ndjson'):
        rows, extras = [], []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            rows.append(entry.get('features') or entry.get('profile') or {})
            extras.append({k: entry.get(k) for k in ('p_standard', 'standard', 'confidence', 'ood', 'shadow', 'at')})
        return pd.DataFrame(rows), pd.DataFrame(extras)
    frame = pd.read_csv(path)
    return modeling.strip_latent(frame), None


def _bands(reference, bins=BINS):
    values = reference.dropna().to_numpy(float)
    if values.size == 0:
        return np.array([])
    edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
    return edges


def _distribution(series, edges):
    """Share of rows in each band, with missing values as their own band."""
    present = series.dropna().to_numpy(float)
    counts = np.histogram(present, bins=edges)[0].astype(float) if edges.size > 1 else np.array([float(present.size)])
    missing = float(series.isna().sum())
    shares = np.append(counts, missing) / max(len(series), 1)
    return shares


def psi(reference_shares, window_shares):
    r = np.clip(reference_shares, EPSILON, None)
    w = np.clip(window_shares, EPSILON, None)
    return float(np.sum((w - r) * np.log(w / r)))


def verdict(value):
    if value is None:
        return 'not measured'
    return 'act' if value >= ACT else 'watch' if value >= WATCH else 'stable'


def feature_drift(reference, window):
    """PSI per catalogue feature, numeric by quantile band and categorical by level."""
    out = {}
    for field in schema.NUMERIC_FIELDS + schema.DERIVED_FIELDS:
        if field not in reference.columns or field not in window.columns:
            continue
        edges = _bands(reference[field])
        value = psi(_distribution(reference[field], edges), _distribution(window[field], edges))
        out[field] = {'psi': round(value, 4), 'verdict': verdict(value),
                      'reference_mean': round(float(reference[field].mean()), 3) if reference[field].notna().any() else None,
                      'window_mean': round(float(window[field].mean()), 3) if window[field].notna().any() else None,
                      'reference_missing': round(float(reference[field].isna().mean()), 3),
                      'window_missing': round(float(window[field].isna().mean()), 3)}
    for field in schema.CATEGORICAL_FIELDS + schema.BOOLEAN_FIELDS:
        if field not in reference.columns or field not in window.columns:
            continue
        levels = sorted(set(reference[field].dropna().astype(str)) | set(window[field].dropna().astype(str)))
        r = np.array([float((reference[field].astype(str) == level).mean()) for level in levels] + [float(reference[field].isna().mean())])
        w = np.array([float((window[field].astype(str) == level).mean()) for level in levels] + [float(window[field].isna().mean())])
        value = psi(r, w)
        out[field] = {'psi': round(value, 4), 'verdict': verdict(value),
                      'levels': {level: {'reference': round(float(rv), 3), 'window': round(float(wv), 3)}
                                 for level, rv, wv in zip(levels, r, w)}}
    return out


def gate_drift(scorer_metrics, extras):
    """What the adapter's own outputs say about the window, when a scoring log was supplied."""
    if extras is None or extras.empty:
        return None
    out = {'scored': int(len(extras))}
    if 'ood' in extras and extras['ood'].notna().any():
        rate = float(extras['ood'].astype(bool).mean())
        target = (scorer_metrics or {}).get('ood', {}).get('train_flag_rate_target', modeling.OOD_TRAIN_FLAG_RATE)
        out['ood_flag_rate'] = {'window': round(rate, 4), 'training_target': target,
                                'verdict': 'act' if rate >= 5 * target else 'watch' if rate >= 2 * target else 'stable'}
    if 'confidence' in extras and extras['confidence'].notna().any():
        passes = (extras['standard'].astype(bool) & (extras['confidence'] >= modeling.STP_CONFIDENCE)
                  & ~extras['ood'].astype(bool))
        holdout = (scorer_metrics or {}).get('stp_operating_point', {}).get('model_pass_rate')
        rate = float(passes.mean())
        out['model_pass_rate'] = {'window': round(rate, 4), 'holdout': holdout,
                                  'verdict': ('not measured' if holdout is None else
                                              'act' if abs(rate - holdout) > 0.15 else
                                              'watch' if abs(rate - holdout) > 0.05 else 'stable')}
    if 'p_standard' in extras and extras['p_standard'].notna().any():
        out['mean_p_standard'] = round(float(extras['p_standard'].mean()), 4)
    shadows = [s for s in extras.get('shadow', pd.Series(dtype=object)).dropna() if isinstance(s, dict)]
    if shadows:
        disagree = sum(1 for s in shadows if s.get('agrees') is False)
        out['shadow'] = {'candidate': shadows[-1].get('candidate'), 'compared': len(shadows),
                         'disagreements': disagree, 'disagreement_rate': round(disagree / len(shadows), 4),
                         'note': 'Gate disagreements between the served champion and the shadow. A rising rate '
                                 'means the two models are drifting apart on live cases, which is the signal to '
                                 'review which one is right.'}
    return out


def report(reference, window, extras=None, scorer_metrics=None, window_name='window'):
    features = feature_drift(reference, window)
    worst = sorted(features.items(), key=lambda kv: -kv[1]['psi'])
    acting = [f for f, e in features.items() if e['verdict'] == 'act']
    watching = [f for f, e in features.items() if e['verdict'] == 'watch']
    summary = {'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
               'window': window_name, 'window_rows': int(len(window)), 'reference_rows': int(len(reference)),
               'features': features, 'gate': gate_drift(scorer_metrics, extras),
               'act': acting, 'watch': watching,
               'overall': 'act' if acting else 'watch' if watching else 'stable',
               'thresholds': {'watch': WATCH, 'act': ACT,
                              'note': 'Conventional PSI bands: below 0.10 stable, 0.10-0.25 watch, above 0.25 act. '
                                      'Demonstration thresholds.'},
               'warnings': []}
    if len(window) < MIN_WINDOW:
        summary['warnings'].append(f'Only {len(window)} rows in the window; PSI on fewer than {MIN_WINDOW} rows '
                                   'is mostly sampling noise.')
    summary['worst'] = [f for f, _ in worst[:5]]
    return summary


def render(summary):
    lines = ['# Drift report', '',
             f"Generated {summary['generated_at']}: {summary['window_rows']} rows in `{summary['window']}` against "
             f"{summary['reference_rows']} training rows. Overall: **{summary['overall']}**.", '',
             summary['thresholds']['note'], '',
             '| Feature | PSI | Verdict | Reference mean | Window mean | Missing ref → window |',
             '|---|---|---|---|---|---|']
    for field, entry in sorted(summary['features'].items(), key=lambda kv: -kv[1]['psi']):
        if 'levels' in entry:
            levels = ', '.join(f"{k} {v['reference']:.2f}→{v['window']:.2f}" for k, v in entry['levels'].items())
            lines.append(f"| {field} | {entry['psi']:.3f} | {entry['verdict']} | {levels} | | |")
        else:
            lines.append(f"| {field} | {entry['psi']:.3f} | {entry['verdict']} | {entry['reference_mean']} | "
                         f"{entry['window_mean']} | {entry['reference_missing']:.2f} → {entry['window_missing']:.2f} |")
    gate = summary.get('gate')
    if gate:
        lines += ['', '## The gate on this window', '']
        for key in ('ood_flag_rate', 'model_pass_rate'):
            if key in gate:
                g = gate[key]
                lines.append(f"- {key.replace('_', ' ')}: {g['window']} in the window against "
                             f"{g.get('training_target', g.get('holdout'))} at training — {g['verdict']}.")
        if 'mean_p_standard' in gate:
            lines.append(f"- mean P(standard) in the window: {gate['mean_p_standard']}.")
        if 'shadow' in gate:
            s = gate['shadow']
            lines.append(f"- shadow `{s['candidate']}` disagreed with the champion on {s['disagreements']} of "
                         f"{s['compared']} gate decisions ({s['disagreement_rate']:.1%}). {s['note']}")
    lines += ['', '## Warnings', ''] + ([f'- {w}' for w in summary['warnings']] or ['- none'])
    lines += ['', 'This report changes nothing. Retraining, recalibrating or withdrawing the model is a decision '
                  'a person records, and the sweep and promotion steps are how it is carried out.', '']
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description='Compare a scoring window with the training population.')
    ap.add_argument('--window', type=Path, required=True, help='CSV of applications or JSON-lines scoring log')
    ap.add_argument('--data', type=Path, default=ROOT / 'data' / 'applications.csv')
    ap.add_argument('--metrics', type=Path, default=ARTIFACTS / 'metrics.json')
    ap.add_argument('--out', type=Path, default=ARTIFACTS / 'drift_report.md')
    ap.add_argument('--fail-on', choices=('watch', 'act'), help='exit 2 when the overall verdict reaches this')
    args = ap.parse_args()

    reference = modeling.temporal_split(modeling.load_frame(args.data))['train']
    reference = modeling.frame_to_features(reference)
    window, extras = load_window(args.window)
    if not window.empty and 'cover_to_income' not in window.columns:
        window = modeling.frame_to_features(window)
    metrics = json.loads(args.metrics.read_text()) if args.metrics.exists() else None
    summary = report(reference, window, extras, metrics, window_name=args.window.name)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(summary))
    args.out.with_suffix('.json').write_text(json.dumps(summary, indent=2, default=str) + '\n')

    print(f"{summary['window_rows']} window rows against {summary['reference_rows']} training rows: "
          f"overall {summary['overall'].upper()}")
    for field in summary['worst']:
        entry = summary['features'][field]
        print(f"  {field:<18} PSI {entry['psi']:.3f}  {entry['verdict']}")
    if summary.get('gate'):
        print('  gate: ' + json.dumps({k: v for k, v in summary['gate'].items() if k != 'shadow'}, default=str))
        if summary['gate'].get('shadow'):
            s = summary['gate']['shadow']
            print(f"  shadow {s['candidate']}: {s['disagreements']} of {s['compared']} disagreements")
    for warning in summary['warnings']:
        print('  warning: ' + warning)
    print(f'Written: {args.out}')
    if args.fail_on and {'watch': 1, 'act': 2}[summary['overall']] >= {'watch': 1, 'act': 2}[args.fail_on] \
            if summary['overall'] != 'stable' else False:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
