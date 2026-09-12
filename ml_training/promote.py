"""Promote one candidate from a sweep to the model the adapter serves.

Selection is a human act with a name attached. This script records who chose, when, from
how many candidates, and why, and it refuses a candidate the sweep marked non-viable
unless the risk is accepted explicitly and in writing.

If the sweep kept a fitted bundle, that exact object is promoted. If it did not, the
candidate is refitted from the recorded seed and the result is checked against the
leaderboard — a mismatch is reported rather than quietly served.
"""
import argparse, json
from datetime import datetime, timezone
from pathlib import Path

import joblib

import experiments
import modeling
import schema

ROOT = Path(__file__).parent
ARTIFACTS = ROOT / 'artifacts'
EXPERIMENTS = ARTIFACTS / 'experiments'
TOLERANCE = 0.02


def load_leaderboard(path):
    if not path.exists():
        raise SystemExit(f'No leaderboard at {path}. Run: python experiments.py')
    return json.loads(path.read_text())


def refit(name, meta, data_path, ood_path, seed):
    spec = experiments.catalogue(seed).get(name)
    if not spec:
        raise SystemExit(f'Unknown candidate {name}.')
    data = modeling.prepare(data_path, ood_path, sample=meta.get('sample'), seed=seed)
    bundle, metrics = experiments.run_one(name, spec, data, seed)
    return bundle, metrics


def main():
    ap = argparse.ArgumentParser(description='Promote a swept candidate to the served model.')
    ap.add_argument('--candidate', required=True, help='candidate name from the leaderboard')
    ap.add_argument('--by', required=True, help='who is making this selection')
    ap.add_argument('--reason', required=True, help='why this candidate, in your own words')
    ap.add_argument('--experiments', type=Path, default=EXPERIMENTS)
    ap.add_argument('--artifacts', type=Path, default=ARTIFACTS)
    ap.add_argument('--data', type=Path, default=ROOT / 'data' / 'applications.csv')
    ap.add_argument('--ood-data', type=Path, default=ROOT / 'data' / 'ood_probe.csv')
    ap.add_argument('--accept-risk', help='reason for promoting a candidate the sweep marked non-viable')
    ap.add_argument('--no-shadow', action='store_true',
                    help='do not keep the previous champion scoring in the shadow')
    args = ap.parse_args()

    board = load_leaderboard(args.experiments / 'leaderboard.json')
    row = next((r for r in board['candidates'] if r['name'] == args.candidate), None)
    if not row:
        raise SystemExit(f"{args.candidate} is not in the leaderboard. Available: "
                         + ', '.join(r['name'] for r in board['candidates']))
    if not row['viable'] and not args.accept_risk:
        raise SystemExit(f"{args.candidate} was marked non-viable: "
                         + '; '.join(row['not_viable_because'])
                         + '\nPromote it only with --accept-risk "your reason", which is recorded.')

    seed = board['meta'].get('seed', 7)
    kept = args.experiments / 'runs' / args.candidate / 'model.joblib'
    if kept.exists():
        stored = joblib.load(kept)
        bundle = {'model': stored['model'], 'preprocessor': stored['preprocessor'],
                  'ood_detector': stored['ood_detector'], 'feature_order': stored['feature_order']}
        metrics = json.loads((args.experiments / 'runs' / args.candidate / 'metrics.json').read_text())
        provenance = 'bundle kept by the sweep'
        drift = []
    else:
        print(f'No bundle kept for {args.candidate}; refitting from seed {seed}.')
        bundle, metrics = refit(args.candidate, board['meta'], args.data, args.ood_data, seed)
        provenance = 'refitted at promotion time'
        drift = [f'{key}: leaderboard {was}, refit {now}'
                 for key, was, now in [
                     ('auc', row['metrics']['discrimination']['auc'], metrics['discrimination']['auc']),
                     ('ece', row['metrics']['calibration']['ece'], metrics['calibration']['ece'])]
                 if abs(float(was) - float(now)) > TOLERANCE]
        if drift:
            print('WARNING: refit does not reproduce the leaderboard within tolerance:')
            for line in drift:
                print('  ' + line)

    selection = {'candidate': args.candidate, 'selected_by': args.by, 'reason': args.reason,
                 'selected_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                 'candidates_compared': len(board['candidates']),
                 'rank_in_sweep': row['rank'], 'score': row['score'], 'components': row['components'],
                 'viable': row['viable'], 'viability': row.get('viability'),
                 'not_viable_because': row['not_viable_because'],
                 'confirmation_note': row.get('confirmation_note'),
                 'unsafe_interval_90pct': row.get('unsafe_interval_90pct'),
                 'coverage_interval_90pct': row.get('coverage_interval_90pct'),
                 'tied_with_leader': row.get('tied_with_leader'),
                 'score_sd_over_seeds': row.get('score_sd'),
                 'risk_accepted': args.accept_risk, 'leaderboard': board['meta'],
                 'artifact_provenance': provenance, 'reproduction_drift': drift}

    full = {**metrics, 'model_version': modeling.MODEL_VERSION, 'candidate': args.candidate,
            'trained_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'data': {'file': str(args.data), 'rows': board['meta']['rows'],
                     'split_rule': board['meta']['split_rule'], 'target': schema.TARGET_DEFINITION},
            'limitations': modeling.LIMITATIONS, 'selection': selection}

    args.artifacts.mkdir(parents=True, exist_ok=True)
    shadow = None
    previous = args.artifacts / 'model.joblib'
    if previous.exists() and not args.no_shadow:
        # The outgoing champion keeps scoring alongside the new one. A promotion is judged
        # on the holdout it was chosen on; the shadow is how it gets judged on live cases.
        outgoing = joblib.load(previous)
        meta = outgoing.get('metadata', {})
        if meta.get('candidate') != args.candidate or meta.get('selected_by') != args.by:
            joblib.dump(outgoing, args.artifacts / 'shadow.joblib')
            shadow = {'candidate': meta.get('candidate'), 'model_version': meta.get('model_version'),
                      'trained_at': meta.get('trained_at'), 'selected_by': meta.get('selected_by'),
                      'path': str(args.artifacts / 'shadow.joblib')}
    selection['shadow'] = shadow
    joblib.dump({'model': bundle['model'], 'preprocessor': bundle['preprocessor'],
                 'ood_detector': bundle['ood_detector'], 'feature_order': bundle['feature_order'],
                 'metadata': modeling.metadata(bundle, metrics, candidate=args.candidate,
                                               extra={'seed': seed, 'promoted': True,
                                                      'selected_by': args.by,
                                                      'selected_at': selection['selected_at']})},
                args.artifacts / 'model.joblib')
    (args.artifacts / 'metrics.json').write_text(json.dumps(full, indent=2, default=str) + '\n')
    (args.artifacts / 'selection.json').write_text(json.dumps(selection, indent=2, default=str) + '\n')
    modeling.write_model_card(args.artifacts / 'model_card.md', full)

    stp = metrics['stp_operating_point']
    print(f"\nPromoted {args.candidate} ({provenance}).")
    print(f"  chosen by {args.by}: {args.reason}")
    print(f"  rank {row['rank']} of {len(board['candidates'])}, score {row['score']}, {row.get('viability', 'viable')}")
    if row.get('confirmation_note'):
        print(f"  {row['confirmation_note']}")
    if row.get('tied_with_leader'):
        print(f"  tied with {row['tied_with_leader']} within what the test set can resolve")
    print(f"  clears {stp['model_pass_rate']:.1%} of holdout cases at confidence "
          f">= {stp['threshold']}, unsafe acceptance "
          + (f"{stp['unsafe_acceptance_rate']:.1%}" if stp['unsafe_acceptance_rate'] is not None else '—'))
    if args.accept_risk:
        print(f"  RISK ACCEPTED: {args.accept_risk}")
    if shadow:
        print(f"  shadow: previous champion {shadow['candidate']} kept at {shadow['path']}; serve.py scores it "
              'alongside and counts disagreements on /health')
    print(f"\nWritten: {args.artifacts / 'model.joblib'}, metrics.json, selection.json, model_card.md")
    print('Restart serve.py to serve the promoted model.')


if __name__ == '__main__':
    main()
