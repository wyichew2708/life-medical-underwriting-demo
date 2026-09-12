"""Train, calibrate and evaluate one model — the default candidate, end to end.

This is the short path: fit `hist-gbm-default`, calibrate it on a later period, measure it
and write the servable artifact. To compare algorithms and architectures instead, run
`experiments.py`, read the top three and promote one with `promote.py`.

Nothing in this file makes the model fit for underwriting. It fits a generated population;
the metrics describe that population.
"""
import argparse, json
from pathlib import Path

import joblib

import experiments
import modeling
import schema

ROOT = Path(__file__).parent
DEFAULT_CANDIDATE = 'hist-gbm-default'


def main():
    ap = argparse.ArgumentParser(description='Train the default mock underwriting classifier.')
    ap.add_argument('--data', type=Path, default=ROOT / 'data' / 'applications.csv')
    ap.add_argument('--ood-data', type=Path, default=ROOT / 'data' / 'ood_probe.csv')
    ap.add_argument('--artifacts', type=Path, default=ROOT / 'artifacts')
    ap.add_argument('--candidate', default=DEFAULT_CANDIDATE,
                    help='any candidate name from experiments.py')
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()

    if not args.data.exists():
        raise SystemExit('No dataset. Run: python data/generate.py')
    spec = experiments.catalogue(args.seed).get(args.candidate)
    if not spec:
        raise SystemExit(f'Unknown candidate {args.candidate}. Names: '
                         + ', '.join(experiments.catalogue(args.seed)))

    data = modeling.prepare(args.data, args.ood_data, seed=args.seed)
    print({k: len(v) for k, v in data['frames'].items()})
    bundle, metrics = experiments.run_one(args.candidate, spec, data, args.seed)

    full = {**metrics, 'model_version': modeling.MODEL_VERSION, 'candidate': args.candidate,
            'trained_at': modeling.metadata(bundle, metrics, args.candidate)['trained_at'],
            'data': {'file': str(args.data), 'rows': data['rows'],
                     'split': {k: len(v) for k, v in data['frames'].items()},
                     'split_rule': modeling.SPLIT_RULE, 'target': schema.TARGET_DEFINITION},
            'limitations': modeling.LIMITATIONS}

    args.artifacts.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': bundle['model'], 'preprocessor': bundle['preprocessor'],
                 'ood_detector': bundle['ood_detector'], 'feature_order': bundle['feature_order'],
                 'metadata': modeling.metadata(bundle, metrics, candidate=args.candidate,
                                               extra={'seed': args.seed, 'promoted': False})},
                args.artifacts / 'model.joblib')
    (args.artifacts / 'metrics.json').write_text(json.dumps(full, indent=2, default=str) + '\n')
    modeling.write_model_card(args.artifacts / 'model_card.md', full)
    print(json.dumps({'candidate': args.candidate,
                      'discrimination': metrics['discrimination'],
                      'ece': metrics['calibration']['ece'],
                      'calibrated': metrics['calibration']['assertion'],
                      'stp_operating_point': metrics['stp_operating_point'],
                      'ood': {k: metrics['ood'][k] for k in ('test_flag_rate', 'shifted_probe_flag_rate')}},
                     indent=2, default=str))
    print('\nCompare algorithms instead: python experiments.py')


if __name__ == '__main__':
    main()
