"""Load the trained artifact and answer in the demo's ML adapter shape.

The adapter contract is deliberately narrow: class, confidence, calibration assertion,
OOD flag and evidence certification. Everything the demo needs to decide whether a case
may go straight through is visible and separately falsifiable here.

Evidence certification is NOT a model output. The model never sees document bytes — the
demo sends only SHA-256 hashes — so `evidence_complete` is answered from a registry of
hashes a human has already verified. An unseen upload therefore cannot be certified, and
the case routes to Vision extraction and review, which is the intended behaviour.
"""
import json
import os
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))  # so joblib can resolve `ood.NoveltyDetector`
import schema  # noqa: E402

ROOT = Path(__file__).parent
DEFAULT_MODEL = ROOT / 'artifacts' / 'model.joblib'
DEFAULT_SHADOW = ROOT / 'artifacts' / 'shadow.joblib'
DEFAULT_REGISTRY = ROOT / 'artifacts' / 'evidence_registry.json'
STP_CONFIDENCE = 0.95


class Scorer:
    def __init__(self, model_path=DEFAULT_MODEL, registry_path=DEFAULT_REGISTRY, trust_all_hashes=False,
                 shadow_path=DEFAULT_SHADOW, log_path=None):
        if not Path(model_path).exists():
            raise FileNotFoundError(f'No trained model at {model_path}. Run data/generate.py then train.py.')
        bundle = joblib.load(model_path)
        self.model = bundle['model']
        self.preprocessor = bundle['preprocessor']
        self.detector = bundle['ood_detector']
        self.metadata = bundle['metadata']
        self.registry_path = Path(registry_path)
        self.trust_all_hashes = bool(trust_all_hashes)
        # Champion/challenger: the previous champion keeps scoring in the shadow so a promotion
        # can be judged on live disagreements, not only on the holdout it was chosen on.
        self.shadow = None
        if shadow_path and Path(shadow_path).exists():
            shadow = joblib.load(shadow_path)
            self.shadow = {'model': shadow['model'], 'preprocessor': shadow['preprocessor'],
                           'detector': shadow['ood_detector'], 'metadata': shadow['metadata']}
        self.shadow_stats = {'compared': 0, 'disagreements': 0}
        # Opt-in scoring log for drift monitoring. It holds applicant attributes, so it is off
        # unless a path is given, and it never holds names or document content.
        self.log_path = Path(log_path) if log_path else None

    def _gate(self, p_standard, ood):
        standard = p_standard >= 0.5
        confidence = min(max(p_standard if standard else 1 - p_standard, 0.001), 0.999)
        return standard, confidence, bool(standard and confidence >= STP_CONFIDENCE and not ood)

    def shadow_score(self, features, champion_passes, champion_standard):
        if self.shadow is None:
            return None
        p = float(self.shadow['model'].predict_proba(features)[0, 1])
        ood = bool(self.shadow['detector'].flags(self.shadow['preprocessor'].transform(features))[0])
        standard, confidence, passes = self._gate(p, ood)
        agrees = passes == champion_passes and standard == champion_standard
        self.shadow_stats['compared'] += 1
        self.shadow_stats['disagreements'] += 0 if agrees else 1
        return {'candidate': self.shadow['metadata'].get('candidate'),
                'model': self.shadow['metadata'].get('model_version'),
                'p_standard': round(p, 4), 'standard': bool(standard), 'confidence': round(confidence, 4),
                'ood': ood, 'gate_passes': passes, 'agrees': agrees}

    def _log(self, features, result, shadow):
        if self.log_path is None:
            return
        from datetime import datetime, timezone
        row = {k: (None if pd.isna(v) else v) for k, v in features.iloc[0].to_dict().items()}
        entry = {'at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'model': result['model'],
                 'features': row, 'p_standard': result['diagnostics']['p_standard'],
                 'standard': result['standard'], 'confidence': result['confidence'], 'ood': result['ood'],
                 'shadow': shadow}
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open('a') as handle:
            handle.write(json.dumps(entry, default=str) + '\n')

    # --- evidence -----------------------------------------------------------
    def registry(self):
        if not self.registry_path.exists():
            return {}
        try:
            data = json.loads(self.registry_path.read_text())
        except ValueError:
            return {}
        return data.get('verified', {}) if isinstance(data, dict) else {}

    def verify_documents(self, documents):
        hashes = [d.get('sha256') for d in documents if isinstance(d, dict) and isinstance(d.get('sha256'), str)]
        if self.trust_all_hashes:
            return hashes, bool(hashes), 'DEMO OVERRIDE: every submitted hash is treated as verified.'
        known = set(self.registry())
        verified = [h for h in hashes if h in known]
        complete = bool(hashes) and len(verified) == len(hashes)
        note = ('All submitted documents match verified evidence records.' if complete else
                'One or more uploads are not in the verified-evidence registry; straight-through is blocked.')
        return verified, complete, note

    # --- model --------------------------------------------------------------
    def score(self, profile, documents=()):
        features = pd.DataFrame([schema.features_from_profile(profile)], columns=schema.FEATURE_ORDER)
        p_standard = float(self.model.predict_proba(features)[0, 1])
        matrix = self.preprocessor.transform(features)
        novelty = self.detector.explain(matrix)[0]
        verified, complete, evidence_note = self.verify_documents(list(documents))
        # Isotonic calibration saturates at exactly 0 and 1 in its end bins. Reporting 1.0
        # would assert certainty the holdout never measured, so the ends are clipped.
        standard, confidence, passes = self._gate(p_standard, bool(novelty['ood']))
        shadow = self.shadow_score(features, passes, standard)
        result = {
            'model': self.metadata['model_version'],
            'standard': bool(standard),
            # The demo gates on confidence in the predicted class.
            'confidence': round(confidence, 4),
            'calibrated': bool(self.metadata['calibrated']),
            'ood': bool(novelty['ood']),
            'evidence_complete': bool(complete),
            'verified_document_hashes': verified,
            'diagnostics': {
                'p_standard': round(p_standard, 4),
                'ood_reasons': novelty['reasons'],
                'isolation_score': novelty['isolation_score'],
                'mahalanobis': novelty['mahalanobis'],
                'calibration_ece': self.metadata['ece'],
                'trained_at': self.metadata['trained_at'],
                'evidence_note': evidence_note,
                'synthetic_model': True,
                'shadow': shadow,
            },
            'note': 'Synthetic demonstration model trained on generated data. Not an underwriting decision.',
        }
        self._log(features, result, shadow)
        return result


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Score profiles with the trained mock model.')
    ap.add_argument('--profiles', type=Path, default=ROOT / 'data' / 'demo_profiles.json')
    ap.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    args = ap.parse_args()
    scorer = Scorer(args.model)
    payload = json.loads(args.profiles.read_text())
    records = payload['profiles'] if isinstance(payload, dict) else payload
    for profile in records:
        result = scorer.score(profile)
        gate = result['standard'] and result['confidence'] >= 0.95 and result['calibrated'] and not result['ood']
        print(f"{profile.get('id', '?'):>4} {profile.get('name', ''):<12} "
              f"standard={result['standard']!s:<5} confidence={result['confidence']:.3f} "
              f"ood={result['ood']!s:<5} model_gate={'pass' if gate else 'blocked'} "
              f"expected={profile.get('expected_route', '')}")
        if result['diagnostics']['ood_reasons']:
            print(f"       ood reasons: {', '.join(result['diagnostics']['ood_reasons'])}")
    print('\nThe model gate is only the first of the demo\'s conditions: evidence certification,'
          '\nbusiness referral rules and custom policy rules still apply in the demo itself.')


if __name__ == '__main__':
    main()
