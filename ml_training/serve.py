"""Loopback HTTP adapter for the demo's `UW_ML_URL`.

Start this, point the demo runner at it, and the straight-through path runs against a
real trained object instead of scripted numbers. The response is exactly the contract in
demo/README.md; the demo re-checks every field and re-derives evidence completeness from
the hashes it uploaded, so a wrong answer here cannot silently create an acceptance.

    export UW_ML_URL=http://127.0.0.1:8099/predict
    python serve.py

Environment:
    UW_ML_MODEL_PATH     trained artifact (default artifacts/model.joblib)
    UW_ML_REGISTRY_PATH  verified-evidence registry (default artifacts/evidence_registry.json)
    UW_ML_SERVE_PORT     default 8099
    UW_ML_SERVE_TOKEN    optional bearer token the caller must present
    UW_ML_TRUST_ALL_HASHES=1  demo-only: certify any submitted hash. Never use this for
                              anything but walking through the STP screen.
    UW_ML_SHADOW_PATH    previous champion to score in the shadow (default artifacts/shadow.joblib)
    UW_ML_SCORING_LOG    opt-in JSON-lines log of scored feature rows for drift.py. Holds
                         applicant attributes (never names or documents); leave unset otherwise.
"""
import json, os, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from predict import DEFAULT_MODEL, DEFAULT_REGISTRY, DEFAULT_SHADOW, Scorer  # noqa: E402

MAX_BODY = 1 * 1024 * 1024
SCORER = None


class Handler(BaseHTTPRequestHandler):
    server_version = 'MockUnderwritingML/1.0'

    def log_message(self, *args):
        pass  # Profiles and hashes are not written to the console.

    def _json(self, payload, status=200):
        raw = json.dumps(payload, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        try:
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _authorised(self):
        token = os.environ.get('UW_ML_SERVE_TOKEN', '')
        return not token or self.headers.get('Authorization', '') == 'Bearer ' + token

    def do_GET(self):
        if self.path.rstrip('/') in ('/health', ''):
            self._json({'status': 'ready', 'model': SCORER.metadata['model_version'],
                        'trained_at': SCORER.metadata['trained_at'],
                        'calibrated': SCORER.metadata['calibrated'],
                        'verified_documents': len(SCORER.registry()),
                        'trust_all_hashes': SCORER.trust_all_hashes,
                        'candidate': SCORER.metadata.get('candidate'),
                        'shadow': ({'candidate': SCORER.shadow['metadata'].get('candidate'),
                                    'model': SCORER.shadow['metadata'].get('model_version'),
                                    **SCORER.shadow_stats,
                                    'disagreement_rate': (round(SCORER.shadow_stats['disagreements']
                                                                / SCORER.shadow_stats['compared'], 4)
                                                          if SCORER.shadow_stats['compared'] else None)}
                                   if SCORER.shadow else None),
                        'scoring_log': str(SCORER.log_path) if SCORER.log_path else None,
                        'synthetic': True})
            return
        self._json({'error': 'Unknown endpoint'}, 404)

    def do_POST(self):
        if not self._authorised():
            self._json({'error': 'Unauthorised'}, 401)
            return
        if self.path.rstrip('/') not in ('/predict', ''):
            self._json({'error': 'Unknown endpoint'}, 404)
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= MAX_BODY:
                raise ValueError('body size')
            body = json.loads(self.rfile.read(length))
            profile = body.get('profile')
            if not isinstance(body, dict) or not isinstance(profile, dict):
                raise ValueError('profile')
            documents = body.get('documents') or []
            if not isinstance(documents, list) or len(documents) > 30:
                raise ValueError('documents')
            self._json(SCORER.score(profile, documents))
        except ValueError:
            self._json({'error': 'Invalid request payload.'}, 400)
        except Exception:
            self._json({'error': 'Scoring failed.'}, 500)


def main():
    global SCORER
    SCORER = Scorer(model_path=os.environ.get('UW_ML_MODEL_PATH') or DEFAULT_MODEL,
                    registry_path=os.environ.get('UW_ML_REGISTRY_PATH') or DEFAULT_REGISTRY,
                    trust_all_hashes=os.environ.get('UW_ML_TRUST_ALL_HASHES') == '1',
                    shadow_path=os.environ.get('UW_ML_SHADOW_PATH') or DEFAULT_SHADOW,
                    log_path=os.environ.get('UW_ML_SCORING_LOG') or None)
    port = int(os.environ.get('UW_ML_SERVE_PORT', '8099'))
    print(f"Mock ML adapter on http://127.0.0.1:{port}/predict "
          f"(model {SCORER.metadata['model_version']}, candidate {SCORER.metadata.get('candidate')}, synthetic)",
          flush=True)
    if SCORER.shadow:
        print(f"  shadow: {SCORER.shadow['metadata'].get('candidate')} scores every case alongside; "
              'disagreements are counted on /health', flush=True)
    if SCORER.log_path:
        print(f'  scoring log: {SCORER.log_path} (applicant attributes; opt-in, for drift.py)', flush=True)
    if SCORER.trust_all_hashes:
        print('WARNING: UW_ML_TRUST_ALL_HASHES=1 — evidence certification is bypassed.', flush=True)
    ThreadingHTTPServer(('127.0.0.1', port), Handler).serve_forever()


if __name__ == '__main__':
    main()
