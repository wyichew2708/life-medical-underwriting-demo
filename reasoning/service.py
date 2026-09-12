"""Loopback HTTP service for the reasoning pipeline.

`POST /context` implements the demo's retrieval and enrichment adapter contract, so the
demo runner can be pointed at this knowledge base:

    export UW_CONTEXT_URL=http://127.0.0.1:8098/context
    python service.py

`POST /assess` runs the whole pipeline for local experimentation, `POST /feedback` records
the underwriter's decision on an assessed case into the case bank, and `GET /knowledge`
lists what is in the base so a reviewer can read the guidance the model will be given.
Nothing here issues a decision, and no credential is stored in this repository.
"""
import json, os, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import Pipeline, PipelineError  # noqa: E402
from pipeline import casebank, knowledge as knowledge_module, product as product_module, retrieval  # noqa: E402
from pipeline.llm import Client  # noqa: E402

MAX_BODY = 4 * 1024 * 1024
PIPELINE = None
BANK_DIR = Path(os.environ.get('UW_CASE_BANK') or casebank.BANK_DIR)


class Handler(BaseHTTPRequestHandler):
    server_version = 'UnderwritingReasoning/1.0'

    def log_message(self, *args):
        pass  # Case content is not written to the console.

    def _json(self, payload, status=200):
        raw = json.dumps(payload, allow_nan=False, default=str).encode()
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
        token = os.environ.get('UW_CONTEXT_SERVE_TOKEN', '')
        return not token or self.headers.get('Authorization', '') == 'Bearer ' + token

    def do_GET(self):
        path = self.path.rstrip('/')
        if path in ('/health', ''):
            self._json({'status': 'ready', 'knowledge': PIPELINE.knowledge.summary(),
                        'llm': PIPELINE.client.describe(),
                        'ml_adapter_configured': bool(os.environ.get('UW_ML_URL')),
                        'configuration': PIPELINE.config.to_dict(),
                        'products': product_module.available(),
                        'note': 'Illustrative knowledge base. Retrieval is deterministic; no web search.'})
            return
        if path == '/products':
            self._json({'products': product_module.available(),
                        'note': 'Draft products are readable but do not route cases. Activation is '
                                'recorded with a person and a reason.'})
            return
        if path == '/knowledge':
            self._json({'cards': [{'id': c['id'], 'type': c['type'], 'title': c['title'],
                                   'collection': c.get('collection'), 'scope': c.get('scope'),
                                   'status': c.get('status'), 'topics': c.get('topics', [])}
                                  for c in PIPELINE.knowledge.cards],
                        'routing_rules': PIPELINE.knowledge.rules,
                        'evidence_grid': PIPELINE.knowledge.evidence_grid})
            return
        self._json({'error': 'Unknown endpoint'}, 404)

    def do_POST(self):
        if not self._authorised():
            self._json({'error': 'Unauthorised'}, 401)
            return
        path = self.path.rstrip('/')
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= MAX_BODY:
                raise ValueError('body size')
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError('object required')
        except ValueError:
            self._json({'error': 'Invalid request payload.'}, 400)
            return

        if path == '/context':
            self._json(self._context(body))
        elif path == '/assess':
            try:
                pipeline = PIPELINE
                if body.get('product_id'):
                    pipeline = Pipeline(client=PIPELINE.client, config=PIPELINE.config,
                                        product=body['product_id'])
                self._json(pipeline.assess(body, offline=not PIPELINE.client.configured))
            except PipelineError as error:
                self._json({'error': str(error)}, 400)
            except Exception:
                self._json({'error': 'Assessment failed.'}, 500)
        elif path == '/feedback':
            try:
                case = casebank.record_feedback(body, directory=BANK_DIR, validate_profile=PIPELINE.validate_profile)
                bank, _ = casebank.load_bank(BANK_DIR, strict=False)
                self._json({'recorded': case.case_id, 'outcome': case.human['outcome'], 'bank_size': len(bank),
                            'note': 'Recorded for evaluation and tuning. Nothing in the pipeline changes until a '
                                    'tuning proposal built on the bank is reviewed and applied.'}, 201)
            except (casebank.CaseBankError, PipelineError) as error:
                self._json({'error': str(error)}, 409 if 'already in the bank' in str(error) else 400)
            except Exception:
                self._json({'error': 'Feedback could not be recorded.'}, 500)
        else:
            self._json({'error': 'Unknown endpoint'}, 404)

    def _context(self, body):
        """The demo's retrieval contract: profile + findings + requested types -> sources."""
        profile = body.get('profile') or {}
        findings = body.get('findings') or []
        requested = body.get('requested_types') or ['internal', 'external', 'knowledge']
        evidence = {'findings': findings if isinstance(findings, list) else [], 'warnings': []}
        found = retrieval.retrieve(PIPELINE.knowledge, profile, evidence,
                                   body.get('instructions', ''),
                                   include_design='external' in requested)
        sources = [s for s in found['sources'] if s['type'] in requested]
        return {'sources': sources,
                'note': ' '.join(found['notes'] + [
                    f'Types requested: {", ".join(requested)}. Returned only types this knowledge base holds '
                    f'({", ".join(sorted({s["type"] for s in sources}))}). No RAG index or web search is '
                    'configured; nothing was retrieved from the internet.']),
                'omitted': found['omitted'],
                'knowledge_revision': PIPELINE.knowledge.rules_revision}


def main():
    global PIPELINE
    PIPELINE = Pipeline(client=Client())
    port = int(os.environ.get('UW_CONTEXT_SERVE_PORT', '8098'))
    summary = PIPELINE.knowledge.summary()
    print(f"Reasoning service on http://127.0.0.1:{port}  "
          f"({summary['cards']} cards, {summary['routing_rules']} routing rules)", flush=True)
    print('  retrieval adapter: POST /context   full pipeline: POST /assess   record a decision: POST /feedback',
          flush=True)
    print('  knowledge: GET /knowledge   products: GET /products', flush=True)
    products = product_module.available()
    if products:
        print('  products: ' + ', '.join(f"{p['product_id']} [{p['status']}]" for p in products), flush=True)
    if PIPELINE.config.provenance:
        print(f"  configuration: tuned proposal applied by "
              f"{PIPELINE.config.provenance.get('applied_by')}", flush=True)
    if not PIPELINE.client.configured:
        print('  UW_LLM_BASE_URL / UW_LLM_MODEL not set: /assess runs the deterministic path only.', flush=True)
    ThreadingHTTPServer(('127.0.0.1', port), Handler).serve_forever()


if __name__ == '__main__':
    main()
