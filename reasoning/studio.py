"""Reasoning Studio: grow the case bank one case at a time and tune the reasoning against it.

    python studio.py            # http://127.0.0.1:8097

The journey it serves: choose a product, describe the applicant, upload the documents,
record what the underwriter decided, and repeat. Then evaluate, and let the tuner search
the configurable surface — rule thresholds, retrieval, the prompt's presentation variant,
and with a model configured, guidance lines the model itself proposes from its misses —
until held-out cases stop improving. Every round is cross-validated, every change is
measured on cases the search never saw, and a change is applied only when the held-out
cases confirm it and the count of less-cautious disagreements has not risen. Nothing here
can make the assistant accept a case: that action does not exist in the pipeline.

Loopback only. Documents stay on this machine; the model endpoint is whatever
UW_LLM_BASE_URL points at, which for a local model is this machine too.
"""
import argparse, base64, contextlib, io, json, os, shutil, sys, tempfile, threading, uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import tune  # noqa: E402
from import_cases import build_record  # noqa: E402
from pipeline import Pipeline, PipelineError  # noqa: E402
from pipeline import casebank, evaluation, extraction, product as product_module  # noqa: E402
from pipeline.casebank import HUMAN_OUTCOMES, CaseBankError  # noqa: E402
from pipeline.config import DEFAULT_PATH, PipelineConfig  # noqa: E402
from pipeline.llm import Client, LLMError  # noqa: E402

ROOT = Path(__file__).parent
STATIC = ROOT / 'studio'
SETTINGS = ROOT / 'studio_settings.json'
ARTIFACTS = ROOT / 'artifacts' / 'tuning'
LAST_EVALUATION = ARTIFACTS / 'last_evaluation.json'
HISTORY = ARTIFACTS / 'studio_history.json'
MAX_BODY = 48 * 1024 * 1024          # five documents of 10 MB, base64-encoded, with headroom
MAX_ROUNDS = 6


class Studio:
    def __init__(self, bank=None, artifacts=None, config_path=None):
        self.bank = Path(bank or casebank.BANK_DIR)
        self.artifacts = Path(artifacts or ARTIFACTS)
        self.config_path = Path(config_path or DEFAULT_PATH)
        self.lock = threading.Lock()
        self.job = None
        self.settings = self._load_settings()
        self._apply_settings_to_env()

    # -- settings ------------------------------------------------------------
    def _load_settings(self):
        if SETTINGS.exists():
            try:
                return json.loads(SETTINGS.read_text())
            except ValueError:
                pass
        return {}

    def _apply_settings_to_env(self):
        # The UI can point the studio at a local model; environment variables already set win.
        for key, env in (('llm_base_url', 'UW_LLM_BASE_URL'), ('llm_model', 'UW_LLM_MODEL')):
            if self.settings.get(key) and not os.environ.get(env):
                os.environ[env] = self.settings[key]

    def save_settings(self, body):
        for key in ('llm_base_url', 'llm_model', 'operator'):
            if key in body:
                value = str(body[key] or '').strip()
                self.settings[key] = value
                if key == 'llm_base_url':
                    os.environ['UW_LLM_BASE_URL'] = value
                if key == 'llm_model':
                    os.environ['UW_LLM_MODEL'] = value
        SETTINGS.write_text(json.dumps(self.settings, indent=2) + '\n')
        return self.settings

    def test_llm(self):
        client = Client()
        if not client.configured:
            return {'ok': False, 'error': 'Set the model endpoint and model name first.'}
        try:
            reply = client.complete([{'role': 'system', 'content': 'Return only JSON.'},
                                     {'role': 'user', 'content': 'Return {"ok": true}.'}])
            return {'ok': isinstance(reply, dict), 'model': client.model, 'reply': reply}
        except LLMError as error:
            return {'ok': False, 'error': str(error)}

    # -- state ---------------------------------------------------------------
    def bank_summary(self):
        try:
            cases, problems = casebank.load_bank(self.bank, strict=False)
        except CaseBankError:
            cases, problems = [], []
        last = self.last_evaluation()
        verdicts = {r['case_id']: r for r in (last or {}).get('rows', [])}
        rows = []
        for case in cases:
            row = case.summary()
            row['human'] = {k: case.human.get(k) for k in ('outcome', 'decided_by', 'rationale', 'rating',
                                                          'evidence_requested', 'decided_at')}
            row['needs_extraction'] = case.needs_extraction()
            last_row = verdicts.get(case.case_id)
            row['last'] = ({'predicted': last_row['predicted'], 'verdict': last_row['verdict'],
                            'quality': (last_row.get('quality') or {}).get('score')} if last_row else None)
            rows.append(row)
        outcomes = {}
        for case in cases:
            outcomes[case.human['outcome']] = outcomes.get(case.human['outcome'], 0) + 1
        return {'cases': rows, 'count': len(cases), 'outcomes': outcomes, 'problems': problems,
                'needing_extraction': sum(1 for c in cases if c.needs_extraction())}

    def last_evaluation(self):
        if LAST_EVALUATION.exists():
            try:
                return json.loads(LAST_EVALUATION.read_text())
            except ValueError:
                return None
        return None

    def history(self):
        if HISTORY.exists():
            try:
                return json.loads(HISTORY.read_text())
            except ValueError:
                return []
        return []

    def _record_history(self, entry):
        history = self.history()
        history.append({'at': datetime.now(timezone.utc).isoformat(timespec='seconds'), **entry})
        HISTORY.parent.mkdir(parents=True, exist_ok=True)
        HISTORY.write_text(json.dumps(history[-100:], indent=2, default=str) + '\n')

    def config_state(self):
        try:
            config = PipelineConfig.load(self.config_path)
        except Exception as error:
            return {'error': str(error)}
        return {'hash': config.fingerprint(), 'tuned': self.config_path.exists(), 'path': str(self.config_path),
                'provenance': config.provenance, 'guidance': config.guidance,
                'rule_overrides': config.rule_overrides, 'prompt_variant': config.prompt_variant,
                'precedent_limit': config.precedent_limit, 'retrieval_source_limit': config.retrieval_source_limit}

    def proposal_state(self):
        path = self.artifacts / 'proposal.json'
        if not path.exists():
            return None
        try:
            p = json.loads(path.read_text())
        except ValueError:
            return None
        return {'generated_at': p.get('generated_at'), 'diff': p.get('diff'), 'comparison': p.get('comparison'),
                'warnings': p.get('warnings'), 'applied': p.get('applied'), 'method': p.get('method'),
                'stability': p.get('stability'), 'not_proposed': p.get('not_proposed'),
                'objective': p.get('objective'), 'prompt_search': p.get('prompt_search'),
                'bank': p.get('bank'), 'markdown': (self.artifacts / 'proposal.md').read_text()
                if (self.artifacts / 'proposal.md').exists() else None}

    def state(self):
        core, extra = extraction.catalogue()
        client = Client()
        return {'llm': client.describe(), 'settings': {k: v for k, v in self.settings.items()},
                'bank': self.bank_summary(), 'products': product_module.available(),
                'catalogue': {'core': core, 'extra': extra},
                'outcomes': {k: v['description'] for k, v in HUMAN_OUTCOMES.items()},
                'config': self.config_state(), 'proposal': self.proposal_state(),
                'last_evaluation': self._trim(self.last_evaluation()), 'history': self.history(),
                'job': self.job_state(), 'bank_path': str(self.bank)}

    @staticmethod
    def _trim(summary):
        if not summary:
            return None
        return {k: v for k, v in summary.items() if k != 'rows'} | {
            'rows': [{k: r.get(k) for k in ('case_id', 'human_outcome', 'predicted', 'verdict', 'credit', 'product',
                                            'rules_fired', 'quality', 'output_excerpt')}
                     for r in summary.get('rows', [])]}

    # -- cases ---------------------------------------------------------------
    def add_case(self, body):
        profile = body.get('profile')
        human = body.get('human')
        if not isinstance(profile, dict) or not isinstance(human, dict):
            raise CaseBankError('A case needs a profile and a human decision.')
        documents = body.get('documents') or []
        if not isinstance(documents, list) or len(documents) > casebank.MAX_DOCUMENTS:
            raise CaseBankError(f'Up to {casebank.MAX_DOCUMENTS} documents per case.')
        Pipeline(client=None, precedents=False).validate_profile(profile)     # before anything is named or written
        case_id = (str(body.get('case_id') or '').strip()
                   or 'S-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')[:-3])
        if (self.bank / case_id).exists():
            raise CaseBankError(f'{case_id} already exists in the bank. Choose another id.')
        product_id = str(body.get('product_id') or '').strip() or None
        if product_id and product_id not in {p['product_id'] for p in product_module.available()}:
            raise CaseBankError(f'Unknown product {product_id}.')

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            names = []
            for index, doc in enumerate(documents, 1):
                name = Path(str(doc.get('name') or f'document-{index}')).name
                try:
                    raw = base64.b64decode(str(doc.get('data') or '').split(',')[-1], validate=True)
                except (ValueError, TypeError):
                    raise CaseBankError(f'{name}: the upload was not readable.') from None
                if not raw:
                    raise CaseBankError(f'{name} is empty.')
                (tmp / name).write_bytes(raw)
                names.append(name)
            row = {'case_id': case_id, 'profile': profile, 'human': human, 'documents': names,
                   'instructions': body.get('instructions') or '', 'product_id': product_id}
            case_id, record, paths = build_record(row, tmp)
            validator = Pipeline(client=None, precedents=False)
            record['profile'] = validator.validate_profile(record['profile'])
            target = self.bank / case_id
            (target / 'documents').mkdir(parents=True)
            for path in paths:
                shutil.copy2(path, target / 'documents' / path.name)
            record['source'] = 'studio'
            (target / 'case.json').write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')

        cases, _ = casebank.load_bank(self.bank, strict=False)
        case = next(c for c in cases if c.case_id == case_id)
        result = {'case_id': case_id, 'documents': len(case.documents), 'extraction': None, 'assessment': None}
        client = Client()
        if body.get('extract', True) and case.documents and client.configured:
            try:
                evidence = casebank.extract_evidence(case, client)
                case.store_evidence(evidence, client.model)
                result['extraction'] = {'findings': len(evidence['findings']), 'values': evidence.get('values', []),
                                        'warnings': evidence.get('warnings', []), 'complete': evidence['complete'],
                                        'pages': evidence.get('pages_processed')}
            except CaseBankError as error:
                result['extraction'] = {'error': str(error)}
        elif case.documents and not client.configured:
            result['extraction'] = {'skipped': 'No model configured; the documents are stored and will be read '
                                               'when a model is set.'}
        if body.get('assess', True):
            result['assessment'] = self.assess_case(case, product_id, client)
        return result

    def assess_case(self, case, product_id, client):
        offline = not client.configured
        if offline:
            case.use_rules_only_stub()
        try:
            pipeline = Pipeline(client=client if client.configured else None, product=product_id)
            assessed = pipeline.assess(case.to_pipeline_case(), offline=offline)
        except PipelineError as error:
            return {'error': str(error)}
        verdict = evaluation.classify(assessed['recommendation'], case.human['outcome'])
        return {'recommendation': assessed['recommendation'], 'verdict': verdict,
                'ideal': HUMAN_OUTCOMES[case.human['outcome']]['ideal'],
                'produced_by': assessed['output'].get('produced_by'), 'offline': offline,
                'rules_only': bool((case.evidence or {}).get('rules_only')),
                'explanation': assessed['output'].get('explanation'), 'reasons': assessed['output'].get('reasons'),
                'missing_information': assessed['output'].get('missing_information'),
                'rules_fired': [r['id'] for r in assessed['rules']['fired']],
                'reconciliation': assessed.get('reconciliation'),
                'quality': evaluation.quality(assessed, case), 'config_hash': assessed['provenance']['config_hash'],
                'stages': assessed['stages']}

    def delete_case(self, case_id):
        case_id = str(case_id or '').strip()
        target = self.bank / case_id
        if not case_id or not target.exists() or not (target / 'case.json').exists():
            raise CaseBankError(f'No case {case_id} in the bank.')
        shutil.rmtree(target)
        return {'deleted': case_id}

    # -- jobs ----------------------------------------------------------------
    def job_state(self):
        if not self.job:
            return None
        return {k: v for k, v in self.job.items() if k != 'thread'}

    def start_job(self, kind, options):
        with self.lock:
            if self.job and self.job['status'] == 'running':
                raise RuntimeError('A job is already running.')
            job = {'id': uuid.uuid4().hex[:8], 'kind': kind, 'options': options, 'status': 'running',
                   'log': [], 'result': None, 'error': None,
                   'started': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'finished': None}
            self.job = job
        thread = threading.Thread(target=self._run_job, args=(job, kind, options), daemon=True)
        job['thread'] = thread
        thread.start()
        return job['id']

    class _Log(io.TextIOBase):
        def __init__(self, job):
            self.job, self.buffer = job, ''

        def write(self, text):
            self.buffer += text
            while '\n' in self.buffer:
                line, self.buffer = self.buffer.split('\n', 1)
                if line.strip():
                    self.job['log'].append(line.rstrip())
                    del self.job['log'][:-400]
            return len(text)

    def _args(self, options, **overrides):
        live = bool(options.get('live')) and Client().configured
        values = dict(
            bank=self.bank, artifacts=self.artifacts, config=self.config_path,
            product=options.get('product_id') or None, report=False, apply=False, live=live,
            folds=int(options.get('folds') or tune.FOLDS), holdout=0.3, seed=int(options.get('seed') or 11),
            passes=int(options.get('passes') or 2), min_delta=float(options.get('min_delta') or tune.MIN_DELTA),
            min_support=int(options.get('min_support') or tune.MIN_SUPPORT),
            allow_disable=bool(options.get('allow_disable')), objective=options.get('objective') or 'severity',
            judge=bool(options.get('judge')), prompt_search=bool(options.get('prompt_search', True)),
            by=None, reason=None, accept_risk=None)
        values.update(overrides)
        return argparse.Namespace(**values)

    def _run_job(self, job, kind, options):
        log = self._Log(job)
        try:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                if kind == 'evaluate':
                    job['result'] = self._evaluate(options)
                elif kind == 'tune':
                    job['result'] = self._tune_round(options, apply=False)
                elif kind == 'autotune':
                    job['result'] = self._autotune(options)
                else:
                    raise ValueError(f'Unknown job {kind}.')
            job['status'] = 'done'
        except SystemExit as stop:
            job['status'], job['error'] = 'failed', str(stop)
        except Exception as error:
            job['status'], job['error'] = 'failed', f'{type(error).__name__}: {error}'
        finally:
            job['finished'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
            log.write('\n')

    def _evaluate(self, options):
        args = self._args(options, report=True)
        summary = tune.run_report(args)
        summary['product_id'] = args.product
        summary['live'] = args.live
        summary['at'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
        LAST_EVALUATION.parent.mkdir(parents=True, exist_ok=True)
        LAST_EVALUATION.write_text(json.dumps(summary, indent=2, default=str) + '\n')
        self._record_history({'kind': 'evaluate', 'cases': summary['cases'], 'severity': summary['severity_score'],
                              'agreement': summary['agreement'], 'unsafe': summary['unsafe_disagreements'],
                              'quality': (summary.get('quality') or {}).get('score'),
                              'config_hash': summary.get('config_hash'), 'live': args.live})
        return self._trim(summary)

    @staticmethod
    def _gate(proposal):
        """The same conditions --apply enforces, read from the proposal."""
        holdout = proposal['comparison']['holdout']
        return {'diff': bool(proposal.get('diff')),
                'holdout_delta': holdout['severity_score']['delta'],
                'unsafe_delta': holdout['unsafe_disagreements']['delta'],
                'confirmed': bool(proposal.get('diff')) and holdout['severity_score']['delta'] > 0
                and holdout['unsafe_disagreements']['delta'] <= 0}

    def _tune_round(self, options, apply, round_number=1):
        args = self._args(options)
        proposal = tune.run_tuning(args)
        gate = self._gate(proposal)
        entry = {'round': round_number, 'diff': proposal.get('diff'), 'gate': gate,
                 'train': proposal['comparison']['train']['severity_score'],
                 'holdout': proposal['comparison']['holdout']['severity_score'],
                 'unsafe': proposal['comparison']['holdout']['unsafe_disagreements'],
                 'warnings': proposal.get('warnings'), 'applied': False, 'live': args.live,
                 'prompt_search': proposal.get('prompt_search')}
        if apply and gate['confirmed']:
            by = str(options.get('by') or self.settings.get('operator') or '').strip()
            if not by:
                raise SystemExit('Auto-apply needs an operator name: set it in the studio settings.')
            reason = (f"studio auto-tune round {round_number}: cross-validated holdout {gate['holdout_delta']:+}, "
                      f"less-cautious disagreements {gate['unsafe_delta']:+}")
            tune.apply_proposal(self._args(options, by=by, reason=reason))
            entry['applied'] = True
            entry['applied_by'] = by
        self._record_history({'kind': 'tune', **{k: v for k, v in entry.items() if k != 'warnings'}})
        return entry

    def _autotune(self, options):
        rounds = []
        limit = max(1, min(int(options.get('max_rounds') or MAX_ROUNDS), 12))
        print(f'Auto-tune: up to {limit} round(s); a round is applied only when held-out cases confirm it and '
              'less-cautious disagreements do not rise.')
        for number in range(1, limit + 1):
            print(f'\n===== round {number} =====')
            entry = self._tune_round(options, apply=True, round_number=number)
            rounds.append(entry)
            if not entry['applied']:
                print('No confirmed improvement this round; stopping.')
                break
            print(f"Round {number} applied: holdout {entry['gate']['holdout_delta']:+}.")
        final = self._evaluate(options)
        return {'rounds': rounds, 'applied_rounds': sum(1 for r in rounds if r['applied']),
                'final': final, 'config': self.config_state()}

    # -- apply / revert --------------------------------------------------------
    def apply(self, body):
        by = str(body.get('by') or self.settings.get('operator') or '').strip()
        reason = str(body.get('reason') or '').strip()
        if not by or not reason:
            raise CaseBankError('Applying records a person and a reason.')
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            try:
                tune.apply_proposal(self._args({}, by=by, reason=reason, accept_risk=body.get('accept_risk') or None))
            except SystemExit as stop:
                raise CaseBankError(str(stop)) from None
        self._record_history({'kind': 'apply', 'by': by, 'reason': reason})
        return {'applied': True, 'log': buffer.getvalue(), 'config': self.config_state()}

    def revert(self):
        existed = self.config_path.exists()
        if existed:
            self.config_path.unlink()
        self._record_history({'kind': 'revert'})
        return {'reverted': existed, 'config': self.config_state()}


STUDIO = None


class Handler(BaseHTTPRequestHandler):
    server_version = 'ReasoningStudio/1.0'

    def log_message(self, *args):
        pass

    def _json(self, payload, status=200):
        raw = json.dumps(payload, default=str).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _page(self):
        raw = (STATIC / 'index.html').read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        path = self.path.split('?')[0].rstrip('/')
        if path in ('', '/index.html'):
            return self._page()
        if path == '/api/state':
            return self._json(STUDIO.state())
        if path == '/api/proposal':
            return self._json(STUDIO.proposal_state() or {})
        if path.startswith('/api/jobs/'):
            job = STUDIO.job_state()
            if not job or job['id'] != path.rsplit('/', 1)[-1]:
                return self._json({'error': 'No such job.'}, 404)
            return self._json(job)
        return self._json({'error': 'Unknown endpoint'}, 404)

    def do_POST(self):
        path = self.path.split('?')[0].rstrip('/')
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 <= length <= MAX_BODY:
                raise ValueError('body size')
            body = json.loads(self.rfile.read(length) or b'{}')
            if not isinstance(body, dict):
                raise ValueError('object required')
        except ValueError:
            return self._json({'error': 'Invalid request payload (or the upload exceeds the size limit).'}, 400)
        try:
            if path == '/api/settings':
                result = STUDIO.save_settings(body)
                if body.get('test'):
                    result = {'settings': result, 'test': STUDIO.test_llm()}
                return self._json(result)
            if path == '/api/llm/test':
                return self._json(STUDIO.test_llm())
            if path == '/api/cases':
                return self._json(STUDIO.add_case(body), 201)
            if path == '/api/cases/delete':
                return self._json(STUDIO.delete_case(body.get('case_id')))
            if path == '/api/jobs':
                kind = body.get('kind')
                if kind not in ('evaluate', 'tune', 'autotune'):
                    return self._json({'error': 'kind must be evaluate, tune or autotune.'}, 400)
                try:
                    return self._json({'job_id': STUDIO.start_job(kind, body.get('options') or {})}, 202)
                except RuntimeError as error:
                    return self._json({'error': str(error)}, 409)
            if path == '/api/apply':
                return self._json(STUDIO.apply(body))
            if path == '/api/revert':
                return self._json(STUDIO.revert())
            return self._json({'error': 'Unknown endpoint'}, 404)
        except (CaseBankError, PipelineError) as error:
            return self._json({'error': str(error)}, 400)
        except Exception as error:
            return self._json({'error': f'{type(error).__name__}: {error}'}, 500)


def main():
    global STUDIO
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--port', type=int, default=int(os.environ.get('UW_STUDIO_PORT', '8097')))
    ap.add_argument('--bank', type=Path, default=casebank.BANK_DIR)
    args = ap.parse_args()
    STUDIO = Studio(bank=args.bank)
    client = Client()
    print(f'Reasoning Studio on http://127.0.0.1:{args.port}', flush=True)
    print(f"  case bank: {STUDIO.bank}   model: {client.model or 'not configured (set it in the page)'}", flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
