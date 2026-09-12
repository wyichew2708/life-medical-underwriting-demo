"""A bank of past cases: the documents, and what a human underwriter actually decided.

This is the only ground truth the reasoning side has. Everything the tuner claims rests on
it, so the loader is strict about provenance: each document is hashed, each extraction
records which model produced it and over which document hashes, and a cached extraction is
reused only while both still match. Change a page in a report and the cache misses.

Limits mirror demo/server.py exactly — at most 5 documents, 10 MB each, 20 MB together, 12
rendered pages per case, PDF/PNG/JPEG/WebP verified by content signature rather than by
file extension. A case bank is not a safer place to relax them.

The human outcome is a recorded decision, not a correct answer. Underwriters disagree with
each other, decisions were made under the guidance of their day, and a bank assembled from
whatever was to hand carries whatever selection produced it. Treat agreement with this bank
as agreement with this bank.
"""
import base64
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from . import extraction
from .extraction import (EXTRACTION_SYSTEM, MAX_PAGES, RENDER_LONG_EDGE, _pymupdf,  # noqa: F401  re-exported
                         check_findings, render_pages)

ROOT = Path(__file__).parent.parent
BANK_DIR = ROOT / 'cases' / 'bank'
MAX_DOCUMENTS = 5
MAX_BYTES_EACH = 10 * 1024 * 1024
MAX_BYTES_TOTAL = 20 * 1024 * 1024

# What an underwriter recorded, and what the pipeline would ideally have recommended.
# `propose_terms` is the strongest thing this pipeline may output; it never accepts.
HUMAN_OUTCOMES = {
    'standard': {'ideal': 'straight_through', 'also_acceptable': ('propose_terms',),
                 'description': 'Accepted at standard rates.'},
    'terms': {'ideal': 'propose_terms', 'also_acceptable': ('refer',),
              'description': 'Accepted on substandard terms: a rating, a flat extra or an exclusion.'},
    'more_evidence': {'ideal': 'request_evidence', 'also_acceptable': (),
                      'description': 'Further evidence requested before any decision.'},
    'refer': {'ideal': 'refer', 'also_acceptable': ('request_evidence',),
              'description': 'Escalated to a senior underwriter or reinsurer.'},
    'postpone': {'ideal': 'refer', 'also_acceptable': ('request_evidence',),
                 'description': 'Postponed until a defined event or interval.'},
    'decline': {'ideal': 'refer', 'also_acceptable': (),
                'description': 'Declined. This pipeline cannot decline; a referral is the correct output.'},
}

SIGNATURES = [
    ('application/pdf', lambda raw: raw.startswith(b'%PDF-')),
    ('image/png', lambda raw: raw.startswith(b'\x89PNG\r\n\x1a\n')),
    ('image/jpeg', lambda raw: raw.startswith(b'\xff\xd8\xff')),
    ('image/webp', lambda raw: raw[:4] == b'RIFF' and raw[8:12] == b'WEBP'),
]


class CaseBankError(extraction.ExtractionError):
    pass




def sniff(raw):
    for mime, test in SIGNATURES:
        if test(raw):
            return mime
    return None


class Case:
    def __init__(self, case_id, data, directory):
        self.case_id = case_id
        self.data = data
        self.directory = Path(directory)
        self.profile = data.get('profile') or {}
        self.instructions = data.get('instructions', '')
        self.ml = data.get('ml')
        self.human = data.get('human') or {}
        self.documents = self._documents()
        self.evidence, self.evidence_source = self._evidence()

    # -- documents
    def _documents(self):
        entries, total = [], 0
        listed = self.data.get('documents') or []
        folder = self.directory / 'documents'
        if not listed and folder.exists():
            listed = [{'file': p.name} for p in sorted(folder.iterdir()) if p.is_file()]
        if len(listed) > MAX_DOCUMENTS:
            raise CaseBankError(f'{self.case_id}: {len(listed)} documents; the limit is {MAX_DOCUMENTS}.')
        for index, entry in enumerate(listed, 1):
            path = folder / entry['file']
            if not path.exists():
                raise CaseBankError(f'{self.case_id}: missing document {entry["file"]}.')
            raw = path.read_bytes()
            total += len(raw)
            if not 0 < len(raw) <= MAX_BYTES_EACH or total > MAX_BYTES_TOTAL:
                raise CaseBankError(f'{self.case_id}: document size limit exceeded.')
            mime = sniff(raw)
            if not mime:
                raise CaseBankError(f'{self.case_id}: {entry["file"]} is not a PDF, PNG, JPEG or WebP.')
            entries.append({'id': f'DOC-{index}', 'name': path.name, 'type': mime, 'path': path,
                            'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)})
        return entries

    def document_manifest(self):
        return [{k: d[k] for k in ('id', 'name', 'type', 'sha256')} for d in self.documents]

    def document_fingerprint(self):
        return hashlib.sha256(''.join(d['sha256'] for d in self.documents).encode()).hexdigest()

    # -- evidence
    def cache_path(self):
        return self.directory / 'extracted.json'

    def _evidence(self):
        cache = self.cache_path()
        if cache.exists():
            cached = json.loads(cache.read_text())
            if cached.get('document_fingerprint') == self.document_fingerprint():
                return cached['evidence'], f"cache ({cached.get('extracted_by', 'unknown')})"
            return None, 'cache stale: the documents changed since extraction'
        if self.data.get('evidence'):
            return self.data['evidence'], self.data.get('evidence_source', 'supplied with the case')
        return None, 'not extracted'

    def needs_extraction(self):
        return self.evidence is None

    def store_evidence(self, evidence, extracted_by):
        self.evidence = evidence
        self.evidence_source = f'extraction ({extracted_by})'
        self.cache_path().write_text(json.dumps({
            'evidence': evidence, 'extracted_by': extracted_by,
            'document_fingerprint': self.document_fingerprint(),
            'documents': [{k: d[k] for k in ('id', 'name', 'sha256')} for d in self.documents],
            'extracted_at': datetime.now(timezone.utc).isoformat(timespec='seconds')}, indent=2) + '\n')

    # -- for the pipeline
    def to_pipeline_case(self):
        case = {'case_id': self.case_id, 'profile': self.profile, 'instructions': self.instructions,
                'documents': self.document_manifest()}
        if self.evidence is not None:
            case['evidence'] = self.evidence
        if self.ml is not None:
            case['ml'] = self.ml
        return case

    def validate(self):
        outcome = self.human.get('outcome')
        if outcome not in HUMAN_OUTCOMES:
            raise CaseBankError(f'{self.case_id}: human outcome {outcome!r} is not one of '
                                f'{", ".join(HUMAN_OUTCOMES)}.')
        if not self.human.get('decided_by'):
            raise CaseBankError(f'{self.case_id}: the human outcome must record who decided it.')
        if not isinstance(self.profile, dict) or not self.profile.get('product'):
            raise CaseBankError(f'{self.case_id}: the case needs a profile with a product.')
        self.human = normalise_human(self.human, self.case_id)
        return self

    def summary(self):
        return {'case_id': self.case_id, 'product': self.profile.get('product'),
                'outcome': self.human.get('outcome'), 'documents': len(self.documents),
                'evidence': self.evidence_source, 'cohort': self.human.get('cohort')}


def _string_list(value):
    if value in (None, ''):
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.replace('\n', ';').split(';') if part.strip()]
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return [v.strip() for v in value if v.strip()]
    raise ValueError


def normalise_human(human, case_id):
    """The recorded decision, with its optional detail typed and bounded.

    A single outcome label is enough to score routing. The detail — what evidence the
    underwriter asked for, the rating, the rationale, what happened once evidence arrived —
    is what lets the reasoning be scored and lets a prior case be read as a precedent.
    """
    out = dict(human)
    try:
        out['evidence_requested'] = _string_list(human.get('evidence_requested'))
        out['rules_cited'] = _string_list(human.get('rules_cited'))
    except ValueError:
        raise CaseBankError(f'{case_id}: evidence_requested and rules_cited must be lists of strings.') from None
    for key in ('rationale', 'rating', 'notes'):
        value = human.get(key)
        if value is not None and (not isinstance(value, str) or len(value) > 2000):
            raise CaseBankError(f'{case_id}: {key} must be a string of at most 2000 characters.')
    later = human.get('later_outcome')
    if later not in (None, ''):
        if isinstance(later, str):
            later = {'outcome': later}
        if not isinstance(later, dict) or later.get('outcome') not in HUMAN_OUTCOMES:
            raise CaseBankError(f'{case_id}: later_outcome must name one of {", ".join(HUMAN_OUTCOMES)}.')
        out['later_outcome'] = {k: later[k] for k in ('outcome', 'decided_at', 'notes') if k in later}
    else:
        out.pop('later_outcome', None)
    return out


def record_feedback(payload, directory=BANK_DIR, validate_profile=None):
    """Append an underwriter's decision on an assessed case to the bank.

    This is how the bank grows from use rather than from imports: the decision, who made
    it and why, the profile as assessed, and a snapshot of what the pipeline said at the
    time — its recommendation, the configuration hash and knowledge revision it ran under,
    the model version, the evidence it read. Later evaluation can then ask not only
    "did the pipeline agree" but "which configuration disagreed, and how".
    """
    if not isinstance(payload, dict):
        raise CaseBankError('Feedback must be an object.')
    human = payload.get('human')
    if not isinstance(human, dict):
        raise CaseBankError('Feedback needs a human decision: {"outcome": ..., "decided_by": ...}.')
    profile = payload.get('profile')
    if not isinstance(profile, dict):
        raise CaseBankError('Feedback needs the profile that was assessed.')
    if validate_profile is not None:
        profile = validate_profile(profile)
    # The decision is checked before anything is written or even named, so a bad outcome
    # is refused as a bad outcome rather than colliding with an earlier record.
    if human.get('outcome') not in HUMAN_OUTCOMES:
        raise CaseBankError(f'Human outcome {human.get("outcome")!r} is not one of {", ".join(HUMAN_OUTCOMES)}.')
    if not human.get('decided_by'):
        raise CaseBankError('The human outcome must record who decided it.')
    human = normalise_human(human, payload.get('case_id') or 'feedback')
    assessment = payload.get('assessment') if isinstance(payload.get('assessment'), dict) else {}
    case_id = str(payload.get('case_id') or '').strip()
    if not case_id:
        digest = hashlib.sha256(json.dumps(profile, sort_keys=True, default=str).encode()).hexdigest()[:6]
        case_id = 'FB-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f') + '-' + digest
    if not case_id.replace('-', '').replace('_', '').isalnum() or len(case_id) > 60:
        raise CaseBankError(f'Invalid case_id {case_id!r}.')
    target = Path(directory) / case_id
    if target.exists():
        raise CaseBankError(f'{case_id} is already in the bank; feedback is appended, never overwritten.')

    evidence = assessment.get('evidence') if isinstance(assessment.get('evidence'), dict) else None
    snapshot = None
    if assessment:
        snapshot = {'recommendation': assessment.get('recommendation'),
                    'produced_by': (assessment.get('output') or {}).get('produced_by'),
                    'provenance': assessment.get('provenance'),
                    'rules_fired': [r.get('id') for r in (assessment.get('rules') or {}).get('fired', [])],
                    'ml_screen': {k: (assessment.get('ml_screen') or {}).get(k)
                                  for k in ('model', 'standard', 'confidence', 'ood', 'evidence_complete')},
                    'guardrail_overrides': assessment.get('guardrail_overrides'),
                    'missing_information': (assessment.get('output') or {}).get('missing_information'),
                    'elapsed_ms': assessment.get('elapsed_ms')}
    documents = payload.get('documents') if isinstance(payload.get('documents'), list) else []
    record = {'case_id': case_id, 'profile': profile,
              'human': {**human, 'recorded_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                        'source': 'feedback'},
              'instructions': payload.get('instructions') or '',
              'documents': [],
              'document_manifest': [{k: d.get(k) for k in ('id', 'name', 'sha256')} for d in documents
                                    if isinstance(d, dict)],
              'assessment': snapshot}
    if evidence is not None:
        record['evidence'] = {k: evidence.get(k) for k in ('complete', 'findings', 'warnings', 'values')
                              if k in evidence}
        record['evidence_source'] = 'assessment snapshot at feedback time'
    case = Case(case_id, record, target).validate()     # same rules as an import
    target.mkdir(parents=True)
    (target / 'case.json').write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str) + '\n')
    return case


def load_bank(directory=BANK_DIR, strict=True):
    directory = Path(directory)
    if not directory.exists():
        raise CaseBankError(f'No case bank at {directory}. Import cases with import_cases.py.')
    cases, problems = [], []
    for path in sorted(directory.iterdir()):
        case_file = path / 'case.json'
        if not path.is_dir() or not case_file.exists():
            continue
        try:
            case = Case(path.name, json.loads(case_file.read_text()), path).validate()
            cases.append(case)
        except (CaseBankError, ValueError) as error:
            problems.append(str(error))
            if strict:
                raise CaseBankError(str(error)) from None
    if not cases:
        raise CaseBankError(f'No usable cases in {directory}.')
    return cases, problems


def split(cases, holdout=0.3, seed=11):
    """Deterministic train/holdout split, stratified by recorded outcome.

    Stratified because a bank is usually unbalanced, and a holdout that happens to contain
    every declined case would make any tuning result meaningless in both directions.
    """
    import random
    buckets = {}
    for case in cases:
        buckets.setdefault(case.human['outcome'], []).append(case)
    rng = random.Random(seed)
    train, test = [], []
    for outcome in sorted(buckets):
        group = sorted(buckets[outcome], key=lambda c: c.case_id)
        rng.shuffle(group)
        cut = max(1, round(len(group) * holdout)) if len(group) > 1 else 0
        test.extend(group[:cut])
        train.extend(group[cut:])
    return sorted(train, key=lambda c: c.case_id), sorted(test, key=lambda c: c.case_id)


def folds(cases, k=5, seed=11):
    """k stratified folds: every case is held out exactly once.

    One train/holdout split measures one draw. Held out in turn across k folds, a tuning
    change has to keep paying off on cases the search did not see, k times over, before
    the proposal will carry it — and each case's holdout verdict is counted exactly once.
    """
    import random
    k = max(2, min(int(k), len(cases)))
    buckets = {}
    for case in cases:
        buckets.setdefault(case.human['outcome'], []).append(case)
    rng = random.Random(seed)
    assigned = [[] for _ in range(k)]
    offset = 0
    for outcome in sorted(buckets):
        group = sorted(buckets[outcome], key=lambda c: c.case_id)
        rng.shuffle(group)
        for position, case in enumerate(group):
            assigned[(offset + position) % k].append(case)
        offset += len(group)
    result = []
    for index in range(k):
        holdout = sorted(assigned[index], key=lambda c: c.case_id)
        train = sorted((c for j, fold in enumerate(assigned) if j != index for c in fold),
                       key=lambda c: c.case_id)
        if holdout and train:
            result.append((train, holdout))
    return result


# --- vision extraction ------------------------------------------------------
def extract_evidence(case, client):
    """Read a case's documents with the vision model, under the same rules as the demo."""
    if not case.documents:
        raise CaseBankError(f'{case.case_id} has no documents to extract.')
    try:
        return extraction.extract(case.documents, case.profile, client)
    except extraction.ExtractionError as error:
        raise CaseBankError(str(error)) from None

def ensure_evidence(cases, client, force=False, verbose=True):
    """Extract for any case that needs it. Returns (ready, skipped)."""
    ready, skipped = [], []
    for case in cases:
        if force or case.needs_extraction():
            try:
                evidence = extract_evidence(case, client)
                model = getattr(client, 'model', 'unknown model')
                case.store_evidence(evidence, model)
                if verbose:
                    print(f"  {case.case_id}: extracted {len(evidence['findings'])} finding(s)")
            except CaseBankError as error:
                skipped.append({'case_id': case.case_id, 'reason': str(error)})
                continue
        ready.append(case)
    return ready, skipped
