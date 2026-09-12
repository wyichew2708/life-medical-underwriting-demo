"""Turn a product specification sheet into reviewable underwriting configuration.

A spec sheet states what a product will and will not cover: issue ages, sum assured
limits, terms, what counts as a smoker, when a medical is required, what is excluded and
for how long. This module reads that document and produces typed rules and knowledge
cards from it — as a **draft**, never as live configuration.

Three things it deliberately will not do:

* It will not activate anything. A draft is inert until a named person activates it, and
  the activation is recorded with their reason.
* It will not generate a rule that accepts. A spec promising automatic acceptance up to
  some limit is recorded verbatim as an unsupported directive and reported to the
  reviewer; the pipeline has no action that grants acceptance, so there is nothing to
  translate it into.
* It will not guess. A field the document does not state stays null and is listed as a
  gap. An extracted value keeps the quote it came from, so a reviewer can check it
  against the page rather than trusting the extraction.

That ordering — generate, review, test, then operationalise — is the pattern the
reinsurer research in the knowledge base describes, and the reason it exists is the
failure mode it describes too: generated rules that look right and are not.
"""
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
PRODUCTS_DIR = ROOT / 'products'
ID_PATTERN = re.compile(r'[a-z0-9][a-z0-9_-]{1,40}')

# Directives a spec may contain that this pipeline cannot honour. Recorded, never applied.
UNSUPPORTED_KEYS = {
    'auto_accept_cover': 'automatic acceptance up to a sum assured',
    'auto_accept': 'automatic acceptance',
    'auto_decline_conditions': 'automatic decline on a condition list',
    'straight_through_limit': 'a product-set straight-through limit',
}

NUMERIC_FIELDS = {
    'issue_age_min': (18, 100), 'issue_age_max': (18, 100),
    'min_cover': (1000, 10000000), 'max_cover': (1000, 10000000),
    'term_years_min': (1, 70), 'term_years_max': (1, 70),
    'smoker_definition_months': (0, 120),
    'max_cover_without_medical': (1000, 10000000),
    'referral_cover': (1000, 10000000),
    'moratorium_months': (0, 120),
    'max_bmi': (10, 70),
}
TEXT_FIELDS = ('product_id', 'name', 'product_type', 'currency', 'notes', 'pre_existing_rule')
LIST_FIELDS = ('exclusions', 'required_evidence', 'benefits')


class ProductError(ValueError):
    pass


def _pymupdf():
    """PyMuPDF, under whichever module name this version prefers."""
    try:
        import pymupdf
        return pymupdf
    except ImportError:
        pass
    try:
        import fitz
        return fitz
    except ImportError:
        raise ProductError('Reading PDFs needs PyMuPDF: pip install -r requirements.txt') from None


# --- reading a spec sheet ---------------------------------------------------
LABELS = [
    (r'product\s*(?:id|code)', 'product_id', 'text'),
    (r'product\s*name|plan\s*name', 'name', 'text'),
    (r'product\s*type|cover\s*type|line\s*of\s*business', 'product_type', 'text'),
    (r'currency', 'currency', 'text'),
    (r'(?:minimum|min\.?)\s*(?:issue\s*)?age', 'issue_age_min', 'number'),
    (r'(?:maximum|max\.?)\s*(?:issue\s*|entry\s*)?age', 'issue_age_max', 'number'),
    (r'(?:minimum|min\.?)\s*(?:sum\s*assured|sum\s*insured|cover)', 'min_cover', 'number'),
    (r'(?:maximum|max\.?)\s*(?:sum\s*assured|sum\s*insured|cover)', 'max_cover', 'number'),
    (r'(?:minimum|min\.?)\s*(?:policy\s*)?term', 'term_years_min', 'number'),
    (r'(?:maximum|max\.?)\s*(?:policy\s*)?term', 'term_years_max', 'number'),
    (r'smoker\s*definition|nicotine[- ]free\s*period|tobacco\s*free', 'smoker_definition_months', 'number'),
    (r'non[- ]medical\s*limit|medical\s*(?:evidence\s*)?(?:required\s*)?(?:above|over)',
     'max_cover_without_medical', 'number'),
    (r'(?:automatic\s*)?referral\s*(?:limit|above|threshold)', 'referral_cover', 'number'),
    (r'moratorium', 'moratorium_months', 'number'),
    (r'(?:maximum|max\.?)\s*bmi', 'max_bmi', 'number'),
    (r'pre[- ]existing', 'pre_existing_rule', 'text'),
]
SECTIONS = [(r'exclusion', 'exclusions'), (r'required\s*evidence|evidence\s*requirement', 'required_evidence'),
            (r'benefit', 'benefits')]


MONTH_FIELDS = ('smoker_definition_months', 'moratorium_months')
# Longest unit first: a lazy alternation reads the "m" in "months" as "million".
UNIT_PATTERN = re.compile(r'(\d[\d,]*(?:\.\d+)?)\s*(million|thousand|months?|years?|m|k)?\b', re.I)


def _number(text, field=None):
    """Read a number out of a spec line, tolerating currency, commas, and units."""
    cleaned = re.sub(r'(?i)\b(sgd|myr|usd|rm|s\$|us\$|\$)\b', ' ', text)
    match = UNIT_PATTERN.search(cleaned)
    if not match:
        return None, None
    value = float(match.group(1).replace(',', ''))
    unit = (match.group(2) or '').lower()
    if unit in ('m', 'million'):
        value *= 1_000_000
    elif unit in ('k', 'thousand'):
        value *= 1_000
    elif unit.startswith('year') and field in MONTH_FIELDS:
        # A smoker window stated in years is the same window stated in months.
        value *= 12
    return (int(value) if value.is_integer() else value), match.group(0).strip()


def parse_text(text):
    """Deterministic label parsing. Returns (fields, report). Nothing is inferred."""
    fields, report = {}, {}
    lines = [line.rstrip() for line in text.splitlines()]
    section = None
    for number, line in enumerate(lines, 1):
        stripped = line.strip().lstrip('#').strip()
        if not stripped:
            continue
        heading = stripped.rstrip(':').lower()
        matched_section = next((name for pattern, name in SECTIONS if re.fullmatch(rf'\W*{pattern}s?\W*', heading)), None)
        if matched_section:
            section = matched_section
            fields.setdefault(section, [])
            report[section] = {'method': 'deterministic', 'quote': stripped, 'line': number}
            continue
        if section and re.match(r'^\s*[-*•]\s+', line):
            fields[section].append(re.sub(r'^\s*[-*•]\s+', '', line).strip())
            continue
        if re.match(r'^\s*[-*•]?\s*\w', line) and ':' in stripped:
            section = None
        label, _, value = stripped.partition(':')
        if not value.strip():
            continue
        for pattern, field, kind in LABELS:
            if re.fullmatch(rf'\W*{pattern}\W*', label.strip(), re.I):
                if field in fields:
                    break
                if kind == 'number':
                    parsed, quote = _number(value, field)
                    if parsed is None:
                        break
                    fields[field] = parsed
                    report[field] = {'method': 'deterministic', 'quote': stripped, 'line': number}
                else:
                    fields[field] = value.strip()
                    report[field] = {'method': 'deterministic', 'quote': stripped, 'line': number}
                break
    return fields, report


def read_document(path):
    """Extract text from a spec sheet. PDFs need PyMuPDF; scanned PDFs need the model."""
    path = Path(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    suffix = path.suffix.lower()
    if suffix == '.json':
        return {'kind': 'json', 'data': json.loads(raw), 'text': raw.decode('utf-8', 'replace'),
                'sha256': digest, 'pages': None}
    if suffix in ('.md', '.txt', '.markdown'):
        return {'kind': 'text', 'text': raw.decode('utf-8', 'replace'), 'sha256': digest, 'pages': None}
    if suffix == '.pdf':
        fitz = _pymupdf()
        text, pages = [], 0
        with fitz.open(stream=raw, filetype='pdf') as pdf:
            if pdf.needs_pass:
                raise ProductError('The spec sheet is encrypted.')
            pages = len(pdf)
            for page in pdf:
                text.append(page.get_text())
        joined = '\n'.join(text)
        if len(joined.strip()) < 40:
            raise ProductError('This PDF has no extractable text; it is probably a scan. Supply a text or '
                               'JSON spec, or run with --use-llm and a configured vision model.')
        return {'kind': 'pdf', 'text': joined, 'sha256': digest, 'pages': pages}
    raise ProductError(f'Unsupported spec sheet type: {suffix}. Use .json, .md, .txt or .pdf.')


# --- the spec itself --------------------------------------------------------
class ProductSpec:
    def __init__(self, fields, report=None, source=None, status='draft', activation=None,
                 created_at=None, unsupported=None):
        self.fields = dict(fields)
        self.report = dict(report or {})
        self.source = dict(source or {})
        self.status = status
        self.activation = activation
        self.created_at = created_at or datetime.now(timezone.utc).isoformat(timespec='seconds')
        self.unsupported = list(unsupported or [])

    # -- validation
    def validate(self):
        errors = []
        product_id = self.fields.get('product_id')
        if not isinstance(product_id, str) or not ID_PATTERN.fullmatch(product_id):
            errors.append('product_id must be a lowercase slug, e.g. term-life-2026.')
        if self.fields.get('product_type') not in ('Life', 'Medical'):
            errors.append('product_type must be Life or Medical.')
        for field, (low, high) in NUMERIC_FIELDS.items():
            value = self.fields.get(field)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(f'{field} must be a number.')
            elif not low <= value <= high:
                errors.append(f'{field} is outside the supported range {low}-{high}.')
        for low_field, high_field in (('issue_age_min', 'issue_age_max'), ('min_cover', 'max_cover'),
                                      ('term_years_min', 'term_years_max')):
            low, high = self.fields.get(low_field), self.fields.get(high_field)
            if isinstance(low, (int, float)) and isinstance(high, (int, float)) and low > high:
                errors.append(f'{low_field} is greater than {high_field}.')
        for field in LIST_FIELDS:
            value = self.fields.get(field)
            if value is not None and (not isinstance(value, list)
                                      or any(not isinstance(x, str) or not x.strip() for x in value)):
                errors.append(f'{field} must be a list of non-empty strings.')
        if errors:
            raise ProductError('; '.join(errors))
        return self

    def gaps(self):
        """Fields this pipeline can use that the document did not state."""
        useful = list(NUMERIC_FIELDS) + ['product_type', 'name', 'currency'] + list(LIST_FIELDS)
        return sorted(f for f in useful if self.fields.get(f) in (None, [], ''))

    # -- generation
    def to_rules(self):
        """Typed routing rules. Every action is refer, request_evidence or note."""
        pid = self.fields['product_id'].upper()
        scope = self.fields.get('product_type', 'Both')
        rules, index = [], 0

        def add(field, operator, value, action, title, cites):
            nonlocal index
            if value is None:
                return
            index += 1
            rules.append({'id': f'PRD-{pid}-{index}', 'title': title, 'field': field,
                          'operator': operator, 'value': value, 'scope': scope, 'action': action,
                          'cites': cites, 'enabled': True, 'origin': 'product spec',
                          'product_id': self.fields['product_id']})

        card = f'PRD-{pid}-CARD-1'
        f = self.fields
        add('age', 'lt', f.get('issue_age_min'), 'refer',
            f"Below the product minimum issue age ({f.get('issue_age_min')})", [card])
        add('age', 'gt', f.get('issue_age_max'), 'refer',
            f"Above the product maximum issue age ({f.get('issue_age_max')})", [card])
        add('cover', 'lt', f.get('min_cover'), 'refer',
            f"Below the product minimum sum assured", [card])
        add('cover', 'gt', f.get('max_cover'), 'refer',
            f"Above the product maximum sum assured", [card])
        add('cover', 'gt', f.get('max_cover_without_medical'), 'request_evidence',
            'Above the product non-medical limit; medical evidence required', [card, 'UW-EVID-01'])
        add('cover', 'gte', f.get('referral_cover'), 'refer',
            'At or above the product referral limit', [card, 'UW-FIN-03'])
        add('termYears', 'lt', f.get('term_years_min'), 'refer', 'Below the product minimum term', [card])
        add('termYears', 'gt', f.get('term_years_max'), 'refer', 'Above the product maximum term', [card])
        add('bmi', 'gt', f.get('max_bmi'), 'refer', 'Above the product build limit', [card, 'UW-BUILD-01'])
        return rules

    def to_cards(self):
        """Knowledge cards so the model can cite the product's own terms."""
        pid = self.fields['product_id'].upper()
        f = self.fields
        products = [f['product_type']] if f.get('product_type') in ('Life', 'Medical') else ['Life', 'Medical']
        limits = []
        for label, key, suffix in [('minimum issue age', 'issue_age_min', ''),
                                   ('maximum issue age', 'issue_age_max', ''),
                                   ('minimum sum assured', 'min_cover', ''),
                                   ('maximum sum assured', 'max_cover', ''),
                                   ('non-medical limit', 'max_cover_without_medical', ''),
                                   ('referral limit', 'referral_cover', ''),
                                   ('minimum term', 'term_years_min', ' years'),
                                   ('maximum term', 'term_years_max', ' years'),
                                   ('smoker definition window', 'smoker_definition_months', ' months'),
                                   ('moratorium', 'moratorium_months', ' months')]:
            if f.get(key) is not None:
                limits.append(f'{label} {f[key]:,}{suffix}' if isinstance(f[key], (int, float))
                              else f'{label} {f[key]}{suffix}')
        cards = [{
            'id': f'PRD-{pid}-CARD-1', 'type': 'internal',
            'title': f"Product limits: {f.get('name') or f['product_id']}",
            'topics': ['product', 'limits'], 'products': products,
            'applies_when': {'always': True},
            'keywords': ['product', 'limit', 'sum assured', 'issue age'],
            'excerpt': (f"Limits declared by the product specification for "
                        f"{f.get('name') or f['product_id']}"
                        + (f" ({f['currency']})" if f.get('currency') else '') + ': '
                        + ('; '.join(limits) if limits else 'no numeric limits were stated in the document')
                        + '. These are the product\'s own terms as extracted from its specification sheet, '
                          'not underwriting judgement, and each was checked against the source quote '
                          'recorded in the extraction report.'),
            'status': 'product specification', 'collection': 'Product specification',
            'product_id': f['product_id']}]
        if f.get('exclusions'):
            cards.append({
                'id': f'PRD-{pid}-CARD-2', 'type': 'internal',
                'title': f"Product exclusions: {f.get('name') or f['product_id']}",
                'topics': ['product', 'exclusions'], 'products': products,
                'applies_when': {'always': True},
                'keywords': ['exclusion', 'not covered', 'waiting period'],
                'excerpt': ('The product specification excludes: ' + '; '.join(f['exclusions'][:25])
                            + '. An exclusion is policy wording. Report which exclusion may bear on a case '
                              'and refer the drafting to an underwriter; do not restate it as a decision.'),
                'status': 'product specification', 'collection': 'Product specification',
                'product_id': f['product_id']})
        if f.get('required_evidence') or f.get('pre_existing_rule'):
            parts = []
            if f.get('required_evidence'):
                parts.append('Evidence the product requires: ' + '; '.join(f['required_evidence'][:25]) + '.')
            if f.get('pre_existing_rule'):
                parts.append('Pre-existing condition treatment: ' + f['pre_existing_rule'])
            cards.append({
                'id': f'PRD-{pid}-CARD-3', 'type': 'internal',
                'title': f"Product evidence and pre-existing terms: {f.get('name') or f['product_id']}",
                'topics': ['product', 'evidence'], 'products': products,
                'applies_when': {'always': True},
                'keywords': ['pre-existing', 'evidence', 'moratorium'],
                'excerpt': ' '.join(parts),
                'status': 'product specification', 'collection': 'Product specification',
                'product_id': f['product_id']})
        return cards

    # -- lifecycle
    def activate(self, by, reason):
        if not by or not reason:
            raise ProductError('Activation records who activated the product and why.')
        self.validate()
        self.status = 'active'
        self.activation = {'activated_by': by, 'reason': reason,
                           'activated_at': datetime.now(timezone.utc).isoformat(timespec='seconds')}
        return self

    def deactivate(self, by, reason=''):
        self.status = 'draft'
        self.activation = {'deactivated_by': by, 'reason': reason,
                           'deactivated_at': datetime.now(timezone.utc).isoformat(timespec='seconds')}
        return self

    def to_dict(self):
        return {'fields': self.fields, 'status': self.status, 'created_at': self.created_at,
                'source': self.source, 'extraction_report': self.report,
                'unsupported_directives': self.unsupported, 'activation': self.activation,
                'generated_rules': self.to_rules(), 'generated_cards': self.to_cards(),
                'gaps': self.gaps(),
                'warning': ('Generated from a specification sheet by automated extraction. Every rule and '
                            'card must be checked against the source document before activation.')}

    @classmethod
    def from_dict(cls, data):
        return cls(data['fields'], data.get('extraction_report'), data.get('source'),
                   data.get('status', 'draft'), data.get('activation'), data.get('created_at'),
                   data.get('unsupported_directives'))

    def save(self, directory=PRODUCTS_DIR):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.fields['product_id']}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + '\n')
        return path


def collect_unsupported(fields):
    """Directives a spec may state that this pipeline refuses to translate into a rule."""
    found = []
    for key, description in UNSUPPORTED_KEYS.items():
        if fields.get(key) is not None:
            found.append({'field': key, 'value': fields.pop(key), 'directive': description,
                          'handling': 'Recorded, not applied. This pipeline has no action that accepts a '
                                      'case, so the directive cannot be translated into a rule. An '
                                      'underwriter decides these cases.'})
    return found


def from_document(path, product_id=None, llm_extractor=None):
    """Read a spec sheet into a draft ProductSpec, keeping the provenance of every field."""
    document = read_document(path)
    if document['kind'] == 'json':
        fields = dict(document['data'])
        report = {k: {'method': 'declared', 'quote': None} for k in fields}
    else:
        fields, report = parse_text(document['text'])
        if llm_extractor is not None:
            extracted = llm_extractor(document['text'])
            for field, entry in (extracted or {}).items():
                if field in fields or entry.get('value') is None:
                    continue   # a deterministic read always wins over a model's read
                fields[field] = entry['value']
                report[field] = {'method': 'model', 'quote': entry.get('quote'),
                                 'note': 'Extracted by the language model. Check against the document.'}
    if product_id:
        fields['product_id'] = product_id
    fields.setdefault('product_id', Path(path).stem.lower().replace(' ', '-')[:40])
    unsupported = collect_unsupported(fields)
    spec = ProductSpec(fields, report,
                       source={'file': str(path), 'sha256': document['sha256'],
                               'kind': document['kind'], 'pages': document['pages'],
                               'read_at': datetime.now(timezone.utc).isoformat(timespec='seconds')},
                       unsupported=unsupported)
    return spec


def load(product_id, directory=PRODUCTS_DIR):
    path = Path(directory) / f'{product_id}.json'
    if not path.exists():
        raise ProductError(f'No product {product_id} in {directory}.')
    return ProductSpec.from_dict(json.loads(path.read_text()))


def available(directory=PRODUCTS_DIR):
    directory = Path(directory)
    if not directory.exists():
        return []
    out = []
    for path in sorted(directory.glob('*.json')):
        data = json.loads(path.read_text())
        out.append({'product_id': data['fields'].get('product_id', path.stem),
                    'name': data['fields'].get('name'), 'type': data['fields'].get('product_type'),
                    'status': data.get('status', 'draft'), 'rules': len(data.get('generated_rules', [])),
                    'gaps': len(data.get('gaps', [])), 'path': str(path)})
    return out
