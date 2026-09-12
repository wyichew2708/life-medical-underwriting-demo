"""Output validation and injection scanning.

The schema and citation checks are the ones demo/server.py applies, kept deliberately
identical so a recommendation produced by this pipeline survives the demo's own
verification step. The additional checks cover what the demo cannot see from inside a
single request: that the deterministic routing outcome was respected, that proposed terms
rest on case guidance rather than on a research citation, and that text which tries to
instruct the model was surfaced rather than obeyed.
"""
import re

RECOMMENDATIONS = ('refer', 'request_evidence', 'propose_terms')
MAX_EXPLANATION = 5000
MAX_ITEMS = 30
MAX_ITEM_LENGTH = 3000

# Patterns that have no business in a medical report or an underwriting rule excerpt.
INJECTION_PATTERNS = [
    (r'ignore (all |any |the )?(previous|prior|above|preceding) (instructions?|rules?|prompts?)', 'instruction override'),
    (r'disregard (the |all |any )?(above|previous|prior|system)', 'instruction override'),
    (r'you are (now )?(a |an )?(different|new)\b', 'role reassignment'),
    (r'\bsystem prompt\b', 'prompt disclosure attempt'),
    (r'(approve|accept|issue)( this| the)? (case|policy|application) (automatically|immediately|without)', 'approval instruction'),
    (r'(pre-?approved|already approved) by (the )?underwrit', 'false authority claim'),
    (r'do not (refer|request|escalate|flag)', 'control suppression'),
    (r'\bas an ai\b.{0,40}\byou must\b', 'instruction override'),
]


def text_list(value, limit=MAX_ITEMS):
    return (isinstance(value, list) and len(value) <= limit
            and all(isinstance(x, str) and 0 < len(x) <= MAX_ITEM_LENGTH for x in value))


def scan_for_injection(texts):
    """Report instruction-like content found in untrusted material."""
    found = []
    for label, text in texts:
        lowered = (text or '').lower()
        for pattern, kind in INJECTION_PATTERNS:
            match = re.search(pattern, lowered)
            if match:
                excerpt = lowered[max(0, match.start() - 40):match.end() + 40].strip()
                found.append({'where': label, 'kind': kind, 'excerpt': excerpt})
                break
    return found


NUMBER = re.compile(r'(?<![\w.])(\d[\d,]*(?:\.\d+)?)(%?)(?![\w])')
# Identifiers carry digits that are not figures: RTE-HBA1C-65, UW-MED-03, DOC-1, p.2.
ID_TOKEN = re.compile(r'\b[A-Za-z]{1,}[A-Za-z0-9]*(?:-[A-Za-z0-9]+)*-\d+[A-Za-z]?\b|\bp\.\s?\d+\b')
SMALL_FIGURES = set(float(n) for n in range(0, 13)) | {100.0}


def figures_in(text):
    """Every number written in a text, as floats, with percentages kept as written."""
    found = set()
    for number, percent in NUMBER.findall(ID_TOKEN.sub(' ', text or '')):
        try:
            value = float(number.replace(',', ''))
        except ValueError:
            continue
        found.add(value)
        if percent:
            found.add(round(value / 100, 6))
    return found


def _walk(value, out):
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        out.add(float(value))
    elif isinstance(value, str):
        out |= figures_in(value)
    elif isinstance(value, dict):
        for item in value.values():
            _walk(item, out)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _walk(item, out)


def allowed_figures(*parts):
    """The set of numbers an explanation may quote: whatever appears in the supplied case data."""
    allowed = set(SMALL_FIGURES)
    for part in parts:
        _walk(part, allowed)
    return allowed


def _supported(figure, allowed):
    for known in allowed:
        if abs(known - figure) <= max(0.05, abs(known) * 0.005):
            return True
    return False


def check_figures(output, allowed):
    """Numbers in the model's text that appear nowhere in the case data or the cited sources.

    A misquoted lab value is the most damaging thing a reasoning step can do quietly. Every
    figure in the explanation, the reasons and the evidence request must be traceable to
    the profile, the evidence, a rule threshold or a supplied source; anything else fails
    validation and the model gets one revision to remove it.
    """
    texts = [output.get('explanation') or '']
    for key in ('reasons', 'missing_information'):
        if isinstance(output.get(key), list):
            texts += [x for x in output[key] if isinstance(x, str)]
    unsupported = set()
    for text in texts:
        for figure in figures_in(text):
            if not _supported(figure, allowed):
                unsupported.add(figure)
    return sorted(unsupported)


def validate_output(output, evidence, sources, rule_outcome='pass', knowledge=None, figures=None):
    """Return a list of validation failures; empty means the output may be used.

    `figures`, when given, is the set of numbers the output may quote; see check_figures.
    """
    if not isinstance(output, dict):
        return ['Output must be a JSON object.']
    errors = []
    recommendation = output.get('recommendation')
    if recommendation not in RECOMMENDATIONS:
        errors.append('Recommendation must be refer, request_evidence or propose_terms; never approve or decline.')
    explanation = output.get('explanation')
    if not isinstance(explanation, str) or not 1 <= len(explanation) <= MAX_EXPLANATION:
        errors.append('Provide a concise explanation.')
    for key in ('reasons', 'citations', 'missing_information'):
        if not text_list(output.get(key)):
            errors.append(f'Invalid {key}.')
    if not output.get('reasons'):
        errors.append('At least one reason is required.')

    known = {s['id'] for s in sources} | {f['id'] for f in (evidence or {}).get('findings', []) or []}
    citations = output.get('citations') if text_list(output.get('citations')) else []
    if not citations:
        errors.append('At least one source citation is required.')
    unknown = [c for c in citations if c not in known]
    if unknown:
        errors.append(f'Unknown source citation(s): {", ".join(sorted(unknown)[:5])}.')

    complete = bool((evidence or {}).get('complete'))
    if not complete and recommendation != 'request_evidence':
        errors.append('Incomplete evidence requires request_evidence.')
    if (evidence or {}).get('warnings') and recommendation == 'propose_terms':
        errors.append('Evidence warnings require referral or further evidence, not proposed terms.')

    if recommendation == 'propose_terms':
        internal = {s['id'] for s in sources if s.get('type') == 'internal'}
        if not internal & set(citations):
            errors.append('Proposed terms require a cited internal rule.')
        if knowledge is not None:
            design = {c['id'] for c in knowledge.cards if c.get('scope') == 'design'}
            if set(citations) and set(citations) <= design:
                errors.append('Research sources cannot be the only support for proposed terms.')

    if rule_outcome == 'request_evidence' and recommendation != 'request_evidence':
        errors.append('A blocking rule requires an evidence request.')
    if rule_outcome == 'refer' and recommendation == 'propose_terms':
        errors.append('A referral rule fired; proposed terms are not available for this case.')

    if figures is not None:
        unsupported = check_figures(output, figures)
        if unsupported:
            shown = ', '.join(f'{f:g}' for f in unsupported[:6])
            errors.append(f'Figure(s) not present in the case data or the supplied sources: {shown}. Quote only '
                          'values that appear in the profile, the evidence, a rule or a cited source.')
    return errors
