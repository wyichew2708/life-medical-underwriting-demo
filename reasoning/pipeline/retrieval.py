"""Select the knowledge a case actually needs.

Deterministic first, keyword second. Structured triggers fire on declared values, so the
same case always retrieves the same guidance; keyword matching then reaches cards that no
intake field can reach — family history, pursuits, alcohol — using the extracted evidence
text and the underwriter's instructions. Nothing is retrieved by similarity score, and
nothing is invented: every source returned is a card that exists in the repository.

The result is capped at the demo's limit of 30 sources. When the cap bites, mandatory
guardrail cards are kept and the truncation is reported rather than hidden.
"""
import re

MAX_SOURCES = 30
DESIGN_SOURCE_LIMIT = 3
# Ordering is by band first, then card ID, so a case retrieves a stable, diffable list.
BAND_ALWAYS = 0        # framework, guardrails: always in context
BAND_TRIGGERED = 1     # a declared value fired the card's condition
BAND_KEYWORD = 2       # evidence text or instructions mentioned it
BAND_DESIGN = 3        # research and architecture context


def _condition_holds(condition, profile):
    from .rules import compare
    result = compare(profile.get(condition['field']), condition['operator'], condition['value'])
    return bool(result)


def _trigger_state(card, profile):
    trigger = card.get('applies_when') or {}
    if trigger.get('always') is True:
        return 'always'
    if trigger.get('any') and any(_condition_holds(c, profile) for c in trigger['any']):
        return 'triggered'
    if trigger.get('all') and all(_condition_holds(c, profile) for c in trigger['all']):
        return 'triggered'
    return 'none'


def _keyword_hits(card, haystack):
    return [k for k in card.get('keywords', []) if k.lower() in haystack]


def case_text(evidence=None, instructions='', profile=None):
    """Everything free-text the case carries, lowercased for matching."""
    parts = [instructions or '']
    for finding in (evidence or {}).get('findings', []) or []:
        parts.extend([str(finding.get('text', '')), str(finding.get('quote', '')), str(finding.get('source', ''))])
    parts.extend(str(w) for w in (evidence or {}).get('warnings', []) or [])
    if profile:
        parts.extend([str(profile.get('occupation', '')), str(profile.get('condition', ''))])
    return re.sub(r'\s+', ' ', ' '.join(parts)).lower()


def retrieve(knowledge, profile, evidence=None, instructions='', include_design=True,
             limit=MAX_SOURCES, design_limit=DESIGN_SOURCE_LIMIT):
    haystack = case_text(evidence, instructions, profile)
    product = profile.get('product')
    selected, trace = [], []

    for card in knowledge.cards:
        if product and product not in card.get('products', ['Life', 'Medical']):
            continue
        state = _trigger_state(card, profile)
        hits = _keyword_hits(card, haystack)
        if card.get('scope') == 'design':
            if not include_design:
                continue
            band, reason = BAND_DESIGN, 'design and research context'
        elif state == 'always':
            band, reason = BAND_ALWAYS, 'always applied'
        elif state == 'triggered':
            band, reason = BAND_TRIGGERED, 'declared values match this rule'
        elif hits:
            band, reason = BAND_KEYWORD, 'mentioned in evidence or instructions: ' + ', '.join(sorted(hits)[:4])
        else:
            continue
        selected.append((band, card['id'], card))
        trace.append({'id': card['id'], 'title': card['title'], 'band': band, 'reason': reason,
                      'keyword_hits': sorted(hits), 'collection': card.get('collection')})

    selected.sort(key=lambda item: (item[0], item[1]))
    design = [item for item in selected if item[0] == BAND_DESIGN][design_limit:]
    dropped_design = {item[1] for item in design}
    selected = [item for item in selected if item[1] not in dropped_design]

    truncated = []
    if len(selected) > limit:
        truncated = [item[1] for item in selected[limit:]]
        selected = selected[:limit]

    by_id = {item[1] for item in selected}
    sources = [knowledge.as_source(card) for _, _, card in selected]
    notes = ['Retrieval is deterministic: structured triggers, then keyword matches on '
             'extracted evidence and underwriter instructions. No similarity search, no web search.']
    if truncated:
        notes.append(f'{len(truncated)} lower-priority source(s) omitted at the 30-source cap: '
                     + ', '.join(truncated))
    if dropped_design:
        notes.append(f'{len(dropped_design)} research source(s) beyond the design limit omitted.')
    return {'sources': sources,
            'cards': [card for _, _, card in selected],
            'trace': [t for t in trace if t['id'] in by_id],
            'omitted': truncated + sorted(dropped_design),
            'notes': notes}
