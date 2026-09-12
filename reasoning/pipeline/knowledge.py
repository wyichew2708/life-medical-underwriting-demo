"""Load and validate the underwriting knowledge base.

Cards are validated against the same constraints the demo runner applies to a retrieval
adapter response in demo/server.py: unique IDs matching a restricted character set, no
`DOC-` prefix (that namespace belongs to uploaded documents), a known source type,
bounded title and excerpt, and HTTP(S) URLs only. A card that would be rejected there is
rejected here, at load time, rather than in front of an underwriter.
"""
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).parent.parent
KNOWLEDGE_DIR = ROOT / 'knowledge'
ROUTING_RULES = KNOWLEDGE_DIR / 'routing_rules.json'

SOURCE_TYPES = ('internal', 'external', 'rag', 'knowledge', 'web')
ID_PATTERN = re.compile(r'[A-Za-z0-9_-]{1,60}')
MAX_TEXT = 6000
# Research and architecture cards answer "why is the system built this way"; they are not
# case guidance, so they stay out of an assessment unless external context is requested.
DESIGN_COLLECTIONS = ('Research and industry evidence',)


class KnowledgeError(ValueError):
    pass


def _check_card(card, collection, seen):
    for key in ('id', 'title', 'excerpt', 'type'):
        value = card.get(key)
        if not isinstance(value, str) or not 1 <= len(value) <= MAX_TEXT:
            raise KnowledgeError(f'{collection}: card {card.get("id", "?")} has an invalid {key}.')
    if not ID_PATTERN.fullmatch(card['id']) or card['id'].startswith('DOC-'):
        raise KnowledgeError(f'Invalid card ID {card["id"]}. DOC- is reserved for uploaded documents.')
    if card['id'] in seen:
        raise KnowledgeError(f'Duplicate card ID {card["id"]}.')
    if card['type'] not in SOURCE_TYPES:
        raise KnowledgeError(f'Card {card["id"]} has unknown type {card["type"]}.')
    url = card.get('url')
    if url is not None and (not isinstance(url, str) or urlsplit(url).scheme not in ('http', 'https')):
        raise KnowledgeError(f'Card {card["id"]} has an invalid URL.')
    products = card.get('products', ['Life', 'Medical'])
    if not isinstance(products, list) or any(p not in ('Life', 'Medical') for p in products):
        raise KnowledgeError(f'Card {card["id"]} has an invalid product scope.')
    keywords = card.get('keywords', [])
    if not isinstance(keywords, list) or any(not isinstance(k, str) or not k for k in keywords):
        raise KnowledgeError(f'Card {card["id"]} has invalid keywords.')
    trigger = card.get('applies_when', {})
    if not isinstance(trigger, dict):
        raise KnowledgeError(f'Card {card["id"]} has an invalid trigger.')
    for group in ('any', 'all'):
        for condition in trigger.get(group, []):
            if not isinstance(condition, dict) or not isinstance(condition.get('field'), str):
                raise KnowledgeError(f'Card {card["id"]} has an invalid {group} condition.')


class KnowledgeBase:
    def __init__(self, cards, rules, evidence_grid, rules_revision=1):
        self.cards = cards
        self.by_id = {c['id']: c for c in cards}
        self.rules = rules
        self.evidence_grid = evidence_grid
        self.rules_revision = rules_revision

    def __len__(self):
        return len(self.cards)

    def topics(self):
        return sorted({t for c in self.cards for t in c.get('topics', [])})

    def of_type(self, *types):
        return [c for c in self.cards if c['type'] in types]

    def cited(self, card_ids):
        return [self.by_id[i] for i in card_ids if i in self.by_id]

    @staticmethod
    def as_source(card):
        """The exact shape the demo's retrieval adapter contract expects."""
        source = {'id': card['id'], 'type': card['type'], 'title': card['title'],
                  'excerpt': card['excerpt']}
        if card.get('url'):
            source['url'] = card['url']
        return source

    def with_cards(self, extra):
        """A copy of the base with extra cards merged in, validated the same way.

        Product cards arrive this way. They are validated at merge time, so a generated
        card that the demo's retrieval contract would reject never reaches a prompt.
        """
        seen = set(self.by_id)
        merged = list(self.cards)
        for card in extra or []:
            _check_card(card, card.get('collection', 'merged'), seen)
            seen.add(card['id'])
            merged.append({**card, 'scope': card.get('scope', 'case'),
                           'status': card.get('status', 'illustrative'),
                           'collection': card.get('collection', 'Merged')})
        return KnowledgeBase(merged, self.rules, self.evidence_grid, self.rules_revision)

    def with_rules(self, extra):
        """A copy with extra routing rules appended. Restrictions only: nothing may accept."""
        for rule in extra or []:
            if rule.get('action') not in ('refer', 'request_evidence', 'note'):
                raise KnowledgeError(f'Rule {rule.get("id")} has action {rule.get("action")}. '
                                     'Only refer, request_evidence and note exist.')
        return KnowledgeBase(self.cards, list(self.rules) + list(extra or []), self.evidence_grid,
                             self.rules_revision)

    def summary(self):
        counts = {}
        for card in self.cards:
            counts[card['type']] = counts.get(card['type'], 0) + 1
        return {'cards': len(self.cards), 'by_type': counts, 'routing_rules': len(self.rules),
                'topics': len(self.topics()), 'rules_revision': self.rules_revision}


def load(knowledge_dir=KNOWLEDGE_DIR, rules_path=ROUTING_RULES):
    cards, seen = [], set()
    for path in sorted(Path(knowledge_dir).rglob('*.json')):
        if path.resolve() == Path(rules_path).resolve():
            continue
        data = json.loads(path.read_text())
        collection = data.get('collection', path.stem)
        for card in data.get('cards', []):
            _check_card(card, collection, seen)
            seen.add(card['id'])
            cards.append({**card, 'collection': collection,
                          'status': card.get('status', data.get('status', 'illustrative')),
                          'scope': 'design' if collection in DESIGN_COLLECTIONS else 'case',
                          'source_file': str(path.relative_to(Path(knowledge_dir).parent))})
    if not cards:
        raise KnowledgeError(f'No knowledge cards found under {knowledge_dir}.')

    rules_data = json.loads(Path(rules_path).read_text())
    rules = rules_data.get('rules', [])
    for rule in rules:
        missing = [c for c in rule.get('cites', []) if c not in seen]
        if missing:
            raise KnowledgeError(f'Rule {rule.get("id")} cites unknown card(s): {missing}')
    return KnowledgeBase(cards, rules, rules_data.get('evidence_grid', {}),
                         rules_data.get('revision', 1))
