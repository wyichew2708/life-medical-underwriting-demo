"""The configurable surface of the pipeline, and the bounds nothing may cross.

Everything a tuning run is allowed to change lives here. The point of collecting it in one
typed object is that the set of things which can move is small, declared and reviewable —
and that the things which cannot move are enforced in code rather than left to the good
behaviour of whatever produced the configuration.

Three invariants hold for every configuration, however it was produced:

* A mandatory rule cannot be disabled, and its action cannot be weakened.
* A rule threshold can only move inside the bounds its own definition declares.
* Validation strictness can only tighten. There is no configuration that permits terms on
  incomplete evidence, terms without an internal citation, or more than one revision.
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
DEFAULT_PATH = ROOT / 'config.json'

ACTION_STRENGTH = {'note': 1, 'refer': 2, 'request_evidence': 3}
BOUNDS = {
    'retrieval_source_limit': (5, 30),
    'design_source_limit': (0, 8),
    'precedent_limit': (0, 5),
    'max_revisions': (0, 1),
}
PROMPT_VARIANTS = ('standard', 'evidence-first', 'concise', 'teaching')


class ConfigError(ValueError):
    pass


class PipelineConfig:
    def __init__(self, retrieval_source_limit=30, include_design_sources=True, design_source_limit=3,
                 precedent_limit=3, prompt_variant='standard', max_revisions=1,
                 require_internal_citation_for_terms=True, warnings_block_terms=True, rule_overrides=None,
                 provenance=None):
        self.retrieval_source_limit = int(retrieval_source_limit)
        self.include_design_sources = bool(include_design_sources)
        self.design_source_limit = int(design_source_limit)
        self.precedent_limit = int(precedent_limit)
        self.prompt_variant = prompt_variant
        self.max_revisions = int(max_revisions)
        self.require_internal_citation_for_terms = bool(require_internal_citation_for_terms)
        self.warnings_block_terms = bool(warnings_block_terms)
        self.rule_overrides = dict(rule_overrides or {})
        self.provenance = provenance or {}

    def to_dict(self):
        return {'retrieval_source_limit': self.retrieval_source_limit,
                'include_design_sources': self.include_design_sources,
                'design_source_limit': self.design_source_limit,
                'precedent_limit': self.precedent_limit,
                'prompt_variant': self.prompt_variant,
                'max_revisions': self.max_revisions,
                'require_internal_citation_for_terms': self.require_internal_citation_for_terms,
                'warnings_block_terms': self.warnings_block_terms,
                'rule_overrides': self.rule_overrides,
                'provenance': self.provenance}

    def replace(self, **changes):
        return PipelineConfig(**{**self.to_dict(), **changes})

    def validate(self, rules=None):
        """Raise unless every field is inside its bounds and every invariant holds."""
        for field, (low, high) in BOUNDS.items():
            value = getattr(self, field)
            if not low <= value <= high:
                raise ConfigError(f'{field} must be between {low} and {high}; got {value}.')
        if self.prompt_variant not in PROMPT_VARIANTS:
            raise ConfigError(f'Unknown prompt variant {self.prompt_variant}. '
                              f'Choose from: {", ".join(PROMPT_VARIANTS)}')
        if not self.require_internal_citation_for_terms:
            raise ConfigError('Proposed terms always require a cited internal rule. This cannot be turned off.')
        if not self.warnings_block_terms:
            raise ConfigError('Evidence warnings always block proposed terms. This cannot be turned off.')
        if rules is not None:
            self._validate_overrides(rules)
        return self

    def _validate_overrides(self, rules):
        index = {r['id']: r for r in rules}
        for rule_id, override in self.rule_overrides.items():
            rule = index.get(rule_id)
            if rule is None:
                raise ConfigError(f'Override for unknown rule {rule_id}.')
            if not isinstance(override, dict):
                raise ConfigError(f'Override for {rule_id} must be an object.')
            for key in override:
                if key not in ('value', 'enabled', 'action'):
                    raise ConfigError(f'Override for {rule_id} cannot change {key}.')
            if 'enabled' in override and not override['enabled']:
                if rule.get('mandatory'):
                    raise ConfigError(f'{rule_id} is mandatory and cannot be disabled.')
            if 'action' in override:
                if ACTION_STRENGTH.get(override['action']) is None:
                    raise ConfigError(f'Override for {rule_id} has an unknown action.')
                if ACTION_STRENGTH[override['action']] < ACTION_STRENGTH[rule['action']]:
                    raise ConfigError(f'{rule_id}: an override may tighten an action, never weaken it '
                                      f'({rule["action"]} -> {override["action"]}).')
            if 'value' in override:
                bounds = rule.get('tunable')
                if not bounds:
                    raise ConfigError(f'{rule_id} has no tunable bounds; its threshold is fixed.')
                value = override['value']
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ConfigError(f'{rule_id}: a tuned threshold must be a number.')
                if not bounds['min'] <= value <= bounds['max']:
                    raise ConfigError(f'{rule_id}: {value} is outside the declared bounds '
                                      f'{bounds["min"]}-{bounds["max"]}.')

    def apply_to_rules(self, rules):
        """Return the effective rule set. Disabled rules are dropped, not silently kept."""
        effective = []
        for rule in rules:
            override = self.rule_overrides.get(rule['id'], {})
            merged = {**rule, **{k: v for k, v in override.items() if k in ('value', 'action')}}
            if override.get('enabled') is False or rule.get('enabled') is False:
                continue
            if override:
                merged['tuned'] = {k: v for k, v in override.items()}
            effective.append(merged)
        return effective

    @classmethod
    def load(cls, path=DEFAULT_PATH, rules=None):
        path = Path(path)
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        return cls(**{k: v for k, v in data.items() if k in cls().to_dict()}).validate(rules)

    def save(self, path=DEFAULT_PATH):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + '\n')
        return path

    def fingerprint(self):
        """A stable hash of everything that affects a run, so a result can name its configuration."""
        import hashlib
        payload = json.dumps({k: v for k, v in self.to_dict().items() if k != 'provenance'}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def prune(self):
        """Drop overrides that cannot affect anything, so a diff shows only real changes.

        A search can arrive at a rule that is both retuned and switched off. Keeping the
        dead threshold in the diff makes a reviewer check a change that does nothing.
        """
        cleaned = {}
        for rule_id, override in self.rule_overrides.items():
            if override.get('enabled') is False:
                cleaned[rule_id] = {'enabled': False}
            elif override:
                cleaned[rule_id] = {k: v for k, v in override.items() if k != 'enabled' or v is not True}
            if cleaned.get(rule_id) == {}:
                cleaned.pop(rule_id)
        return self.replace(rule_overrides=cleaned)

    def diff(self, other):
        """What changed between two configurations, field by field."""
        changes = {}
        mine, theirs = self.to_dict(), other.to_dict()
        for key in mine:
            if key == 'provenance':
                continue
            if key == 'rule_overrides':
                for rule_id in sorted(set(mine[key]) | set(theirs[key])):
                    before, after = mine[key].get(rule_id), theirs[key].get(rule_id)
                    if before != after:
                        changes[f'rule {rule_id}'] = {'from': before, 'to': after}
            elif mine[key] != theirs[key]:
                changes[key] = {'from': mine[key], 'to': theirs[key]}
        return changes
