"""The pipeline itself: validate, route, screen, retrieve, reason, verify.

Order matters and is fixed. Deterministic rules run before the model so the model is told
what the restrictions already are; they run again after it so nothing the model wrote can
loosen them. The LLM stage is skipped entirely when the ML screen clears every gate, and
it is optional in general: with no model configured the pipeline still produces the rule
outcome, the retrieved guidance and the evidence request, clearly labelled as having had
no model involvement rather than silently degraded.
"""
import json
import math
import os
import time
from pathlib import Path

from . import extraction, knowledge as knowledge_module, precedents as precedents_module, reconcile
from . import prompts, product as product_module, rules, retrieval, validators
from .config import PipelineConfig
from .llm import Client, LLMError

ATTRIBUTES = Path(__file__).parent.parent.parent / 'demo' / 'dist' / 'attributes.json'
STP_CONFIDENCE = 0.95


class PipelineError(ValueError):
    pass


def _catalogue():
    if not ATTRIBUTES.exists():
        raise PipelineError(f'Field catalogue not found at {ATTRIBUTES}. The demo folder is the source of record.')
    data = json.loads(ATTRIBUTES.read_text())
    return data['core'], data['extra']


class Pipeline:
    def __init__(self, knowledge=None, client=None, config=None, product=None,
                 include_design=None, source_limit=None, precedents=None):
        self.base_knowledge = knowledge or knowledge_module.load()
        # None: load the case bank lazily when precedents are wanted. False: never. An index: use it.
        self._precedents = precedents
        self.client = client if client is not None else Client()
        self.core, self.extra = _catalogue()
        # With no configuration passed, an applied tuning proposal (config.json) is in force.
        # That file is the only way a tuned setting reaches a run; defaults apply when it is absent.
        if config is None:
            config = PipelineConfig.load(rules=self.base_knowledge.rules)
        self.config = config.validate(self.base_knowledge.rules)
        if include_design is not None:
            self.config = self.config.replace(include_design_sources=include_design)
        if source_limit is not None:
            self.config = self.config.replace(retrieval_source_limit=source_limit)
        self.product = self._load_product(product)
        self.knowledge = self._effective_knowledge()

    # -- product configuration ----------------------------------------------
    @staticmethod
    def _load_product(product):
        if product is None or isinstance(product, product_module.ProductSpec):
            return product
        if isinstance(product, str):
            return product_module.load(product)
        if isinstance(product, dict):
            return product_module.ProductSpec.from_dict(product)
        raise PipelineError('product must be a product id, a ProductSpec or None.')

    def _effective_knowledge(self):
        base = self.base_knowledge
        if self.product is None:
            return base
        base = base.with_cards(self.product.to_cards())
        # A draft product is inert. Its cards are available to read; its rules do not route.
        if self.product.status == 'active':
            base = base.with_rules(self.product.to_rules())
        return base

    def using(self, config):
        """A copy with a different configuration, sharing the loaded knowledge base.

        Tuning evaluates hundreds of configurations; reloading and revalidating the whole
        knowledge base for each one would make the search slow enough to discourage
        running it, which is the wrong incentive.
        """
        clone = object.__new__(Pipeline)
        clone.__dict__.update(self.__dict__)
        clone.config = config.validate(self.knowledge.rules)
        return clone

    def precedent_index(self):
        if self._precedents is False:
            return None
        if self._precedents is None:
            self._precedents = precedents_module.PrecedentIndex.from_bank(self.core, self.extra)
        return self._precedents

    def product_summary(self):
        if self.product is None:
            return None
        fields = self.product.fields
        applied = len(self.product.to_rules()) if self.product.status == 'active' else 0
        return {'product_id': fields.get('product_id'), 'name': fields.get('name'),
                'type': fields.get('product_type'), 'status': self.product.status,
                'rules_applied': applied, 'rules_available': len(self.product.to_rules()),
                'gaps': self.product.gaps(),
                'unsupported_directives': [d['field'] for d in self.product.unsupported],
                'note': ('Product rules are applied as additional restrictions.'
                         if self.product.status == 'active' else
                         'Product is a DRAFT: its cards are readable but its rules were not applied. '
                         'Activate it after review to route on them.')}

    # --- input -------------------------------------------------------------
    def validate_profile(self, profile):
        if not isinstance(profile, dict):
            raise PipelineError('Missing customer profile.')
        clean = {}
        for field, spec in self.core.items():
            value = profile.get(field)
            if spec['type'] == 'number':
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
                        or not spec['min'] <= value <= spec['max']:
                    raise PipelineError(f'Invalid profile field: {field}')
            elif spec['type'] == 'boolean' and not isinstance(value, bool):
                raise PipelineError(f'Invalid profile field: {field}')
            elif spec['type'] == 'enum' and value not in spec['values']:
                raise PipelineError(f'Invalid profile field: {field}')
            clean[field] = value
        if profile.get('product') not in ('Life', 'Medical'):
            raise PipelineError('Product must be Life or Medical.')
        clean['product'] = profile['product']
        for field in ('name', 'sex', 'occupation'):
            value = profile.get(field)
            if value is not None and not isinstance(value, str):
                raise PipelineError(f'Invalid {field}.')
            clean[field] = value
        for field, spec in self.extra.items():
            value = profile.get(field)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value) or not spec['min'] <= value <= spec['max']):
                raise PipelineError(f'Invalid optional attribute: {field}')
            clean[field] = value
        return clean

    # --- stages ------------------------------------------------------------
    def ml_screen(self, profile, documents, supplied=None):
        """Use a supplied ML result, or call the configured adapter, or report absence."""
        if supplied is not None:
            return {**supplied, 'source': 'supplied with the case'}
        url = os.environ.get('UW_ML_URL', '')
        if not url:
            return {'available': False, 'source': 'none',
                    'error': 'No ML adapter configured (UW_ML_URL). Straight-through is disabled.'}
        import urllib.error, urllib.request
        payload = {'profile': profile, 'documents': documents or [], 'policy_revision': self.knowledge.rules_revision}
        headers = {'Content-Type': 'application/json'}
        key = os.environ.get('UW_ML_API_KEY', '')
        if key:
            headers['Authorization'] = 'Bearer ' + key
        request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.loads(response.read(1024 * 1024))
            if not isinstance(result, dict):
                raise ValueError
            return {**result, 'available': True, 'source': url}
        except Exception:
            # A model outage is a routing exception, never an acceptance.
            return {'available': False, 'source': url,
                    'error': 'ML adapter unreachable or invalid; the case takes the reasoning path.'}

    def straight_through(self, ml, rule_outcome, evidence):
        reasons = []
        if not ml.get('available'):
            reasons.append('No ML screen available.')
        else:
            if not ml.get('standard'):
                reasons.append('Model did not classify the case as standard.')
            if not ml.get('calibrated'):
                reasons.append('Model did not assert calibration evidence.')
            confidence = ml.get('confidence')
            if not isinstance(confidence, (int, float)) or confidence < STP_CONFIDENCE:
                reasons.append(f'Confidence below {STP_CONFIDENCE}.')
            if ml.get('ood') is not False:
                reasons.append('Input reported as out of distribution.')
            if not ml.get('evidence_complete'):
                reasons.append('Evidence not certified against the uploaded document hashes.')
        if rule_outcome != 'pass':
            reasons.append(f'Deterministic rules require: {rule_outcome}.')
        if evidence is not None and not evidence.get('complete', False):
            reasons.append('Extracted evidence is incomplete.')
        if evidence is not None and evidence.get('warnings'):
            reasons.append('Evidence warnings are unresolved.')
        return {'eligible': not reasons, 'blocked_by': reasons}

    def deterministic_recommendation(self, rule_outcome, fired_rules, requirements, missing, reconciliation=None):
        """The answer the pipeline gives with no model in the loop."""
        recommendation = 'request_evidence' if rule_outcome == 'request_evidence' else 'refer'
        reasons = [f"{r['id']}: {r.get('title', r['id'])} ({r['status']})" for r in fired_rules] or \
                  ['No routing rule fired; referred because no model assessment was produced.']
        for record in (reconciliation or {}).get('discrepancies', []):
            reasons.append(f"Declared {record['field']} {record['declared']} contradicted by {record['source']}; "
                           f"routed on {record['routed']}.")
        citations = sorted({c for r in fired_rules for c in r.get('cites', [])}) or \
                    [c['id'] for c in self.knowledge.cards if c['id'] == 'UW-GOV-01']
        return {
            'recommendation': recommendation,
            'explanation': ('Produced by the deterministic layer only. No language model was called, so this '
                            'is the routing outcome of the typed rules and the evidence grid, not a reasoned '
                            'assessment of the case.'),
            'reasons': reasons,
            'citations': citations,
            'missing_information': [r['requirement'] if isinstance(r, dict) else r for r in requirements] + missing,
            'produced_by': 'deterministic rules',
        }

    # --- orchestration -----------------------------------------------------
    def assess(self, case, offline=False):
        started = time.perf_counter()
        stages = []

        def stage(name, summary, t0):
            stages.append({'stage': name, 'summary': summary, 'ms': round((time.perf_counter() - t0) * 1000, 1)})

        t0 = time.perf_counter()
        profile = self.validate_profile(case.get('profile'))
        instructions = case.get('instructions') or ''
        if not isinstance(instructions, str) or len(instructions) > 3000:
            raise PipelineError('Instructions must be a string of at most 3000 characters.')
        if self.product is not None and self.product.status == 'active':
            declared = self.product.fields.get('product_type')
            if declared in ('Life', 'Medical') and profile['product'] != declared:
                raise PipelineError(f"Case is a {profile['product']} application but product "
                                    f"{self.product.fields.get('product_id')} covers {declared}.")
        documents = case.get('documents') or []
        manifest = [{k: d[k] for k in ('id', 'name', 'type', 'sha256') if k in d} for d in documents]
        evidence = case.get('evidence')
        stage('intake', f'Profile validated against the demo field catalogue; '
                        f'{len((evidence or {}).get("findings", []))} evidence finding(s) supplied, '
                        f'{len(documents)} document(s).', t0)

        # Documents supplied as files, no evidence supplied, a model configured: read them here.
        # The case bank does this itself with a cache; this is the path for a live case.
        readable = documents and all(d.get('path') for d in documents)
        can_extract = not offline and self.client is not None and getattr(self.client, 'configured', False)
        if evidence is None and readable and can_extract:
            t0 = time.perf_counter()
            try:
                evidence = extraction.extract(documents, profile, self.client)
                stage('extraction', f"{len(evidence['findings'])} finding(s) and {len(evidence['values'])} "
                                    f"typed value(s) read from {evidence['pages_processed']} page(s).", t0)
            except extraction.ExtractionError as error:
                evidence = {'complete': False, 'findings': [], 'values': [],
                            'warnings': [f'Document extraction failed: {error}']}
                stage('extraction', f'Failed: {error}', t0)
        elif evidence is None:
            evidence = {'complete': False, 'findings': [], 'values': [], 'warnings': [],
                        'note': 'No extracted evidence supplied to the pipeline.'}
        evidence_supplied = case.get('evidence') is not None or evidence.get('findings')

        t0 = time.perf_counter()
        effective_rules = self.config.apply_to_rules(self.knowledge.rules)
        reconciliation = reconcile.reconcile(profile, evidence, self.core, self.extra, effective_rules)
        routed = reconciliation['profile']
        if reconciliation['warnings']:
            evidence = {**evidence, 'warnings': list(evidence.get('warnings') or []) + reconciliation['warnings']}
        stage('reconciliation', reconciliation['summary'], t0)

        t0 = time.perf_counter()
        rule_results = rules.evaluate(routed, effective_rules)
        fired = rules.fired(rule_results)
        rule_outcome = rules.outcome(rule_results)
        if reconciliation['discrepancies'] and rules.OUTCOME_PRIORITY[rule_outcome] < rules.OUTCOME_PRIORITY['refer']:
            rule_outcome = 'refer'
        requirements = rules.evidence_requirements(routed, self.knowledge.evidence_grid)
        missing_fields = rules.missing_attributes(routed, self.extra)
        stage('routing rules', f'{len(fired)} of {len(rule_results)} rule(s) fired on the routed profile; '
                               f'outcome {rule_outcome}.', t0)

        t0 = time.perf_counter()
        ml = self.ml_screen(profile, manifest, case.get('ml'))
        stp = self.straight_through(ml, rule_outcome, evidence if evidence_supplied else None)
        stage('ml screen', ('Straight-through eligible.' if stp['eligible']
                            else 'Blocked: ' + '; '.join(stp['blocked_by'])), t0)

        if stp['eligible']:
            return self._result(profile, instructions, evidence, ml, stp, rule_results, fired, rule_outcome,
                                requirements, missing_fields, None, None,
                                {'recommendation': 'straight_through',
                                 'explanation': 'Every straight-through gate passed, so the reasoning path was '
                                                'not used. The case still carries the demo\'s own checks and an '
                                                'audit record.',
                                 'reasons': ['ML screen standard, calibrated, confident, in distribution and '
                                             'evidence-certified', 'No routing rule blocked the case'],
                                 'citations': ['UW-GOV-01'], 'missing_information': [],
                                 'produced_by': 'straight-through gates'},
                                [], stages, started, reconciliation=reconciliation)

        t0 = time.perf_counter()
        found = retrieval.retrieve(self.knowledge, routed, evidence, instructions,
                                   include_design=self.config.include_design_sources,
                                   limit=self.config.retrieval_source_limit,
                                   design_limit=self.config.design_source_limit)
        condition_evidence = rules.condition_requirements(found['cards'])
        stage('retrieval', f'{len(found["sources"])} source(s) selected from {len(self.knowledge)} cards.', t0)

        t0 = time.perf_counter()
        prior = []
        index = self.precedent_index() if self.config.precedent_limit > 0 else None
        if index is not None and len(index):
            matches = index.nearest(routed, self.config.precedent_limit, exclude=case.get('case_id'))
            room = max(self.config.retrieval_source_limit - len(found['sources']), 0)
            prior = [index.as_source(m) for m in matches][:room]
            found['sources'] = found['sources'] + prior
            found['trace'] = found['trace'] + [{'id': s['id'], 'title': s['title'], 'band': 4,
                                                'reason': f"nearest prior decision (distance {s['distance']})",
                                                'keyword_hits': [], 'collection': 'case bank'} for s in prior]
            if len(matches) > len(prior):
                found['notes'].append(f'{len(matches) - len(prior)} prior decision(s) omitted at the source cap.')
            found['notes'].append('Prior decisions are retrieved by declared-profile distance from the case bank, '
                                  'de-identified, as context for consistency. They are not rules and cannot '
                                  'support proposed terms.')
        stage('precedents', f'{len(prior)} prior decision(s) attached from a bank of '
                            f'{len(index) if index is not None else 0}.', t0)

        t0 = time.perf_counter()
        scan_targets = [('operator instructions', instructions)]
        scan_targets += [(f'evidence {f.get("id", "?")}', f.get('text', '')) for f in evidence.get('findings', [])]
        scan_targets += [(f'evidence quote {f.get("id", "?")}', f.get('quote', '')) for f in evidence.get('findings', [])]
        scan_targets += [(f'source {s["id"]}', s['excerpt']) for s in found['sources']]
        injection = validators.scan_for_injection(scan_targets)
        if injection:
            # Surfaced, never obeyed, and treated as an evidence warning so the conservative
            # rules downstream refuse to propose terms on it.
            evidence = {**evidence, 'warnings': list(evidence.get('warnings') or [])
                        + [f'Instruction-like text in {i["where"]} ({i["kind"]}). Not acted on; review the source.'
                           for i in injection]}
        stage('injection scan', f'{len(injection)} instruction-like passage(s) found in untrusted text.', t0)

        payload = {
            'profile': profile,
            'reconciliation': {'summary': reconciliation['summary'], 'records': reconciliation['records'],
                               'routed_profile': routed},
            'product': profile['product'],
            'ml_screen': {k: ml.get(k) for k in ('available', 'model', 'standard', 'confidence', 'calibrated',
                                                 'ood', 'evidence_complete', 'error') if k in ml},
            'straight_through': stp,
            'evidence': evidence,
            'sources': found['sources'],
            'deterministic_rules': [{k: r[k] for k in ('id', 'title', 'action', 'status', 'actual', 'blocks', 'cites')}
                                    for r in fired],
            'deterministic_outcome': rule_outcome,
            'routine_evidence_requirements': requirements,
            'impairment_evidence_requirements': condition_evidence,
            'unsupplied_optional_attributes': missing_fields,
            'retrieval_notes': found['notes'],
            'precedents_note': ('Sources whose id starts with PREC- are prior recorded decisions on similar '
                                'declared profiles. Use them for consistency and to notice what this book '
                                'usually asks for; they are not rules, they do not override any supplied '
                                'rule, and they cannot support proposed terms on their own.'),
        }

        t0 = time.perf_counter()
        if offline or not self.client or not getattr(self.client, 'configured', False):
            output = self.deterministic_recommendation(
                rule_outcome, fired, requirements['requirements'] + [c['requirement'] for c in condition_evidence],
                missing_fields, reconciliation)
            errors, iterations = [], 0
            stage('reasoning', 'Skipped: no model configured or offline requested. Deterministic output only.', t0)
        else:
            # Every number the model may write must already exist somewhere in what it was given.
            figures = validators.allowed_figures(profile, routed, evidence, found['sources'], instructions,
                                                 [r['value'] for r in effective_rules], requirements,
                                                 condition_evidence, [ml.get('confidence'), STP_CONFIDENCE])
            output, errors, iterations = self._reason(payload, instructions, evidence, found, rule_outcome, figures)
            stage('reasoning', f'{iterations} model call(s); '
                               f'{"validated" if not errors else str(len(errors)) + " validation failure(s)"}.', t0)

        return self._result(profile, instructions, evidence, ml, stp, rule_results, fired, rule_outcome,
                            requirements, missing_fields, found, condition_evidence, output, errors, stages,
                            started, injection, reconciliation)

    def _reason(self, payload, instructions, evidence, found, rule_outcome, figures=None):
        messages = prompts.build(payload, instructions, self.knowledge.summary(),
                                 variant=self.config.prompt_variant, guidance=self.config.guidance)
        errors = []
        for attempt in range(1, self.config.max_revisions + 2):
            try:
                output = self.client.complete(messages)
                errors = validators.validate_output(output, evidence, found['sources'], rule_outcome, self.knowledge,
                                                    figures=figures)
            except LLMError as error:
                if str(error) not in ('Model output was not valid JSON.', 'Model must return a JSON object.'):
                    raise
                output, errors = {}, [str(error)]
            if not errors:
                return {**output, 'produced_by': 'model, validated'}, [], attempt
            if attempt <= self.config.max_revisions:
                messages = messages + [{'role': 'assistant', 'content': json.dumps(output)},
                                       prompts.revision_message(errors)]
        return ({'recommendation': 'request_evidence' if rule_outcome == 'request_evidence' else 'refer',
                 'explanation': 'Model output failed validation after every permitted attempt. A human '
                                'underwriter must assess the case from the original evidence; the model '
                                'output was discarded.',
                 'reasons': errors, 'citations': [], 'missing_information': [],
                 'produced_by': 'guardrail fallback'}, errors, self.config.max_revisions + 1)

    def _result(self, profile, instructions, evidence, ml, stp, rule_results, fired, rule_outcome, requirements,
                missing_fields, found, condition_evidence, output, errors, stages, started, injection=None,
                reconciliation=None):
        recommendation = output.get('recommendation')
        # Final conservative pass: the deterministic layer has the last word, not the model.
        overrides = []
        if rule_outcome == 'request_evidence' and recommendation not in ('request_evidence', 'straight_through'):
            recommendation, _ = 'request_evidence', overrides.append(
                'Recommendation raised to request_evidence: a blocking rule needs data that was not supplied.')
        elif rule_outcome == 'refer' and recommendation == 'propose_terms':
            recommendation, _ = 'refer', overrides.append(
                'Recommendation raised to refer: a referral rule fired for this case.')
        if evidence.get('warnings') and recommendation == 'propose_terms':
            recommendation, _ = 'refer', overrides.append(
                'Recommendation raised to refer: evidence warnings are unresolved.')

        # Straight-through is the one route where no underwriter is asked to decide, and the
        # acceptance is executed by the demo's decision layer rather than here.
        straight = recommendation == 'straight_through'
        return {
            'recommendation': recommendation,
            'output': {**output, 'recommendation': recommendation},
            'guardrail_overrides': overrides,
            'validation_errors': errors or [],
            'human_decision_required': not straight,
            'decision_authority': ('Straight-through: acceptance rests on the gates above and the case remains '
                                   'auditable. This pipeline issues nothing.' if straight else
                                   'This pipeline recommends. An authorised underwriter decides, records the '
                                   'reason and owns the outcome.'),
            'profile': profile,
            'routed_profile': (reconciliation or {}).get('profile', profile),
            'reconciliation': {k: v for k, v in (reconciliation or {}).items() if k != 'profile'},
            'instructions': instructions,
            'ml_screen': ml,
            'straight_through': stp,
            'rules': {'outcome': rule_outcome, 'revision': self.knowledge.rules_revision,
                      'fired': fired, 'evaluated': len(rule_results)},
            'evidence': evidence,
            'evidence_requirements': {'routine': requirements, 'impairment': condition_evidence or [],
                                      'unsupplied_optional_attributes': missing_fields},
            'retrieval': {'sources': (found or {}).get('sources', []), 'trace': (found or {}).get('trace', []),
                          'omitted': (found or {}).get('omitted', []), 'notes': (found or {}).get('notes', [])},
            'injection_findings': injection or [],
            'knowledge': self.knowledge.summary(),
            'product': self.product_summary(),
            'config': self.config.to_dict(),
            'provenance': {'config_hash': self.config.fingerprint(),
                           'rules_revision': self.knowledge.rules_revision,
                           'product_id': (self.product_summary() or {}).get('product_id'),
                           'ml_model': ml.get('model'),
                           'llm': getattr(self.client, 'model', None) if output.get('produced_by', '').startswith('model') else None,
                           'produced_by': output.get('produced_by')},
            'stages': stages,
            'elapsed_ms': round((time.perf_counter() - started) * 1000, 1),
            'disclaimer': 'Demonstration pipeline over an illustrative knowledge base. Thresholds are demo '
                          'settings, not medical, actuarial or legal advice, and no policy is issued.',
        }


def assess(case, offline=False, **kwargs):
    return Pipeline(**kwargs).assess(case, offline=offline)
