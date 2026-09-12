"""Tests for product ingestion, the configurable surface, the case bank and the tuner.

These are the guardrail tests for the two new inputs a user can supply. A product spec
sheet and a case bank are both documents from outside: one describes a product, the other
records decisions, and neither may quietly widen what the pipeline is allowed to do. The
assertions below are mostly about what stays impossible.
"""
import json, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import casebank, evaluation, knowledge as knowledge_module, product as product_module  # noqa: E402
from pipeline.config import ConfigError, PipelineConfig  # noqa: E402
from pipeline.engine import Pipeline, PipelineError  # noqa: E402
from pipeline.prompts import EMPHASIS, system_message  # noqa: E402
from pipeline.product import ProductError, ProductSpec  # noqa: E402

SPEC_TEXT = """Product ID: test-term
Product Name: Test Term Plan
Product Type: Life
Currency: SGD
Minimum issue age: 18 years
Maximum entry age: 60 years
Maximum sum assured: SGD 2,000,000
Non-medical limit: SGD 500,000
Smoker definition: 12 months nicotine free
Maximum BMI: 36

Exclusions
- Suicide within 12 months
- War risks
"""

PROFILE = {'name': 'Fictional', 'age': 40, 'sex': 'Female', 'occupation': 'Accountant', 'product': 'Life',
           'cover': 600000, 'bmi': 24.0, 'smoker': False, 'condition': 'none', 'annualIncome': 120000,
           'existingCover': 0, 'termYears': 20, 'systolic': 120, 'diastolic': 78, 'hba1c': 5.3,
           'egfr': 95, 'ldl': 2.5, 'diagnosisYears': 0, 'hospitalisations': 0, 'reportAgeDays': 30}
EVIDENCE = {'complete': True, 'warnings': [],
            'findings': [{'id': 'DOC-1', 'source': 'r', 'page': 1, 'quote': 'q', 'text': 'No findings.'}]}


def write(directory, name, text):
    path = Path(directory) / name
    path.write_text(text)
    return path


class SpecParsing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = write(self.tmp.name, 'spec.md', SPEC_TEXT)

    def tearDown(self):
        self.tmp.cleanup()

    def spec(self):
        return product_module.from_document(self.path)

    def test_fields_are_read_with_the_line_they_came_from(self):
        spec = self.spec().validate()
        self.assertEqual(spec.fields['issue_age_max'], 60)
        self.assertEqual(spec.fields['max_cover'], 2000000)
        self.assertEqual(spec.fields['max_cover_without_medical'], 500000)
        self.assertIn('Maximum entry age', spec.report['issue_age_max']['quote'])

    def test_months_are_not_read_as_millions(self):
        self.assertEqual(self.spec().fields['smoker_definition_months'], 12)

    def test_a_list_section_is_collected(self):
        self.assertEqual(len(self.spec().fields['exclusions']), 2)

    def test_fields_the_document_does_not_state_are_reported_as_gaps(self):
        spec = self.spec().validate()
        self.assertIn('term_years_min', spec.gaps())
        self.assertNotIn('term_years_min', spec.fields)

    def test_a_spec_is_a_draft_until_someone_activates_it(self):
        spec = self.spec().validate()
        self.assertEqual(spec.status, 'draft')
        self.assertIsNone(spec.activation)

    def test_activation_records_a_person_and_a_reason(self):
        spec = self.spec().validate()
        with self.assertRaises(ProductError):
            spec.activate('', 'no name given')
        spec.activate('reviewer', 'checked against page 1')
        self.assertEqual(spec.status, 'active')
        self.assertEqual(spec.activation['activated_by'], 'reviewer')

    def test_an_inconsistent_spec_is_rejected(self):
        path = write(self.tmp.name, 'bad.md', 'Product ID: bad\nProduct Type: Life\n'
                                              'Minimum issue age: 60\nMaximum entry age: 30\n')
        with self.assertRaises(ProductError):
            product_module.from_document(path).validate()

    def test_an_unknown_product_type_is_rejected(self):
        path = write(self.tmp.name, 'odd.md', 'Product ID: odd\nProduct Type: Pet Insurance\n')
        with self.assertRaises(ProductError):
            product_module.from_document(path).validate()


class GeneratedConfiguration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.spec = product_module.from_document(write(self.tmp.name, 'spec.md', SPEC_TEXT)).validate()

    def tearDown(self):
        self.tmp.cleanup()

    def test_generated_rules_can_only_restrict(self):
        for rule in self.spec.to_rules():
            self.assertIn(rule['action'], ('refer', 'request_evidence', 'note'), rule['id'])

    def test_generated_rules_cite_a_generated_card(self):
        card_ids = {c['id'] for c in self.spec.to_cards()}
        for rule in self.spec.to_rules():
            self.assertTrue(set(rule['cites']) & (card_ids | {'UW-EVID-01', 'UW-FIN-03', 'UW-BUILD-01'}))

    def test_generated_cards_pass_the_knowledge_validator(self):
        base = knowledge_module.load()
        merged = base.with_cards(self.spec.to_cards())
        self.assertEqual(len(merged), len(base) + len(self.spec.to_cards()))

    def test_a_spec_demanding_automatic_acceptance_is_recorded_not_applied(self):
        path = write(self.tmp.name, 'auto.json', json.dumps({
            'product_id': 'auto-accept', 'product_type': 'Life', 'max_cover': 1000000,
            'auto_accept_cover': 500000}))
        spec = product_module.from_document(path).validate()
        self.assertTrue(spec.unsupported)
        self.assertEqual(spec.unsupported[0]['field'], 'auto_accept_cover')
        for rule in spec.to_rules():
            self.assertIn(rule['action'], ('refer', 'request_evidence', 'note'))

    def test_a_rule_with_an_accepting_action_cannot_be_merged(self):
        base = knowledge_module.load()
        with self.assertRaises(knowledge_module.KnowledgeError):
            base.with_rules([{'id': 'PRD-X-1', 'action': 'approve', 'field': 'age',
                              'operator': 'gt', 'value': 1, 'scope': 'Both'}])

    def test_a_draft_product_does_not_route_cases(self):
        draft = Pipeline(client=None, config=PipelineConfig(), product=self.spec)
        active = Pipeline(client=None, config=PipelineConfig(),
                          product=ProductSpec.from_dict({**self.spec.to_dict(), 'status': 'active'}))
        self.assertEqual(draft.product_summary()['rules_applied'], 0)
        self.assertGreater(active.product_summary()['rules_applied'], 0)

    def test_an_active_product_adds_restrictions(self):
        active = ProductSpec.from_dict({**self.spec.to_dict(), 'status': 'active'})
        pipeline = Pipeline(client=None, config=PipelineConfig(), product=active)
        result = pipeline.assess({'profile': PROFILE, 'evidence': EVIDENCE}, offline=True)
        fired = [r['id'] for r in result['rules']['fired']]
        self.assertTrue(any(r.startswith('PRD-TEST-TERM') for r in fired),
                        'cover above the product non-medical limit should fire a product rule')

    def test_a_case_for_another_product_line_is_refused(self):
        active = ProductSpec.from_dict({**self.spec.to_dict(), 'status': 'active'})
        pipeline = Pipeline(client=None, config=PipelineConfig(), product=active)
        with self.assertRaises(PipelineError):
            pipeline.assess({'profile': {**PROFILE, 'product': 'Medical'}, 'evidence': EVIDENCE}, offline=True)


class Configuration(unittest.TestCase):
    def setUp(self):
        self.rules = knowledge_module.load().rules

    def test_defaults_validate(self):
        PipelineConfig().validate(self.rules)

    def test_a_mandatory_rule_cannot_be_switched_off(self):
        mandatory = next(r['id'] for r in self.rules if r.get('mandatory'))
        with self.assertRaises(ConfigError):
            PipelineConfig(rule_overrides={mandatory: {'enabled': False}}).validate(self.rules)

    def test_an_action_may_tighten_but_never_loosen(self):
        rule = next(r for r in self.rules if r['action'] == 'request_evidence')
        PipelineConfig(rule_overrides={rule['id']: {'action': 'request_evidence'}}).validate(self.rules)
        with self.assertRaises(ConfigError):
            PipelineConfig(rule_overrides={rule['id']: {'action': 'note'}}).validate(self.rules)

    def test_a_threshold_cannot_leave_its_declared_bounds(self):
        rule = next(r for r in self.rules if r.get('tunable'))
        PipelineConfig(rule_overrides={rule['id']: {'value': rule['tunable']['max']}}).validate(self.rules)
        with self.assertRaises(ConfigError):
            PipelineConfig(rule_overrides={rule['id']: {'value': rule['tunable']['max'] + 1}}).validate(self.rules)

    def test_a_rule_without_declared_bounds_is_fixed(self):
        rule = next(r for r in self.rules if not r.get('tunable'))
        with self.assertRaises(ConfigError):
            PipelineConfig(rule_overrides={rule['id']: {'value': 1}}).validate(self.rules)

    def test_validation_controls_cannot_be_switched_off(self):
        with self.assertRaises(ConfigError):
            PipelineConfig(require_internal_citation_for_terms=False).validate(self.rules)
        with self.assertRaises(ConfigError):
            PipelineConfig(warnings_block_terms=False).validate(self.rules)

    def test_the_revision_budget_is_bounded(self):
        with self.assertRaises(ConfigError):
            PipelineConfig(max_revisions=5).validate(self.rules)

    def test_an_unknown_prompt_variant_is_rejected(self):
        with self.assertRaises(ConfigError):
            PipelineConfig(prompt_variant='do-whatever').validate(self.rules)

    def test_every_prompt_variant_keeps_every_mandatory_control(self):
        baseline = system_message()['content']
        controls = [line for line in baseline.split('\n') if line and line[0].isdigit()]
        for variant in EMPHASIS:
            content = system_message(variant=variant)['content']
            for control in controls:
                self.assertIn(control, content, f'{variant} lost a control')

    def test_disabling_a_rule_removes_it_from_evaluation(self):
        optional = next(r for r in self.rules if not r.get('mandatory'))
        config = PipelineConfig(rule_overrides={optional['id']: {'enabled': False}}).validate(self.rules)
        effective = {r['id'] for r in config.apply_to_rules(self.rules)}
        self.assertNotIn(optional['id'], effective)

    def test_pruning_drops_overrides_that_do_nothing(self):
        rule = next(r for r in self.rules if r.get('tunable') and not r.get('mandatory'))
        config = PipelineConfig(rule_overrides={rule['id']: {'value': rule['tunable']['min'],
                                                             'enabled': False}})
        self.assertEqual(config.prune().rule_overrides[rule['id']], {'enabled': False})

    def test_a_diff_lists_only_real_changes(self):
        base = PipelineConfig()
        changed = base.replace(retrieval_source_limit=12)
        self.assertEqual(base.diff(changed), {'retrieval_source_limit': {'from': 30, 'to': 12}})


class CaseBank(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.bank = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def add(self, case_id, **overrides):
        directory = self.bank / case_id
        directory.mkdir(parents=True, exist_ok=True)
        case = {'profile': PROFILE, 'evidence': EVIDENCE,
                'human': {'outcome': 'standard', 'decided_by': 'an underwriter'}, **overrides}
        (directory / 'case.json').write_text(json.dumps(case))
        return directory

    def test_a_case_without_a_recorded_decider_is_rejected(self):
        self.add('CASE-1', human={'outcome': 'standard'})
        with self.assertRaises(casebank.CaseBankError):
            casebank.load_bank(self.bank)

    def test_an_unknown_outcome_is_rejected(self):
        self.add('CASE-1', human={'outcome': 'accepted-ish', 'decided_by': 'x'})
        with self.assertRaises(casebank.CaseBankError):
            casebank.load_bank(self.bank)

    def test_a_case_with_no_documents_still_loads_when_evidence_is_supplied(self):
        self.add('CASE-1')
        cases, _ = casebank.load_bank(self.bank)
        self.assertEqual(len(cases), 1)
        self.assertFalse(cases[0].needs_extraction())

    def test_a_file_that_is_not_a_document_is_refused_by_content_not_extension(self):
        directory = self.add('CASE-1')
        (directory / 'documents').mkdir()
        (directory / 'documents' / 'report.pdf').write_bytes(b'this is not a pdf')
        with self.assertRaises(casebank.CaseBankError):
            casebank.load_bank(self.bank)

    def test_cached_extraction_is_reused_only_while_the_documents_match(self):
        directory = self.add('CASE-1', evidence=None)
        (directory / 'extracted.json').write_text(json.dumps(
            {'evidence': EVIDENCE, 'extracted_by': 'a model', 'document_fingerprint': 'stale'}))
        cases, _ = casebank.load_bank(self.bank)
        self.assertTrue(cases[0].needs_extraction())
        self.assertIn('stale', cases[0].evidence_source)

    def test_extraction_needs_a_configured_model(self):
        self.add('CASE-1', evidence=None)
        cases, _ = casebank.load_bank(self.bank)
        with self.assertRaises(casebank.CaseBankError):
            casebank.extract_evidence(cases[0], None)

    def test_the_split_is_deterministic_and_stratified(self):
        for index in range(8):
            outcome = 'standard' if index % 2 else 'refer'
            self.add(f'CASE-{index}', human={'outcome': outcome, 'decided_by': 'x'})
        cases, _ = casebank.load_bank(self.bank)
        first = casebank.split(cases, 0.25, seed=3)
        second = casebank.split(cases, 0.25, seed=3)
        self.assertEqual([c.case_id for c in first[1]], [c.case_id for c in second[1]])
        self.assertEqual({c.human['outcome'] for c in first[1]}, {'standard', 'refer'})


class DocumentExtraction(unittest.TestCase):
    """The PDF path, exercised with a stubbed vision model.

    Rendering, page counting and finding validation all run for real; only the model call
    is scripted, so a finding that cites a page the document does not have still fails here.
    """
    @classmethod
    def setUpClass(cls):
        try:
            casebank._pymupdf()
        except casebank.CaseBankError as error:
            raise unittest.SkipTest(str(error))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self.tmp.name) / 'CASE-X'
        (self.directory / 'documents').mkdir(parents=True)
        fitz = casebank._pymupdf()
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((60, 80), 'SYNTHETIC REPORT: HbA1c 6.1%', fontsize=12, fontname='helv')
        doc.save(self.directory / 'documents' / 'report.pdf')
        doc.close()
        (self.directory / 'case.json').write_text(json.dumps(
            {'profile': PROFILE, 'human': {'outcome': 'standard', 'decided_by': 'an underwriter'}}))

    def tearDown(self):
        self.tmp.cleanup()

    def case(self):
        cases, _ = casebank.load_bank(self.directory.parent)
        return cases[0]

    def client(self, response):
        return type('Stub', (), {'configured': True, 'model': 'stub-vision',
                                 'complete': lambda self, messages: response})()

    def good_response(self):
        return {'complete': True, 'warnings': [],
                'findings': [{'id': 'DOC-1', 'page': 1, 'quote': 'HbA1c 6.1%',
                              'text': 'Report records HbA1c 6.1 percent.'}]}

    def test_a_document_is_hashed_and_typed_by_content(self):
        case = self.case()
        self.assertEqual(len(case.documents), 1)
        self.assertEqual(case.documents[0]['type'], 'application/pdf')
        self.assertEqual(len(case.documents[0]['sha256']), 64)

    def test_pages_are_rendered_and_sent_with_their_identifiers(self):
        pages = casebank.render_pages(self.case().documents)
        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0]['doc_id'], 'DOC-1')
        self.assertTrue(pages[0]['data_uri'].startswith('data:image/png;base64,'))

    def test_extraction_result_is_cached_against_the_document_hashes(self):
        case = self.case()
        self.assertTrue(case.needs_extraction())
        evidence = casebank.extract_evidence(case, self.client(self.good_response()))
        case.store_evidence(evidence, 'stub-vision')
        reloaded = self.case()
        self.assertFalse(reloaded.needs_extraction())
        self.assertIn('stub-vision', reloaded.evidence_source)

    def test_a_finding_citing_a_page_that_does_not_exist_is_rejected(self):
        bad = self.good_response()
        bad['findings'][0]['page'] = 7
        with self.assertRaises(casebank.CaseBankError):
            casebank.extract_evidence(self.case(), self.client(bad))

    def test_a_finding_citing_an_unknown_document_is_rejected(self):
        bad = self.good_response()
        bad['findings'][0]['id'] = 'DOC-9'
        with self.assertRaises(casebank.CaseBankError):
            casebank.extract_evidence(self.case(), self.client(bad))

    def test_a_finding_without_a_source_excerpt_is_rejected(self):
        bad = self.good_response()
        bad['findings'][0].pop('quote')
        with self.assertRaises(casebank.CaseBankError):
            casebank.extract_evidence(self.case(), self.client(bad))


class Scoring(unittest.TestCase):
    def test_being_less_cautious_than_the_underwriter_is_unsafe(self):
        self.assertEqual(evaluation.classify('straight_through', 'decline'), 'unsafe')
        self.assertEqual(evaluation.classify('propose_terms', 'more_evidence'), 'unsafe')

    def test_being_more_cautious_is_a_different_kind_of_miss(self):
        self.assertEqual(evaluation.classify('request_evidence', 'standard'), 'conservative')

    def test_an_exact_or_acceptable_answer_is_credited(self):
        self.assertEqual(evaluation.classify('straight_through', 'standard'), 'match')
        self.assertEqual(evaluation.classify('refer', 'terms'), 'acceptable')

    def test_a_decline_is_matched_by_a_referral_since_nothing_here_declines(self):
        self.assertEqual(evaluation.classify('refer', 'decline'), 'match')

    def test_unsafe_and_conservative_misses_are_never_averaged_together(self):
        rows = [{'case_id': 'a', 'product': 'Life', 'human_outcome': 'standard', 'ideal': 'straight_through',
                 'predicted': 'request_evidence', 'verdict': 'conservative', 'credit': 0.5, 'cohort': None,
                 'rules_fired': [], 'evidence_source': 'x'},
                {'case_id': 'b', 'product': 'Life', 'human_outcome': 'decline', 'ideal': 'refer',
                 'predicted': 'straight_through', 'verdict': 'unsafe', 'credit': 0.0, 'cohort': None,
                 'rules_fired': [], 'evidence_source': 'x'}]
        summary = evaluation.summarise(rows)
        self.assertEqual(summary['unsafe_disagreements'], 1)
        self.assertEqual(summary['conservative_disagreements'], 1)

    def test_a_small_bank_is_flagged_in_the_rendered_report(self):
        rows = [{'case_id': 'a', 'product': 'Life', 'human_outcome': 'standard', 'ideal': 'straight_through',
                 'predicted': 'straight_through', 'verdict': 'match', 'credit': 1.0, 'cohort': None,
                 'rules_fired': [], 'evidence_source': 'x'}]
        self.assertIn('indicative', evaluation.render(evaluation.summarise(rows)))


class Tuning(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(ROOT))
        import tune
        self.tune = tune
        self.rules = knowledge_module.load().rules

    def test_offline_search_leaves_prompt_and_retrieval_alone(self):
        offline = {k['kind'] for k in self.tune.search_space(self.rules, offline=True)}
        live = {k['kind'] for k in self.tune.search_space(self.rules, offline=False)}
        self.assertNotIn('field', offline)
        self.assertIn('field', live)

    def test_the_search_space_never_offers_a_mandatory_rule_for_disabling(self):
        mandatory = {r['id'] for r in self.rules if r.get('mandatory')}
        knobs = self.tune.search_space(self.rules, offline=False, allow_disable=True)
        self.assertTrue(any(k['kind'] == 'rule_enabled' for k in knobs))
        for knob in knobs:
            if knob['kind'] == 'rule_enabled':
                self.assertNotIn(knob['rule'], mandatory)

    def test_disabling_a_rule_is_not_searched_unless_asked_for(self):
        kinds = {k['kind'] for k in self.tune.search_space(self.rules, offline=False)}
        self.assertNotIn('rule_enabled', kinds)

    def test_folds_hold_every_case_out_exactly_once_and_keep_outcomes_spread(self):
        cases, _ = casebank.load_bank(strict=False)
        folds = casebank.folds(cases, k=5, seed=3)
        self.assertEqual(len(folds), 5)
        held = [c.case_id for _, holdout in folds for c in holdout]
        self.assertEqual(sorted(held), sorted(c.case_id for c in cases))
        for train, holdout in folds:
            self.assertFalse({c.case_id for c in train} & {c.case_id for c in holdout})
            self.assertEqual(len(train) + len(holdout), len(cases))
        # the commonest outcome lands in every fold rather than piling into one
        commonest = max({c.human['outcome'] for c in cases},
                        key=lambda o: sum(1 for c in cases if c.human['outcome'] == o))
        for _, holdout in folds:
            self.assertTrue(any(c.human['outcome'] == commonest for c in holdout))

    def test_a_change_resting_on_too_few_cases_is_not_carried(self):
        """One case improving is one case's opinion: the gate is exercised with a scripted evaluation."""
        import types
        from unittest import mock
        rule = next(r for r in self.rules if r.get('tunable'))
        fake_pipeline = types.SimpleNamespace(knowledge=types.SimpleNamespace(rules=[rule]), using=lambda cfg: cfg)

        def scripted(config, cases, offline=True, judge=None):
            override = config.rule_overrides.get(rule['id'], {}).get('value')
            rows = [{'case_id': f'C{i}', 'product': 'Life', 'human_outcome': 'terms', 'ideal': 'propose_terms',
                     'predicted': 'refer', 'verdict': 'acceptable', 'credit': 0.8, 'rules_fired': []}
                    for i in range(10)]
            if override is not None and override != rule['value']:
                rows[0].update(predicted='propose_terms', verdict='match', credit=1.0)   # exactly one case better
            return evaluation.summarise(rows)

        with mock.patch.object(self.tune.evaluation, 'evaluate', scripted):
            strict_config, _, strict_trials, strict_accepted = self.tune.search(
                fake_pipeline, [], PipelineConfig(), offline=True, min_delta=0.01, verbose=False, min_support=3)
            self.assertEqual(strict_accepted, [])
            self.assertEqual(strict_config.diff(PipelineConfig()), {})
            self.assertTrue(any('improves only 1 case' in t.get('rejected', '') for t in strict_trials))
            _, _, _, loose_accepted = self.tune.search(
                fake_pipeline, [], PipelineConfig(), offline=True, min_delta=0.01, verbose=False, min_support=1)
            self.assertEqual(len(loose_accepted), 1)
            self.assertEqual(loose_accepted[0]['improved'], 1)

    def test_support_counts_cases_by_credit_not_by_average(self):
        before = {'rows': [{'case_id': 'A', 'credit': 0.5}, {'case_id': 'B', 'credit': 1.0},
                           {'case_id': 'C', 'credit': 0.8}]}
        after = {'rows': [{'case_id': 'A', 'credit': 1.0}, {'case_id': 'B', 'credit': 0.0},
                          {'case_id': 'C', 'credit': 0.8}]}
        backing = evaluation.support(before, after)
        self.assertEqual(backing, {'improved': ['A'], 'worsened': ['B']})

    def test_rule_evidence_reports_what_humans_decided_where_a_rule_fired(self):
        rows = [{'case_id': 'A', 'rules_fired': ['R1'], 'human_outcome': 'terms', 'verdict': 'conservative'},
                {'case_id': 'B', 'rules_fired': ['R1', 'R2'], 'human_outcome': 'refer', 'verdict': 'match'},
                {'case_id': 'C', 'rules_fired': [], 'human_outcome': 'standard', 'verdict': 'match'}]
        evidence = evaluation.rule_evidence(rows)
        self.assertEqual(evidence['R1']['fired'], 2)
        self.assertEqual(evidence['R1']['human_outcomes'], {'terms': 1, 'refer': 1})
        self.assertEqual(evidence['R2']['cases'], ['B'])
        self.assertNotIn('R3', evidence)

    def test_only_changes_found_in_at_least_half_the_folds_are_stable(self):
        records = [{'fold': 1, 'diff': {'rule X': {'from': None, 'to': {'value': 7.5}}}},
                   {'fold': 2, 'diff': {'rule X': {'from': None, 'to': {'value': 7.0}}}},
                   {'fold': 3, 'diff': {}},
                   {'fold': 4, 'diff': {'rule Y': {'from': None, 'to': {'value': 45}}}}]
        diff = {'rule X': {'from': None, 'to': {'value': 7.5}}, 'rule Y': {'from': None, 'to': {'value': 45}}}
        stab = self.tune.stability(records, diff)
        self.assertTrue(stab['rule X']['stable'])
        self.assertEqual(stab['rule X']['folds_adopting'], 2)
        self.assertEqual(stab['rule X']['folds_same_value'], 1)
        self.assertEqual(stab['rule X']['values_seen'], ['{"value": 7.0}', '{"value": 7.5}'])
        self.assertFalse(stab['rule Y']['stable'])

    def test_restrict_carries_only_the_named_changes(self):
        baseline = PipelineConfig()
        tuned = PipelineConfig(retrieval_source_limit=10,
                               rule_overrides={'RTE-EGFR-60': {'value': 45}, 'RTE-HBA1C-65': {'value': 7.5}})
        kept = self.tune.restrict(baseline, tuned, ['rule RTE-EGFR-60'])
        self.assertEqual(baseline.diff(kept), {'rule RTE-EGFR-60': {'from': None, 'to': {'value': 45}}})
        kept = self.tune.restrict(baseline, tuned, ['retrieval_source_limit'])
        self.assertEqual(list(baseline.diff(kept)), ['retrieval_source_limit'])

    def test_cross_validation_measures_each_fold_on_cases_it_never_saw(self):
        cases, _ = casebank.load_bank(strict=False)
        usable = [c for c in cases if c.evidence is not None]
        pipeline = Pipeline(client=None, config=PipelineConfig())
        records = self.tune.cross_validate(pipeline, usable, PipelineConfig(), offline=True, min_delta=0.01,
                                           min_support=1, passes=1, k=3, seed=5, verbose=False)
        self.assertEqual(len(records), 3)
        self.assertEqual(sum(r['holdout'] for r in records), len(usable))
        pooled = self.tune.pooled_holdout(records)
        self.assertEqual(pooled['folds'], 3)
        self.assertEqual(len(pooled['per_fold']), 3)
        self.assertIn('severity_score', pooled)

    def test_threshold_candidates_stay_inside_the_declared_bounds(self):
        for rule in self.rules:
            if not rule.get('tunable'):
                continue
            for value in self.tune.threshold_candidates(rule):
                self.assertGreaterEqual(value, rule['tunable']['min'])
                self.assertLessEqual(value, rule['tunable']['max'])

    def test_every_configuration_the_search_can_reach_is_valid(self):
        config = PipelineConfig()
        for knob in self.tune.search_space(self.rules, offline=False):
            for value in knob['values']:
                self.tune.with_knob(config, knob, value).validate(self.rules)

    def test_applying_a_proposal_requires_a_person_and_a_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            (artifacts / 'proposal.json').write_text(json.dumps(
                {'diff': {'retrieval_source_limit': {'from': 30, 'to': 10}},
                 'proposed_config': PipelineConfig(retrieval_source_limit=10).to_dict(),
                 'comparison': {'holdout': {'severity_score': {'delta': 0.1},
                                            'unsafe_disagreements': {'delta': 0}}},
                 'generated_at': 'now', 'bank': {}}))
            args = type('Args', (), {'artifacts': artifacts, 'by': None, 'reason': None,
                                     'accept_risk': None, 'config': artifacts / 'config.json'})()
            with self.assertRaises(SystemExit):
                self.tune.apply_proposal(args)

    def test_a_proposal_the_holdout_did_not_confirm_needs_the_risk_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp)
            (artifacts / 'proposal.json').write_text(json.dumps(
                {'diff': {'retrieval_source_limit': {'from': 30, 'to': 10}},
                 'proposed_config': PipelineConfig(retrieval_source_limit=10).to_dict(),
                 'comparison': {'holdout': {'severity_score': {'delta': -0.05},
                                            'unsafe_disagreements': {'delta': 0}}},
                 'generated_at': 'now', 'bank': {}}))
            args = type('Args', (), {'artifacts': artifacts, 'by': 'tester', 'reason': 'wanted it',
                                     'accept_risk': None, 'config': artifacts / 'config.json'})()
            with self.assertRaises(SystemExit) as caught:
                self.tune.apply_proposal(args)
            self.assertIn('--accept-risk', str(caught.exception))


if __name__ == '__main__':
    unittest.main()
