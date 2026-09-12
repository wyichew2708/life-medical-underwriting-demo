"""Prior decisions as context, richer recorded decisions, and scoring the reasoning itself."""
import json, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import casebank, evaluation, extraction, precedents  # noqa: E402
from pipeline.config import ConfigError, PipelineConfig  # noqa: E402
from pipeline.engine import Pipeline  # noqa: E402
from pipeline.judge import Judge  # noqa: E402

CORE, EXTRA = extraction.catalogue()


def profile(**overrides):
    base = {'name': 'Someone', 'age': 44, 'sex': 'Female', 'occupation': 'Teacher', 'product': 'Life',
            'cover': 400000, 'bmi': 26.0, 'smoker': False, 'condition': 'none', 'annualIncome': 90000,
            'existingCover': 0, 'termYears': 20, 'systolic': 124, 'diastolic': 80, 'hba1c': 5.4, 'egfr': 98,
            'ldl': 2.4, 'diagnosisYears': 0, 'hospitalisations': 0, 'reportAgeDays': 40}
    return {**base, **overrides}


class FakeCase:
    def __init__(self, case_id, prof, human):
        self.case_id, self.profile, self.human = case_id, prof, human


def bank():
    return [FakeCase('CASE-A', profile(), {'outcome': 'standard', 'decided_by': 'Ann Person', 'decided_at': '2026-01-02',
                                           'rationale': 'Clean profile, standard rates.'}),
            FakeCase('CASE-B', profile(hba1c=7.4, condition='controlled'),
                     {'outcome': 'terms', 'decided_by': 'Bob Person', 'rating': '+50%',
                      'rationale': 'HbA1c 7.4 with a controlled condition.'}),
            FakeCase('CASE-C', profile(smoker=True, cover=1500000),
                     {'outcome': 'refer', 'decided_by': 'Cy Person', 'evidence_requested': ['cotinine test']}),
            FakeCase('CASE-D', profile(product='Medical'), {'outcome': 'standard', 'decided_by': 'Di Person'})]


class PrecedentRetrieval(unittest.TestCase):
    def setUp(self):
        self.index = precedents.PrecedentIndex(bank(), CORE, EXTRA)

    def test_nearest_is_by_declared_profile_distance_on_the_same_product_line(self):
        matches = self.index.nearest(profile(hba1c=7.0, condition='controlled'), k=2)
        self.assertEqual([m['entry']['case_id'] for m in matches], ['CASE-B', 'CASE-A'])
        self.assertTrue(all(m['entry']['product'] == 'Life' for m in matches))

    def test_a_case_is_never_its_own_precedent(self):
        matches = self.index.nearest(profile(), k=3, exclude='CASE-A')
        self.assertNotIn('CASE-A', [m['entry']['case_id'] for m in matches])
        identical = self.index.nearest(profile(), k=1)
        self.assertTrue(identical[0]['identical_profile'])

    def test_sources_are_de_identified_and_typed_as_context(self):
        source = self.index.as_source(self.index.nearest(profile(hba1c=7.4, condition='controlled'), k=1)[0])
        self.assertEqual(source['type'], 'rag')
        self.assertTrue(source['id'].startswith('PREC-'))
        for secret in ('Someone', 'Bob Person', 'Female', 'Teacher'):
            self.assertNotIn(secret, source['excerpt'])
        self.assertIn('Rating: +50%', source['excerpt'])
        self.assertIn('HbA1c 7.4', source['excerpt'])
        self.assertIn('age 36-50', source['excerpt'])

    def test_a_limit_of_zero_attaches_nothing(self):
        self.assertEqual(self.index.nearest(profile(), k=0), [])

    def test_the_engine_attaches_precedents_within_the_source_cap(self):
        pipeline = Pipeline(client=None, config=PipelineConfig(precedent_limit=2), precedents=self.index)
        case = {'case_id': 'CASE-A', 'profile': profile(condition='controlled'),
                'evidence': {'complete': True, 'findings': [], 'warnings': [], 'values': []}}
        result = pipeline.assess(case, offline=True)
        prior = [s for s in result['retrieval']['sources'] if s['id'].startswith('PREC-')]
        self.assertEqual(len(prior), 2)
        self.assertNotIn('PREC-CASE-A', [s['id'] for s in prior])
        self.assertLessEqual(len(result['retrieval']['sources']), 30)
        self.assertTrue(any(t['band'] == 4 for t in result['retrieval']['trace']))
        self.assertEqual(result['provenance']['config_hash'], PipelineConfig(precedent_limit=2).fingerprint())
        off = Pipeline(client=None, config=PipelineConfig(precedent_limit=0), precedents=self.index)
        result = off.assess(case, offline=True)
        self.assertFalse([s for s in result['retrieval']['sources'] if s['id'].startswith('PREC-')])

    def test_the_precedent_limit_is_bounded(self):
        with self.assertRaises(ConfigError):
            PipelineConfig(precedent_limit=9).validate()

    def test_a_precedent_cannot_be_the_internal_citation_terms_need(self):
        from pipeline import validators
        sources = [{'id': 'PREC-CASE-B', 'type': 'rag', 'excerpt': 'Prior decision: terms'}]
        output = {'recommendation': 'propose_terms', 'explanation': 'Consistent with a prior decision.',
                  'reasons': ['precedent'], 'citations': ['PREC-CASE-B'], 'missing_information': []}
        errors = validators.validate_output(output, {'complete': True}, sources, 'pass')
        self.assertTrue(any('internal rule' in e for e in errors))


class RicherDecisions(unittest.TestCase):
    def test_optional_detail_is_typed_and_lists_accept_semicolons(self):
        human = casebank.normalise_human({'outcome': 'more_evidence', 'decided_by': 'x',
                                          'evidence_requested': 'HbA1c; GP report', 'rating': '+25%',
                                          'later_outcome': 'terms'}, 'CASE-1')
        self.assertEqual(human['evidence_requested'], ['HbA1c', 'GP report'])
        self.assertEqual(human['later_outcome'], {'outcome': 'terms'})
        self.assertEqual(human['rules_cited'], [])

    def test_a_later_outcome_must_be_a_recognised_outcome(self):
        with self.assertRaises(casebank.CaseBankError):
            casebank.normalise_human({'outcome': 'refer', 'decided_by': 'x', 'later_outcome': 'approved'}, 'CASE-1')
        with self.assertRaises(casebank.CaseBankError):
            casebank.normalise_human({'outcome': 'refer', 'decided_by': 'x', 'rationale': 'x' * 2001}, 'CASE-1')

    def test_the_sample_bank_records_rationale_and_requested_evidence(self):
        cases, _ = casebank.load_bank(strict=False)
        asked = [c for c in cases if c.human['outcome'] == 'more_evidence']
        self.assertTrue(asked)
        self.assertTrue(all(c.human['evidence_requested'] for c in asked))
        self.assertTrue(all(c.human.get('rationale') for c in cases))
        self.assertIn('case_id', cases[0].to_pipeline_case())


class ReasoningQuality(unittest.TestCase):
    def result(self, explanation, reasons, missing, citations, fired):
        return {'output': {'recommendation': 'request_evidence', 'explanation': explanation, 'reasons': reasons,
                           'missing_information': missing, 'citations': citations},
                'rules': {'fired': fired}, 'profile': profile(hba1c=7.2), 'routed_profile': profile(hba1c=7.2),
                'evidence': {'findings': []}, 'retrieval': {'sources': []}, 'instructions': '',
                'evidence_requirements': {}}

    def test_evidence_request_matching_is_token_based_and_symmetric(self):
        match = evaluation.evidence_match(['repeat HbA1c', 'GP report covering the declared condition'],
                                          ['Repeat HbA1c within 3 months', 'Paramedical examination'])
        self.assertEqual(match['recall'], 0.5)
        self.assertEqual(match['precision'], 0.5)
        self.assertEqual(match['unmatched'], ['GP report covering the declared condition'])
        self.assertIsNone(evaluation.evidence_match([], ['anything']))

    def test_quality_rewards_naming_the_rule_and_citing_its_card(self):
        fired = [{'id': 'RTE-HBA1C-65', 'title': 'HbA1c at or above 6.5', 'cites': ['UW-DM-01'], 'value': 6.5}]
        case = FakeCase('C', profile(), {'outcome': 'more_evidence', 'evidence_requested': ['repeat HbA1c'],
                                         'rationale': 'HbA1c 7.2 needs a repeat test before terms.'})
        good = evaluation.quality(self.result('RTE-HBA1C-65 fired on HbA1c 7.2; a repeat test is needed.',
                                              ['HbA1c 7.2 at or above 6.5'], ['repeat HbA1c'], ['UW-DM-01'], fired),
                                  case)
        poor = evaluation.quality(self.result('The case needs more information.', ['unclear'],
                                              ['medical records'], ['UW-GOV-01'], fired), case)
        self.assertEqual(good['rules_named'], 1.0)
        self.assertEqual(good['citations_cover_rules'], 1.0)
        self.assertEqual(good['evidence_request_f1'], 1.0)
        self.assertGreater(good['rationale_overlap'], poor['rationale_overlap'])
        self.assertGreater(good['score'], poor['score'])
        self.assertEqual(poor['rules_named'], 0.0)

    def test_an_invented_figure_costs_quality(self):
        fired = []
        case = FakeCase('C', profile(), {'outcome': 'refer'})
        invented = evaluation.quality(self.result('HbA1c of 9.9 is far above range.', ['x'], [], ['UW-GOV-01'], fired),
                                      case)
        self.assertEqual(invented['figures_supported'], 0.0)
        self.assertEqual(invented['unsupported_figures'], [9.9])

    def test_summaries_carry_quality_and_the_combined_objective_only_when_asked(self):
        cases, _ = casebank.load_bank(strict=False)
        summary = evaluation.evaluate(Pipeline(client=None, config=PipelineConfig()), cases[:6], offline=True)
        self.assertIsNotNone(summary['quality']['score'])
        self.assertEqual(evaluation.objective(summary, 'severity'), summary['severity_score'])
        self.assertGreater(evaluation.objective(summary, 'combined'), summary['severity_score'])
        rendered = evaluation.render(summary)
        self.assertIn('reasoning quality', rendered)

    def test_the_judge_grades_against_the_recorded_rationale_and_reports_scores_only(self):
        stub = type('Stub', (), {'configured': True, 'model': 'stub',
                                 'complete': lambda self, messages: {'faithfulness': 2, 'rule_named': 1,
                                                                     'evidence_specificity': 2, 'comment': 'fine'}})()
        judge = Judge(stub)
        case = FakeCase('C', profile(), {'outcome': 'refer', 'rationale': 'smoker with large cover'})
        graded = judge({'output': {'explanation': 'x', 'reasons': [], 'missing_information': [], 'citations': []},
                        'rules': {'fired': []}}, case)
        self.assertEqual(graded['score'], round(5 / 6, 3))
        self.assertIsNone(Judge(None)({}, case))
        wild = type('Stub', (), {'configured': True, 'model': 'stub',
                                 'complete': lambda self, messages: {'faithfulness': 7, 'rule_named': 'yes'}})()
        graded = Judge(wild)({'output': {}, 'rules': {'fired': []}}, case)
        self.assertIsNone(graded['score'])


if __name__ == '__main__':
    unittest.main()
