"""Record an underwriter's decision on an assessed case into the case bank.

    python run_case.py cases/controlled_condition.json --out /tmp/result.json
    python feedback.py --assessment /tmp/result.json --outcome terms --by "A. Underwriter" \
        --rationale "Controlled for 6 years, +25%" --rating "+25%"

The same thing `POST /feedback` does on the service, from the command line. The decision
is appended to the bank with a snapshot of what the pipeline said and under which
configuration, so the next tuning run can measure this case too. Nothing else changes.
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import Pipeline  # noqa: E402
from pipeline import casebank  # noqa: E402
from pipeline.casebank import HUMAN_OUTCOMES  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--assessment', type=Path, required=True, help='result JSON written by run_case.py --out')
    ap.add_argument('--outcome', required=True, choices=list(HUMAN_OUTCOMES))
    ap.add_argument('--by', required=True, help='who decided')
    ap.add_argument('--rationale', help='why, in the underwriter\'s words')
    ap.add_argument('--rating', help='rating or terms applied, if any')
    ap.add_argument('--evidence-requested', help='semicolon-separated items requested')
    ap.add_argument('--later-outcome', choices=list(HUMAN_OUTCOMES), help='what happened once evidence arrived')
    ap.add_argument('--case-id', help='identifier; generated when omitted')
    ap.add_argument('--bank', type=Path, default=casebank.BANK_DIR)
    args = ap.parse_args()

    assessment = json.loads(args.assessment.read_text())
    human = {'outcome': args.outcome, 'decided_by': args.by}
    for key, value in (('rationale', args.rationale), ('rating', args.rating),
                       ('evidence_requested', args.evidence_requested), ('later_outcome', args.later_outcome)):
        if value:
            human[key] = value
    pipeline = Pipeline(client=None)
    case = casebank.record_feedback({'case_id': args.case_id, 'profile': assessment.get('profile'), 'human': human,
                                     'assessment': assessment, 'instructions': assessment.get('instructions')},
                                    directory=args.bank, validate_profile=pipeline.validate_profile)
    print(f"Recorded {case.case_id}: {args.outcome} by {args.by} "
          f"(pipeline said {(assessment.get('recommendation'))}, "
          f"config {((assessment.get('provenance') or {}).get('config_hash'))}).")
    print(f'Bank: {args.bank}. Score it with: python tune.py --report')


if __name__ == '__main__':
    main()
