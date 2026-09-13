"""Import past cases — their documents and the decision a human made — into the case bank.

    python import_cases.py --manifest intake/outcomes.csv --documents-dir intake/documents
    python import_cases.py --manifest intake/cases.json --extract     # also read the documents now
    python import_cases.py --stats

A manifest is CSV or JSON. Each row needs a case_id, the profile fields the intake form
collects, and what the underwriter decided: outcome (standard, terms, more_evidence,
refer, postpone, decline), decided_by, and optionally decided_at, notes and cohort.
Documents are matched by an explicit `documents` column, or by filename prefix.

Every profile is validated against the same field catalogue the demo uses, so a bank
cannot be built out of rows the pipeline could never assess. Documents are copied, hashed
and type-checked by content. Nothing is extracted at import time unless you ask for it:
reading documents costs model calls, and it is cached against the document hashes.
"""
import argparse, csv, json, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import Pipeline, PipelineError  # noqa: E402
from pipeline import casebank, extraction  # noqa: E402
from pipeline.casebank import BANK_DIR, HUMAN_OUTCOMES, CaseBankError  # noqa: E402
from pipeline.llm import Client  # noqa: E402

PROFILE_STRINGS = ('name', 'sex', 'occupation', 'product', 'condition')
HUMAN_COLUMNS = ('outcome', 'decided_by', 'decided_at', 'notes', 'cohort', 'rating',
                 'rationale', 'evidence_requested', 'rules_cited', 'later_outcome')
TRUE = {'y', 'yes', 'true', '1', 't', 'smoker'}
FALSE = {'n', 'no', 'false', '0', 'f', 'non-smoker', 'nonsmoker'}


def coerce(value):
    """CSV gives strings. Turn them into the types the catalogue expects, or None."""
    if value is None:
        return None
    if isinstance(value, (int, float, bool)) or isinstance(value, (list, dict)):
        return value
    text = str(value).strip()
    if not text or text.lower() in ('none', 'null', 'na', 'n/a', '-'):
        return None
    lowered = text.lower()
    if lowered in TRUE:
        return True
    if lowered in FALSE:
        return False
    cleaned = text.replace(',', '')
    try:
        number = float(cleaned)
    except ValueError:
        return text
    return int(number) if number.is_integer() and '.' not in cleaned else number


def read_manifest(path):
    path = Path(path)
    if path.suffix.lower() == '.json':
        data = json.loads(path.read_text())
        return data['cases'] if isinstance(data, dict) else data
    if path.suffix.lower() in ('.csv', '.tsv'):
        delimiter = '\t' if path.suffix.lower() == '.tsv' else ','
        with path.open(newline='') as handle:
            return list(csv.DictReader(handle, delimiter=delimiter))
    raise SystemExit(f'Unsupported manifest type {path.suffix}. Use .json, .csv or .tsv.')


def build_record(row, documents_dir):
    """One manifest row -> (case_id, case.json content, list of document paths)."""
    case_id = str(row.get('case_id') or row.get('id') or '').strip()
    if not case_id or not case_id.replace('-', '').replace('_', '').isalnum():
        raise CaseBankError(f'Invalid or missing case_id: {case_id!r}')

    human = {key: row.get(key) for key in HUMAN_COLUMNS if row.get(key) not in (None, '')}
    if isinstance(row.get('human'), dict):
        human = {**human, **row['human']}
    if human.get('outcome') not in HUMAN_OUTCOMES:
        raise CaseBankError(f'{case_id}: outcome {human.get("outcome")!r} must be one of '
                            f'{", ".join(HUMAN_OUTCOMES)}')
    if not human.get('decided_by'):
        raise CaseBankError(f'{case_id}: decided_by is required — a recorded decision has an owner.')

    profile = dict(row.get('profile') or {})
    for key, value in row.items():
        if key in ('case_id', 'id', 'documents', 'evidence', 'ml', 'profile', 'human', 'instructions',
                   'product_id', 'evidence_source') or key in HUMAN_COLUMNS:
            continue
        profile.setdefault(key, value)
    profile = {k: (str(v).strip() if k in PROFILE_STRINGS and v is not None else coerce(v))
               for k, v in profile.items()}
    profile = {k: v for k, v in profile.items() if v is not None or k in PROFILE_STRINGS}

    listed = row.get('documents')
    if isinstance(listed, str):
        listed = [part.strip() for part in listed.replace(';', ',').split(',') if part.strip()]
    if not listed:
        listed = [p.name for p in sorted(Path(documents_dir).glob(f'{case_id}*')) if p.is_file()]
    paths = []
    for name in listed or []:
        path = Path(documents_dir) / name
        if not path.exists():
            raise CaseBankError(f'{case_id}: document not found: {name}')
        paths.append(path)

    case = {'case_id': case_id, 'profile': profile, 'human': human,
            'instructions': row.get('instructions') or '',
            'documents': [{'file': p.name} for p in paths]}
    if row.get('product_id'):
        case['product_id'] = str(row['product_id']).strip()
    if row.get('evidence'):
        case['evidence'] = row['evidence']
        case['evidence_source'] = row.get('evidence_source', 'supplied with the manifest')
    if row.get('ml'):
        case['ml'] = row['ml']
    return case_id, case, paths


def verify_supplied(evidence, paths):
    documents = []
    for index, path in enumerate(paths, 1):
        raw = path.read_bytes()
        mime = casebank.sniff(raw)
        if mime:
            documents.append({'id': f'DOC-{index}', 'name': path.name, 'type': mime, 'path': path})
    try:
        return extraction.verify_quotes(dict(evidence), documents)
    except extraction.ExtractionError:
        return evidence


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--manifest', type=Path, help='CSV or JSON file of cases and outcomes')
    ap.add_argument('--documents-dir', type=Path, help='folder holding the case documents')
    ap.add_argument('--bank', type=Path, default=BANK_DIR)
    ap.add_argument('--extract', action='store_true', help='read the documents now with the configured model')
    ap.add_argument('--dry-run', action='store_true', help='validate without writing anything')
    ap.add_argument('--stats', action='store_true', help='describe the bank that already exists')
    args = ap.parse_args()

    if args.stats:
        cases, problems = casebank.load_bank(args.bank, strict=False)
        outcomes, products, evidence = {}, {}, {}
        for case in cases:
            outcomes[case.human['outcome']] = outcomes.get(case.human['outcome'], 0) + 1
            products[case.profile.get('product')] = products.get(case.profile.get('product'), 0) + 1
            key = case.evidence_source.split('(')[0].strip()
            evidence[key] = evidence.get(key, 0) + 1
        print(f'{len(cases)} case(s) in {args.bank}')
        print('  outcomes: ' + ', '.join(f'{k} {v}' for k, v in sorted(outcomes.items())))
        print('  products: ' + ', '.join(f'{k} {v}' for k, v in sorted(products.items(), key=str)))
        print('  evidence: ' + ', '.join(f'{k} {v}' for k, v in sorted(evidence.items())))
        print('  documents: ' + str(sum(len(c.documents) for c in cases)))
        if problems:
            print(f'  {len(problems)} unusable case(s):')
            for problem in problems[:5]:
                print('    ' + problem)
        smallest = min(outcomes.values()) if outcomes else 0
        if len(cases) < 50 or smallest < 5:
            print('\n  This bank is small. Tuning against it will overfit; treat any gain as a hypothesis\n'
                  '  to test on more cases, not as a measured improvement.')
        return

    if not args.manifest:
        raise SystemExit('Give --manifest (or --stats).')
    documents_dir = args.documents_dir or args.manifest.parent / 'documents'
    rows = read_manifest(args.manifest)
    validator = Pipeline(client=None)

    imported, skipped = [], []
    for row in rows:
        try:
            case_id, case, paths = build_record(row, documents_dir)
            validator.validate_profile(case['profile'])          # same catalogue as the demo
        except (CaseBankError, PipelineError) as error:
            skipped.append(str(error))
            continue
        if args.dry_run:
            imported.append(case_id)
            continue
        target = args.bank / case_id
        (target / 'documents').mkdir(parents=True, exist_ok=True)
        for path in paths:
            shutil.copy2(path, target / 'documents' / path.name)
        if case.get('evidence') and paths:
            # Supplied evidence is checked against the documents like extracted evidence is:
            # a quote that is not on the page leaves its value unverified.
            case['evidence'] = verify_supplied(case['evidence'], paths)
        (target / 'case.json').write_text(json.dumps(case, indent=2, ensure_ascii=False) + '\n')
        imported.append(case_id)

    print(f"{'Would import' if args.dry_run else 'Imported'} {len(imported)} case(s) into {args.bank}")
    if skipped:
        print(f'{len(skipped)} row(s) skipped:')
        for reason in skipped[:10]:
            print('  ' + reason)
    if args.dry_run or not imported:
        return

    cases, _ = casebank.load_bank(args.bank, strict=False)
    needing = [c for c in cases if c.needs_extraction()]
    if args.extract:
        client = Client()
        print(f'\nExtracting evidence from documents for {len(needing)} case(s)...')
        _, failed = casebank.ensure_evidence(cases, client)
        for entry in failed:
            print(f"  {entry['case_id']}: {entry['reason']}")
    elif needing:
        print(f'\n{len(needing)} case(s) have no evidence yet. Either supply it in the manifest, or run:')
        print('  python import_cases.py --manifest ... --extract     (needs a vision model configured)')
    print('\nNext: python tune.py --report     to score the current configuration against this bank.')


if __name__ == '__main__':
    main()
