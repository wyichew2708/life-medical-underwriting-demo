"""Run one case through the reasoning pipeline and print what happened at each stage.

Offline by default: without UW_LLM_BASE_URL and UW_LLM_MODEL the pipeline still evaluates
the typed rules, retrieves guidance and assembles the evidence request, and says plainly
that no model was called. `--live` requires a configured OpenAI-compatible vision/text
endpoint, the same variables the demo runner uses.
"""
import argparse, hashlib, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import Pipeline, PipelineError  # noqa: E402
from pipeline.casebank import sniff  # noqa: E402
from pipeline.llm import Client  # noqa: E402

CASES = Path(__file__).parent / 'cases'


def render(result):
    out = []
    out.append(f"RECOMMENDATION  {result['recommendation']}   "
               f"(produced by {result['output'].get('produced_by', 'unknown')})")
    out.append(f"Human decision required: {result['human_decision_required']}")
    config = result.get('config') or {}
    if config.get('provenance'):
        p = config['provenance']
        out.append(f"Configuration: tuned proposal applied by {p.get('applied_by')} "
                   f"on {str(p.get('applied_at'))[:10]} — {p.get('reason')}")
    if result.get('product'):
        product = result['product']
        out.append(f"Product: {product['product_id']} [{product['status']}] — "
                   f"{product['rules_applied']} of {product['rules_available']} product rule(s) applied")
    out.append('')
    out.append('STAGES')
    for stage in result['stages']:
        out.append(f"  {stage['stage']:<16} {stage['ms']:>7.1f} ms  {stage['summary']}")
    reconciliation = result.get('reconciliation') or {}
    if reconciliation.get('records'):
        out.append('')
        out.append('DECLARED VERSUS DOCUMENTED')
        for record in reconciliation['records']:
            line = f"  {record.get('field', '?'):<16} {record['status']:<26}"
            if 'declared' in record:
                line += f" declared {record['declared']!r:<10} document {record['extracted']!r:<10}"
            if 'routed' in record:
                line += f" routed {record['routed']!r}"
            if record.get('verified') is not None:
                line += '  (verified)' if record['verified'] else '  (unverified)'
            out.append(line)
    out.append('')
    ml = result['ml_screen']
    out.append(f"ML SCREEN  available={ml.get('available')}  " +
               (f"model={ml.get('model')} standard={ml.get('standard')} confidence={ml.get('confidence')} "
                f"calibrated={ml.get('calibrated')} ood={ml.get('ood')} evidence_complete={ml.get('evidence_complete')}"
                if ml.get('available') else str(ml.get('error'))))
    if result['straight_through']['blocked_by']:
        for reason in result['straight_through']['blocked_by']:
            out.append(f"  blocked: {reason}")
    out.append('')
    out.append(f"RULES  outcome={result['rules']['outcome']}  "
               f"({len(result['rules']['fired'])} of {result['rules']['evaluated']} fired, "
               f"revision {result['rules']['revision']})")
    for rule in result['rules']['fired']:
        out.append(f"  {rule['id']:<22} {rule['status']:<12} {rule['action']:<16} "
                   f"actual={rule['actual']}  cites={','.join(rule.get('cites', []))}")
    out.append('')
    routine = result['evidence_requirements']['routine']
    out.append(f"ROUTINE EVIDENCE (exposure {routine.get('total_exposure')})")
    for item in routine['requirements']:
        out.append(f"  - {item}")
    for item in result['evidence_requirements']['impairment']:
        out.append(f"  - {item['requirement']}  [{item['source']}]")
    out.append('')
    out.append(f"RETRIEVED {len(result['retrieval']['sources'])} SOURCE(S)")
    for entry in result['retrieval']['trace']:
        out.append(f"  {entry['id']:<18} {entry['reason']}")
    if result['injection_findings']:
        out.append('')
        out.append('INSTRUCTION-LIKE TEXT FOUND IN UNTRUSTED MATERIAL (reported, not followed)')
        for finding in result['injection_findings']:
            out.append(f"  {finding['where']}: {finding['kind']}")
    out.append('')
    output = result['output']
    out.append('OUTPUT')
    out.append(f"  explanation: {output.get('explanation', '')}")
    for reason in output.get('reasons', []):
        out.append(f"  reason: {reason}")
    for citation in output.get('citations', []):
        out.append(f"  cites: {citation}")
    for item in output.get('missing_information', []):
        out.append(f"  missing: {item}")
    for override in result['guardrail_overrides']:
        out.append(f"  GUARDRAIL: {override}")
    for error in result['validation_errors']:
        out.append(f"  VALIDATION FAILURE: {error}")
    out.append('')
    out.append(result['disclaimer'])
    return '\n'.join(out)


def resolve_documents(entries, base):
    """Turn document entries in a case file into what the pipeline needs.

    An entry may be a manifest line (id, name, sha256) as the demo sends, or a file name
    relative to the case file. Files are hashed and typed by content here, so a live run
    can read them; manifest-only entries pass through unchanged and nothing is read.
    """
    resolved = []
    for index, entry in enumerate(entries, 1):
        name = entry if isinstance(entry, str) else entry.get('file') or entry.get('path')
        if not name:
            resolved.append(entry)
            continue
        file = Path(name) if Path(name).is_absolute() else base / name
        if not file.exists() and (base / 'documents' / name).exists():
            file = base / 'documents' / name          # the case-bank layout
        if not file.exists():
            raise SystemExit(f'Document not found: {file}')
        raw = file.read_bytes()
        mime = sniff(raw)
        if not mime:
            raise SystemExit(f'{file.name} is not a PDF, PNG, JPEG or WebP.')
        resolved.append({'id': (entry.get('id') if isinstance(entry, dict) else None) or f'DOC-{index}',
                         'name': file.name, 'type': mime, 'path': file,
                         'sha256': hashlib.sha256(raw).hexdigest()})
    return resolved


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('case', nargs='?', default=str(CASES / 'controlled_condition.json'),
                    help='path to a case JSON file, or a name in cases/')
    ap.add_argument('--live', action='store_true', help='call the configured LLM instead of running offline')
    ap.add_argument('--json', action='store_true', help='print the full result as JSON')
    ap.add_argument('--out', type=Path, help='write the full result JSON to this path')
    ap.add_argument('--no-research', action='store_true', help='exclude research and design sources')
    ap.add_argument('--product', help='assess under a configured product id')
    args = ap.parse_args()

    path = Path(args.case)
    if not path.exists():
        candidate = CASES / (args.case if args.case.endswith('.json') else args.case + '.json')
        if not candidate.exists():
            raise SystemExit(f'No such case: {args.case}. Available: '
                             + ', '.join(sorted(p.stem for p in CASES.glob("*.json"))))
        path = candidate

    case = json.loads(path.read_text())
    case['documents'] = resolve_documents(case.get('documents') or [], path.parent)
    client = Client()
    if args.live and not client.configured:
        raise SystemExit('--live needs UW_LLM_BASE_URL and UW_LLM_MODEL. Run without --live for the '
                         'deterministic path.')
    pipeline = Pipeline(client=client, include_design=not args.no_research, product=args.product)
    try:
        result = pipeline.assess(case, offline=not args.live)
    except PipelineError as error:
        raise SystemExit(f'Case rejected: {error}')

    if args.out:
        args.out.write_text(json.dumps(result, indent=2, default=str) + '\n')
    print(json.dumps(result, indent=2, default=str) if args.json else render(result))
    if case.get('label'):
        print(f"\nCase: {case['label']}")


if __name__ == '__main__':
    main()
