"""Tune the reasoning layers against recorded human decisions — and propose, never apply.

    python tune.py --report                       score the current configuration
    python tune.py                                search, then write a proposal
    python tune.py --apply --by "you" --reason "reviewed the diff"

What can move: rule thresholds inside the bounds each rule declares, how much guidance is
retrieved, the revision budget, and which presentation variant the prompt uses. Switching a
non-mandatory rule off is possible only with --allow-disable. What cannot: any mandatory
rule, any action in the loosening direction, any validation control, and the absence of an
accept action.

The search optimises the severity score under two hard constraints. A change that
increases the number of cases where the pipeline was less cautious than the underwriter is
rejected however much it improves the average. And a change must improve at least
--min-support cases before it counts: one case moving is one case's opinion.

The search is cross-validated. It runs once per fold on the cases the fold keeps, is
re-measured on the cases the fold holds out, and every case is held out exactly once. A
change is proposed only when at least half the folds found it independently. When the
pooled holdout does not confirm the gain, the proposal says so and applying it is refused
unless the risk is accepted in writing.

On a bank of a few dozen cases, none of this is a measurement. It is a hypothesis with a
diff attached.
"""
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import Pipeline  # noqa: E402
from pipeline import casebank, evaluation  # noqa: E402
from pipeline.config import PROMPT_VARIANTS, ConfigError, PipelineConfig  # noqa: E402
from pipeline.judge import Judge  # noqa: E402
from pipeline.llm import Client  # noqa: E402

ARTIFACTS = Path(__file__).parent / 'artifacts' / 'tuning'
MIN_DELTA = 0.01          # below this a "gain" is noise on a bank this size
MIN_SUPPORT = 3           # improved cases a change must show before it is carried
FOLDS = 5
SMALL_BANK = 50


def threshold_candidates(rule):
    """A few values spanning the rule's declared bounds, including where it is now."""
    bounds = rule['tunable']
    low, high, current = bounds['min'], bounds['max'], rule['value']
    span = [low, low + (high - low) / 4, (low + high) / 2, high - (high - low) / 4, high, current]
    rounded = []
    for value in span:
        value = round(value, 1) if isinstance(current, float) or high - low < 20 else int(round(value))
        if low <= value <= high and value not in rounded:
            rounded.append(value)
    return sorted(rounded)


def search_space(rules, offline, allow_disable=False):
    """Knobs to try, in a fixed order. Offline, only the ones that can change the outcome."""
    knobs = []
    for rule in rules:
        if rule.get('tunable'):
            knobs.append({'kind': 'rule_value', 'rule': rule['id'],
                          'label': f"{rule['id']} threshold ({rule['field']} {rule['operator']} …)",
                          'values': threshold_candidates(rule)})
    if allow_disable:
        # Off by default. Disabling a control removes it from every future case on the
        # evidence of the few it fired on here; a threshold shift is the reviewable change.
        for rule in rules:
            if not rule.get('mandatory'):
                knobs.append({'kind': 'rule_enabled', 'rule': rule['id'],
                              'label': f"{rule['id']} enabled", 'values': [True, False]})
    if not offline:
        # These change what the model is asked and how many attempts it gets. Offline there
        # is no model, so tuning them would be tuning nothing and reporting a tie as a result.
        knobs += [
            {'kind': 'field', 'field': 'retrieval_source_limit',
             'label': 'sources retrieved', 'values': [10, 20, 30]},
            {'kind': 'field', 'field': 'include_design_sources',
             'label': 'include research sources', 'values': [True, False]},
            {'kind': 'field', 'field': 'design_source_limit',
             'label': 'research sources kept', 'values': [0, 2, 3, 5]},
            {'kind': 'field', 'field': 'precedent_limit',
             'label': 'prior decisions attached', 'values': [0, 2, 3, 5]},
            {'kind': 'field', 'field': 'max_revisions',
             'label': 'revision budget', 'values': [0, 1]},
            {'kind': 'field', 'field': 'prompt_variant',
             'label': 'prompt presentation variant', 'values': list(PROMPT_VARIANTS)},
        ]
    return knobs


def with_knob(config, knob, value):
    if knob['kind'] == 'field':
        return config.replace(**{knob['field']: value})
    overrides = {k: dict(v) for k, v in config.rule_overrides.items()}
    entry = overrides.setdefault(knob['rule'], {})
    entry['value' if knob['kind'] == 'rule_value' else 'enabled'] = value
    return config.replace(rule_overrides=overrides)


def current_value(config, knob, rules):
    if knob['kind'] == 'field':
        return getattr(config, knob['field'])
    override = config.rule_overrides.get(knob['rule'], {})
    rule = next(r for r in rules if r['id'] == knob['rule'])
    return override.get('value' if knob['kind'] == 'rule_value' else 'enabled',
                        rule['value'] if knob['kind'] == 'rule_value' else rule.get('enabled', True))


def search(pipeline, train, baseline_config, offline, min_delta, passes=1, verbose=True,
           min_support=MIN_SUPPORT, allow_disable=False, objective='severity', judge=None):
    rules = pipeline.knowledge.rules
    best_config = baseline_config
    best = evaluation.evaluate(pipeline.using(best_config), train, offline=offline, judge=judge)
    baseline_unsafe = best['unsafe_disagreements']
    score_of = lambda summary: evaluation.objective(summary, objective)
    trials, accepted = [], []
    if verbose:
        print(f"baseline on {best['cases']} training case(s): severity {best['severity_score']}, "
              f"agreement {best['agreement']}, less-cautious disagreements {baseline_unsafe}"
              + (f", objective {score_of(best)}" if objective != 'severity' else ''))

    for pass_number in range(1, passes + 1):
        improved_this_pass = False
        for knob in search_space(rules, offline, allow_disable):
            now = current_value(best_config, knob, rules)
            for value in knob['values']:
                if value == now:
                    continue
                try:
                    candidate_config = with_knob(best_config, knob, value).validate(rules)
                except ConfigError as error:
                    trials.append({'knob': knob['label'], 'value': value, 'rejected': str(error)})
                    continue
                result = evaluation.evaluate(pipeline.using(candidate_config), train, offline=offline, judge=judge)
                delta = round(score_of(result) - score_of(best), 4)
                unsafe_delta = result['unsafe_disagreements'] - baseline_unsafe
                backing = evaluation.support(best, result)
                trial = {'pass': pass_number, 'knob': knob['label'], 'from': now, 'value': value,
                         'severity_score': result['severity_score'], 'objective': score_of(result), 'delta': delta,
                         'quality_score': (result.get('quality') or {}).get('score'),
                         'unsafe_disagreements': result['unsafe_disagreements'],
                         'agreement': result['agreement'],
                         'improved': len(backing['improved']), 'worsened': len(backing['worsened']),
                         'improved_cases': backing['improved'], 'worsened_cases': backing['worsened']}
                if unsafe_delta > 0:
                    # The constraint, not a penalty term: a change that makes the pipeline
                    # less cautious than the underwriter more often is not a better change.
                    trial['rejected'] = (f'increases less-cautious disagreements from {baseline_unsafe} '
                                         f"to {result['unsafe_disagreements']}")
                elif delta > min_delta and len(backing['improved']) < min_support:
                    trial['rejected'] = (f"improves only {len(backing['improved'])} case(s); a change needs "
                                         f'{min_support} before it is carried')
                elif delta > min_delta:
                    trial['accepted'] = True
                    best, best_config = result, candidate_config
                    now = value
                    improved_this_pass = True
                    accepted.append(trial)
                    if verbose:
                        print(f"  adopt  {knob['label']}: {trial['from']} -> {value}   "
                              f"severity {result['severity_score']} ({delta:+}), "
                              f"{trial['improved']} case(s) better, {trial['worsened']} worse")
                trials.append(trial)
        if not improved_this_pass:
            break
    return best_config, best, trials, accepted


def cross_validate(pipeline, cases, baseline_config, offline, min_delta, min_support, passes, k, seed,
                   allow_disable=False, verbose=True, objective='severity', judge=None):
    """Run the whole search once per fold and measure each result on the fold it never saw."""
    rules = pipeline.knowledge.rules
    records = []
    for number, (train, holdout) in enumerate(casebank.folds(cases, k, seed), 1):
        tuned, _, _, accepted = search(pipeline, train, baseline_config, offline, min_delta, passes,
                                       verbose=False, min_support=min_support, allow_disable=allow_disable,
                                       objective=objective, judge=judge)
        tuned = tuned.prune().validate(rules)
        base = evaluation.evaluate(pipeline.using(baseline_config), holdout, offline=offline, judge=judge)
        after = evaluation.evaluate(pipeline.using(tuned), holdout, offline=offline, judge=judge)
        comparison = evaluation.compare(base, after)
        changed = comparison.pop('changed_cases')
        record = {'fold': number, 'train': len(train), 'holdout': len(holdout),
                  'diff': baseline_config.diff(tuned), 'comparison': comparison,
                  'changed_cases': changed, 'adopted': len(accepted),
                  'holdout_rows': {'from': base['rows'], 'to': after['rows']}}
        records.append(record)
        if verbose:
            print(f"  fold {number}: trained on {len(train)}, held out {len(holdout)}; "
                  f"{len(accepted)} change(s) adopted; holdout severity "
                  f"{comparison['severity_score']['from']} -> {comparison['severity_score']['to']} "
                  f"({comparison['severity_score']['delta']:+}), less-cautious "
                  f"{comparison['unsafe_disagreements']['from']} -> {comparison['unsafe_disagreements']['to']}")
    return records


def pooled_holdout(records):
    """The cross-validated estimate: every case's holdout verdict, counted once, pooled."""
    before = [row for r in records for row in r['holdout_rows']['from']]
    after = [row for r in records for row in r['holdout_rows']['to']]
    comparison = evaluation.compare(evaluation.summarise(before), evaluation.summarise(after))
    comparison['per_fold'] = [{'fold': r['fold'], 'holdout': r['holdout'], 'adopted': r['adopted'],
                               'severity_delta': r['comparison']['severity_score']['delta'],
                               'unsafe_delta': r['comparison']['unsafe_disagreements']['delta']}
                              for r in records]
    comparison['folds'] = len(records)
    return comparison


def stability(records, diff):
    """How many folds found each change on their own."""
    folds = len(records)
    out = {}
    for key, change in diff.items():
        adopting = [r['fold'] for r in records if key in r['diff']]
        same = [r['fold'] for r in records if r['diff'].get(key, {}).get('to') == change['to']]
        values = sorted({json.dumps(r['diff'][key]['to']) for r in records if key in r['diff']})
        out[key] = {'to': change['to'], 'folds_adopting': len(adopting), 'folds_same_value': len(same),
                    'of': folds, 'values_seen': values, 'stable': 2 * len(adopting) >= folds}
    return out


def restrict(baseline, tuned, keys):
    """A configuration carrying only the listed diff keys from the tuned one."""
    overrides = {k: dict(v) for k, v in baseline.rule_overrides.items()}
    fields = {}
    for key in keys:
        if key.startswith('rule '):
            rule_id = key[len('rule '):]
            if rule_id in tuned.rule_overrides:
                overrides[rule_id] = dict(tuned.rule_overrides[rule_id])
            else:
                overrides.pop(rule_id, None)
        else:
            fields[key] = getattr(tuned, key)
    return baseline.replace(rule_overrides=overrides, **fields)


def proposal_markdown(proposal):
    p = proposal
    bank = p['bank']
    method = (f"{bank['folds']}-fold cross-validation, every case held out once"
              if bank.get('folds') else f"{bank['train']} training, {bank['holdout']} holdout")
    lines = ['# Tuning proposal', '',
             f"Generated {p['generated_at']} against {bank['cases']} recorded case(s) ({method}).", '',
             '**Nothing has been applied.** Read the diff, then apply it deliberately.', '',
             '## Measured', '',
             '| Split | Metric | Current | Proposed | Change |', '|---|---|---|---|---|']
    labels = {'train': 'all cases' if bank.get('folds') else 'train',
              'holdout': 'cross-validated holdout' if bank.get('folds') else 'holdout'}
    for split in ('train', 'holdout'):
        for metric in ('severity_score', 'agreement', 'unsafe_disagreements', 'conservative_disagreements',
                       'quality_score'):
            entry = p['comparison'][split].get(metric)
            if not entry or entry.get('delta') is None:
                continue
            lines.append(f"| {labels[split]} | {metric.replace('_', ' ')} | {entry['from']} | {entry['to']} | "
                         f"{entry['delta']:+} |")
    if p.get('objective', 'severity') != 'severity':
        lines.append('')
        lines.append(f"Objective: {p['objective']} (severity plus {evaluation.QUALITY_WEIGHT} × reasoning quality). "
                     'Safety is still a hard constraint, not part of the objective.')
    per_fold = p['comparison']['holdout'].get('per_fold')
    if per_fold:
        lines += ['', '| Fold | Held out | Changes adopted | Holdout severity Δ | Less-cautious Δ |',
                  '|---|---|---|---|---|']
        for fold in per_fold:
            lines.append(f"| {fold['fold']} | {fold['holdout']} | {fold['adopted']} | "
                         f"{fold['severity_delta']:+} | {fold['unsafe_delta']:+} |")
    lines += ['', '## Configuration diff', '']
    if not p['diff']:
        lines.append('No change is proposed: nothing in the search space improved the score by more than '
                     f"the {p['min_delta']} threshold, on at least {p['min_support']} cases, without costing "
                     'safety' + (', and held up in at least half the folds' if bank.get('folds') else '') + '.')
    for key, change in p['diff'].items():
        stable = p.get('stability', {}).get(key)
        note = (f" — found independently in {stable['folds_adopting']} of {stable['of']} folds"
                + (f", same value in {stable['folds_same_value']}" if stable['folds_same_value'] != stable['folds_adopting']
                   else '') if stable else '')
        lines.append(f"- `{key}`: {json.dumps(change['from'])} → {json.dumps(change['to'])}{note}")
    if p.get('not_proposed'):
        lines += ['', '### Found on all cases but not proposed', '']
        for key, entry in p['not_proposed'].items():
            lines.append(f"- `{key}` → {json.dumps(entry['to'])}: adopted in only {entry['folds_adopting']} of "
                         f"{entry['of']} folds" + (f" (values seen: {', '.join(entry['values_seen'])})"
                                                   if len(entry['values_seen']) > 1 else ''))
    lines += ['', '## Cases that changed', '']
    if not p['comparison']['holdout_changed_cases']:
        lines.append('No held-out case changed its recommendation.')
    for row in p['comparison']['holdout_changed_cases']:
        lines.append(f"- `{row['case_id']}` (human: {row['human']}): {row['from']} → {row['to']} "
                     f"[{row['verdict']}]")
    if p.get('rule_evidence'):
        lines += ['', '## Rule evidence', '',
                  'What the humans decided on the cases each rule fired on, under the current configuration.', '',
                  '| Rule | Fired | Human outcomes | Pipeline verdicts |', '|---|---|---|---|']
        for rule_id, entry in p['rule_evidence'].items():
            outcomes = ', '.join(f'{k} {v}' for k, v in sorted(entry['human_outcomes'].items()))
            verdicts = ', '.join(f'{k} {v}' for k, v in sorted(entry['verdicts'].items()))
            lines.append(f"| `{rule_id}` | {entry['fired']} | {outcomes} | {verdicts} |")
    lines += ['', '## Warnings', '']
    lines += [f'- {w}' for w in p['warnings']] or ['- none']
    lines += ['', '## Trials', '',
              f"{p['trials_run']} configuration(s) evaluated on all cases, {len(p['accepted'])} adopted by that "
              f"search (minimum {p['min_support']} improved cases per change). Only changes the folds also found "
              'were carried into the diff above.', '',
              '| Knob | From | To | Severity | Δ | Better | Worse | Outcome |', '|---|---|---|---|---|---|---|---|']
    for trial in p['accepted']:
        lines.append(f"| {trial['knob']} | {trial['from']} | {trial['value']} | {trial['severity_score']} | "
                     f"{trial['delta']:+} | {trial.get('improved', '')} | {trial.get('worsened', '')} | adopted |")
    blocked = [t for t in p.get('trials', []) if 'rejected' in t and 'improves only' in t['rejected']]
    if blocked:
        lines += ['', f"{len(blocked)} change(s) improved the score but rested on too few cases to carry:", '']
        for trial in blocked[:10]:
            lines.append(f"- {trial['knob']}: {trial.get('from')} → {trial['value']} ({trial['delta']:+}, "
                         f"{trial['improved']} case(s) better)")
    lines += ['', 'Apply with:', '',
              '```', 'python tune.py --apply --by "your name" --reason "what you checked"', '```', '']
    return '\n'.join(lines)


def run_tuning(args):
    cases, problems = casebank.load_bank(args.bank, strict=False)
    client = Client() if args.live else None
    offline = not args.live
    if args.live and not client.configured:
        raise SystemExit('--live needs UW_LLM_BASE_URL and UW_LLM_MODEL.')
    if args.live:
        print('Extracting evidence for any case that lacks it...')
        cases, skipped = casebank.ensure_evidence(cases, client)
        for entry in skipped:
            print(f"  skipped {entry['case_id']}: {entry['reason']}")
    usable = [c for c in cases if c.evidence is not None]
    if not usable:
        raise SystemExit('No case in the bank has evidence. Supply it in the manifest or run with --live '
                         'and a configured vision model.')

    pipeline = Pipeline(client=client, product=args.product)
    rules = pipeline.knowledge.rules
    baseline_config = PipelineConfig.load(args.config, rules) if args.config else PipelineConfig()
    judge = Judge(client) if (args.live and args.judge) else None
    objective = args.objective
    if objective == 'combined' and offline:
        print('Offline, the deterministic output does not vary in quality; objective falls back to severity.')
        objective = 'severity'

    if args.report:
        summary = evaluation.evaluate(pipeline.using(baseline_config), usable, offline=offline, judge=judge)
        print(evaluation.render(summary, f'Current configuration over {len(usable)} case(s)'))
        evidence = evaluation.rule_evidence(summary['rows'])
        if evidence:
            print('\nrules fired, with what the humans decided on those cases:')
            for rule_id, entry in evidence.items():
                outcomes = ', '.join(f'{k} {v}' for k, v in sorted(entry['human_outcomes'].items()))
                print(f"  {rule_id:<26} {entry['fired']:>3}   {outcomes}")
        if problems:
            print(f'\n{len(problems)} case(s) in the bank are unusable: ' + '; '.join(problems[:3]))
        return

    k = max(int(args.folds), 1)
    if k >= 2 and len(usable) < 2 * k:
        k = max(2, len(usable) // 2)
        print(f'Too few cases for {args.folds} folds; using {k}.')
    if args.live and k >= 2:
        knobs = sum(len(x['values']) - 1 for x in search_space(rules, offline, args.allow_disable))
        print(f'Live cross-validation: roughly {(k + 1) * knobs * len(usable) * args.passes} model calls. '
              'Use --folds 1 for a single split if that is too many.')

    records, warnings, stab, not_proposed = [], [], {}, {}
    if k >= 2:
        print(f'{len(usable)} case(s); {k}-fold cross-validation of the search, then a final search on all.\n')
        records = cross_validate(pipeline, usable, baseline_config, offline, args.min_delta, args.min_support,
                                 args.passes, k, args.seed, args.allow_disable, objective=objective, judge=judge)
        print('\nfinal search on all cases:')
        tuned_all, _, trials, accepted = search(pipeline, usable, baseline_config, offline, args.min_delta,
                                                passes=args.passes, min_support=args.min_support,
                                                allow_disable=args.allow_disable, objective=objective, judge=judge)
        tuned_all = tuned_all.prune().validate(rules)
        stab = stability(records, baseline_config.diff(tuned_all))
        stable_keys = [key for key, entry in stab.items() if entry['stable']]
        not_proposed = {key: entry for key, entry in stab.items() if not entry['stable']}
        tuned_config = restrict(baseline_config, tuned_all, stable_keys).prune().validate(rules)
        train, holdout = usable, usable
        train_base = evaluation.evaluate(pipeline.using(baseline_config), usable, offline=offline)
        train_tuned = evaluation.evaluate(pipeline.using(tuned_config), usable, offline=offline)
        comparison = {'train': evaluation.compare(train_base, train_tuned), 'holdout': pooled_holdout(records)}
        comparison['holdout_changed_cases'] = [row for r in records for row in r['changed_cases']]
        comparison['train'].pop('changed_cases', None)
        comparison['holdout'].pop('changed_cases', None)
        bank = {'path': str(args.bank), 'cases': len(usable), 'train': len(usable), 'holdout': len(usable),
                'folds': k, 'unusable': len(problems)}
        if not_proposed:
            warnings.append(f'{len(not_proposed)} change(s) the search found on all cases were adopted in fewer '
                            'than half the folds and are not proposed: ' + ', '.join(not_proposed) + '.')
        bad_folds = [f['fold'] for f in comparison['holdout']['per_fold'] if f['unsafe_delta'] > 0]
        if bad_folds:
            warnings.append('The search was less cautious than the current configuration on the held-out '
                            f"cases of fold(s) {', '.join(map(str, bad_folds))}. Do not apply it.")
    else:
        train, holdout = casebank.split(usable, holdout=args.holdout, seed=args.seed)
        print(f'{len(train)} training case(s), {len(holdout)} holdout case(s); single split.\n')
        tuned_config, train_best, trials, accepted = search(
            pipeline, train, baseline_config, offline, args.min_delta, passes=args.passes,
            min_support=args.min_support, allow_disable=args.allow_disable, objective=objective, judge=judge)
        tuned_config = tuned_config.prune().validate(rules)
        train_base = evaluation.evaluate(pipeline.using(baseline_config), train, offline=offline)
        hold_base = evaluation.evaluate(pipeline.using(baseline_config), holdout, offline=offline)
        hold_tuned = evaluation.evaluate(pipeline.using(tuned_config), holdout, offline=offline)
        comparison = {'train': evaluation.compare(train_base, train_best),
                      'holdout': evaluation.compare(hold_base, hold_tuned)}
        comparison['holdout_changed_cases'] = comparison['holdout'].pop('changed_cases')
        comparison['train'].pop('changed_cases', None)
        bank = {'path': str(args.bank), 'cases': len(usable), 'train': len(train), 'holdout': len(holdout),
                'folds': None, 'unusable': len(problems)}
        warnings.append('Single train/holdout split (--folds 1): one draw of the holdout. Cross-validation '
                        'is the default for a reason.')

    diff = baseline_config.diff(tuned_config)
    if len(usable) < SMALL_BANK:
        warnings.append(f'Only {len(usable)} usable cases. A gain measured here is as likely to be noise '
                        'as signal; treat this as a hypothesis to test on more cases.')
    holdout_delta = comparison['holdout']['severity_score']['delta']
    train_delta = comparison['train']['severity_score']['delta']
    if diff and holdout_delta <= 0:
        warnings.append(f'The gain did not transfer: severity improved {train_delta:+} where the search '
                        f'looked but {holdout_delta:+} on held-out cases. This is what overfitting looks like.')
    if diff and holdout_delta > 0 and train_delta > 2 * max(holdout_delta, 1e-9):
        warnings.append('The gain where the search looked is much larger than the held-out gain, which '
                        'suggests part of it is fitted to those cases.')
    if comparison['holdout']['unsafe_disagreements']['delta'] > 0:
        warnings.append('The tuned configuration is less cautious than the current one on held-out cases. '
                        'Do not apply it.')
    if offline:
        warnings.append('Tuned offline: only rule thresholds were searched. Retrieval, revision budget and '
                        'prompt variant cannot change a deterministic outcome, so they were left alone.')
    if not args.allow_disable:
        warnings.append('Switching rules off was not searched (pass --allow-disable to include it).')
    if not diff:
        warnings.append('No change is proposed. The current configuration is the best the search could '
                        'justify on this bank.')

    proposal = {'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                'bank': bank, 'method': f'{k}-fold cross-validation' if k >= 2 else 'single split',
                'objective': objective, 'judge': bool(judge),
                'offline': offline, 'min_delta': args.min_delta, 'min_support': args.min_support,
                'allow_disable': args.allow_disable, 'seed': args.seed, 'product': args.product,
                'baseline_config': baseline_config.to_dict(), 'proposed_config': tuned_config.to_dict(),
                'diff': diff, 'stability': stab, 'not_proposed': not_proposed,
                'comparison': comparison, 'accepted': accepted, 'trials_run': len(trials), 'trials': trials,
                'rule_evidence': evaluation.rule_evidence(train_base['rows']),
                'folds': [{k_: v for k_, v in r.items() if k_ != 'holdout_rows'} for r in records],
                'warnings': warnings, 'applied': False}

    args.artifacts.mkdir(parents=True, exist_ok=True)
    (args.artifacts / 'proposal.json').write_text(json.dumps(proposal, indent=2, default=str) + '\n')
    (args.artifacts / 'proposal.md').write_text(proposal_markdown(proposal))

    print('\n' + '=' * 76)
    print('PROPOSAL — nothing has been applied')
    print('=' * 76)
    where = 'all cases' if k >= 2 else 'train'
    held = 'cross-validated holdout' if k >= 2 else 'holdout'
    print(f"{where:<24} severity {comparison['train']['severity_score']['from']} -> "
          f"{comparison['train']['severity_score']['to']} ({train_delta:+})")
    print(f"{held:<24} severity {comparison['holdout']['severity_score']['from']} -> "
          f"{comparison['holdout']['severity_score']['to']} ({holdout_delta:+})")
    print(f"{held:<24} less-cautious disagreements "
          f"{comparison['holdout']['unsafe_disagreements']['from']} -> "
          f"{comparison['holdout']['unsafe_disagreements']['to']}")
    print('\nchanges:')
    for key, change in (diff or {'(none)': {'from': None, 'to': None}}).items():
        stable = stab.get(key)
        print(f"  {key}: {json.dumps(change['from'])} -> {json.dumps(change['to'])}"
              + (f"   [{stable['folds_adopting']} of {stable['of']} folds]" if stable else ''))
    for key, entry in not_proposed.items():
        print(f"  not proposed: {key} -> {json.dumps(entry['to'])}   [{entry['folds_adopting']} of {entry['of']} folds]")
    print('\nwarnings:')
    for warning in warnings:
        print('  - ' + warning)
    print(f"\nWritten: {args.artifacts / 'proposal.md'}")
    print('Apply with: python tune.py --apply --by "your name" --reason "what you checked"')


def apply_proposal(args):
    path = args.artifacts / 'proposal.json'
    if not path.exists():
        raise SystemExit(f'No proposal at {path}. Run python tune.py first.')
    proposal = json.loads(path.read_text())
    if not args.by or not args.reason:
        raise SystemExit('Applying a proposal records a person and a reason: --by and --reason.')
    if not proposal['diff']:
        raise SystemExit('The proposal contains no change to apply.')
    holdout_delta = proposal['comparison']['holdout']['severity_score']['delta']
    unsafe_delta = proposal['comparison']['holdout']['unsafe_disagreements']['delta']
    if (holdout_delta <= 0 or unsafe_delta > 0) and not args.accept_risk:
        raise SystemExit(
            f'This proposal did not improve the holdout (severity {holdout_delta:+}, less-cautious '
            f'disagreements {unsafe_delta:+}).\nApply it only with --accept-risk "your reason", '
            'which is recorded in the configuration.')

    pipeline = Pipeline(client=None)
    config = PipelineConfig(**{k: v for k, v in proposal['proposed_config'].items() if k != 'provenance'})
    config.provenance = {'source': 'tune.py proposal', 'generated_at': proposal['generated_at'],
                         'applied_by': args.by, 'reason': args.reason,
                         'applied_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                         'bank': proposal['bank'], 'method': proposal.get('method'),
                         'holdout_severity_delta': holdout_delta,
                         'risk_accepted': args.accept_risk, 'diff': proposal['diff']}
    config.validate(pipeline.knowledge.rules)
    target = config.save(args.config or Path(__file__).parent / 'config.json')
    proposal['applied'] = {'by': args.by, 'at': config.provenance['applied_at'], 'config': str(target)}
    path.write_text(json.dumps(proposal, indent=2, default=str) + '\n')
    print(f'Applied to {target}. The pipeline will load this configuration from now on.')
    print('Recorded: ' + json.dumps(config.provenance['applied_by']) + f' — {args.reason}')
    print('Revert by deleting that file; the defaults in pipeline/config.py take over again.')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--bank', type=Path, default=casebank.BANK_DIR)
    ap.add_argument('--artifacts', type=Path, default=ARTIFACTS)
    ap.add_argument('--config', type=Path, help='configuration file to start from and write to')
    ap.add_argument('--product', help='assess under this product id')
    ap.add_argument('--report', action='store_true', help='score the current configuration and stop')
    ap.add_argument('--apply', action='store_true', help='apply the written proposal')
    ap.add_argument('--live', action='store_true', help='use the configured model instead of running offline')
    ap.add_argument('--folds', type=int, default=FOLDS,
                    help='cross-validation folds; 1 for a single train/holdout split')
    ap.add_argument('--holdout', type=float, default=0.3, help='holdout share when --folds 1')
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--passes', type=int, default=2)
    ap.add_argument('--min-delta', type=float, default=MIN_DELTA)
    ap.add_argument('--min-support', type=int, default=MIN_SUPPORT,
                    help='cases a change must improve before it is carried')
    ap.add_argument('--allow-disable', action='store_true',
                    help='let the search switch non-mandatory rules off')
    ap.add_argument('--objective', choices=('severity', 'combined'), default='severity',
                    help='what the search maximises: routing severity alone, or severity plus reasoning quality')
    ap.add_argument('--judge', action='store_true',
                    help='with --live, also grade explanations against the recorded rationale with the model')
    ap.add_argument('--by', help='who is applying the proposal')
    ap.add_argument('--reason', help='why')
    ap.add_argument('--accept-risk', help='reason for applying a proposal the holdout did not confirm')
    args = ap.parse_args()
    apply_proposal(args) if args.apply else run_tuning(args)


if __name__ == '__main__':
    main()
