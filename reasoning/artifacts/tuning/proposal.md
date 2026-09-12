# Tuning proposal

Generated 2026-09-12T14:31:17+00:00 against 30 recorded case(s) (5-fold cross-validation, every case held out once).

**Nothing has been applied.** Read the diff, then apply it deliberately.

## Measured

| Split | Metric | Current | Proposed | Change |
|---|---|---|---|---|
| all cases | severity score | 0.83 | 0.83 | +0.0 |
| all cases | agreement | 0.8 | 0.8 | +0.0 |
| all cases | unsafe disagreements | 1 | 1 | +0 |
| all cases | conservative disagreements | 5 | 5 | +0 |
| all cases | quality score | 0.8 | 0.8 | +0.0 |
| cross-validated holdout | severity score | 0.83 | 0.7967 | -0.0333 |
| cross-validated holdout | agreement | 0.8 | 0.7667 | -0.0333 |
| cross-validated holdout | unsafe disagreements | 1 | 2 | +1 |
| cross-validated holdout | conservative disagreements | 5 | 5 | +0 |
| cross-validated holdout | quality score | 0.8 | 0.787 | -0.013 |

| Fold | Held out | Changes adopted | Holdout severity Δ | Less-cautious Δ |
|---|---|---|---|---|
| 1 | 6 | 0 | +0.0 | +0 |
| 2 | 6 | 0 | +0.0 | +0 |
| 3 | 6 | 0 | +0.0 | +0 |
| 4 | 6 | 1 | -0.1667 | +1 |
| 5 | 6 | 0 | +0.0 | +0 |

## Configuration diff

No change is proposed: nothing in the search space improved the score by more than the 0.01 threshold, on at least 3 cases, without costing safety, and held up in at least half the folds.

## Cases that changed

- `CASE-029` (human: more_evidence): request_evidence → refer [match -> unsafe]

## Rule evidence

What the humans decided on the cases each rule fired on, under the current configuration.

| Rule | Fired | Human outcomes | Pipeline verdicts |
|---|---|---|---|
| `RTE-BMI-40` | 1 | more_evidence 1 | match 1 |
| `RTE-BP-DIA-100` | 9 | decline 1, more_evidence 6, postpone 1, refer 1 | acceptable 2, conservative 1, match 6 |
| `RTE-BP-SYS-160` | 8 | decline 2, more_evidence 6 | conservative 2, match 6 |
| `RTE-CONDITION` | 20 | decline 2, more_evidence 6, postpone 2, refer 5, standard 1, terms 4 | acceptable 8, conservative 5, match 7 |
| `RTE-CONDITION-COMPLEX` | 5 | decline 2, more_evidence 1, postpone 2 | acceptable 2, conservative 2, match 1 |
| `RTE-COVER-2M` | 4 | refer 4 | acceptable 3, match 1 |
| `RTE-EGFR-45` | 7 | decline 1, more_evidence 6 | conservative 1, match 6 |
| `RTE-EGFR-60` | 10 | decline 2, more_evidence 6, refer 2 | acceptable 2, conservative 2, match 6 |
| `RTE-HBA1C-65` | 18 | decline 2, more_evidence 7, postpone 2, refer 4, standard 1, terms 2 | acceptable 6, conservative 5, match 7 |
| `RTE-HBA1C-85` | 8 | decline 2, more_evidence 6 | conservative 2, match 6 |
| `RTE-HOSPITAL-2` | 4 | decline 2, more_evidence 1, postpone 1 | acceptable 1, conservative 2, match 1 |
| `RTE-LDL-49` | 8 | decline 2, more_evidence 6 | conservative 2, match 6 |
| `RTE-LIFE-COVER-1M` | 7 | decline 1, more_evidence 1, refer 5 | acceptable 4, conservative 1, match 2 |
| `RTE-MEDICAL-PRODUCT` | 5 | decline 1, more_evidence 3, terms 1 | conservative 2, match 3 |
| `RTE-REPORT-STALE` | 6 | more_evidence 5, postpone 1 | acceptable 1, match 5 |
| `RTE-SMOKER` | 9 | decline 2, more_evidence 1, refer 6 | acceptable 4, conservative 2, match 3 |

## Warnings

- The search was less cautious than the current configuration on the held-out cases of fold(s) 4. Do not apply it.
- Only 30 usable cases. A gain measured here is as likely to be noise as signal; treat this as a hypothesis to test on more cases.
- The tuned configuration is less cautious than the current one on held-out cases. Do not apply it.
- Tuned offline: only rule thresholds were searched. Retrieval, revision budget and prompt variant cannot change a deterministic outcome, so they were left alone.
- Switching rules off was not searched (pass --allow-disable to include it).
- No change is proposed. The current configuration is the best the search could justify on this bank.

## Trials

64 configuration(s) evaluated on all cases, 0 adopted by that search (minimum 3 improved cases per change). Only changes the folds also found were carried into the diff above.

| Knob | From | To | Severity | Δ | Better | Worse | Outcome |
|---|---|---|---|---|---|---|---|

Apply with:

```
python tune.py --apply --by "your name" --reason "what you checked"
```
