# Model card — mock-underwriting-ml-2026-09

**Synthetic demonstration model. Not approved for underwriting use.**

Candidate `logistic-l2`, trained 2026-09-12T14:23:46+00:00 on generated data (20000 rows, temporal by application_month: train <=17, calibration 18-20, test >=21).

## Intended use

Supplies the `UW_ML_URL` adapter of the underwriting demo so the straight-through path can be
exercised end to end with a real model object instead of scripted numbers. Any other use,
including pricing, acceptance or medical inference, is out of scope.

## Label

standard = the synthetic historical process accepted the case at standard rates. It is a generated routing label, not mortality, not a claims outcome and not an underwriter adjudication.

## Measured on the latest-period holdout

| Metric | Value |
|---|---|
| AUC | 0.8059 |
| Average precision | 0.9388 |
| Accuracy @0.5 | 0.8688 |
| Brier (calibrated) | 0.1024 |
| Brier (raw) | 0.1023 |
| ECE (calibrated) | 0.0083 |
| ECE (raw) | 0.0173 |
| Standard base rate | 0.8361 |

## Calibration in the tail

Where the gate decides. A positive gap is over-confidence.

| Bin | n | Predicted standard | Observed standard | Gap |
|---|---|---|---|---|
| 0.900-0.925 | 440 | 0.9024 | 0.9023 | 0.0001 |
| 0.925-0.950 | 300 | 0.9412 | 0.9333 | 0.0078 |
| 0.950-0.975 | 782 | 0.9527 | 0.954 | -0.0013 |
| 0.975-1.000 | 28 | 0.9997 | 1.0 | -0.0003 |

Tail ECE 0.0022 over 1550 holdout cases at or above 0.9.

## Straight-through operating point

At confidence >= 0.95 and no OOD flag the model alone clears 32.5% of holdout cases, of which 36 were labelled non-standard (unsafe acceptance 0.0447).

## Out-of-distribution behaviour

Flag rate 0.0194 on holdout, 1.0 on the shifted probe cohort. A flag means unfamiliar input, not high risk. It routes the case away from STP.

## Limitations

- Synthetic generator, not portfolio experience. No mortality or claims validation.
- The label is a generated acceptance decision, not an adjudicated underwriter outcome.
- Labs are missing not at random; subgroup AUC for missing-lab cases reflects that design.
- No fairness evaluation. Sex and name are excluded from features but proxies are not audited.

## Selection

Chosen from 14 candidates by demo setup (placeholder selection) on 2026-09-12T14:23:46+00:00. Reason: Leads the three-seed sweep, tied with logistic-elasticnet and hist-gbm-shallow-monotone within what the test set resolves, 6 KB to serve and its coefficients can be reviewed. Replace this with your own selection.


## Subgroups

| Cohort | n | AUC | Brier |
|---|---|---|---|
| age 18-35 | 62 | 0.7737 | 0.0693 |
| age 36-50 | 845 | 0.7209 | 0.0791 |
| age 51-65 | 1075 | 0.8126 | 0.0937 |
| age 66+ | 495 | 0.779 | 0.1652 |
| product Life | 1539 | 0.8013 | 0.1039 |
| product Medical | 938 | 0.8145 | 0.0999 |
| smoker | 531 | 0.8002 | 0.1391 |
| non-smoker | 1946 | 0.7956 | 0.0924 |
| labs missing | 436 | 0.7722 | 0.0821 |
| labs present | 2041 | 0.8082 | 0.1067 |

Subgroup figures are synthetic-population diagnostics, not a fairness assessment.
