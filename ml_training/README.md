# ml_training — mock historical-risk model

Trains the model that sits behind the demo's `UW_ML_URL` adapter, so the straight-through
path can be exercised against a real fitted object instead of scripted numbers.

**Everything here is synthetic.** The fields are placeholders, the applicants are
generated, and the label is a simulated acceptance decision — not mortality, not claims,
not an adjudicated underwriter outcome. Metrics below describe the generator. They say
nothing about underwriting reality and must not be quoted as model performance.

## Quick start

```bash
python -m venv ../.venv && source ../.venv/bin/activate
pip install -r requirements.txt
python data/generate.py --rows 20000     # writes data/applications.csv + ood_probe.csv
python experiments.py                    # fits 12 candidates, ranks them, prints the top 3
python promote.py --candidate <name> --by "your name" --reason "why"
python predict.py                        # scores the demo's six built-in profiles
python -m unittest discover -s tests -p 'test_*.py'
```

`train.py` is still there for the short path — it fits one candidate (`hist-gbm-default` by
default, or any `--candidate` from the sweep) and writes the artifact directly.

## Wiring it into the demo

```bash
python serve.py                          # loopback adapter on :8099
```

In the demo runner's environment:

```bash
export UW_ML_URL=http://127.0.0.1:8099/predict
```

Then start `demo/server.py` and choose **Local API testing**. The demo re-validates every
field of the response and re-derives evidence completeness from the hashes it uploaded, so
a wrong answer here cannot silently create an acceptance.

### Certifying evidence

The adapter receives document hashes, never document bytes, so it cannot judge a report's
contents. `evidence_complete` is answered from a registry of hashes a human already
verified — an unseen upload is uncertified by design and the case routes to Vision
extraction and review.

```bash
python evidence_registry.py add /path/to/report.pdf --by "your name" --note "checked pages 1-3"
python evidence_registry.py list
```

`UW_ML_TRUST_ALL_HASHES=1` certifies anything submitted. It exists only to walk through
the STP screen during a presentation; it makes the evidence gate meaningless.

## Choosing a model

`experiments.py` fits fourteen candidates through identical machinery — the same
preprocessor, the same temporal splits, the same calibration on the same held-out period,
the same novelty detector, the same metric definitions. Only the classifier differs, so
the comparison is about the classifier and not about incidental differences in preparation.

| Family | Candidates |
|---|---|
| Baseline | `baseline-prior` — predicts the base rate, and anchors the table |
| Linear | `logistic-l2`, `logistic-elasticnet` |
| Bagged trees | `random-forest`, `extra-trees` |
| Boosted trees | `hist-gbm-default`, `hist-gbm-deep`, `hist-gbm-shallow` |
| Boosted trees, monotone | `hist-gbm-monotone`, `hist-gbm-shallow-monotone` — the same configurations with the manual's directions enforced: higher HbA1c, lower eGFR, smoking, higher blood pressure or cover can never raise P(standard) |
| Neural | `mlp-64-32` |
| Model ensembles | `voting-soft`, `stacking-lr`, `stacking-trees` |

**Ranking is not by AUC.** What this system does with a model is clear cases straight
through at a confidence threshold, so the score measures what that costs:

- **Viability first.** A candidate outside the unsafe-acceptance budget (default 5% of the
  cases it clears), or below the coverage floor (default 5% of cases cleared), ranks below
  every viable candidate whatever its AUC. The floor exists because the base-rate baseline
  is perfectly safe and completely useless: without it, it wins the sweep.
- **Then a weighted score** over safety (0.40), coverage (0.25), calibration (0.20) and
  discrimination (0.15). Every weight is a demonstration choice, settable with `--weight
  coverage=0.4`, and the components are printed separately so you can see what drove the rank.
- **Intervals before ranks.** Every unsafe rate and coverage figure carries a 90% bootstrap
  interval over the test rows. Viability has three tiers: *confirmed* when the top of the
  interval is inside the budget, *provisional* when only the point estimate is, *not viable*
  otherwise — and the tiers rank in that order. A viable candidate whose intervals both
  overlap the leader's is marked tied, because the order between them is noise. The board
  also says how many cleared cases it would take to confirm the leader's rate.
- **Three seeds by default.** Each candidate is fitted on three consecutive seeds; the score
  is the mean, the spread is printed, and a candidate that left the budget on any seed is
  held at provisional however good its first seed looked. `--repeats 1` turns this off.
- **Calibration is reported where the gate decides.** A global ECE averages over bins the
  gate never looks at. The board also reports reliability at and above 0.9 in narrow bins,
  a tail ECE weighted over those cases only, and the observed non-standard share among
  cases each model put at 0.95 or higher — which is the unsafe rate before the OOD filter.
- **The oracle floor is printed.** The generator records its own probability for every row,
  so the sweep can state what a classifier that knew everything, hidden labs included,
  would achieve at the same threshold. A candidate is read against that ceiling, not only
  against the others.
- **Near-duplicates are flagged.** Two configurations of one family landing within noise of
  each other is a real result — the extra complexity bought nothing — but it is not two
  options to choose between, so the second is labelled as indistinguishable from the first.

The sweep never selects. It prints three candidates with the trade-off each one asks you to
accept, and the command to promote one. `promote.py` records who chose, when, from how many
candidates and why, into `artifacts/selection.json` and the model card. It refuses a
candidate the sweep marked non-viable unless you pass `--accept-risk "reason"`, which is
recorded too.

Fitted bundles for the top three are kept under `artifacts/experiments/runs/` so promotion
serves the exact object that was measured. They are large — a 400-tree forest is 50-80 MB —
so they are gitignored; `--keep 0` skips storing them and promotion refits from the recorded
seed instead, checking the result against the leaderboard and reporting any drift.

## After promotion: shadow, log, drift

Holdout numbers describe the period a model was tested on. Three things keep them
meaningful once it is serving.

- **Champion and challenger.** `promote.py` keeps the outgoing champion as
  `artifacts/shadow.joblib`. `serve.py` scores every case with both, returns only the
  champion's answer in the contract fields, records the shadow's gate decision under
  `diagnostics.shadow`, and counts disagreements on `/health`. A promotion was chosen on a
  holdout; the shadow is how it gets judged on live cases.
- **An opt-in scoring log.** Set `UW_ML_SCORING_LOG` to a path and the adapter appends one
  JSON line per scored case: the feature row (never a name, never document content), the
  probability, the gate outputs and the shadow's verdict. It holds applicant attributes, so
  it is off unless asked for.
- **A drift report.** `drift.py --window <csv or scoring log>` compares the window with the
  training period feature by feature using the population stability index, with the
  conventional bands (below 0.10 stable, 0.10–0.25 watch, above 0.25 act), and — when the
  window is the scoring log — the novelty flag rate against its training target, the gate
  pass rate against the holdout, and the shadow disagreement rate. It writes
  `artifacts/drift_report.md` and changes nothing: retraining or withdrawing a model is a
  decision a person records, carried out through the sweep and promotion steps.

On the shifted probe every moved field is flagged `act` with PSI above 5; on the holdout
period of the training data every field is `stable` with PSI below 0.01.

## What the four signals mean

The demo's gate needs class, confidence, calibration and distribution status kept apart,
because they fail independently. This package keeps them apart.

| Output | Produced by | What it does not mean |
|---|---|---|
| `standard` | Gradient-boosted classifier on the declared intake fields | That the case is acceptable |
| `confidence` | Isotonic calibration fitted on a later period the classifier never saw | Probability the whole decision is right |
| `calibrated` | Holdout ECE ≤ 0.05 on synthetic data | Calibration on any real portfolio |
| `ood` | Isolation Forest ∪ Mahalanobis distance, thresholds fixed at a 1% training flag rate | Elevated risk — it means unfamiliar input |
| `evidence_complete` | Hash lookup against the verified-evidence registry | That a document was read or is correct |

Confidence is clipped away from 1.0: isotonic regression saturates at exactly 0 and 1 in
its end bins, and reporting certainty the holdout never measured would be a false claim.

## Design decisions worth knowing

- **Temporal split.** Train on the oldest 18 months, calibrate on months 18–20, report on
  months 21+. Random splits would leak period effects into the headline number.
- **`name` and `sex` are never features.** The demo excludes both from custom rules and
  states that demographic inputs are not proof of individual risk; the same line is held
  here so the model cannot learn a sex proxy by accident. Occupation enters only as a
  four-level mock hazard class.
- **Missing is not zero.** Unknown optional attributes stay `None`, are median-imputed
  *with* a missingness indicator, and a declared value outside the catalogue range is
  treated as unknown rather than clipped.
- **The label knows more than the model.** The generator draws the label from latent lab
  values that the observed row often hides, so perfect separation is impossible. That is
  deliberate: unknown critical data should cost confidence, not be rewarded.
- **The OOD probe keeps every value.** A shifted cohort whose labs were blanked would be
  median-imputed back into the middle of the distribution and detected as ordinary.

## Latest sweep

Fourteen candidates, three seeds each, 20,000 synthetic rows, 2,477 in the holdout period.
Seven were viable at the point estimate; none was confirmed; seven were not viable, and the
table says why for each.

| # | Candidate | Score (mean ± sd) | Clears [90%] | Unsafe [90%] | Tail ECE | AUC | Size |
|---|---|---|---|---|---|---|---|
| 1 | `logistic-l2` | 0.382 ± 0.000 | 32.5% [31.0–34.0] | 4.47% [3.34–5.78] | 0.002 | 0.806 | 6 KB |
| 2 | `logistic-elasticnet` | 0.380 ± 0.001 | 32.4% [30.9–33.9] | 4.49% [3.35–5.81] | 0.003 | 0.806 | 6 KB |
| 3 | `extra-trees` | 0.370 ± 0.023 | 28.0% [26.4–29.4] | 4.18% [2.98–5.54] | 0.013 | 0.798 | 76 MB |
| 4 | `hist-gbm-monotone` | 0.362 ± 0.090 | 28.6% [27.1–30.1] | 4.65% [3.39–5.95] | 0.004 | 0.796 | 448 KB |
| 7 | `hist-gbm-shallow-monotone` | 0.329 ± 0.025 | 32.2% [30.6–33.7] | 4.39% [3.25–5.62] | 0.008 | 0.799 | 198 KB |
| — | `hist-gbm-shallow` | 0.378 ± 0.067 | 38.4% [36.8–40.1] | 5.15% [4.02–6.35] | 0.010 | 0.797 | 185 KB |
| — | `hist-gbm-default` | 0.321 ± 0.035 | 40.0% [38.3–41.5] | 5.86% [4.63–7.15] | 0.017 | 0.792 | 325 KB |

Five things in that table are worth more than the ranking.

**Nothing is confirmed, and the board says how far it is from being confirmed.** The
leader cleared 805 holdout cases, which resolves its unsafe rate to about ±1.2 points.
Showing a 4.5% rate sits under a 5% budget would take roughly 4,100 cleared cases, five
times this holdout. Every viable candidate is inside the budget at the point estimate only.

**The unsafe-acceptance floor is the label, not the algorithm.** The oracle — the
generator's own probabilities, hidden labs included — clears 31.8% of the holdout with 4.7%
unsafe, and its own interval runs from 3.5% to 6.0%. The best candidate's 4.1% sits inside
that interval, so this holdout cannot measure any headroom between the best model and the
ceiling. The board marks `logistic-elasticnet` and `hist-gbm-shallow-monotone` as tied with
the leader for that reason.

**Monotone constraints repair boosting's tail.** Unconstrained, both boosting configurations
clear more cases than anything else and pay for it with 5.2% and 5.9% unsafe, over budget.
The same configurations with the manual's directions enforced come inside the budget at
4.4% and 4.7%: the constraints remove the leaves where a worse lab value had been raising
P(standard), which is exactly where over-confident tail probabilities were coming from.
The shallow constrained model clears as many cases as the linear leader at the same unsafe
rate. Both are seed-sensitive — each crossed the budget on one seed of three — so they are
held at provisional and the board says so.

**Seeds move trees more than the ranking suggests.** `extra-trees` led the single-seed
sweep; over three seeds its coverage ranges from 23.9% to 28.1% and it drops to third.
`mlp-64-32` clears between 8% and 29% depending on the seed. The linear models do not move.

**Size is a real cost.** `extra-trees` is 12,000 times larger than the linear model above
it, for less score. The repository ships `logistic-l2` promoted, recorded as a placeholder
selection — replace it with your own, for your own stated reason.

`artifacts/experiments/leaderboard.md` holds the full table with the seed-repeat spread,
the resolution paragraph, the oracle floor and the tail-calibration table;
`artifacts/metrics.json` the reliability tables and subgroup detail; `artifacts/selection.json`
who chose what, at which viability tier, and against which interval.

## Files

| Path | Purpose |
|---|---|
| `schema.py` | Mock feature catalogue, mirrored from `demo/dist/attributes.json`; profile → feature row |
| `modeling.py` | Shared splits, preprocessing, calibration, monotone constraints, tail reliability, intervals, oracle floor, metrics and model card — identical for every candidate |
| `experiments.py` | Candidate catalogue, sweep, viability rules, scoring, leaderboard |
| `promote.py` | Promotes one candidate to the served artifact, records the selection, keeps the outgoing champion as the shadow |
| `data/generate.py` | Synthetic population, missing-not-at-random blanks, latent label process with its true probability recorded, shifted probe |
| `data/demo_profiles.json` | The demo's six built-in profiles, for routing checks |
| `train.py` | The short path: fit one candidate end to end |
| `ood.py` | Novelty detector (Isolation Forest + Mahalanobis) |
| `predict.py` | Artifact loading, adapter-shaped scoring, evidence lookup |
| `serve.py` | Loopback HTTP service implementing the `UW_ML_URL` contract, with shadow scoring and the opt-in log |
| `evidence_registry.py` | Verified-hash registry CLI |
| `drift.py` | Population stability per feature, gate and shadow behaviour over a scoring window |
| `tests/` | Feature-schema, generator, adapter-contract, uncertainty, monotone, sweep-ranking, promotion, drift and shadow tests (62) |

## Before this could be anything but a demo

Replace the generator with governed historical extracts and an agreed label and
observation window; investigate selection bias and censoring; validate separately by
product and cohort with uncertainty intervals; run a real fairness assessment rather than
relying on field exclusion; register the model, feature schema, data period and approval;
and evaluate a product-specific operating threshold instead of inheriting the demo's 0.95.
