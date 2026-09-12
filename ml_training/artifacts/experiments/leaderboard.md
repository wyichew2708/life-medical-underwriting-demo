# Model experiment leaderboard

Generated 2026-09-12T14:23:24+00:00 over 20000 synthetic rows (temporal by application_month: train <=17, calibration 18-20, test >=21), 3 seed(s) per candidate.

Ranked by viability first — confirmed inside the unsafe-acceptance budget, then inside it at the point estimate only, then outside it or below the coverage floor — and within a tier by the composite score. Weights: safety 0.4, coverage 0.25, calibration 0.2, discrimination 0.15; unsafe budget 0.05; coverage floor 0.05. Brackets are 90% bootstrap intervals over the test rows.

| # | Candidate | Architecture | Score | Coverage [90%] | Unsafe [90%] | ECE | AUC | Fit (s) | Size | Viability |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `logistic-l2` | single | 0.3822 ±0.0001 | 32.5% [31.0%–34.0%] | 0.0447 [0.0334–0.0578] | 0.0083 | 0.8059 | 0.36 | 6 KB | provisional |
| 2 | `logistic-elasticnet` | single | 0.3797 ±0.0013 | 32.4% [30.9%–33.9%] | 0.0449 [0.0335–0.0581] | 0.0084 | 0.8062 | 1.25 | 6 KB | provisional; tied with logistic-l2 |
| 3 | `extra-trees` | ensemble (bagging) | 0.3704 ±0.0227 | 28.0% [26.4%–29.4%] | 0.0418 [0.0298–0.0554] | 0.0073 | 0.798 | 0.96 | 76 MB | provisional |
| 4 | `hist-gbm-monotone` | ensemble (boosting, monotone) | 0.3615 ±0.0895 | 28.6% [27.1%–30.1%] | 0.0465 [0.0339–0.0595] | 0.0169 | 0.7964 | 1.59 | 448 KB | provisional — exceeded the unsafe budget on 1 of 3 seeds; fell below the coverage floor on 1 of 3 seeds |
| 5 | `stacking-lr` | ensemble (stacking) | 0.3547 ±0.0076 | 36.5% [34.9%–38.1%] | 0.0497 [0.0377–0.0620] | 0.0058 | 0.8034 | 5.73 | 25 MB | provisional |
| 6 | `mlp-64-32` | single (2 hidden layers) | 0.3358 ±0.0276 | 29.2% [27.6%–30.6%] | 0.0470 [0.0346–0.0604] | 0.0204 | 0.8014 | 0.79 | 125 KB | provisional |
| 7 | `hist-gbm-shallow-monotone` | ensemble (boosting, monotone) | 0.3290 ±0.0247 | 32.2% [30.6%–33.7%] | 0.0439 [0.0325–0.0562] | 0.0171 | 0.799 | 0.96 | 198 KB | provisional — exceeded the unsafe budget on 1 of 3 seeds; tied with logistic-l2 |
| 8 | `baseline-prior` | single | 0.5828 ±0.0000 | 0.0% [0.0%–0.0%] | — | 0.0043 | 0.5 | 0.35 | 5 KB | not viable — clears 0.0% of cases, below the 5% floor (safety is undefined when nothing is cleared) |
| 9 | `hist-gbm-deep` | ensemble (boosting) | 0.3982 ±0.0854 | 4.9% [4.2%–5.6%] | 0.0413 [0.0108–0.0763] | 0.0149 | 0.7877 | 2.99 | 1021 KB | not viable — clears 4.9% of cases, below the 5% floor |
| 10 | `hist-gbm-shallow` | ensemble (boosting) | 0.3778 ±0.0669 | 38.4% [36.8%–40.1%] | 0.0515 [0.0402–0.0635] | 0.0089 | 0.7967 | 0.85 | 185 KB | not viable — unsafe acceptance 0.0515 exceeds the budget of 0.05 |
| 11 | `voting-soft` | ensemble (soft voting) | 0.3473 ±0.0159 | 35.4% [33.8%–36.9%] | 0.0502 [0.0380–0.0632] | 0.0059 | 0.8027 | 1.83 | 25 MB | not viable — unsafe acceptance 0.0502 exceeds the budget of 0.05 |
| 12 | `stacking-trees` | ensemble (stacking) | 0.3322 ±0.0074 | 35.0% [33.4%–36.6%] | 0.0519 [0.0396–0.0645] | 0.0113 | 0.798 | 12.33 | 39 MB | not viable — unsafe acceptance 0.0519 exceeds the budget of 0.05 |
| 13 | `hist-gbm-default` | ensemble (boosting) | 0.3212 ±0.0347 | 40.0% [38.3%–41.5%] | 0.0586 [0.0463–0.0715] | 0.0175 | 0.7923 | 1.24 | 325 KB | not viable — unsafe acceptance 0.0586 exceeds the budget of 0.05 |
| 14 | `random-forest` | ensemble (bagging) | 0.3177 ±0.0836 | 4.1% [3.5%–4.8%] | 0.0490 [0.0180–0.0909] | 0.0095 | 0.7969 | 1.31 | 50 MB | not viable — clears 4.1% of cases, below the 5% floor |

## What this test set can resolve

The leader cleared 805 test cases, which resolves its unsafe rate to [0.0334, 0.0578] at 90%, about ±0.0122.
No candidate is confirmed inside the 0.05 budget at this test-set size: every viable one is inside it at the point estimate only. Confirming the leader's rate would take about 4114 cleared cases.
Statistically tied with the leader on both unsafe rate and coverage: logistic-elasticnet, hist-gbm-shallow-monotone. The order among these is noise.

## Oracle floor

Oracle floor on this data: a classifier that knew the generator's own probabilities, hidden labs included, would clear 31.8% with 0.0470 unsafe (AUC 0.8149). No model can beat that here.
The lowest unsafe rate any candidate reached is 0.0413, inside the floor's own 90% interval [0.0347, 0.0599]. This test set cannot measure any headroom between the best candidate and the ceiling.

## Seed repeats

Score is the mean over seeds; the first seed is the bundle kept. A candidate whose unsafe rate left the budget on any seed is held at provisional.

| Candidate | Score mean ± sd | Unsafe min–max | Coverage min–max | AUC min–max |
|---|---|---|---|---|
| `logistic-l2` | 0.3822 ± 0.0001 | 0.0447–0.0447 | 32.5%–32.5% | 0.8059–0.8059 |
| `logistic-elasticnet` | 0.3797 ± 0.0013 | 0.0448–0.0449 | 32.4%–32.4% | 0.8058–0.8062 |
| `extra-trees` | 0.3704 ± 0.0227 | 0.0418–0.0430 | 23.9%–28.1% | 0.7962–0.7980 |
| `hist-gbm-monotone` | 0.3615 ± 0.0895 | 0.0217–0.0530 | 1.9%–33.5% | 0.7940–0.8021 |
| `stacking-lr` | 0.3547 ± 0.0076 | 0.0459–0.0497 | 33.4%–37.6% | 0.8034–0.8050 |
| `mlp-64-32` | 0.3358 ± 0.0276 | 0.0392–0.0470 | 8.2%–29.2% | 0.7953–0.8036 |
| `hist-gbm-shallow-monotone` | 0.3290 ± 0.0247 | 0.0437–0.0521 | 25.8%–35.6% | 0.7966–0.7990 |
| `baseline-prior` | 0.5828 ± 0.0000 | — | 0.0%–0.0% | 0.5000–0.5000 |
| `hist-gbm-deep` | 0.3982 ± 0.0854 | 0.0213–0.0413 | 3.8%–4.9% | 0.7859–0.7929 |
| `hist-gbm-shallow` | 0.3778 ± 0.0669 | 0.0279–0.0527 | 11.6%–38.4% | 0.7945–0.8002 |
| `voting-soft` | 0.3473 ± 0.0159 | 0.0472–0.0529 | 31.6%–35.4% | 0.8024–0.8031 |
| `stacking-trees` | 0.3322 ± 0.0074 | 0.0508–0.0543 | 34.2%–36.5% | 0.7957–0.8002 |
| `hist-gbm-default` | 0.3212 ± 0.0347 | 0.0357–0.0586 | 1.1%–40.0% | 0.7896–0.7929 |
| `random-forest` | 0.3177 ± 0.0836 | 0.0364–0.0652 | 4.1%–16.6% | 0.7969–0.7993 |

## Calibration in the tail

Global ECE averages over bins the gate never looks at. This is reliability at and above 0.9, where straight-through decisions are made; the last column is the observed share of non-standard cases among those the model put at 0.95 or higher.

| Candidate | Cases ≥0.9 | Tail ECE | 0.950–0.975 observed | 0.975–1.0 observed | Non-standard among ≥0.95 |
|---|---|---|---|---|---|
| `logistic-l2` | 1550 | 0.0022 | 0.954 (n=782) | 1.000 (n=28) | 0.0444 |
| `logistic-elasticnet` | 1556 | 0.0029 | 0.954 (n=779) | 1.000 (n=28) | 0.0446 |
| `extra-trees` | 1329 | 0.0131 | 0.959 (n=678) | 0.952 (n=21) | 0.0415 |
| `hist-gbm-monotone` | 1354 | 0.0041 | 0.953 (n=704) | 1.000 (n=11) | 0.0462 |
| `stacking-lr` | 1360 | 0.0065 | 0.948 (n=872) | 1.000 (n=38) | 0.0494 |
| `mlp-64-32` | 1279 | 0.0051 | 0.950 (n=684) | 0.978 (n=45) | 0.0480 |
| `hist-gbm-shallow-monotone` | 1420 | 0.008 | 0.955 (n=791) | 1.000 (n=15) | 0.0447 |
| `baseline-prior` | 0 | None | — | — | — |
| `hist-gbm-deep` | 1197 | 0.0164 | 0.958 (n=119) | 1.000 (n=3) | 0.0410 |
| `hist-gbm-shallow` | 1339 | 0.0102 | 0.948 (n=959) | — | 0.0521 |
| `voting-soft` | 1367 | 0.0087 | 0.947 (n=837) | 0.978 (n=45) | 0.0510 |
| `stacking-trees` | 1222 | 0.0078 | 0.947 (n=873) | 1.000 (n=3) | 0.0525 |
| `hist-gbm-default` | 1270 | 0.0174 | 0.940 (n=986) | 1.000 (n=15) | 0.0589 |
| `random-forest` | 1226 | 0.0037 | 0.949 (n=79) | 0.957 (n=23) | 0.0490 |

## The three to choose between

### 1. `logistic-l2` — L2-regularised logistic regression on the standardised features.

- Cheap, stable and inspectable. Coefficients can be read by a reviewer.
- best of the three on coverage; weaker on safety; clears 32.5% [31.0%–34.0%] of cases with 4.5% [3.3%–5.8%] of those labelled non-standard, ECE 0.0083 (tail 0.0022), AUC 0.8059, 0.36s to fit, 6 KB to serve.
- Viability: provisional — inside the budget at the point estimate, but the 90% interval reaches 0.0578; confirming a rate of 0.0447 needs about 4114 cleared cases and this test set cleared 805.
- Promote with: `python promote.py --candidate logistic-l2 --by "your name" --reason "why you chose it"`

### 2. `logistic-elasticnet` — Elastic-net logistic regression (saga), which can zero out features.

- Shows whether a sparser feature set performs as well as the full catalogue.
- best of the three on discrimination; weaker on safety, calibration; tied with logistic-l2 within what this test set can resolve; indistinguishable from logistic-l2 on this data — prefer whichever is simpler or cheaper to run; clears 32.4% [30.9%–33.9%] of cases with 4.5% [3.4%–5.8%] of those labelled non-standard, ECE 0.0084 (tail 0.0029), AUC 0.8062, 1.25s to fit, 6 KB to serve.
- Viability: provisional — inside the budget at the point estimate, but the 90% interval reaches 0.0581; confirming a rate of 0.0449 needs about 4462 cleared cases and this test set cleared 802.
- Promote with: `python promote.py --candidate logistic-elasticnet --by "your name" --reason "why you chose it"`

### 3. `extra-trees` — 400 extremely randomised trees.

- More variance reduction than a random forest, at the cost of sharper probabilities.
- best of the three on safety, calibration; weaker on coverage; clears 28.0% [26.4%–29.4%] of cases with 4.2% [3.0%–5.5%] of those labelled non-standard, ECE 0.0073 (tail 0.0131), AUC 0.798, 0.96s to fit, 76 MB to serve.
- Viability: provisional — inside the budget at the point estimate, but the 90% interval reaches 0.0554; confirming a rate of 0.0418 needs about 1612 cleared cases and this test set cleared 694.
- Promote with: `python promote.py --candidate extra-trees --by "your name" --reason "why you chose it"`

Scores compare candidates on one synthetic dataset. They are not evidence that any of these
models is fit for underwriting, and the ranking will move with the data, the weights and the
confidence threshold. Read the intervals and the trade-off, not only the rank.
