"""Mock application generator.

Produces a synthetic population for the demo's ML adapter. There is NO real portfolio
here: every row, correlation and label is invented by the process documented below, so
any accuracy measured on it describes this generator, not underwriting reality.

Simulation choices that are deliberately awkward, because real intake is awkward:

* Labs are missing-not-at-random. Blood results mostly exist when the case crossed a
  mock non-medical limit or declared a condition, so missingness itself carries signal.
* The label is drawn from a latent score that reads the TRUE lab values even when the
  observed row has them blank. The model therefore cannot reach perfect separation,
  which is the point: unknown critical data is a reason to abstain, not a free pass.
* A separate shifted cohort is written for out-of-distribution probing. It is never
  part of training or of the reported holdout.
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).parent
OCCUPATIONS = ['Office professional', 'Teacher', 'Accountant', 'Designer', 'Engineer', 'Nurse',
               'Business owner', 'Sales executive', 'Delivery rider', 'Construction supervisor',
               'Offshore technician', 'Commercial diver']
OCCUPATION_WEIGHTS = np.array([.17, .09, .09, .07, .12, .08, .10, .11, .07, .06, .025, .015])
HAZARD = {'Office professional': 1, 'Teacher': 1, 'Accountant': 1, 'Designer': 1, 'Engineer': 2,
          'Nurse': 2, 'Business owner': 2, 'Sales executive': 2, 'Delivery rider': 3,
          'Construction supervisor': 3, 'Offshore technician': 4, 'Commercial diver': 4}
INCOME_BASE = {1: 78000, 2: 95000, 3: 52000, 4: 120000}
MONTHS = 24


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _population(rng, n):
    age = np.clip(rng.gamma(9.0, 4.2, n) + 18, 18, 92).round()
    sex = rng.choice(['Female', 'Male'], n, p=[.49, .51])
    occupation = rng.choice(OCCUPATIONS, n, p=OCCUPATION_WEIGHTS / OCCUPATION_WEIGHTS.sum())
    hazard = np.array([HAZARD[o] for o in occupation], float)
    product = rng.choice(['Life', 'Medical'], n, p=[.62, .38])
    smoker = rng.random(n) < np.clip(.22 - .0015 * (age - 40), .05, .35)

    bmi = np.clip(rng.normal(23.8 + .035 * (age - 30), 3.9), 15, 55)
    systolic = np.clip(rng.normal(104 + .46 * (age - 18) + .85 * (bmi - 24) + 4.5 * smoker, 9), 85, 215)
    diastolic = np.clip(rng.normal(.58 * systolic + 8 + .25 * (bmi - 24), 6), 50, 135)
    hba1c = np.clip(rng.normal(5.05 + .013 * (age - 18) + .052 * np.maximum(bmi - 23, 0), .42)
                    + rng.gamma(1.1, .35, n) * (rng.random(n) < .13), 3.8, 16)
    egfr = np.clip(rng.normal(119 - .58 * age - .35 * np.maximum(hba1c - 6, 0) * 9, 11), 8, 140)
    ldl = np.clip(rng.normal(2.35 + .021 * (age - 18) + .035 * np.maximum(bmi - 23, 0), .72), .8, 8)

    # Declaration follows the underlying picture, imperfectly: people disclose what they know.
    severity = (.9 * np.maximum(hba1c - 6.4, 0) + .03 * np.maximum(systolic - 145, 0)
                + .04 * np.maximum(55 - egfr, 0) + .5 * (rng.random(n) < .07))
    declared = rng.random(n) < sigmoid(-1.1 + 2.2 * severity + .015 * (age - 40))
    complex_case = declared & (rng.random(n) < sigmoid(-1.3 + 1.4 * severity))
    condition = np.where(complex_case, 'complex', np.where(declared, 'controlled', 'none'))
    diagnosis_years = np.where(condition == 'none', 0.0, np.clip(rng.gamma(2.0, 2.6, n), 0, 45).round(1))
    hospitalisations = rng.poisson(np.where(condition == 'complex', .9, np.where(condition == 'controlled', .25, .06)))

    income = np.clip(np.array([INCOME_BASE[int(h)] for h in hazard])
                     * rng.lognormal(0, .45, n) * (1 + .006 * (age - 30)), 12000, 4000000).round(-2)
    multiple = np.clip(rng.normal(np.where(age < 35, 12, np.where(age < 50, 9, 5)), 4), .5, 30)
    cover = np.clip((income * multiple).round(-3), 1000, 10000000)
    existing = np.where(rng.random(n) < .45, np.clip((income * rng.uniform(.5, 8, n)).round(-3), 0, 10000000), 0.0)
    term = np.clip(rng.choice([10, 15, 20, 25, 30, 35], n, p=[.12, .18, .28, .21, .15, .06])
                   .astype(float), 1, 70)
    report_age = np.clip(rng.gamma(2.2, 28, n), 0, 900).round()
    month = rng.integers(0, MONTHS, n)
    return dict(application_month=month, age=age, sex=sex, occupation=occupation, product=product,
                cover=cover, bmi=bmi.round(1), smoker=smoker, condition=condition,
                annualIncome=income, existingCover=existing, termYears=term,
                systolic=systolic.round(), diastolic=diastolic.round(), hba1c=hba1c.round(2),
                egfr=egfr.round(), ldl=ldl.round(2), diagnosisYears=diagnosis_years,
                hospitalisations=hospitalisations.astype(float), reportAgeDays=report_age,
                _hazard=hazard)


LABEL_NOISE_SD = 0.28   # irreducible noise in the acceptance process, on the logit scale


def true_probability_standard(latent):
    """P(standard | everything the generator knows), averaged over the label noise.

    This is the ceiling: a classifier that saw every true value, including the labs the
    row later hides, could do no better than this. The sweep prints it as the oracle floor
    so a candidate is read against what is achievable rather than against the others.
    Quadrature, not sampling, so it consumes no random draws and leaves the label unchanged.
    """
    nodes, weights = np.polynomial.hermite.hermgauss(40)
    shifts = np.sqrt(2) * LABEL_NOISE_SD * nodes
    p_non_standard = sum(w * sigmoid(np.asarray(latent)[:, None] + shifts[None, :])[:, i]
                         for i, w in enumerate(weights)) / np.sqrt(np.pi)
    return 1 - p_non_standard


def _label(rng, pop):
    """Latent acceptance process. Reads true values, including ones later hidden."""
    cover_to_income = pop['cover'] / np.maximum(pop['annualIncome'], 1)
    z = (-4.40
         + .043 * (pop['age'] - 35)
         + .11 * np.maximum(pop['bmi'] - 27.5, 0) + .06 * np.maximum(21 - pop['bmi'], 0)
         + 1.05 * pop['smoker']
         + .85 * (pop['condition'] == 'controlled') + 2.35 * (pop['condition'] == 'complex')
         + .034 * np.maximum(pop['systolic'] - 132, 0) + .03 * np.maximum(pop['diastolic'] - 86, 0)
         + 1.25 * np.maximum(pop['hba1c'] - 6.0, 0)
         + .021 * np.maximum(62 - pop['egfr'], 0)
         + .32 * np.maximum(pop['ldl'] - 3.4, 0)
         + .46 * pop['hospitalisations']
         + .028 * pop['diagnosisYears']
         + .27 * (pop['_hazard'] - 1)
         + .11 * np.maximum(cover_to_income - 15, 0)
         + .18 * (pop['cover'] > 1500000))
    p_non_standard = sigmoid(z + rng.normal(0, LABEL_NOISE_SD, len(z)))
    standard = rng.random(len(z)) > p_non_standard
    return standard.astype(int), z


def _hide_optionals(rng, frame):
    """Apply missing-not-at-random blanks to the optional catalogue fields."""
    n = len(frame)
    lab_ordered = (frame['cover'] > 600000) | (frame['condition'] != 'none') | (frame['age'] > 50)
    lab_present = lab_ordered & (rng.random(n) < .88) | (~lab_ordered & (rng.random(n) < .22))
    for field in ('systolic', 'diastolic', 'hba1c', 'egfr', 'ldl'):
        frame.loc[~lab_present, field] = np.nan
    frame.loc[rng.random(n) < .12, 'annualIncome'] = np.nan
    frame.loc[rng.random(n) < .18, 'existingCover'] = np.nan
    frame.loc[rng.random(n) < .08, 'termYears'] = np.nan
    # Not blanked when nothing is declared: the demo sends 0, and training data that never
    # pairs condition=none with a 0 would make every clean demo case look unfamiliar.
    frame.loc[rng.random(n) < .15, 'hospitalisations'] = np.nan
    frame.loc[~lab_present | (rng.random(n) < .10), 'reportAgeDays'] = np.nan
    return frame


def build(rows, seed, shift=False):
    rng = np.random.default_rng(seed)
    pop = _population(rng, rows)
    if shift:
        # Deliberately outside the trained population: very old, very large cover,
        # extreme metabolic and renal values, unseen occupations.
        pop['age'] = np.clip(pop['age'] + rng.uniform(28, 45, rows), 18, 100).round()
        pop['bmi'] = np.clip(pop['bmi'] + rng.uniform(14, 26, rows), 10, 70).round(1)
        pop['hba1c'] = np.clip(pop['hba1c'] + rng.uniform(5, 12, rows), 3.8, 30).round(2)
        pop['egfr'] = np.clip(pop['egfr'] - rng.uniform(60, 95, rows), 4, 250).round()
        pop['cover'] = np.clip(pop['cover'] * rng.uniform(6, 12, rows), 1000, 10000000).round(-3)
        pop['occupation'] = rng.choice(['Stunt performer', 'Deep-sea fisher', 'Blast technician'], rows)
        pop['_hazard'] = np.full(rows, 4.0)
    label, latent = _label(rng, pop)
    frame = pd.DataFrame({k: v for k, v in pop.items() if k != '_hazard'})
    frame['standard'] = label
    frame['_latent_risk'] = latent.round(3)
    frame['_p_standard'] = true_probability_standard(latent).round(4)
    # The shifted probe keeps every value, otherwise blanked labs would hide the shift
    # behind median imputation and the novelty check would have nothing to see.
    if not shift:
        frame = _hide_optionals(rng, frame)
    frame['smoker'] = frame['smoker'].astype(bool)
    return frame.sort_values('application_month').reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description='Generate mock underwriting applications.')
    ap.add_argument('--rows', type=int, default=12000)
    ap.add_argument('--ood-rows', type=int, default=600)
    ap.add_argument('--seed', type=int, default=20260911)
    ap.add_argument('--out', type=Path, default=OUT_DIR)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    train = build(args.rows, args.seed)
    ood = build(args.ood_rows, args.seed + 1, shift=True)
    train.to_csv(args.out / 'applications.csv', index=False)
    ood.to_csv(args.out / 'ood_probe.csv', index=False)
    summary = {
        'generator_seed': args.seed,
        'rows': int(len(train)),
        'ood_rows': int(len(ood)),
        'standard_rate': round(float(train['standard'].mean()), 4),
        'months': MONTHS,
        'missing_rate': {c: round(float(train[c].isna().mean()), 3)
                         for c in train.columns if train[c].isna().any()},
        'warning': 'Synthetic data. No real applicants, no mortality experience, no claims.',
    }
    (args.out / 'dataset_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
