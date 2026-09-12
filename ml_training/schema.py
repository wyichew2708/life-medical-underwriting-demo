"""Mock feature schema for the demo's historical ML adapter.

Every field here is a PLACEHOLDER. It mirrors the intake catalogue the demo already
sends to `UW_ML_URL` (demo/dist/attributes.json plus the core profile fields) so that a
model trained here is wire-compatible with the demo. It is not a validated feature set:
a real programme agrees features with actuarial and model-risk review, on historical
data, with a defined label and observation window.

Deliberate exclusions: `name` and `sex` never become features. The demo excludes both
from custom rules and states that demographic inputs are not proof of individual risk;
the same stance is kept here so the mock model cannot learn a sex proxy by accident.
Occupation enters only as a coarse mock hazard class, never as free text.
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent
DEMO_ATTRIBUTES = ROOT.parent / 'demo' / 'dist' / 'attributes.json'

# The demo is the source of record for field names, types and bounds.
CATALOGUE = json.loads(DEMO_ATTRIBUTES.read_text())
CORE = CATALOGUE['core']
EXTRA = CATALOGUE['extra']

EXCLUDED_FROM_FEATURES = ('name', 'sex')

# Declared numeric intake fields (core + optional). Optional ones are frequently null.
NUMERIC_FIELDS = ['age', 'cover', 'bmi'] + list(EXTRA)
CATEGORICAL_FIELDS = ['product', 'condition', 'occupation_class']
BOOLEAN_FIELDS = ['smoker']

# Mock-only engineered features. Ratios an underwriter would compute by hand; they are
# demonstration conveniences, not evidence that these ratios predict anything.
DERIVED_FIELDS = ['cover_to_income', 'total_exposure', 'pulse_pressure', 'occupation_hazard']

FEATURE_ORDER = NUMERIC_FIELDS + DERIVED_FIELDS + BOOLEAN_FIELDS + CATEGORICAL_FIELDS

# Mock occupation grouping. A real manual maps hundreds of occupation codes to classes
# with agreed extra-mortality loadings; this is a four-bucket stand-in for the demo.
OCCUPATION_CLASSES = {
    'Office professional': ('office', 1),
    'Teacher': ('office', 1),
    'Accountant': ('office', 1),
    'Designer': ('office', 1),
    'Engineer': ('technical', 2),
    'Nurse': ('technical', 2),
    'Business owner': ('proprietor', 2),
    'Sales executive': ('proprietor', 2),
    'Delivery rider': ('manual', 3),
    'Construction supervisor': ('manual', 3),
    'Offshore technician': ('hazardous', 4),
    'Commercial diver': ('hazardous', 4),
}
DEFAULT_OCCUPATION = ('unclassified', 2)

TARGET = 'standard'
TARGET_DEFINITION = (
    'standard = the synthetic historical process accepted the case at standard rates. '
    'It is a generated routing label, not mortality, not a claims outcome and not an '
    'underwriter adjudication.'
)


def occupation_class(value):
    """Return (class name, mock hazard level 1-4) for an occupation string."""
    return OCCUPATION_CLASSES.get((value or '').strip(), DEFAULT_OCCUPATION)


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value == value and abs(value) != float('inf') else None


def features_from_profile(profile):
    """Build one feature row from an intake profile. Unknowns stay None, never zero.

    Out-of-range declared values are treated as unknown rather than clipped: the demo
    validates bounds server-side, and a value outside the catalogue range reaching the
    model means the input was not the one the model was trained on.
    """
    row = {}
    for field in NUMERIC_FIELDS:
        spec = CORE.get(field) or EXTRA[field]
        value = _number(profile.get(field))
        if value is not None and not spec['min'] <= value <= spec['max']:
            value = None
        row[field] = value
    # Encoded 1.0/0.0 rather than True/False so an undeclared answer can stay None.
    row['smoker'] = float(profile['smoker']) if isinstance(profile.get('smoker'), bool) else None
    row['product'] = profile.get('product') if profile.get('product') in ('Life', 'Medical') else 'unknown'
    row['condition'] = profile.get('condition') if profile.get('condition') in CORE['condition']['values'] else 'unknown'
    name, hazard = occupation_class(profile.get('occupation'))
    row['occupation_class'] = name
    row['occupation_hazard'] = float(hazard)

    income, cover = row['annualIncome'], row['cover']
    row['cover_to_income'] = cover / income if income and income > 0 and cover is not None else None
    existing = row['existingCover']
    row['total_exposure'] = (cover or 0.0) + existing if existing is not None and cover is not None else cover
    sys_bp, dia_bp = row['systolic'], row['diastolic']
    row['pulse_pressure'] = sys_bp - dia_bp if sys_bp is not None and dia_bp is not None else None
    return row
