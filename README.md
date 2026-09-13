# Life and medical underwriting workspace

Three parts of one hybrid underwriting system, kept separate because they fail
differently and need reviewing differently.

| Folder | What it is |
|---|---|
| [`demo/`](demo/) | The cloned interactive underwriting studio: UI, local runner, Vision extraction, reasoning harness, governance rules and the research report. Unchanged apart from its location. |
| [`ml_training/`](ml_training/) | Sweeps twelve model architectures, ranks them for a human to choose, and serves the promoted one behind the demo's `UW_ML_URL` adapter. Synthetic data, placeholder fields. |
| [`reasoning/`](reasoning/) | The underwriting knowledge base and the LLM reasoning pipeline behind the demo's `UW_CONTEXT_URL` adapter — plus product-spec ingestion, tuning against recorded human decisions, and a studio page to train it case by case. |

The demo previously said two things about itself that are no longer true here: that no ML
training was included, and that no retrieval adapter was available. `ml_training/` and
`reasoning/` fill those two sockets — with synthetic data and an illustrative knowledge
base, which is exactly what they claim to be and nothing more.

```
 intake ──► ml_training (UW_ML_URL) ──► all STP gates pass? ──yes──► standard acceptance
                                             │
                                             no / model unavailable
                                             ▼
             demo Vision extraction ──► reasoning (UW_CONTEXT_URL) ──► reasoning harness
                                             │
                                             ▼
                              recommendation ──► human underwriter decides
```

## Running the whole stack

Three terminals, from this directory.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r ml_training/requirements.txt -r reasoning/requirements.txt -r demo/requirements.txt
python ml_training/data/generate.py --rows 20000
python ml_training/experiments.py            # 12 candidates x 3 seeds, intervals, top 3
python ml_training/promote.py --candidate <name> --by "you" --reason "why"
```

```bash
python ml_training/serve.py        # mock ML adapter on :8099; scores the shadow model alongside
```

```bash
python reasoning/service.py        # knowledge retrieval on :8098
```

```bash
export UW_ML_URL=http://127.0.0.1:8099/predict
export UW_CONTEXT_URL=http://127.0.0.1:8098/context
export UW_LLM_BASE_URL=http://localhost:8000/v1   # your own vision-capable endpoint
export UW_LLM_MODEL=your-vision-model
python demo/server.py              # studio on :8080, choose "Local API testing"
```

Each piece is independently useful: `ml_training/predict.py` scores the demo's six
profiles from the command line, and `reasoning/run_case.py` runs a full case with no model
configured at all.

## Configuring it for your own product and your own decisions

Two inputs turn the generic pipeline into one that matches a particular book.

```bash
cd reasoning
python ingest_spec.py ingest specs/term-life-2026.pdf     # spec sheet -> draft rules and cards
python ingest_spec.py compare protectmax-term-2026        # what would route differently
python ingest_spec.py activate protectmax-term-2026 --by "you" --reason "checked page 2"

python import_cases.py --manifest cases/sample_intake/cases.json   # past cases + outcomes
python tune.py --report                                   # how it agrees with your underwriters
python tune.py                                            # propose configuration changes
python tune.py --apply --by "you" --reason "reviewed the diff"
python feedback.py --assessment result.json --outcome terms --by "you"   # grow the bank from use
python studio.py                                          # the training page on :8097

cd ../ml_training
python drift.py --window artifacts/scoring_log.jsonl      # has the scored population moved?
```

Both follow the same shape as model promotion, and for the same reason: **generate, review,
test, then operationalise.** A spec sheet becomes a draft that routes nothing until someone
activates it by name. A tuning run writes a proposal with a diff, a cross-validated holdout
check, the evidence behind every rule it touched and its own warnings, and refuses to apply
itself when the held-out cases do not confirm the gain.
Neither can reach an action that accepts a case, because no such action exists.

### Tests

The demo's Python tests import `server` and `governance` as top-level modules, so they run
from inside `demo/`. The other two suites run from here.

```bash
(cd demo && python -m unittest discover -s tests -p 'test_*.py' && npm test)
python -m unittest discover -s ml_training/tests -p 'test_*.py'
python -m unittest discover -s reasoning/tests -p 'test_*.py'
```

## What is real and what is not

Real: twelve model architectures are fitted three times each, calibrated on a later period
and compared on identical splits with intervals around every figure, and the promoted one
is served over HTTP; the knowledge base is loaded,
validated and retrieved deterministically; document values are checked against the page text
and reconciled with the declaration by code before any rule runs; a spec sheet really is parsed into typed rules
with the source line attached to each value; the tuner really does search a bounded
configuration space against recorded outcomes and measure the result by cross-validation; the
guardrails are executed and tested; prior decisions are retrieved from the bank as
de-identified context and the reasoning is scored against the recorded rationale; feedback
is appended to the bank with the configuration it disagreed with; the outgoing model scores
in the shadow and a drift report compares what is scored with what was trained on; the two
adapters satisfy the contracts the demo runner re-validates on every call.

Not real: the applicants, the training data, the label, the sample product, the sample case
bank, and every underwriting threshold in both new folders. There is no portfolio
experience, no mortality or claims validation, no clinical or actuarial review, no fairness
assessment, no maker-checker approval workflow, no immutable audit store, and no compliance
conclusion for any jurisdiction. Each folder's README states its own limits;
`demo/dist/research.html` states the evidence behind the architecture and where that
evidence stops.

Two numbers worth carrying. From `ml_training/`: across twelve architectures, the lowest
unsafe-acceptance rate any of them reached at the 95% confidence gate was 4.1%, and the
oracle that knows the generator's own probabilities reaches 4.7% with an interval that
contains every serious candidate — the floor is set by the label, not by the algorithm,
and no candidate can be confirmed under the 5% budget on a holdout this size. From
`reasoning/`: on the sample case bank the pipeline agrees with the recorded human decision
about 80% of the time, its one disagreement in the unsafe direction is a case the
underwriter escalated on a file note that was never in the structured data, and every
change a single train/holdout split once proposed is withdrawn under cross-validation.

Evidence certification, the routing rules and human review are what stand between those
figures and an acceptance. That is the argument for the whole architecture, and it is why
neither new folder can accept anything by itself.

## Layout note

The clone was moved wholesale into `demo/`, so its internal paths are unchanged. The one
edit outside that move is `.openai/hosting.json`, whose static directory now points at
`demo/dist` so the hosted demo still resolves.
