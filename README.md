# 15 — Adaptive Practice Loop

A competency-based adaptive practice engine: readiness scoring **with uncertainty**, a
next-item recommender that explains itself, spaced-repetition scheduling, and a FastAPI service
layer.

> **Two questions this answers**, and they are the two halves of an assessment product:
>
> 1. *"What should this candidate practise next?"* — a ranking function, whose specification is
>    usually written out in prose. §3 of [`THEORY.md`](THEORY.md) shows the five signals such a
>    function needs, implemented from scratch as five terms, with three baselines and an
>    ablation to measure whether they earn their place.
> 2. *"Why was my answer wrong?"* — grounded in **approved content**, cited, and **refused**
>    when the syllabus does not cover it. See [`THEORY.md`](THEORY.md) §4.

⚠️ **Start with [`THEORY.md`](THEORY.md).** It explains item response theory, spaced repetition
and how the JD's own sentence is a ranking function — in plain English, with the comparisons to
software engineering. This README is the index and the results.

---

## What is here

| File | What it does |
|---|---|
| `THEORY.md` | **The plain-English explanation.** IRT, spaced repetition, and whether this closes the recommendation gap |
| `src/irt.py` | Item Response Theory from scratch: the 2PL model, item information, MAP ability estimation, item selection, and calibration |
| `src/scheduler.py` | Spaced repetition: the forgetting curve, interval inversion, and state updates — plus SM-2 for comparison |
| `src/competency.py` | The competency graph: exam → domain → competency → concept → question, with validation |
| `src/selector.py` | **The recommender.** The five-term ranking function the JD specifies, three baselines, and an ablation |
| `src/retrieval.py` | Retrieval over the **approved** content: BM25, corpus-known coverage, and an abstention gate |
| `src/tutor.py` | **The explanation engine**: a named misconception, a grounded explanation, a citation — or a refusal |
| `src/api.py` | **The FastAPI service layer**: typed models, validation that rejects, abstention as a first-class response |
| `content/syllabus.md` | The approved corpus. Fixed, vetted, and inspectable at `GET /content/sources` |
| `content/items.json` | Authored questions, with each **wrong option mapped to the misconception it encodes** |

```bash
python src/irt.py --selftest         # 14 checks — the estimator recovers known ability
python src/scheduler.py --selftest   # 17 checks — and hits its stated retention target
python src/competency.py             # validates the graph, prints coverage
python src/selector.py --selftest    # the five-term ranking, baselines, and an ablation
python src/selector.py --explain     # the top 8 recommendations with their terms
python src/retrieval.py --selftest   # 26 checks — finds what is covered, refuses what is not
python src/tutor.py --selftest       # 12 checks — and two wrong answers give two diagnoses
python src/api.py --selftest         # 23 checks — HTTP agrees with the library

python src/tutor.py --item q-002 --chosen C     # one explanation, without a server
```

---

## See it work in 30 seconds

The API is a real running service. ⚠️ **Nothing below is illustrative** — it is a transcript of
the actual output, captured from `uvicorn api:app` on this repo.

```bash
pip install fastapi uvicorn pydantic
uvicorn api:app --app-dir src --port 8000     # interactive docs at /docs
```

**What is loaded.** Note that `/health` reports *what it has*, not merely that the process is
up — a health check that only says "ok" tells a client nothing about whether the thing it needs
is present, which is how a broken dependency comes to look like a slow service.

```
$ curl -s localhost:8000/health
{
  "status": "ok",
  "items": 34,
  "competencies": 8,
  "domains": 4,
  "graph_valid": true
}
```

**The recommendation, and the reason for it.** This is the whole point of the repository: the
API returns *why*, not just *what*.

```
$ curl -s -X POST localhost:8000/recommend -d @candidate.json
{
  "item_id": "q-001",
  "competency_id": "c-rag",
  "difficulty": -0.4,
  "basis": "measured",
  "probing": true,
  "score": 1.169
}

  terms (why this item):
    info     +0.349
    cover    +0.214
    weight   +0.200
    gap      +0.174
    unc      +0.100
    perf     +0.078
    rec      +0.053
```

⚠️ `basis` is the field that matters most. It says **`measured`** when the pick is driven by
this candidate's history, and **`cold_start`** when it rests on the prior — because a
recommendation and a guess must not look the same to a client.

**Readiness, with uncertainty.** Every figure carries its standard error, and a `reliable` flag
that is false when the estimate rests on too few items.

```
$ curl -s -X POST localhost:8000/readiness -d @candidate.json
  exam_weighted_mean       0.483
  exam_weighted_bottleneck 0.458

  competency                        theta     SE   n  reliable
  Chunk content for retrieval      -0.58   2.03  1  False
  Build a retrieval system         -0.48   2.00  1  False
  Write a scoring rubric           +0.26   2.29  1  False
  Fine-tune a model                +0.46   2.01  1  False
  Cut inference cost               +0.00   1.00  0  False
  Curate and clean training data   +0.00   1.00  0  False
```

⚠️ **Read the SE column, because it is the honest part.** After one item the standard error is
about **2.0 logits** — the estimate is barely distinguishable from the prior, and `reliable` is
`False` for every competency. That is what a readiness score from four questions actually is, and
the service says so rather than rendering "72% ready".

---

## Why the answer was wrong

⚠️ **Submit a wrong answer, not a corpus.** The corpus is fixed and pre-approved, because
*"grounded in reliable and validated sources"* is only a meaningful promise if **the set of
sources is fixed** — a system that answers from whatever document it is handed has no notion of
a validated source. `GET /content/sources` enumerates exactly what the tutor may teach from.

### ⚠️ The test that proves this is not a document chatbot

**The same question with two different wrong answers produces two different diagnoses.** A
language model cannot do that from the question alone — it does not know which option the
candidate ticked. This is **distractor analysis**, and it is what the role means by *"identify
the reasons behind incorrect answers."*

```
$ curl -s localhost:8000/content/sources
{
  "approved_passages": 10,
  "authored_questions": 8
}
  passages: d-retrieval#1, d-retrieval#2, d-retrieval#3, d-chunk#1, d-chunk#2 ...

$ curl -s -X POST localhost:8000/explain -d '{"item_id":"q-002","chosen":"B"}'
{
  "item_id": "q-002", "chosen": "B", "correct": "A",
  "misconception": "attributes the problem to a sign cancellation that does not occur,
                    rather than to the difference in scale",
  "citations": [{"passage_id": "d-retrieval#2",
                 "source": "Fusing scores you cannot compare"}],
  "grounded": true, "generator": "template"
}

$ curl -s -X POST localhost:8000/explain -d '{"item_id":"q-002","chosen":"C"}'
{
  "chosen": "C",
  "misconception": "states a units objection, which is true in spirit but is not the
                    mechanism that makes the sum unsafe in practice",
  "citations": [{"passage_id": "d-retrieval#2", ...}]
}

  explanation: Choosing C: states a units objection, which is true in spirit but is not the
  mechanism that makes the sum unsafe in practice

  The correct answer is A. Fusing scores you cannot compare says: "BM25 scores are unbounded
  and grow with term frequency and corpus size. Cosine similarity is bounded to `[-1, 1]`."

  That is what rules out C.
```

### Two design decisions that are not negotiable

**1. ⚠️ The diagnosis is authored, not generated.** The misconception label comes from a subject
expert via `content/items.json`. The model's job is to explain it in prose, grounded in a cited
passage. **A model must not invent the diagnosis** — if it could, the product would be telling
candidates they misunderstood something they did not.

**2. ⚠️ No source, no answer.** If the approved content does not cover the question, the tutor
**refuses**. It does not answer from general knowledge, which is the failure that makes an
educational product unsafe. And a citation the model *invented* is **rejected in code after the
call**, not requested in the prompt — a prompt instruction is a preference; a validator is a
guarantee.

### ⚠️ Three bugs this section found, all in the sufficiency gate

The gate decides *"is this question in the syllabus?"*, and it took three attempts. Each failure
is pinned as a test so none can return:

| Version | Failure | Measured |
|---|---|---|
| Coverage over the whole query | An expert wrote *"states a **units** objection"*; the syllabus never uses "units" | coverage **32% → 22%** → the tutor refused a question whose passage it had already ranked **first** |
| Gate on the question stem alone | A scenario-style stem is mostly scenario words (`system`, `document`) | **3 of 8** questions abstained for the same dilution reason |
| **Corpus-known terms only** | ✅ | the question above is **100%**; a framing-only question (*"how does a search system work in Peru"*) is correctly refused |

**The lesson:** coverage must measure *"of the words this corpus understands, how many does this
passage use?"* Measuring it over words the corpus has **never seen** measures the author's
vocabulary, not the candidate's understanding — and the resulting refusal looks like a content
gap rather than a wording mismatch.

### ⚠️ And the bottleneck is not the AI

The engine holds **34 items**. This repository has authored content for **8**. Writing a
distractor and naming the misconception it encodes is subject-expert work, and **that is the
scarce resource in this product** — not the model, and not the retrieval.

---

## The measured results

### IRT recovers ability it was not given (`irt.py`)

Simulated candidates with **known** ability, estimated from their responses:

| Metric | Result |
|---|---|
| Pearson r vs true ability | **0.9168** |
| Mean absolute error | **0.312 logits** |
| Systematic bias | **+0.011** (none) |
| SE at n = 2 items | **1.55** |
| SE at n = 21 items | **0.42** |
| A perfect score | θ = 2.33, **bounded by the prior and labelled as such** (MLE would return +∞) |

⚠️ Bias is checked **separately** from error: an estimator can be accurate on average and
systematically generous, which would make every candidate look ready.

### Item information collapses away from the candidate's level

```
b=0 → 0.2500        b=5 → 0.00665        38x less
```

**A question everybody gets right and one everybody gets wrong both measure nothing.** This is
why adaptive beats a fixed order, and why the selector never serves an item far from the
candidate's estimate.

### Spaced repetition hits its target — and has a measurable bound (`scheduler.py`)

| Cadence | Achieved retention (target 90%) |
|---|---|
| sub-day (continuous) | **90.0%** — the inversion is exact |
| once a day | **85.2%** — a **4.8-point** shortfall |

⚠️ **The shortfall is structural, not a bug.** A freshly lapsed card has stability 0.2 days, so
the scheduler wants to review it in 0.2 days — a once-a-day app cannot, and serving it at day 1
makes it five times overdue, where recall is **67.8%**. **Retention targets are bounded by
review granularity.**

And the counter-intuitive result: a **hard-but-successful** recall builds more stability than an
easy one. Measured: on-time **18.15** vs overdue **23.27**.

### ⚠️ The selector does not beat the baselines, and that is recorded

| policy | mean objective | bottleneck objective |
|---|---|---|
| **adaptive selector** | **+0.0416** | +0.0412 |
| random | +0.0483 | +0.0544 |
| least-recently-seen | +0.0477 | +0.0460 |
| **"practise your worst competency"** | **+0.1037** | +0.0857 |

**The trivial baseline wins by more than double.** Three findings came from diagnosing it, and
they are worth more than a tuned winner:

1. **Four of the five terms the JD names are constants on a fresh candidate** — measured spread
   across competencies: `gap` 0.000, `performance` 0.000, `uncertainty` 0.000, `recency` 0.000.
   ⚠️ **A term with no spread is not a term.**
2. **The objective was gameable.** Readiness within a domain was the *mean* of its competencies,
   and a mean responds strongly to raising one member — the winning baseline served **one
   competency 75 times out of 120**. **A metric a degenerate policy maximises is a broken
   metric.**
3. **The decision-relevant metric is a bottleneck, not a mean.** A certification is passed as a
   whole; a candidate who is outstanding at retrieval and hopeless at evaluation **fails**, with
   a reassuring mean right up to that moment. So the API returns both, and says which is which.

⚠️ And the **ablation came back degenerate**: dropping any single term changed the outcome by
exactly `0.00000` while the served sequences genuinely differed. The mechanism is measured too —
the simulated learning model only fires for `0.25 < p < 0.95`, so at θ = −1.4 only **1 of 7**
items can teach anything. **The harness's learning model decides the result, not the selector.**

⚠️ **What would settle it: real candidate response data.** Every number above comes from a
learner whose memory, learning rate and competency structure were chosen by me. A simulation can
show a selector is internally inconsistent; it cannot show any policy helps a person.

---

## The service layer (`api.py`)

FastAPI, because this is a product with clients rather than a notebook.

| Endpoint | What it returns |
|---|---|
| `GET /health` | what is **loaded** — item count, graph validity — not merely that the process is up |
| `GET /bank/items` · `GET /bank/competencies` | the graph, and per-competency coverage |
| `POST /readiness` | θ **and its standard error** per competency, plus mean and bottleneck |
| `POST /recommend` | the pick, the terms that produced it, and its **evidential basis** |
| `POST /schedule` | next due date per item, and current recall probability |

**Four decisions that make it a service rather than a wrapper:**

1. **Validation rejects.** An unknown `item_id` is **422** with the field named, not a silent
   skip — a typo must not look like a candidate who answered fewer questions. Weights that do
   not sum to 1.0 are **refused rather than renormalised**, because a silently-rescaled score is
   indistinguishable from a correct one.
2. **Abstention is first-class.** A candidate with no history gets **200** with
   `basis: "cold_start"` and a note saying the pick rests on the prior. ⚠️ Not a 404 (that would
   blame the route) and not a confident pick (that would be a measurement it does not have).
3. **Uncertainty travels with the number.** `reliable: false` when the estimate rests on too few
   items, because the server is the only place that knows.
4. **⚠️ A parity test.** The HTTP response and the direct library call must return the **same
   item, the same score to 1e-12, and the same term decomposition** — verified. Same move as the
   Inspect/DeepEval cross-check in track 13: *two implementations, one answer*. Without it a
   service layer silently becomes a second implementation of the thing it wraps.

```bash
pip install fastapi pydantic httpx
cd src && uvicorn api:app --reload --port 8000     # docs at /docs
```

⚠️ **`fastapi` is an optional extra.** The core maths (`irt`, `scheduler`, `selector`,
`competency`) runs on the **standard library alone**, so it is verifiable on a fresh clone before
anything is installed — the same rule the rest of this repository follows.

---

## ⚠️ What is NOT verified

- **No real candidate data.** The learner is simulated, with memory, a learning rate and a
  competency structure that I chose. **This is the single largest limitation.** Calibrating item
  parameters and validating the selector both need thousands of real attempts.
- **Item parameters are assigned, not calibrated.** `competency.py` says so at the point of use,
  because that is the exact trap `irt.py` documents.
- **The learning model is a simplification that turned out to dominate** (the `0.25 < p < 0.95`
  band made a weak competency almost unteachable). It needs replacing before any selector result
  from this harness means anything.
- **The selector loses to the baselines.** Measured, reported, and not tuned away.
- **No database.** The API holds state per request. A real system needs PostgreSQL — the JD names
  it — and the tables are implied by the dataclasses (`CandidateState`, `Card`, the graph).
- **No auth, no rate limiting, no multi-tenancy.** Those are product concerns and their absence is
  deliberate rather than overlooked.
- **⚠️ The LLM explainer has never been run against a live API.** It is implemented, uses
  `response_format: json_object`, and its citation validator is tested — but every explanation in
  this repository was produced by the **deterministic** explainer. Set `TUTOR_API_KEY` to run the
  model path; nothing here should claim more than that.
- **No dense retrieval.** BM25 only, so the repo runs on the standard library. Production would
  fuse BM25 with an encoder exactly as `RaggyEditor` does, and would need a stemmer better than
  the plural strip in `retrieval.py`.
- **The official answer key is per-question, not per-option.** Distractors are mapped to
  misconceptions by hand for 8 items; the other 26 have calibrated parameters and no authored
  text at all.

---

## Where this sits in the job description

| JD requirement | Status |
|---|---|
| *"competency models linking exams, domains, competencies, concepts, questions, performance"* | ✅ built, validated, and refuses dangling references |
| *"dynamic readiness and competency profiles"* | ✅ θ **with a standard error**, per competency and exam-weighted |
| *"mechanisms that determine what a candidate should practice next"* | ✅ built — ⚠️ **but it loses to a simple baseline, and that is documented** |
| *"competency gaps, previous performance, exam weighting, uncertainty, recency"* | ✅ all five implemented as terms; ⚠️ four are constants on a cold start, measured |
| *"personalized learning and recommendation systems"* | ✅ `THEORY.md` §3 explains why this **is** a recommender system |
| *"Develop AI-powered features using foundation-model APIs"* | ✅ `LLMExplainer` — ⚠️ **implemented but never run live; the deterministic explainer is what produced every result here** |
| *"Implement structured outputs"* | ✅ `response_format: json_object` plus a validator that **rejects an invented citation** |
| *"Design and develop RAG systems"* | ✅ `retrieval.py` — BM25 over approved content, corpus-known coverage, abstention |
| *"grounded in reliable and validated sources"* | ✅ every claim carries a citation that resolves to `GET /content/sources`, or the tutor refuses |
| *"identify the reasons behind incorrect answers"* | ✅ **distractor analysis** — the diagnosis is authored per wrong option, not generated |
| *"detect potential misconceptions"* | ✅ the authored `misconceptions` map |
| *"Ensure AI systems are reliable and maintainable"* | ✅ every module self-tests offline on the standard library |
| *"Python / FastAPI"* | ✅ `api.py`, with a parity test against the library |
| *"PostgreSQL"* | ❌ not used — SQLite/`dict` here; the JD names it and this does not demonstrate it |
| *"AWS / React / Next.js"* | ❌ out of scope for this track |
