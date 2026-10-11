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

## The idempotent write path (`idempotency.py`)

    python src/idempotency.py --selftest           # 22 checks, single process
    python tests/test_idempotency_http.py          # 14 checks, through the server

⚠️ **The problem.** A client sends a write, the network drops the response, and the
client retries — correctly, because it cannot know whether the write happened. A
naive server records the answer twice and every number downstream is wrong. This is
the defining problem of a payments system: **a retried transfer must not send money
twice.**

### The four rules

| Situation | Behaviour |
|---|---|
| Same key, same payload | return the **original response**, do not execute again |
| Same key, **different** payload | **409 conflict** — a key identifies one request |
| Same key, still in flight | **409** — a second write must not start |
| No key | executes every time, and says so |

⚠️ Rule 2 is the one most implementations get wrong. Returning the stored response
would silently discard the second request's intent; writing again would double-apply
under one key. It is a conflict, not a replay.

⚠️ It is not "check if it exists, then write". That has a race: two requests with
the same key both look, both find nothing, both write. The claim is a separate
committed transaction, so a concurrent request can **see** it.

### Measured, over HTTP

```
first call            200  written=True   replayed=False
4 retries             replayed=[True, True, True, True]  written=[False, ...]
5 calls total         -> 1 attempt row
same key, new body    409  conflict
no key                200  written=True   (not idempotent, by design)
```

**Six HTTP calls for one answer produced two attempt rows** — one keyed, one
deliberately unkeyed. The four retries wrote nothing and returned the original
response.

### Reconciliation

    GET /reconcile

⚠️ **This is the half that gets skipped.** An idempotency key prevents the *known*
retry; reconciliation catches the ones nobody saw. It reports:

| Finding | Meaning |
|---|---|
| `double_applied` | ⚠️ two attempts under one key — the failure the module prevents |
| `stuck` | claimed, never completed — a process died between the two phases |
| `ghost` | marked done with no attempt behind it |
| `unkeyed` | attempts written with no key — not wrong, but visible |

### ⚠️ Three bugs, and only the HTTP test found them

The module selftest passed the whole time. It runs single-threaded, in one process,
and never crossed a thread or a request boundary.

1. **`SQLite objects created in a thread can only be used in that same thread.`**
   FastAPI runs a *synchronous* endpoint in a threadpool, so a connection made at
   import time is used from a worker thread. Fixed with a connection per thread.
2. **`no such table: idempotency_keys`.** With a connection per thread, the schema
   created on the bootstrap thread does not exist for the worker. Every new
   connection now creates it — `IF NOT EXISTS`, so it is free after the first time.
3. ⚠️ **Every retry WROTE AGAIN, with no error at all.** The store was `":memory:"`,
   and an in-memory database is **per connection** — so each worker thread had its
   own empty database, the lookup never found anything, and every response honestly
   said `written=True`.

⚠️ Failures 1 and 2 raise. Failure 3 returns **200 and looks correct**. That is the
shape of bug that costs money: nothing crashes, the dashboard is green, and the
write happened twice. **A module selftest proves the logic; only a test through the
server proves the wiring.**

## The same retrieval, in LangChain (`langchain_pipeline.py`)

    python src/langchain_pipeline.py --selftest     # 18 checks
    python src/langchain_pipeline.py --parity       # the comparison table

⚠️ **This is not a rewrite. It is a comparison, and the comparison is the
deliverable.** `retrieval.py` already does hybrid retrieval with a hand-rolled BM25
and a coverage gate. The question is not *"can it be done in LangChain"* — obviously
it can — but **"what does the framework buy, and what does it cost?"**

What LangChain provides: `Document` for the corpus, `BM25Retriever` as a second
independent BM25, `BaseRetriever` for adapting our index, and RRF composed on top.

### ⚠️ Measured: LangChain's retriever cannot abstain

The most useful row is a query that shares **no term** with the corpus — no passage
contains `abstain`, `abstention` or `tutor`:

```
"when should the tutor abstain"

  our Index            -> []            (it filters score == 0)
  LangChain BM25       -> 3 passages    (none contain the terms)
  raw rank_bm25 scores -> all 10 non-zero, max 2.708
```

⚠️ **LangChain's retriever returns k documents for any query, including one the
corpus has never seen.** Abstention is the product decision this repository is built
around — it is what stops the tutor inventing an answer — and **a retriever that
always returns k makes abstention undetectable downstream, because there is nothing
to detect.**

⚠️ That is a property of the retriever, not a defect in the framework. It is the kind
of thing a comparison finds and a rewrite removes. Pinned as a test, so a future
version that starts abstaining fails loudly instead of changing this quietly.

### Parity, over the same queries

| | |
|---|---|
| Adapter path identical to a direct call | **5/5** |
| LangChain BM25 agrees on the top-1 | 4/5 |
| LangChain BM25 agrees on the whole set | **1/5** |

⚠️ The second and third rows are the point. **Two retrievers that agree because they
are the same retriever add nothing to a fusion.** Different tokenisers and different
`k1`/`b` produce genuinely different rankings, and RRF is what turns that into a
strength rather than a conflict.

⚠️ The adapter row is the only place an equality assertion is honest: it wraps
`Index.search` and holds no logic of its own, so if it ever disagrees the fault is in
the adapter or the framework — never in the ranking.

### Why RRF is composed here, not `EnsembleRetriever`

1. `EnsembleRetriever` moved out of `langchain-community` to the main `langchain`
   meta-package. Pulling that for one class is a poor trade when the algorithm is
   fifteen lines.
2. ⚠️ **RRF keeps ranks and discards scores, which is exactly why it works.** BM25
   scores and cosine similarities are not on a comparable scale — `device-search`
   measured a garbage query at 0.649 cosine against 0.632 for the worst *real* match
   — so score-level fusion is fusing noise. Writing it here makes that visible and
   puts the k constant somewhere it can be justified (60, from the original paper).

⚠️ `langchain-community` also emits a deprecation warning: it is being sunset in
favour of standalone integration packages. The ecosystem is mid-reorganisation,
which is a reason to depend on less of it.

## Vector search: exact, approximate, and a real Postgres

    python src/vector_index.py --selftest      # 17 checks, stable across runs
    python src/vector_index.py --curve         # FAISS recall vs latency
    python tests/test_pgvector.py              # parity against pgvector (skips without one)

### FAISS

⚠️ **This measures the INDEX, not the embeddings.** FAISS has no opinion about
whether "nearest" means anything useful, so a recall figure over vectors that
represent nothing is not a claim about retrieval quality. The vectors here are
generated deterministically — the property under test is the index's behaviour, and
that is independent of where the vectors came from.

**The hard property:** exact `IndexFlatIP` agrees with brute-force NumPy on **every
query**. *"It's supposed to be exact"* is not a measurement, and an approximate
index silently returning the wrong neighbour looks exactly like a correct one.

```
efSearch   recall@10    p50 ms    p95 ms
      10      0.3190     0.044     0.060
      64      0.7800     0.141     0.158
     256      0.9930     0.377     0.445
     512      1.0000     0.644     0.728
```

#### ⚠️ And the important part is what got measured wrong, twice

| Asserted | Measured | |
|---|---|---|
| "a seeded HNSW run **is** reproducible" | 0.970 / 0.980 / 0.980 | ✗ |
| "a seeded HNSW run is **not** reproducible" | 0.990 / 0.990 / 0.990 | ✗ |
| "uniform vectors are easier for ANN" | clustered 0.610 / uniform 0.527 | ✗ |
| "clustered is easier" (the correction) | clustered 0.573 / uniform 0.577 | ✗ |

**Three measurements, three different directions.** Every one was a single reading
of a noisy quantity.

⚠️ Asserting any of them would have shipped a **flaky test and a false claim** — and a
flaky test gets deleted, taking the measurement with it. The checks no longer assert
a direction: they assert the effect **does not exceed its own noise** and report the
spread. At this sample size there is no clustered-vs-uniform result to report, and
clustering is chosen for **realism**, not difficulty.

### pgvector

⚠️ **A real server, not a re-implementation of myself.** The in-process index and
pgvector were written by different people, in different languages, with different
rounding and a different execution engine. **Parity between them is evidence that
neither is wrong**; agreement with my own NumPy code is evidence of nothing.

| Check | Result |
|---|---|
| Vectors round-trip bit-identically | **2000/2000**, max diff 0.000e+00 |
| pgvector's exact top-10 == brute force | **25/25 queries** |
| `<#>` is negative inner product | `[1,2,3] <#> [1,2,4] = -17.0` |
| A wrong-length vector is refused, not truncated | ✅ |

```
pgvector 0.8.7, 2,000 vectors, 128 dims, recall@10
ef_search   recall@10     p50 ms     p95 ms
       10      0.4680      0.538      0.840
       32      0.7800      0.617      0.865
       64      1.0000      0.871      1.548
      128      1.0000      0.834      1.073
```

⚠️ **Recall and latency are reported together, never separately** — the acceptance
rule is that a latency number alone is meaningless, and it cuts both ways.

⚠️ And parity is re-checked **with the HNSW index dropped**. If the planner had used
the approximate index for an `ORDER BY <#>` query, the "exact" parity above would
have been measuring an approximate index — and it would have passed, because the
recall is high. That is the classic ANN benchmark error, and this check rules it out.

## Open it and use it

```bash
./run.sh ui        # → http://127.0.0.1:8000
```

⚠️ **One HTML file, no build step, no CDN, no npm.** The page is served by the API it
calls, so there is no CORS problem, no second process and no framework — and it works
offline. A demo that needs `npm install` before it shows anything is a demo nobody runs.

### The three panels, and why those three

⚠️ A README can claim anything. These are the three things that are hard to believe
without using them.

⚠️ **And if you paste nothing, the page says so.** It reads:

> ⚠️ **No document supplied — using my bundled 10-passage syllabus.** Everything below
> answers from that. Paste your own text above to replace it.

⚠️ Without that notice every answer below would look like it came from the reviewer's
own content when it did not — the panel would be **misleading rather than empty**. A
fallback that does not name itself is worse than no fallback.
`tests/test_ui_browser.py` asserts it appears on load and returns after a reset.

**1. The tutor will refuse.** Pick a question, pick an answer, and the verdict is
either *grounded in approved content* (with citations) or **abstained**. ⚠️ Abstention
is a product decision, not an error — a tutor that cannot abstain will invent an
explanation, and a candidate told they misunderstood something they did not is worse
off than one told nothing.

**2. A write sent five times applies once.** Press the button and the table shows five
requests, one write, four replays, one attempt row, and a clean reconciliation. Then
the second button reuses the **same key with a different body** and gets a **409**.
⚠️ That second button is the interesting one: returning the first response would
silently discard the second request, and writing again would double-apply under one
key. It is the decision most implementations get wrong.

**3. What to practise next.** Not "the weakest topic" — a weighted function over five
named terms, each readable in `src/selector.py`, with the terms printed as the
breakdown. ⚠️ And `basis: cold_start` appears when the pick rests on the prior rather
than this candidate's history, so a guess is never presented as a recommendation.

### Verified, not asserted

| The UI calls | Result |
|---|---|
| `/bank/items` + `/content/sources` | 34 items, **8** with authored content |
| `/explain` | 200, `grounded=True` |
| `/attempts` ×5 on one key | `written=[True, False, False, False, False]` |
| `/reconcile` | **1** attempt row, clean |
| `/attempts`, same key, different body | **409** |
| `/recommend` | `q-003`, score 1.060, `cold_start`, 7 terms |

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
