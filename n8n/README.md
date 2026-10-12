# n8n — the practice loop, orchestrated

    python tests/test_workflow.py        # validates the artefact and replays its logic

```
POST /webhook/practice            ← a bank item; can NEVER abstain
  → Explain → Grounded? ──true──→ Recommend → Respond (answered)
                        └─false─→ Queue for human review → Respond

POST /webhook/ask                 ← ⚠️ a candidate's own question; CAN abstain
  → Ask     → Grounded? ──true──→ Respond (answered)
                        └─false─→ Queue for human review → Respond (abstained)
```

⚠️ **TWO TRIGGERS, AND THE SECOND ONE IS THE FIX.** `/explain` only serves items a
subject expert wrote a diagnosis for, and an item only gets authored content if its
concept is *in the syllabus* — so grounding succeeded every time and the queue branch
was unreachable. A **candidate's own question** is the input that can reach it.

⚠️ **The branch is the point.** `POST /explain` returns `grounded`, and a grounded
answer goes to the candidate while an **abstention is routed to a human queue**. That
is the product decision this repository is built around: a tutor that cannot abstain
will invent an explanation, and a candidate told they misunderstood something they
did not is worse off than one told nothing.

## Running it

⚠️ **No container required.** n8n is an npm package (⚠️ but it is 2.6 GB of
`node_modules`, and installing it took 11 minutes — twice, the first time failing on a
dropped connection):

```bash
npm install n8n && npx n8n import:workflow --input=n8n/practice-loop.workflow.json
```

The Docker route below also works, and is what most documentation shows:

```bash
docker run -d --name n8n-practice -p 5678:5678 n8nio/n8n

docker cp n8n/practice-loop.workflow.json n8n-practice:/data/
docker exec n8n-practice n8n import:workflow --input=/data/practice-loop.workflow.json

docker exec -e API_BASE=http://host.docker.internal:8000 \
            -e QUEUE_URL=http://host.docker.internal:8001/queue \
            n8n-practice n8n execute --id <workflow-id>
```

⚠️ **`host.docker.internal`, not `localhost`.** From inside the container, `localhost`
is the container — the API is on the host, and `localhost` resolves to nothing. This
is the single most common reason a self-hosted n8n workflow fails on its first run.

⚠️ `API_BASE` and `QUEUE_URL` are read from the environment. A hardcoded host in a
committed workflow is a workflow that only runs on one machine.

## What is asserted, and what is measured

| | |
|---|---|
| Every connection names a real node | ✅ |
| No orphaned node; everything reachable from the webhook | ✅ |
| The IF node has **two** outputs | ✅ |
| The IF branches on `grounded` | ✅ |
| The false branch does **not** reach `Recommend` | ✅ |
| The false branch **does** reach the human queue | ✅ |
| A malformed `chosen` is refused (422), so `grounded` can be trusted | ✅ |

⚠️ **n8n resolves connections by node NAME, and a name that does not exist is not an
error — the branch simply never fires.** That is why the structure is checked as a
graph and not by eye.

## ✅ Verified running in real n8n

**n8n 2.35.7 — imported, activated, and both branches fired.**

```
POST /webhook/ask  {"question": "how do I bake sourdough bread"}
  -> {"grounded":false,"abstained":true,"queued_for_review":true,
      "reason":"no passage shares any content word with the question","coverage":0}
  ⚠️ and the human queue received 1 message

POST /webhook/ask  {"question": "why does hybrid retrieval exist"}
  -> {"grounded":true,"coverage":1,
      "citations":[{"passage_id":"d-retrieval#1","source":"Why hybrid retrieval exists"}, …]}
```

## ⚠️ Two things only running n8n could reveal

**1. `SQLITE_CONSTRAINT: NOT NULL constraint failed: workflow_entity.id`**

My workflow JSON was missing a **top-level `id`**. Every structural check passed —
connections, orphans, reachability, the two-output IF — because **the graph was valid
and the record was not.** A well-formed graph is not a well-formed n8n record.

**2. `access to env vars denied` → `undefined`**

⚠️ **n8n ≥ 2.x blocks `$env` in expressions by default.** `{{ $env.API_BASE }}`
resolved to `undefined`, so the HTTP node called `undefined/ask` and the execution
errored — while the webhook still returned **200 with an empty body.**

⚠️ That combination is the worst kind: a green status code and a dead workflow. The
fix is `N8N_BLOCK_ENV_ACCESS_IN_NODE=false`, or workflow variables instead of env.

⚠️ **Neither of these is findable by reading the JSON**, which is why the workflow sat
"graph-validated, never executed" for so long. It needed to be run.

## ⚠️ The abstention branch — FIXED

**The branch is now reachable.** It was wired correctly and could not be triggered,
because `/explain` only serves items a subject expert wrote a diagnosis for, and those
are in the syllabus by construction — so grounding succeeded every time.

⚠️ **The missing entry point was the candidate's own question.** `POST /ask` takes free
text and runs the same gate, so the refusal is now demonstrable:

    POST /ask  {"question": "why does hybrid retrieval exist"}
      -> grounded=true,  coverage 1.000

    POST /ask  {"question": "how do I bake sourdough bread"}
      -> grounded=false, coverage 0.000   ⚠️ abstained

⚠️ **The fix belongs in the workflow, not in the test.** Add a second trigger for the
candidate's free-text question and route it on the same `grounded` field. The branch
logic was always right; the input that could reach it did not exist.

## ⚠️ The original finding, kept because it is the interesting part

**The abstention branch is wired correctly and cannot currently be triggered through
`/explain`.**

Measured across the item bank: **8 items have authored content, 26 return 404, and
all 8 authored items are grounded. None abstains.**

⚠️ The reason is structural, not a bug. `/explain` only serves items a subject expert
has written a diagnosis for, and an item only gets authored content if its concept is
in the syllabus — so grounding succeeds by construction. **An abstention happens when
a *candidate* asks about something outside the corpus**, which is the tutor's
free-text entry point, not a bank item.

⚠️ The gate itself does abstain — checked directly:

```
ground(Index(load_passages()), "how do I bake sourdough bread")
  -> grounded=False, best coverage 0.000
```

So the false branch is a **real state**, not dead code. It is reachable in principle
and not reachable from this entry point with this content.

⚠️ **A workflow claim whose trigger cannot be demonstrated is a claim to make
carefully.** A test asserting only "the branch exists" would have hidden this, which
is why the check asserts what was *measured* instead.

## What the test does NOT do

It does not start n8n. The `n8nio/n8n` image is large, and a test that cannot run on
a fresh clone is a test that gets deleted. So it checks the two things that actually
break — **the artefact** as a graph, and **the assumptions** replayed against the
live API — and leaves the container to the commands above.
