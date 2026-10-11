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

⚠️ **No container required.** n8n is an npm package:

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
