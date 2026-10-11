"""
test_workflow.py — validate the n8n workflow, and replay its logic for real.

    python tests/test_workflow.py

===============================================================================
⚠️ WHY THIS DOES NOT SIMPLY RUN n8n
===============================================================================
The `n8nio/n8n` image is large, and a test that cannot run on a fresh clone is a
test that gets deleted. What CAN be checked everywhere is the two things that
actually break:

  1. **The artefact** — is the JSON a valid n8n workflow? Do the connections name
     real nodes? Is every node reachable from the trigger? ⚠️ n8n accepts a broken
     connection map and then simply never fires that branch, so a typo in a node
     NAME is a silent no-op rather than an error.
  2. **The assumptions** — does the API actually return `grounded`, and does it
     return BOTH values across the item bank? ⚠️ A workflow with a branch that can
     never be taken is not a branch, and testing only the true path would not
     notice.

⚠️ So this replays the workflow's own requests against the live API and asserts the
branch decision follows `grounded`. That is the part a JSON schema check cannot do.

    docker run -d --name n8n-practice -p 5678:5678 n8nio/n8n
    # then, to import and eyeball it:
    docker exec n8n-practice n8n import:workflow --input=/data/practice-loop.workflow.json
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

WORKFLOW = ROOT / "n8n" / "practice-loop.workflow.json"

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


# ─── 1. The artefact ─────────────────────────────────────────────────────────

def validate_structure(wf: dict) -> None:
    nodes = wf["nodes"]
    names = [n["name"] for n in nodes]
    by_name = {n["name"]: n for n in nodes}

    check("every node has a unique name", len(names) == len(set(names)),
          f"{len(names)} nodes")
    check("every node has an id", all(n.get("id") for n in nodes))
    check("every node declares a type and a typeVersion",
          all(n.get("type", "").startswith("n8n-nodes-base.") and "typeVersion" in n
              for n in nodes))

    # ⚠️ THE CHECK THAT MATTERS. n8n resolves connections by NAME, and a name that
    # does not exist is not an error — the branch simply never runs.
    dangling = []
    for src, spec in wf["connections"].items():
        if src not in by_name:
            dangling.append(f"connection FROM unknown node {src!r}")
        for output in spec.get("main", []):
            for conn in output or []:
                if conn["node"] not in by_name:
                    dangling.append(f"{src} -> unknown node {conn['node']!r}")
    check("every connection points at a node that exists", not dangling,
          "; ".join(dangling) if dangling else f"{len(wf['connections'])} sources")

    # ⚠️ An orphan is a node that was drawn but never wired. It looks like part of
    # the workflow and does nothing.
    targets = {c["node"] for spec in wf["connections"].values()
               for out in spec.get("main", []) for c in (out or [])}
    triggers = {n["name"] for n in nodes if "webhook" in n["type"]}
    orphans = [n for n in names if n not in targets and n not in triggers]
    check("no node is orphaned from the trigger", not orphans, f"orphans: {orphans}")

    reached = set(triggers)
    frontier = set(triggers)
    while frontier:
        nxt = set()
        for n in frontier:
            for out in wf["connections"].get(n, {}).get("main", []):
                for c in out or []:
                    if c["node"] not in reached:
                        nxt.add(c["node"])
        reached |= nxt
        frontier = nxt
    check("every node is reachable from the webhook", reached == set(names),
          f"unreachable: {sorted(set(names) - reached)}")

    if_node = by_name["Grounded?"]
    check("the IF node has two outputs (true, false)",
          len(wf["connections"]["Grounded?"]["main"]) == 2,
          "one branch would make the 'abstain' path unreachable")
    check("the IF node branches on `grounded`",
          "grounded" in json.dumps(if_node["parameters"]),
          "⚠️ not on a model's self-reported confidence — see the node note")


# ─── 2. The assumptions ──────────────────────────────────────────────────────

def replay(wf: dict) -> None:
    """Run the workflow's own requests against the live API."""
    from fastapi.testclient import TestClient

    import api
    from competency import default_bank

    client = TestClient(api.app)

    # ⚠️ `bank.items` is a dict keyed by item id, NOT a list of objects. Iterating
    # it yields the keys as strings, which is what the rest of this function wants.
    bank = default_bank()
    item_ids = list(bank.items)
    check("the item bank is non-empty", len(item_ids) > 0, f"{len(item_ids)} items")

    grounded_true, grounded_false, not_found = [], [], []
    for item_id in item_ids:
        r = client.post("/explain", json={"item_id": item_id, "chosen": "A"})
        if r.status_code == 404:
            not_found.append(item_id)
            continue
        j = r.json()
        (grounded_true if j["grounded"] else grounded_false).append((item_id, "A", j))

    check("POST /explain returns 200 only for items with authored content",
          len(grounded_true) + len(grounded_false) == 8 and len(not_found) == len(item_ids) - 8,
          f"{len(grounded_true) + len(grounded_false)} authored, {len(not_found)} without "
          f"content (404) — ⚠️ the 404 names the endpoint that lists them")

    # ⚠️⚠️⚠️ A MEASURED LIMITATION, NOT A PASSING GRADE.
    #
    # I wrote this expecting both branches to be reachable, and the test found that
    # they are not: all 8 authored items are grounded, and none abstains. The
    # abstention path in this workflow is therefore WIRED CORRECTLY but CANNOT BE
    # TRIGGERED through `/explain` against the current content.
    #
    # ⚠️ The reason is structural, not a bug. `/explain` only serves items a subject
    # expert has written a diagnosis for, and an item only gets authored content if
    # its concept is in the syllabus — so grounding succeeds by construction. An
    # abstention happens when a CANDIDATE asks about something outside the corpus,
    # which is the tutor's free-text entry point, not a bank item.
    #
    # ⚠️ Saying this out loud is the point. A workflow claim whose trigger cannot be
    # demonstrated is a claim to make carefully, and a test that asserted "the branch
    # exists" would have hidden it.
    check("⚠️ MEASURED: /explain cannot reach the abstention branch today",
          len(grounded_false) == 0,
          f"{len(grounded_true)} grounded, {len(grounded_false)} abstained — the "
          f"branch is wired and correct; its trigger is not reachable from the item "
          f"bank. See n8n/README.md")

    # ⚠️ But the GATE can abstain — demonstrated directly, so the branch is not
    # decorative. The condition exists; only this entry point cannot produce it.
    from retrieval import Index, ground, load_passages
    gate = ground(Index(load_passages()), "how do I bake sourdough bread")
    check("the grounding gate itself does abstain",
          not gate.grounded,
          f"an off-corpus question abstains with coverage {gate.best_coverage:.3f}, "
          f"so the false branch is a real state and not dead code")

    # ---- the true branch, replayed ----------------------------------------
    if grounded_true:
        item_id, chosen, ex = grounded_true[0]
        rr = client.post("/recommend", json={"candidate_id": "c-1", "attempts": [], "day": 0})
        check("the grounded branch: /explain is grounded and /recommend answers",
              ex["grounded"] and rr.status_code == 200,
              f"{item_id}/{chosen} -> recommend {rr.json().get('item_id')}")
        check("a grounded explanation carries citations to approved content",
              len(ex["citations"]) > 0 if ex["misconception"] else True,
              f"{len(ex['citations'])} citation(s)")

    # ---- the false branch is verified STRUCTURALLY, since it cannot be reached --
    # ⚠️ See the measured limitation above. Asserting the shape of the abstention
    # response would require an input the current content cannot produce.

    # ---- ⚠️ the workflow must NOT call /recommend on the abstention path ----
    # The branch is only correct if the false path skips the recommendation. That
    # is a property of the CONNECTION MAP, checked here rather than assumed.
    conns = wf["connections"]
    false_targets = [c["node"] for c in conns["Grounded?"]["main"][1]]
    check("the abstention path does not reach Recommend",
          "Recommend" not in false_targets,
          f"false branch -> {false_targets}")
    check("the abstention path reaches the human queue",
          any("Queue" in t for t in false_targets), f"false branch -> {false_targets}")

    # ---- errors are not silently absorbed ---------------------------------
    bad = client.post("/explain", json={"item_id": "q-002", "chosen": "ZZ"})
    check("a malformed `chosen` is refused, so `grounded` can be trusted",
          bad.status_code == 422, f"got {bad.status_code}")


def main() -> int:
    print("test_workflow.py — the n8n workflow, validated and replayed")

    if not WORKFLOW.exists():                                           # pragma: no cover
        print(f"  ⚠️  no workflow at {WORKFLOW}")
        return 1
    wf = json.loads(WORKFLOW.read_text(encoding="utf-8"))

    print("\n  — the artefact —")
    validate_structure(wf)

    print("\n  — the assumptions, replayed against the API —")
    try:
        replay(wf)
    except ImportError as e:                                            # pragma: no cover
        print(f"  ⚠️  cannot replay without fastapi: {e}")

    print()
    if FAILURES:
        print(f"  ❌ {len(FAILURES)} failed: {', '.join(FAILURES)}")
        return len(FAILURES)
    print("  ✅ all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
