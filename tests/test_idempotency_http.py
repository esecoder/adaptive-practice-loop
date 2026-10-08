"""
test_idempotency_http.py — the idempotent write path, over real HTTP.

    python tests/test_idempotency_http.py

===============================================================================
⚠️ WHY THIS FILE EXISTS SEPARATELY FROM `idempotency.py --selftest`
===============================================================================
The module selftest exercises the store's functions directly. It runs on one
thread, in one process, and it passed — while the HTTP endpoint was broken in two
different ways:

1. **"SQLite objects created in a thread can only be used in that same thread."**
   FastAPI runs a *synchronous* endpoint in a threadpool, so the connection made at
   import time is used from a worker thread.

2. **"no such table: idempotency_keys"** — after fixing (1) with a connection per
   thread, the schema created on the bootstrap thread did not exist for the worker.

3. ⚠️ And then the worst one: every retry WROTE AGAIN, with **no error at all**.
   The store was `":memory:"`, and an in-memory database is per connection — so each
   worker thread had its own empty database, the idempotency lookup never found
   anything, and every response honestly said `written=True`.

⚠️ Failures 1 and 2 raise. Failure 3 does not: it returns 200 and looks correct.
That is the case this file exists for, and it is the shape of bug that costs money
in a payments system.

⚠️ So: a module selftest proves the logic. Only a test through the server proves
the wiring. Both are here.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    from fastapi.testclient import TestClient
except ImportError:                                            # pragma: no cover
    print("  ⚠️  fastapi[testclient] not installed — skipping the HTTP checks")
    raise SystemExit(0)

import api

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    print("test_idempotency_http.py — the write path through the server")
    client = TestClient(api.app)
    body = {"candidate_id": "C1", "item_id": "ir-01", "correct": True, "day": 0.0}
    H = {"Idempotency-Key": "order-abc-123"}

    # ---- 1. the first call writes ------------------------------------------
    r1 = client.post("/attempts", json=body, headers=H)
    check("the first keyed call writes", r1.status_code == 200 and r1.json()["written"])
    check("the first keyed call is not a replay", r1.json()["replayed"] is False)

    # ---- 2. ⚠️ FOUR RETRIES, NO SECOND WRITE --------------------------------
    retries = [client.post("/attempts", json=body, headers=H) for _ in range(4)]
    check("every retry is reported as a replay",
          all(r.json()["replayed"] for r in retries),
          f"replayed={[r.json()['replayed'] for r in retries]}")
    check("no retry reports a write",
          all(not r.json()["written"] for r in retries))
    check("every retry returns a 200, not an error",
          all(r.status_code == 200 for r in retries),
          "a client that retried must get the same answer, not a failure")
    check("⚠️ FIVE CALLS LEAVE EXACTLY ONE ATTEMPT",
          client.get("/reconcile").json()["attempts"] == 1,
          f"{client.get('/reconcile').json()['attempts']} row(s)")

    # ---- 3. the same key with a different body is a conflict ---------------
    r3 = client.post("/attempts", json=dict(body, correct=False), headers=H)
    check("the same key with a different body is refused with 409",
          r3.status_code == 409, f"got {r3.status_code}")
    check("the conflicting call wrote nothing",
          client.get("/reconcile").json()["attempts"] == 1)

    # ---- 4. no key is not idempotent ---------------------------------------
    before = client.get("/reconcile").json()["attempts"]
    client.post("/attempts", json=body)
    client.post("/attempts", json=body)
    after = client.get("/reconcile").json()["attempts"]
    check("an unkeyed call writes every time", after - before == 2,
          f"{after - before} row(s) for 2 calls — which is what non-idempotent means")

    # ---- 5. reconciliation is clean, and separates keyed from unkeyed ------
    rec = client.get("/reconcile").json()
    check("reconciliation reports no double-apply", rec["findings"]["double_applied"] == [])
    check("reconciliation reports no ghosts", rec["findings"]["ghost"] == [])
    check("reconciliation reports no stuck keys", rec["findings"]["stuck"] == [])
    check("reconciliation counts the unkeyed writes separately",
          rec["findings"]["unkeyed"][0]["count"] == 2)

    # ---- 6. the endpoint advertises the header -----------------------------
    schema = client.get("/openapi.json").json()
    params = schema["paths"]["/attempts"]["post"].get("parameters", [])
    check("Idempotency-Key is a declared header, so a client can find it",
          any(p.get("name") == "Idempotency-Key" for p in params))

    print()
    if FAILURES:
        print(f"  ❌ {len(FAILURES)} failed: {', '.join(FAILURES)}")
        return len(FAILURES)
    print("  ✅ all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
