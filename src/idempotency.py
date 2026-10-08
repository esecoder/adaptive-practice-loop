"""
idempotency.py — an idempotent write path, and the reconciliation that makes it
trustworthy.

    python src/idempotency.py --selftest

===============================================================================
WHY THIS MODULE EXISTS
===============================================================================
A client sends a request. The network drops the response. The client does not know
whether the write happened, so it retries — because that is the correct thing to
do. If the server is naive, the learner now has two attempts recorded for one
answer, and every number downstream is wrong.

⚠️ **This is the defining problem of a payments system.** A cross-border transfer
that is retried must not send money twice, and the reconciliation job that catches
it when it does is a first-class part of the product, not an afterthought.

⚠️ The pattern is small and it is not "check if it exists, then write". That has a
race: two requests with the same key both look, both find nothing, both write.

The rules implemented here
--------------------------
1. **Same key, same payload** → return the ORIGINAL response, do not execute again.
2. **Same key, different payload** → REFUSE. A key identifies one request, and a
   replay with different content is either a client bug or an attack. Returning the
   first response would silently drop the second request's intent.
3. **Same key, still in flight** → REFUSE. A second write must not start while the
   first is unresolved.
4. **No key** → execute every time. That is what non-idempotent means, and it is
   the behaviour this module exists to make optional rather than mandatory.

⚠️ The record and the effect are written in ONE transaction. A key marked "done"
with no attempt behind it is a lie the reconciliation job will find, and an attempt
with no key behind it cannot be retried safely. Either both or neither.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# ⚠️ How long an in-flight claim may sit before reconciliation calls it stuck. A
# process that dies between claiming a key and recording the result leaves exactly
# this state, and without a timeout the key is dead forever — the client retries,
# gets "still processing", and retries again.
STUCK_AFTER_SECONDS = 30.0


class IdempotencyError(Exception):
    """Base class, so a caller can catch the whole family."""


class IdempotencyConflict(IdempotencyError):
    """The key exists and this request is not the request that created it."""


class IdempotencyInFlight(IdempotencyError):
    """The key exists and its first execution has not finished."""


# ─── Fingerprinting ──────────────────────────────────────────────────────────

def fingerprint(payload: Any) -> str:
    """A stable hash of the request.

    ⚠️ `sort_keys=True` matters. Without it, two dicts with the same content in a
    different insertion order hash differently, and every replay is reported as a
    CONFLICT — a correctness bug that looks exactly like a security feature.
    """
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


# ─── Schema ──────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS idempotency_keys (
    key          TEXT PRIMARY KEY,
    fingerprint  TEXT NOT NULL,
    status       TEXT NOT NULL CHECK (status IN ('in_flight', 'done')),
    response     TEXT,
    claimed_at   REAL NOT NULL,
    completed_at REAL
);

CREATE TABLE IF NOT EXISTS attempts (
    attempt_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    idempotency_key TEXT,                       -- NULL for a keyless write
    learner_id      TEXT NOT NULL,
    question_id     TEXT NOT NULL,
    correct         INTEGER NOT NULL,
    created_at      REAL NOT NULL
);

-- ⚠️ The index that makes double-apply detectable. Without a UNIQUE constraint the
-- database will happily hold two attempts for one key, and the only thing that
-- would notice is a human reading the reconciliation report.
CREATE INDEX IF NOT EXISTS idx_attempts_key ON attempts(idempotency_key);
"""


@dataclass(frozen=True)
class Outcome:
    """What a call to `execute` produced."""
    result: dict[str, Any]
    replayed: bool
    attempt_id: int | None

    @property
    def executed(self) -> bool:
        return not self.replayed


# ─── The store ───────────────────────────────────────────────────────────────

class IdempotencyStore:
    """A SQLite-backed idempotent write path.

    ⚠️ `:memory:` is fine for tests but a file is what makes the crash behaviour
    real: a process that dies mid-transaction leaves the file as evidence.
    """

    def __init__(self, path: str | Path | None = None):
        # ⚠️⚠️ A CONNECTION PER THREAD, AND THIS IS NOT OPTIONAL.
        #
        # FastAPI runs a synchronous endpoint in a THREADPOOL, so a connection made
        # at import time is used from a different thread and sqlite3 refuses it:
        # "SQLite objects created in a thread can only be used in that same thread."
        #
        # ⚠️ The module's own selftest passed — it runs single-threaded and never
        # crossed a thread. Only the HTTP test found this, which is the argument for
        # testing through the real server rather than only through the functions.
        #
        # ⚠️ A temp FILE rather than ":memory:", because an in-memory database is
        # PER CONNECTION: with one connection per thread, `:memory:` would give every
        # thread its own empty database and the idempotency check would never fire.
        # Silently. It would look like it worked.
        if path is None:
            fd, tmp = tempfile.mkstemp(prefix="idem-", suffix=".db")
            os.close(fd)
            path = tmp
            self._owns_file = True
        else:
            self._owns_file = False
        self.path = str(path)
        self._local = threading.local()
        self._conn()   # creates this thread's connection, and its schema

    def _conn(self) -> sqlite3.Connection:
        """This thread's connection, created on first use.

        ⚠️ `isolation_level=None` gives explicit transaction control: the claim is
        committed on its own, so another thread can SEE it. If the claim were held
        open until the write finished, two concurrent requests would both read "no
        key" — the race this module exists to close.
        """
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, isolation_level=None, timeout=10)
            conn.row_factory = sqlite3.Row
            # WAL + a busy timeout is what stops the second writer failing with
            # "database is locked" instead of reading the claim.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=8000")
            # ⚠️ THE SCHEMA IS CREATED ON EVERY NEW CONNECTION, not once at init.
            # A connection is per-thread, so a schema made on the bootstrap thread
            # does not exist for the threadpool worker that serves the request —
            # measured: "no such table: idempotency_keys" on the first HTTP call,
            # while the single-threaded selftest passed.
            # `IF NOT EXISTS` makes this free after the first time.
            conn.executescript(SCHEMA)
            self._local.conn = conn
        return conn

    # ── the write path ───────────────────────────────────────────────────────

    def execute(
        self,
        key: str | None,
        payload: dict[str, Any],
        operation: Callable[[], dict[str, Any]],
        *,
        learner_id: str,
        question_id: str,
        correct: bool,
        now: float | None = None,
    ) -> Outcome:
        """Run `operation` at most once for `key`.

        `operation` returns the response body. It is only called when this is a
        genuinely new request.
        """
        now = time.time() if now is None else now

        # ⚠️ No key: no promise of idempotency, so no lookup. This path is here to
        # be *contrasted* with the keyed one, not to be a fallback.
        if key is None:
            result = operation()
            attempt_id = self._record_attempt(None, learner_id, question_id, correct, now)
            return Outcome(result=result, replayed=False, attempt_id=attempt_id)

        fp = fingerprint(payload)

        # ── PHASE 1: CLAIM. ─────────────────────────────────────────────────
        # ⚠️ This COMMITS on its own, before the work is done. If it were part of
        # the same transaction as the write below, two concurrent requests would
        # both read "no key" and both proceed — the exact race this module exists
        # to close.
        try:
            self._conn().execute(
                "INSERT INTO idempotency_keys (key, fingerprint, status, claimed_at) "
                "VALUES (?, ?, 'in_flight', ?)",
                (key, fp, now),
            )
        except sqlite3.IntegrityError:
            return self._handle_existing(key, fp, now)

        # ── PHASE 2: EXECUTE AND RECORD, IN ONE TRANSACTION. ────────────────
        # ⚠️ The attempt row and the key's completion are one unit. A crash between
        # them would leave a key saying "done" with no attempt behind it — a
        # reconciliation finding, and a claim that cannot be trusted.
        try:
            self._conn().execute("BEGIN IMMEDIATE")
            result = operation()
            attempt_id = self._record_attempt(key, learner_id, question_id, correct, now)
            self._conn().execute(
                "UPDATE idempotency_keys SET status='done', response=?, completed_at=? "
                "WHERE key=?",
                (json.dumps(result, default=str), now, key),
            )
            self._conn().execute("COMMIT")
            return Outcome(result=result, replayed=False, attempt_id=attempt_id)
        except Exception:
            # ⚠️ We do NOT delete the claim. The key stays in_flight, which is the
            # truth: the outcome is unknown, and reconciliation must surface it
            # rather than a retry silently re-running a write that may have partly
            # applied.
            self._conn().execute("ROLLBACK")
            raise

    def _handle_existing(self, key: str, fp: str, now: float) -> Outcome:
        row = self._conn().execute(
            "SELECT fingerprint, status, response, claimed_at FROM idempotency_keys WHERE key=?",
            (key,),
        ).fetchone()

        # ⚠️ RULE 2, and the one most implementations get wrong. Same key, different
        # body is NOT a replay. Returning the stored response would silently discard
        # this request; treating it as new would double-apply under one key. It is a
        # conflict.
        if row["fingerprint"] != fp:
            raise IdempotencyConflict(
                f"key {key!r} was used for a different request "
                f"(fingerprint {row['fingerprint'][:8]}… vs {fp[:8]}…)"
            )

        # ⚠️ RULE 3. The first execution has not finished. It may have written and
        # be about to respond; it may have crashed. Either way a second write now is
        # unsafe.
        if row["status"] == "in_flight":
            age = now - row["claimed_at"]
            raise IdempotencyInFlight(
                f"key {key!r} is still in flight ({age:.1f}s) — not a replay, "
                f"and not safe to execute twice"
            )

        # ⚠️ RULE 1. The genuine replay: the original response, replayed verbatim.
        # This is what makes a retry safe — the client gets the same answer it would
        # have got, without a second write.
        return Outcome(
            result=json.loads(row["response"]),
            replayed=True,
            attempt_id=None,
        )

    def _record_attempt(
        self, key: str | None, learner_id: str, question_id: str, correct: bool, now: float
    ) -> int:
        cur = self._conn().execute(
            "INSERT INTO attempts (idempotency_key, learner_id, question_id, correct, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (key, learner_id, question_id, int(correct), now),
        )
        return int(cur.lastrowid)

    # ── reconciliation ───────────────────────────────────────────────────────

    def reconcile(self, now: float | None = None) -> dict[str, list]:
        """Compare what was promised against what happened.

        ⚠️ This is the half that gets skipped. An idempotency key stops the *known*
        retry; reconciliation catches the ones nobody saw, including the case where
        the key mechanism itself failed.
        """
        now = time.time() if now is None else now
        findings: dict[str, list] = {
            "double_applied": [],   # ⚠️ the failure the whole module exists to prevent
            "stuck": [],            # claimed, never finished
            "ghost": [],            # marked done, no attempt behind it
            "unkeyed": [],          # attempts written with no key — not wrong, but visible
        }

        # ⚠️ TWO attempts under ONE key. If this is ever non-empty the guard failed,
        # and it is the first thing to check before trusting any other number.
        dupes = self._conn().execute(
            "SELECT idempotency_key, COUNT(*) n FROM attempts "
            "WHERE idempotency_key IS NOT NULL "
            "GROUP BY idempotency_key HAVING n > 1"
        ).fetchall()
        for r in dupes:
            findings["double_applied"].append({"key": r["idempotency_key"], "attempts": r["n"]})

        # claimed but never completed, past the timeout
        for r in self._conn().execute(
            "SELECT key, claimed_at FROM idempotency_keys WHERE status='in_flight'"
        ).fetchall():
            age = now - r["claimed_at"]
            findings["stuck"].append({"key": r["key"], "age_seconds": round(age, 3),
                                      "over_timeout": age > STUCK_AFTER_SECONDS})

        # marked done with nothing written
        for r in self._conn().execute(
            "SELECT k.key FROM idempotency_keys k "
            "LEFT JOIN attempts a ON a.idempotency_key = k.key "
            "WHERE k.status='done' AND a.attempt_id IS NULL"
        ).fetchall():
            findings["ghost"].append({"key": r["key"]})

        # attempts with no key at all
        for r in self._conn().execute(
            "SELECT COUNT(*) n FROM attempts WHERE idempotency_key IS NULL"
        ).fetchall():
            findings["unkeyed"].append({"count": r["n"]})

        return findings

    def attempt_count(self, key: str | None = None) -> int:
        if key is None:
            return int(self._conn().execute("SELECT COUNT(*) n FROM attempts").fetchone()["n"])
        return int(self._conn().execute(
            "SELECT COUNT(*) n FROM attempts WHERE idempotency_key=?", (key,)
        ).fetchone()["n"])

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
        # ⚠️ The temp file is ours to remove. A service on a real path must not.
        if getattr(self, "_owns_file", False):
            for suffix in ("", "-wal", "-shm"):
                try:
                    os.unlink(self.path + suffix)
                except OSError:
                    pass


# ─── Demonstration: the same answer, sent twice ──────────────────────────────

def _demo_record(learner_id: str, question_id: str, correct: bool) -> dict[str, Any]:
    """Stand-in for the real write: record a learner's answer."""
    return {"learner_id": learner_id, "question_id": question_id, "correct": correct,
            "recorded": True}


# ─── Self-test ───────────────────────────────────────────────────────────────

def selftest(verbose: bool = True) -> int:
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        if verbose:
            print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")
        if not ok:
            failures.append(name)

    print("idempotency.py — self-test")

    PAY = {"learner_id": "L1", "question_id": "Q1", "correct": True}

    def op_factory(counter: list):
        def op():
            counter.append(1)
            return _demo_record("L1", "Q1", True)
        return op

    # ---- 1. the headline: N retries, ONE attempt ---------------------------
    store = IdempotencyStore()
    calls: list = []
    first = store.execute("k-1", PAY, op_factory(calls), learner_id="L1",
                          question_id="Q1", correct=True)
    replays = [
        store.execute("k-1", PAY, op_factory(calls), learner_id="L1",
                      question_id="Q1", correct=True)
        for _ in range(5)
    ]
    check("a retry does not execute again", len(calls) == 1,
          f"operation ran {len(calls)}× for 6 requests")
    check("six requests leave exactly ONE attempt", store.attempt_count("k-1") == 1,
          f"{store.attempt_count('k-1')} attempt row(s)")
    check("the first call reports it executed", first.executed)
    check("every retry reports a replay", all(r.replayed for r in replays))
    check("a replay returns the ORIGINAL result", all(r.result == first.result for r in replays))
    check("a replay returns no new attempt id", all(r.attempt_id is None for r in replays))

    # ---- 2. the same key with a DIFFERENT body is a conflict, not a replay --
    other = dict(PAY, question_id="Q9")
    try:
        store.execute("k-1", other, op_factory(calls), learner_id="L1",
                      question_id="Q9", correct=True)
        check("a key reused for a different request is refused", False, "it was accepted")
    except IdempotencyConflict as e:
        check("a key reused for a different request is refused", True, str(e)[:52] + "…")
    check("the conflicting request wrote nothing", store.attempt_count("k-1") == 1)

    # ---- 3. fingerprint is ORDER-INSENSITIVE -------------------------------
    a = fingerprint({"x": 1, "y": 2})
    b = fingerprint({"y": 2, "x": 1})
    check("key order does not change the fingerprint", a == b,
          "otherwise every replay is misread as a conflict")

    # ---- 4. concurrent call while in flight --------------------------------
    store2 = IdempotencyStore()

    # ⚠️ Claim a key WITHOUT completing it, by hand. This is exactly the state a
    # process leaves if it dies between claiming and recording — the case a single
    # transaction cannot protect against, and the one reconciliation exists for.
    store2._conn().execute(
        "INSERT INTO idempotency_keys (key, fingerprint, status, claimed_at) VALUES (?,?,?,?)",
        ("k-3", fingerprint(PAY), "in_flight", time.time()),
    )
    try:
        store2.execute("k-3", PAY, op_factory(calls), learner_id="L1",
                       question_id="Q1", correct=True)
        check("a call while the key is in flight is refused", False, "it was accepted")
    except IdempotencyInFlight as e:
        check("a call while the key is in flight is refused", True, str(e)[:52] + "…")

    # ---- 5. reconciliation sees it -----------------------------------------
    findings = store2.reconcile()
    check("reconciliation reports the stuck key",
          any(f["key"] == "k-3" for f in findings["stuck"]))
    check("a fresh claim is not yet over the timeout",
          not findings["stuck"][0]["over_timeout"])
    old = store2.reconcile(now=time.time() + STUCK_AFTER_SECONDS + 1)
    check("an aged claim IS over the timeout",
          old["stuck"][0]["over_timeout"], f"{old['stuck'][0]['age_seconds']}s")

    # ---- 6. reconciliation is CLEAN when everything is consistent -----------
    clean = store.reconcile()
    check("no double-apply on a healthy store", clean["double_applied"] == [])
    check("no ghosts on a healthy store", clean["ghost"] == [])
    check("no stuck keys on a healthy store", clean["stuck"] == [])

    # ---- 7. reconciliation DETECTS a double-apply ---------------------------
    # force the failure the module prevents, and confirm the check catches it
    store._conn().execute(
        "INSERT INTO attempts (idempotency_key, learner_id, question_id, correct, created_at) "
        "VALUES ('k-1','L1','Q1',1,?)", (time.time(),),
    )
    dirty = store.reconcile()
    check("reconciliation catches a double-apply",
          any(d["key"] == "k-1" for d in dirty["double_applied"]),
          "the check that would have caught a real duplicated transfer")

    # ---- 8. reconciliation catches a ghost ----------------------------------
    store3 = IdempotencyStore()
    store3._conn().execute(
        "INSERT INTO idempotency_keys (key, fingerprint, status, response, claimed_at, completed_at) "
        "VALUES ('k-ghost', ?, 'done', '{}', ?, ?)",
        (fingerprint(PAY), time.time(), time.time()),
    )
    check("reconciliation catches a key marked done with no attempt",
          any(g["key"] == "k-ghost" for g in store3.reconcile()["ghost"]))

    # ---- 9. no key means no promise -----------------------------------------
    store4 = IdempotencyStore()
    for _ in range(3):
        store4.execute(None, PAY, op_factory(calls), learner_id="L1",
                       question_id="Q1", correct=True)
    check("an unkeyed write executes every time",
          store4.attempt_count() == 3, f"{store4.attempt_count()} attempts for 3 calls")
    check("reconciliation counts unkeyed writes",
          store4.reconcile()["unkeyed"][0]["count"] == 3)

    # ---- 10. a failing operation does not mark the key done -------------------
    store5 = IdempotencyStore()

    def explode():
        raise ValueError("downstream failed")

    try:
        store5.execute("k-5", PAY, explode, learner_id="L1", question_id="Q1", correct=True)
    except ValueError:
        pass
    check("a failed write leaves the key in_flight, not done",
          store5.reconcile()["stuck"] != [],
          "the outcome is unknown, so a silent retry must not be allowed")
    check("a failed write leaves no attempt behind", store5.attempt_count() == 0)

    print()
    if failures:
        print(f"  ❌ {len(failures)} failed: {', '.join(failures)}")
        return len(failures)
    print("  ✅ all checks passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="An idempotent write path, and its reconciliation.")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
