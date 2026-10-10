"""
test_pgvector.py — the in-process vector index, checked against a real Postgres.

    python tests/test_pgvector.py
    python tests/test_pgvector.py --curve

⚠️ REQUIRES A RUNNING POSTGRES WITH pgvector, and SKIPS CLEANLY WITHOUT ONE.

    docker run -d --name pgvector-idx \\
      -e POSTGRES_PASSWORD=test -e POSTGRES_DB=vectors \\
      -p 5433:5432 pgvector/pgvector:pg16

    PGVECTOR_DSN="host=localhost port=5433 user=postgres password=test dbname=vectors" \\
      python tests/test_pgvector.py

===============================================================================
WHY A REAL SERVER, WHEN THE SAME TEST PASSES AGAINST NUMPY
===============================================================================
⚠️ The in-process index and pgvector answer the same question and are written by
different people in different languages with different rounding, different
comparison operators and a different execution engine. **Parity between them is
evidence that neither is wrong**; agreement with a NumPy re-implementation of
myself is evidence of nothing.

⚠️ And the acceptance criterion is explicit — *"a latency number without a recall
number is meaningless"* — so both are reported together, never separately.

WHAT IS ASSERTED, AND WHAT IS ONLY REPORTED
-------------------------------------------
ASSERTED  exact parity — pgvector's top-k by inner product is IDENTICAL to brute
          force, on every query. This is a hard property and it either holds or the
          query is wrong.
ASSERTED  the vectors round-trip exactly through the `vector` type.
REPORTED  HNSW recall against exact search, with latency. ⚠️ Not asserted equal,
          for the reason `vector_index.py` measured: an approximate index is
          supposed to differ, and pgvector's HNSW is not bit-reproducible either.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np                                                      # noqa: E402

from vector_index import brute_force, clustered_vectors, normalize       # noqa: E402

try:
    import psycopg2
    import psycopg2.extras
except ImportError:                                                     # pragma: no cover
    psycopg2 = None

DSN = os.environ.get(
    "PGVECTOR_DSN",
    "host=localhost port=5433 user=postgres password=test dbname=vectors connect_timeout=3",
)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


# ─── Connection ──────────────────────────────────────────────────────────────

def connect():
    """⚠️ Returns None rather than raising, so the file SKIPS instead of failing.

    A test that cannot run in an environment without Docker must not make the
    repository red there — that is how a good test gets deleted by someone who
    never had the dependency.
    """
    if psycopg2 is None:                                                # pragma: no cover
        return None
    try:
        return psycopg2.connect(DSN)
    except Exception:                                                   # pragma: no cover
        return None


def literal(vec: np.ndarray) -> str:
    """A pgvector literal. ⚠️ `::vector` cast is required — a bare string is text
    and Postgres will not compare it to a vector column."""
    return "[" + ",".join(f"{float(x):.9g}" for x in vec) + "]"


# ─── The parity measurement ──────────────────────────────────────────────────

def parity_setup(conn, vectors: np.ndarray, table: str = "items") -> None:
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(f"DROP TABLE IF EXISTS {table}")
        # ⚠️ The dimension is fixed in the column type. pgvector will reject a
        # vector of the wrong length rather than truncating it, which is the
        # behaviour you want — a silent truncation would return confident nonsense.
        cur.execute(f"CREATE TABLE {table} (id INT PRIMARY KEY, embedding vector({vectors.shape[1]}))")
        psycopg2.extras.execute_values(
            cur,
            f"INSERT INTO {table} (id, embedding) VALUES %s",
            [(int(i), literal(v)) for i, v in enumerate(vectors)],
            template="(%s, %s::vector)",
        )
    conn.commit()


def roundtrip(conn, vectors: np.ndarray, table: str = "items") -> tuple[int, float]:
    """⚠️ Do the vectors come back BIT-IDENTICAL? If not, the parity test below is
    comparing different inputs and proves nothing.

    Returns (n_equal, max_abs_diff).
    """
    with conn.cursor() as cur:
        cur.execute(f"SELECT id, embedding::text FROM {table} ORDER BY id")
        rows = cur.fetchall()
    equal = 0
    worst = 0.0
    for i, txt in rows:
        back = np.array([float(x) for x in txt.strip("[]").split(",")], dtype="float32")
        worst = max(worst, float(np.max(np.abs(back - vectors[i]))))
        equal += bool(np.array_equal(back, vectors[i]))
    return equal, worst


def parity(conn, vectors: np.ndarray, queries: np.ndarray, k: int = 10, table: str = "items"):
    """⚠️ THE HARD PROPERTY: pgvector's exact top-k == brute force, every query.

    ⚠️ `<#>` is NEGATIVE INNER PRODUCT, so ASCENDING order is descending
    similarity. Using `<=>` (cosine distance) instead would also work on normalised
    vectors — and would silently disagree the moment they were not normalised, which
    is why the operator is pinned and the vectors are normalised explicitly.
    """
    agree = 0
    first_disagreement = None
    with conn.cursor() as cur:
        for q in queries:
            cur.execute(
                f"SELECT id FROM {table} ORDER BY embedding <#> %s::vector LIMIT %s",
                (literal(q), k),
            )
            got = [r[0] for r in cur.fetchall()]
            want = brute_force(q, vectors, k)[1].tolist()
            if got == want:
                agree += 1
            elif first_disagreement is None:
                first_disagreement = (want, got)
    return agree, len(queries), first_disagreement


def hnsw_curve(conn, vectors: np.ndarray, queries: np.ndarray, k: int = 10,
               ef_values: tuple[int, ...] = (10, 32, 64, 128), table: str = "items"):
    """Recall AND latency together, at several `hnsw.ef_search` values.

    ⚠️ `SET LOCAL hnsw.ef_search` is a per-transaction setting. Setting it globally
    and forgetting to reset it leaves a benchmark that silently measures the last
    configuration for every subsequent query.
    """
    truth = [set(brute_force(q, vectors, k)[1].tolist()) for q in queries]
    out = []

    with conn.cursor() as cur:
        cur.execute(f"DROP INDEX IF EXISTS {table}_hnsw")
        # ⚠️ vector_cosine_ops matches the normalised inner-product ordering. The
        # ops class must match the operator used in the query, or Postgres silently
        # ignores the index and does an exact seq scan — which shows up as perfect
        # recall and terrible latency, a classic benchmark error.
        cur.execute(
            f"CREATE INDEX {table}_hnsw ON {table} "
            f"USING hnsw (embedding vector_cosine_ops)"
        )
    conn.commit()

    for ef in ef_values:
        hits = 0
        times: list[float] = []
        with conn.cursor() as cur:
            cur.execute(f"SET hnsw.ef_search = {int(ef)}")
            for q, want in zip(queries, truth):
                t0 = time.perf_counter()
                cur.execute(
                    f"SELECT id FROM {table} ORDER BY embedding <=> %s::vector LIMIT %s",
                    (literal(q), k),
                )
                times.append((time.perf_counter() - t0) * 1000.0)
                got = {r[0] for r in cur.fetchall()}
                hits += len(want & got) / k
        times.sort()
        out.append({
            "ef_search": ef,
            "recall": hits / len(queries),
            "p50_ms": statistics.median(times),
            "p95_ms": times[min(len(times) - 1, int(np.ceil(0.95 * len(times))) - 1)],
            "index_used": None,
        })
    return out


def main() -> int:
    print("test_pgvector.py — the vector index against a real Postgres")

    conn = connect()
    if conn is None:
        print()
        print("  ⚠️  No pgvector instance reachable — SKIPPING, not failing.")
        print("      docker run -d --name pgvector-idx -e POSTGRES_PASSWORD=test \\")
        print("        -e POSTGRES_DB=vectors -p 5433:5432 pgvector/pgvector:pg16")
        print()
        return 0

    with conn.cursor() as cur:
        cur.execute("SELECT extversion FROM pg_extension WHERE extname='vector'")
        row = cur.fetchone()
    if row is None:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
        with conn.cursor() as cur:
            cur.execute("SELECT extversion FROM pg_extension WHERE extname='vector'")
            row = cur.fetchone()
    print(f"  pgvector {row[0]} reachable\n")

    vectors = normalize(clustered_vectors(2000, 128, seed=20261003))
    rng = np.random.default_rng(20261004)
    queries = normalize(rng.normal(size=(25, 128)))

    parity_setup(conn, vectors)

    # ---- 1. ⚠️ the inputs round-trip exactly, or nothing below means anything --
    equal, worst = roundtrip(conn, vectors)
    check("every vector round-trips bit-identically through the `vector` type",
          equal == len(vectors),
          f"{equal}/{len(vectors)} exact, max abs diff {worst:.3e}")

    # ---- 2. ⚠️ THE HARD PROPERTY ---------------------------------------------
    agree, total, first = parity(conn, vectors, queries, k=10)
    check("pgvector's exact top-10 is IDENTICAL to brute force on every query",
          agree == total,
          f"{agree}/{total} queries" + ("" if agree == total else f"; first mismatch {first}"))

    # ---- 3. the operator is the one we think it is ---------------------------
    with conn.cursor() as cur:
        cur.execute("SELECT '[1,2,3]'::vector <#> '[1,2,4]'::vector")
        neg_ip = float(cur.fetchone()[0])
    check("<#> is the NEGATIVE inner product, so ASC is descending similarity",
          abs(neg_ip - (-17.0)) < 1e-6,
          f"[1,2,3] <#> [1,2,4] = {neg_ip} (dot product is 17.0)")

    # ---- 4. a dimension mismatch is REFUSED, not truncated -------------------
    with conn.cursor() as cur:
        try:
            cur.execute("SELECT '[1,2]'::vector <#> (SELECT embedding FROM items LIMIT 1)")
            refused = False
        except Exception:
            conn.rollback()
            refused = True
    check("a wrong-length vector is refused rather than truncated", refused,
          "silent truncation would return confident nonsense from a shorter vector")

    # ---- 5. ⚠️ RECALL AND LATENCY, REPORTED TOGETHER -------------------------
    curve = hnsw_curve(conn, vectors, queries, k=10)
    print()
    print(f"  {'ef_search':>9}  {'recall@10':>10}  {'p50 ms':>9}  {'p95 ms':>9}")
    for c in curve:
        print(f"  {c['ef_search']:>9}  {c['recall']:>10.4f}  {c['p50_ms']:>9.3f}  {c['p95_ms']:>9.3f}")
    print()

    # ⚠️ In a corpus this small an HNSW index frequently has recall 1.0 at every
    # setting, which means the benchmark has not measured anything. Saying so is
    # more useful than printing a perfect number and letting it pass for a result.
    if all(c["recall"] >= 0.999 for c in curve):
        check("the HNSW curve is informative at this corpus size", False,
              "recall is 1.0000 at every ef_search — 2,000 vectors is too few for "
              "the recall difference to appear, so the latency column is the only "
              "real measurement here")
    else:
        check("recall increases with ef_search",
              curve[0]["recall"] <= curve[-1]["recall"],
              " < ".join(f"ef{c['ef_search']}={c['recall']:.4f}" for c in curve))
        check("latency increases with ef_search",
              curve[0]["p50_ms"] <= curve[-1]["p50_ms"],
              " < ".join(f"ef{c['ef_search']}={c['p50_ms']:.3f}ms" for c in curve))

    # ---- 6. exact search must be EXPLAINed as exact, not as the HNSW index ---
    # ⚠️ If the planner used the approximate index for an `ORDER BY <#>` query, the
    # "exact" parity above would have been measuring an approximate index — and it
    # would have passed, because the recall is high. That is the benchmark error
    # this check exists to rule out.
    with conn.cursor() as cur:
        cur.execute("DROP INDEX IF EXISTS items_hnsw")
    conn.commit()
    agree2, total2, _ = parity(conn, vectors, queries, k=10)
    check("parity holds with NO index at all (a sequential scan)",
          agree2 == total2,
          f"{agree2}/{total2} — so the parity result was not the approximate index "
          f"in disguise")

    conn.close()

    print()
    if FAILURES:
        print(f"  ❌ {len(FAILURES)} failed: {', '.join(FAILURES)}")
        return len(FAILURES)
    print("  ✅ all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
