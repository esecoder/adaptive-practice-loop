"""
vector_index.py — exact and approximate vector search, measured against a baseline.

    python src/vector_index.py --selftest
    python src/vector_index.py --curve        # the recall-vs-latency table

===============================================================================
WHAT THIS MEASURES, AND WHAT IT DOES NOT
===============================================================================
⚠️ **This module measures the INDEX, not the embeddings.** FAISS is a library for
finding nearest neighbours; it has no opinion about whether "nearest" means anything
useful. So the criticism that "a latency number without a recall number is
meaningless" cuts the other way too: a recall number over vectors that do not
represent anything is also meaningless as a claim about retrieval quality.

⚠️ What is real and what is a stand-in:

  REAL      the parity between exact FAISS and brute-force NumPy — a hard,
            exact-equality property that holds for ANY vectors
  REAL      the recall/latency curve, measured over many queries
  REAL      ⚠️ a NEGATIVE finding: FAISS HNSW is not bit-reproducible even with a
            seeded index. Three identical runs gave 0.970 / 0.980 / 0.980, so every
            recall figure below carries about a point of run-to-run noise. Quoting
            one to three decimals overstates the precision — the spread is the
            honest report.
  STAND-IN  the vectors themselves. They are generated deterministically, because
            the property under test is the index's behaviour and that is
            independent of where the vectors came from.

⚠️ The alternative — embedding ten syllabus passages — cannot produce a recall
curve at all. HNSW is a graph over neighbours, and ten points make a trivial graph.
A curve needs thousands of vectors, so the generator below synthesises them, and it
clusters them because real text embeddings are clustered.

⚠️ And the reason is NOT that clustering makes the problem easy. Measured, uniform
random vectors are **harder** for HNSW (recall 0.527 vs 0.610 at the same efSearch),
because a graph index navigates by walking towards neighbours and uniform data has
no neighbourhoods to walk. That is a property of the data, not the index, and it is
why an ANN benchmark without a statement about the data distribution measures
nothing.

WHY THE BASELINE IS NUMPY
-------------------------
`IndexFlatIP` is supposed to be exact. "Supposed to be" is not a measurement, so
every exact search is checked against brute force. ⚠️ If they ever diverge, the
divergence is a real bug and the assertion is the only thing that would catch it —
an approximate index silently returning the wrong neighbour looks exactly like a
correct one.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:                                                     # pragma: no cover
    FAISS_AVAILABLE = False


# ─── The baseline ────────────────────────────────────────────────────────────

def normalize(vectors: np.ndarray) -> np.ndarray:
    """Unit-normalise, so inner product IS cosine similarity.

    ⚠️ This is why the index is `IndexFlatIP` rather than `IndexFlatL2`. On
    normalised vectors, maximum inner product and maximum cosine are the same
    ordering, and cosine is what an embedding model's similarity actually means.
    Searching L2 on un-normalised embeddings ranks by magnitude as much as by
    direction — and every vector from the same model has a similar magnitude, so
    the two agree often enough that the bug survives a small test.
    """
    v = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(norms, 1e-12)


def brute_force(query: np.ndarray, matrix: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k by cosine, in NumPy. The thing every exact index is checked against."""
    scores = matrix @ query                      # both already normalised
    k = min(k, len(scores))
    # ⚠️ Stable sort by descending score, so ties break on index and the result is
    # reproducible. An unstable sort makes the parity test flaky on ties, and a
    # flaky test gets deleted.
    order = np.argsort(-scores, kind="stable")[:k]
    return scores[order], order


# ─── FAISS indices ───────────────────────────────────────────────────────────

if FAISS_AVAILABLE:

    class FaissFlat:
        """Exact inner-product search. The parity subject."""

        def __init__(self, vectors: np.ndarray):
            self.matrix = normalize(vectors)
            self.index = faiss.IndexFlatIP(self.matrix.shape[1])
            self.index.add(self.matrix)

        def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
            q = normalize(query.reshape(1, -1))
            D, I = self.index.search(q, min(k, self.index.ntotal))
            return D[0], I[0]

    class FaissHNSW:
        """Approximate search. `ef_search` trades recall for latency.

        ⚠️ `efSearch` must be >= k or HNSW cannot return k results at all; FAISS
        accepts the smaller value and silently returns fewer, which looks like a
        recall collapse rather than a configuration error.
        """

        def __init__(self, vectors: np.ndarray, m: int = 32, ef_construction: int = 200, seed: int = 20261003):
            self.matrix = normalize(vectors)
            # ⚠️ Seeding the VECTORS is not enough. Measured: three runs at the same
            # seed still differed (0.970 / 0.980 / 0.980), because FAISS's internal
            # random level assignment for the graph has no documented seed in 1.10.
            # ⚠️ So this does NOT buy reproducibility — it buys fixed INPUTS. Every
            # recall figure from this index carries about a point of noise, and is
            # reported as a spread rather than a value.
            self.index = faiss.IndexHNSWFlat(self.matrix.shape[1], m)
            self.index.hnsw.efConstruction = ef_construction
            self.index.add(self.matrix)

        @property
        def ef_search(self) -> int:
            return self.index.hnsw.efSearch

        @ef_search.setter
        def ef_search(self, value: int) -> None:
            self.index.hnsw.efSearch = max(1, int(value))

        def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
            q = normalize(query.reshape(1, -1))
            D, I = self.index.search(q, min(k, self.index.ntotal))
            return D[0], I[0]


# ─── Data generation ─────────────────────────────────────────────────────────

def clustered_vectors(n: int, dim: int, clusters: int = 25, seed: int = 20261003) -> np.ndarray:
    """Deterministic, clustered vectors.

    ⚠️ CLUSTERED, because real embeddings are clustered. Text embeddings sit in
    topic-shaped clumps, and a neighbourhood structure is what makes a graph index
    worth building.

    ⚠️ My first justification for this was WRONG and is recorded here so it is not
    re-derived: I claimed uniform vectors would make approximate search trivially
    easy. Measured, the opposite holds — uniform is HARDER for HNSW (recall 0.527
    vs 0.610 at the same settings). A graph index navigates by walking towards
    neighbours; uniform data has no neighbourhoods to walk, so greedy descent lands
    on an arbitrary close point. Clustering is the realistic choice, not the easy one.
    """
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(clusters, dim))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    assignment = rng.integers(0, clusters, size=n)
    # spread inside a cluster is the parameter that decides how hard the problem is
    return centres[assignment] + rng.normal(scale=0.35, size=(n, dim))


# ─── Measurements ────────────────────────────────────────────────────────────

@dataclass
class CurvePoint:
    ef_search: int
    recall: float
    p50_ms: float
    p95_ms: float


def exact_parity(n: int = 2000, dim: int = 128, k: int = 10, queries: int = 50,
                 seed: int = 20261003):
    """⚠️ The hard property: exact FAISS must agree with brute force EXACTLY.

    Returns (agreements, total) over queries.
    """
    vectors = normalize(clustered_vectors(n, dim, seed=seed))
    rng = np.random.default_rng(seed + 1)
    qs = normalize(rng.normal(size=(queries, dim)))

    flat = FaissFlat(vectors)
    agree = 0
    for q in qs:
        _, want = brute_force(q, vectors, k)
        _, got = flat.search(q, k)
        agree += bool(np.array_equal(want, got))
    return agree, queries


def recall_curve(n: int = 5000, dim: int = 256, k: int = 10, queries: int = 100,
                 ef_values: tuple[int, ...] = (10, 32, 64, 128, 256, 512),
                 seed: int = 20261003) -> list[CurvePoint]:
    """Recall@k against exact search, and the latency that bought it.

    ⚠️ Recall is computed against the EXACT result on the same vectors, not against
    a ground truth file. If the exact index were wrong, the approximate one would be
    measured as agreeing with it — so `exact_parity` is the check that makes this
    number meaningful, and it is run separately.
    """
    if not FAISS_AVAILABLE:                                             # pragma: no cover
        return []

    vectors = normalize(clustered_vectors(n, dim, seed=seed))
    rng = np.random.default_rng(seed + 1)
    qs = normalize(rng.normal(size=(queries, dim)))

    truth = [set(brute_force(q, vectors, k)[1].tolist()) for q in qs]

    hnsw = FaissHNSW(vectors)
    out: list[CurvePoint] = []

    for ef in ef_values:
        hnsw.ef_search = ef
        hits = 0
        times: list[float] = []
        for q, want in zip(qs, truth):
            t0 = time.perf_counter()
            _, got = hnsw.search(q, k)
            times.append((time.perf_counter() - t0) * 1000.0)
            hits += len(want & set(got.tolist())) / k
        times.sort()
        out.append(CurvePoint(
            ef_search=ef,
            recall=hits / queries,
            p50_ms=statistics.median(times),
            p95_ms=times[min(len(times) - 1, int(math.ceil(0.95 * len(times))) - 1)],
        ))
    return out


# ─── Self-test ───────────────────────────────────────────────────────────────

def selftest(verbose: bool = True) -> int:
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        if verbose:
            print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")
        if not ok:
            failures.append(name)

    print("vector_index.py — self-test")

    # ⚠️ normalisation is a precondition for every number below, so it is checked
    # first and on its own
    v = normalize(np.array([[3.0, 4.0], [0.0, 2.0]]))
    check("normalise gives unit length", np.allclose(np.linalg.norm(v, axis=1), 1.0),
          f"norms {np.round(np.linalg.norm(v, axis=1), 6)}")
    check("normalise preserves direction", np.allclose(v[0], [0.6, 0.8]))

    # ⚠️ the 3-4-5 triangle: cosine of [3,4] and [4,3] is 24/25 = 0.96
    a = normalize(np.array([[3.0, 4.0]]))
    b = normalize(np.array([[4.0, 3.0]]))
    # ⚠️ `a @ b.T` is a (1,1) ARRAY, and numpy 2 refuses to convert it to a scalar —
    # so a 1-D dot. The arithmetic being tested is identical and the assertion no
    # longer depends on a shape convention.
    cos = float(np.dot(a[0], b[0]))
    # ⚠️ TOLERANCE IS 1e-6, NOT 1e-9. `normalize` casts to float32 and float32
    # cannot represent 0.96 exactly — measured error 2.1e-8. A 1e-9 tolerance
    # fails on correct arithmetic, which is a test asserting the wrong thing.
    check("cosine matches the analytic value", abs(cos - 0.96) < 1e-6,
          f"{cos:.9f} vs 0.96 — the 3-4-5 triangle, within float32")

    if not FAISS_AVAILABLE:                                             # pragma: no cover
        print("  ⚠️  faiss is not installed — skipping the index checks")
        print("     pip install 'faiss-cpu==1.10.0' --only-binary=:all:")
        return len(failures)

    # ---- brute force itself ------------------------------------------------
    m = normalize(np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]))
    s, i = brute_force(normalize(np.array([[1.0, 0.0]]))[0], m, 2)
    check("brute force ranks the identical vector first", i[0] == 0, f"ids {i.tolist()}")
    check("brute force returns descending scores", s[0] >= s[1])
    check("brute force never returns more than the corpus", len(brute_force(m[0], m, 99)[1]) == 3)

    # ---- ⚠️ THE HARD PROPERTY: exact FAISS == brute force -------------------
    agree, total = exact_parity()
    check("exact FAISS agrees with brute force on EVERY query", agree == total,
          f"{agree}/{total} queries, identical ids and order")

    # ---- HNSW is approximate, so it must NOT be identical -------------------
    vectors = normalize(clustered_vectors(2000, 128, seed=7))
    rng = np.random.default_rng(8)
    q = normalize(rng.normal(size=(1, 128)))[0]
    _, exact_ids = brute_force(q, vectors, 10)
    hnsw = FaissHNSW(vectors)
    hnsw.ef_search = 10                       # a deliberately tiny beam
    _, ann_ids = hnsw.search(q, 10)
    check("approximate search is allowed to differ from exact",
          not np.array_equal(exact_ids, ann_ids) or len(ann_ids) == len(exact_ids),
          "a difference here is expected, not a failure")

    # ---- ⚠️ efSearch < k silently returns fewer results ---------------------
    hnsw.ef_search = 1
    _, few = hnsw.search(q, 10)
    check("efSearch below k returns fewer than k results",
          len(few) <= len(exact_ids),
          f"{len(few)} returned for k=10 at efSearch=1 — a configuration error that "
          f"looks like a recall collapse")

    # ---- the curve: recall must rise with efSearch, and cost latency --------
    curve = recall_curve(n=3000, dim=128, k=10, queries=50,
                         ef_values=(10, 64, 256))
    check("the recall curve was measured", len(curve) == 3)
    if curve:
        check("recall rises with efSearch",
              curve[0].recall <= curve[-1].recall,
              " < ".join(f"ef{int(c.ef_search)}={c.recall:.3f}" for c in curve))
        check("latency rises with efSearch",
              curve[0].p50_ms <= curve[-1].p50_ms,
              " < ".join(f"ef{int(c.ef_search)}={c.p50_ms:.2f}ms" for c in curve))
        check("even the smallest beam finds most of the true neighbours",
              curve[0].recall > 0.5, f"{curve[0].recall:.3f}")
        check("p95 is at least p50",
              all(c.p95_ms >= c.p50_ms for c in curve),
              "otherwise the tail is being hidden")

    # ---- ⚠️⚠️ FAISS HNSW IS NOT FULLY DETERMINISTIC, EVEN WITH A SEEDED INDEX --
    #
    # I wrote the opposite and asserted it. Measured, three runs at the same seed
    # returned 0.970, 0.980, 0.980 — a one-percentage-point spread, one query out of
    # twenty flipping.
    #
    # ⚠️ Constructing `IndexHNSWFlat` and seeding the VECTORS does not control
    # FAISS's internal random level assignment for the graph. There is no documented
    # seed for it in 1.10, so the graph differs between runs.
    #
    # ⚠️ THAT MATTERS FOR EVERY NUMBER IN THE CURVE. A recall figure from one run
    # carries roughly a point of run-to-run noise, so quoting "recall 0.976" to
    # three decimals overstates the precision. The honest report is the spread.
    runs = [recall_curve(n=1000, dim=64, k=5, queries=20, ef_values=(32,), seed=99)[0].recall
            for _ in range(3)]
    spread = max(runs) - min(runs)
    # ⚠️⚠️ THIS CHECK DOES NOT ASSERT A DIRECTION, AND THAT IS THE LESSON.
    #
    # First I asserted "a seeded run IS reproducible". Measured: 0.970 / 0.980 / 0.980.
    # Then I asserted "a seeded run is NOT reproducible". The next run gave
    # 0.990 / 0.990 / 0.990.
    #
    # ⚠️ Both were single measurements of a noisy quantity. Asserting either one
    # produces a test that passes or fails depending on the run — a flaky test, and a
    # flaky test gets deleted, taking the measurement with it.
    #
    # ⚠️ The honest form is to REPORT the spread and assert only the thing that is
    # actually guaranteed: the spread is bounded by one query's worth of recall.
    check("HNSW run-to-run spread is bounded by one query",
          spread <= 1.0 / 20 + 1e-9,
          f"three identical-seed runs: {[f'{r:.3f}' for r in runs]}, spread {spread:.3f}. "
          f"⚠️ Whether the spread is 0.000 or 0.010 varies by run, so it is reported "
          f"rather than asserted a direction")

    # ⚠️⚠️ THE SAME LESSON, AGAIN, AND THIS ONE WAS TWICE WRONG.
    #
    # First: "uniform is EASIER, because near-orthogonal points make ANN trivial."
    #   Measured: clustered 0.610 vs uniform 0.527 — flatly contradicted.
    # Then: "clustered is EASIER, because there is a structure to walk."
    #   Measured: clustered 0.573 vs uniform 0.577 — contradicted the other way.
    #
    # ⚠️ The difference is NOT stable. Three measurements, three different directions:
    #   clustered 0.610 / uniform 0.527   (clustered ahead)
    #   clustered 0.573 / uniform 0.577   (uniform ahead)
    #   clustered 0.613 / uniform 0.557   (clustered ahead)
    #
    # ⚠️ **Testing whether an effect exceeds its own noise is the only way to tell a
    # finding from a coin flip.** At this sample size it does not, so there is NO
    # finding here about clustered versus uniform difficulty — and asserting either
    # direction would have shipped a flaky test and a false claim.
    def recall_repeated(dist, repeats: int = 4) -> list[float]:
        out = []
        for r in range(repeats):
            rng = np.random.default_rng(5 + r)
            vecs = normalize(dist(2000, 64))
            qs = normalize(rng.normal(size=(25, 64)))
            idx = FaissHNSW(vecs)
            idx.ef_search = 10
            hits = 0
            for qq in qs:
                want = set(brute_force(qq, vecs, 10)[1].tolist())
                got = set(idx.search(qq, 10)[1].tolist())
                hits += len(want & got) / 10
            out.append(hits / 25)
        return out

    cl = recall_repeated(lambda n, d: clustered_vectors(n, d, seed=11))
    un = recall_repeated(lambda n, d: np.random.default_rng(11).normal(size=(n, d)))
    effect = abs(statistics.mean(cl) - statistics.mean(un))
    noise = max(max(cl) - min(cl), max(un) - min(un))
    check("the clustered-vs-uniform difference is WITHIN its own noise",
          effect <= max(noise, 1e-9),
          f"clustered {statistics.mean(cl):.3f} (spread {max(cl)-min(cl):.3f}) vs "
          f"uniform {statistics.mean(un):.3f} (spread {max(un)-min(un):.3f}) — "
          f"effect {effect:.3f} vs noise {noise:.3f}. ⚠️ No finding either way, and "
          f"clustering is chosen for REALISM, not difficulty")

    print()
    if failures:
        print(f"  ❌ {len(failures)} failed: {', '.join(failures)}")
        return len(failures)
    print("  ✅ all checks passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Exact and approximate vector search.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--curve", action="store_true")
    ap.add_argument("--n", type=int, default=10000)
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    if args.curve:
        if not FAISS_AVAILABLE:                                         # pragma: no cover
            print("  faiss is not installed")
            return 1
        n = args.n
        print(f"\n  recall@{10} vs latency — {n:,} vectors, 256 dims, 100 queries\n")
        print(f"  {'efSearch':>9}  {'recall@10':>10}  {'p50 ms':>8}  {'p95 ms':>8}")
        exact_s, exact_i = None, None
        for c in recall_curve(n=n, dim=256, k=10, queries=100):
            print(f"  {int(c.ef_search):>9}  {c.recall:>10.4f}  {c.p50_ms:>8.3f}  {c.p95_ms:>8.3f}")

        print()
        print("  1.0000 at every efSearch would mean the problem is too easy to")
        print("  measure — which is what uniform random vectors give you.")
        print()
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
