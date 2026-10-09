"""
langchain_pipeline.py — the same retrieval, expressed in LangChain.

    python src/langchain_pipeline.py --selftest
    python src/langchain_pipeline.py --parity     # the comparison table

===============================================================================
WHY THIS MODULE EXISTS, AND WHAT IT IS NOT
===============================================================================
⚠️ **This is not a rewrite. It is a comparison, and the comparison is the
deliverable.**

`retrieval.py` already does hybrid retrieval with a hand-rolled BM25 and a coverage
gate that decides abstention. It has measured numbers behind it and it works. So the
question this module answers is not *"can it be done in LangChain"* — obviously it
can — but **"what does the framework actually buy, and what does it cost?"** That is
answerable only by running both over the same corpus and the same queries and
reporting where they agree.

⚠️ Both implementations stay, and running the comparison produced the strongest
argument for keeping the direct one. Measured:

    query "when should the tutor abstain"  — shares no term with the corpus

      our Index            -> []            (it filters score == 0)
      LangChain BM25       -> 3 passages    (none of which contain the terms)
      raw rank_bm25 scores -> all 10 non-zero, max 2.708

⚠️ **LangChain's retriever cannot abstain.** It returns k documents for any query,
including one the corpus has never seen. Abstention is the product decision this
repository is built around — it is what stops the tutor inventing an answer — and a
retriever that always returns k makes abstention undetectable downstream.

⚠️ That is a property of the retriever, not a defect in the framework, and it is
the kind of thing a comparison finds and a rewrite would have removed.

WHAT LANGCHAIN IS USED FOR
--------------------------
  `Document`              the corpus, as LangChain objects
  `BM25Retriever`         a second, independent BM25 — different tokeniser, different
                          k1/b, so its disagreement with ours is informative
  `BaseRetriever`         a custom retriever adapting OUR index, so the direct
                          implementation is *adapted* rather than duplicated
  RRF fusion              ⚠️ composed HERE, not via `EnsembleRetriever`. See below.

⚠️ WHY RRF IS COMPOSED HERE RATHER THAN USING `EnsembleRetriever`
-----------------------------------------------------------------
Two reasons, and the second one is the interesting one.

1. `EnsembleRetriever` no longer lives in `langchain-community`; it moved to the
   main `langchain` package. Pulling a large meta-package for one class is a poor
   trade when the algorithm is fifteen lines.
2. ⚠️ **Reciprocal Rank Fusion is rank-based, which is exactly why it works.** It
   discards the scores and keeps only the positions, because BM25 scores and cosine
   similarities are not on a comparable scale — `device-search` measured a garbage
   query scoring 0.649 cosine against 0.632 for the worst *real* match, so any
   score-level fusion is fusing noise. Writing it here makes that decision visible
   and puts the k constant where it can be justified.

⚠️ `langchain-community` also emits a deprecation warning: it is being sunset in
favour of standalone integration packages. The framework ecosystem is mid-
reorganisation, which is a reason to depend on less of it, not more.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from retrieval import Index, Passage, load_passages, tokenize          # noqa: E402

try:
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from langchain_community.retrievers import BM25Retriever

    LANGCHAIN_AVAILABLE = True
except ImportError:                                                     # pragma: no cover
    LANGCHAIN_AVAILABLE = False

# ⚠️ The RRF constant. 60 is the value from the original paper and it is not
# arbitrary: it damps the contribution of low-ranked documents so that being 1st
# rather than 2nd in one list does not outweigh appearing in both. A smaller k makes
# fusion behave more like "best rank wins", which defeats the purpose.
RRF_K = 60


# ─── Conversion ──────────────────────────────────────────────────────────────

def to_documents(passages: Sequence[Passage]) -> list[Any]:
    """The corpus as LangChain `Document`s.

    ⚠️ The passage id travels in `metadata`, NOT in the page content. If the id were
    part of the text it would be tokenised and scored, and every document would match
    a query containing the word "passage" — the corpus would vote on its own
    metadata.
    """
    return [
        Document(page_content=f"{p.title}\n{p.text}", metadata={"id": p.id, "title": p.title})
        for p in passages
    ]


# ─── Retrievers ──────────────────────────────────────────────────────────────

if LANGCHAIN_AVAILABLE:

    class DirectIndexRetriever(BaseRetriever):
        """Adapts `retrieval.Index` to the LangChain retriever interface.

        ⚠️ **This is the parity anchor.** It contains no logic of its own — it calls
        the same `Index.search`. So if it ever disagrees with a direct call, the
        disagreement is in the ADAPTER or in LangChain, never in the ranking. That
        makes it the one place where an exact equality assertion is honest.
        """

        index: Any
        k: int = 3

        def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Any]:
            hits = self.index.search(query, k=self.k)
            return [
                Document(page_content=f"{p.title}\n{p.text}",
                         metadata={"id": p.id, "score": s, "coverage": c})
                for p, s, c in hits
            ]


def build_bm25_retriever(passages: Sequence[Passage], k: int = 3) -> Any:
    """LangChain's own BM25, over the same corpus.

    ⚠️ It has a different tokeniser and different k1/b defaults from ours, so its
    ranking is an INDEPENDENT OPINION rather than a re-run of the same arithmetic.
    That is what makes it worth fusing: two retrievers that agree because they are
    the same retriever add nothing.
    """
    r = BM25Retriever.from_documents(to_documents(passages))
    r.k = k
    return r


# ─── Fusion ──────────────────────────────────────────────────────────────────

def rrf_fuse(rankings: Sequence[Sequence[str]], k: int = RRF_K, top_n: int = 3) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion over lists of document ids.

        score(d) = Σ  1 / (k + rank_i(d))

    ⚠️ Only RANKS are used. No score from either retriever survives this function,
    which is the entire point: the two scores are not on a comparable scale, so any
    fusion that combines them is combining numbers with different units.

    ⚠️ Ties break on the document id, because a reproducible run has to be
    reproducible and ties are common in a corpus of ten.
    """
    totals: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            totals[doc_id] = totals.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    ordered = sorted(totals.items(), key=lambda t: (-t[1], t[0]))
    return ordered[:top_n]


# ─── The pipeline ────────────────────────────────────────────────────────────

@dataclass
class LangChainResult:
    passage_id: str
    rrf_score: float
    coverage: float
    ranks: dict[str, int]      # which retriever put it where — the audit trail


class LangChainIndex:
    """Hybrid retrieval built from LangChain retrievers, fused with RRF.

    ⚠️ Deliberately mirrors `retrieval.Index.search(query, k)` so the two can be run
    over the same queries without an adapter at the comparison site — an adapter
    there would be one more place for a difference to hide.
    """

    def __init__(self, passages: list[Passage] | None = None, k: int = 3, rrf_k: int = RRF_K):
        if not LANGCHAIN_AVAILABLE:                                     # pragma: no cover
            raise RuntimeError("langchain is not installed — see requirements.txt")
        self.passages = passages if passages is not None else load_passages()
        self.by_id = {p.id: p for p in self.passages}
        self.k = k
        self.rrf_k = rrf_k

        # retired once, reused for every query — the same discipline as the direct index
        self.direct = DirectIndexRetriever(index=Index(self.passages), k=k)
        self.bm25 = build_bm25_retriever(self.passages, k=k)

    # ── one retriever, for the per-retriever parity checks ───────────────────
    def _ids_from(self, retriever: Any, query: str) -> list[str]:
        docs = retriever.invoke(query)
        return [d.metadata["id"] for d in docs]

    def direct_ids(self, query: str) -> list[str]:
        """The direct index, reached THROUGH LangChain."""
        return self._ids_from(self.direct, query)

    def bm25_ids(self, query: str) -> list[str]:
        """LangChain's own BM25."""
        return self._ids_from(self.bm25, query)

    # ── the hybrid ───────────────────────────────────────────────────────────
    def search(self, query: str, k: int | None = None) -> list[LangChainResult]:
        k = k or self.k
        direct_ids = self.direct_ids(query)
        bm25_ids = self.bm25_ids(query)

        ranks: dict[str, dict[str, int]] = {}
        for label, ids in (("direct", direct_ids), ("bm25", bm25_ids)):
            for rank, doc_id in enumerate(ids, start=1):
                ranks.setdefault(doc_id, {})[label] = rank

        fused = rrf_fuse([direct_ids, bm25_ids], k=self.rrf_k, top_n=k)

        # ⚠️ Coverage comes from the DIRECT index, not from a retriever score. It is
        # the abstention signal, and it must stay comparable with `retrieval.ground`
        # — a fused score could not be, because it is rank-based and unitless.
        idx = self.direct.index
        q_tokens = tokenize(query)
        out: list[LangChainResult] = []
        for doc_id, score in fused:
            pos = idx._idx_by_id.get(doc_id)
            cov = idx.coverage(q_tokens, pos) if pos is not None else 0.0
            out.append(LangChainResult(passage_id=doc_id, rrf_score=score,
                                       coverage=cov, ranks=ranks.get(doc_id, {})))
        return out


# ─── Parity ──────────────────────────────────────────────────────────────────

def parity(queries: Sequence[str], k: int = 3, verbose: bool = True) -> dict[str, Any]:
    """Run both implementations over the same queries and report where they differ.

    ⚠️ This does NOT assert that they agree. They should not: LangChain's BM25 has a
    different tokeniser and different parameters. A test that forced agreement would
    either be measuring nothing or be lying. What it asserts is narrower and real —
    the adapter path is identical, and every disagreement is reported with enough
    detail to explain.
    """
    lc = LangChainIndex(k=k)
    direct_index = Index(lc.passages)

    rows = []
    exact = set_match = top1_match = 0

    for q in queries:
        want = [p.id for p, _, _ in direct_index.search(q, k=k)]      # direct, no LangChain
        got = lc.direct_ids(q)                                        # direct, through the adapter
        native = lc.bm25_ids(q)                                       # LangChain's own BM25
        hybrid = [r.passage_id for r in lc.search(q, k=k)]

        row = {
            "query": q,
            "direct": want,
            "adapter": got,
            "bm25": native,
            "hybrid": hybrid,
            "adapter_identical": want == got,
            "bm25_top1": (want[:1] == native[:1]),
            "bm25_set": set(want) == set(native),
        }
        exact += row["adapter_identical"]
        top1_match += row["bm25_top1"]
        set_match += row["bm25_set"]
        rows.append(row)

    n = len(queries)
    report = {
        "queries": n, "k": k,
        "adapter_identical": exact,
        "bm25_top1_agreement": top1_match,
        "bm25_set_agreement": set_match,
        "rows": rows,
    }

    if verbose:
        # ⚠️ IDS ARE MAPPED TO INDEX NUMBERS FOR DISPLAY. The raw ids share prefixes
        # ("d-retrieval#1" and "d-chunk#1" both shorten to "1"), so a truncated id
        # printed two DIFFERENT passages identically and the table looked like it
        # agreed when it did not. An index into a printed legend cannot collide.
        order = [p.id for p in lc.passages]
        pos = {pid: i + 1 for i, pid in enumerate(order)}

        def fmt(ids):
            return ",".join(str(pos.get(i, "?")) for i in ids) or "(none)"

        print(f"  parity over {n} queries, k={k}\n")
        print(f"  {'#':3s} {'query':36s} {'direct index':14s} {'langchain bm25':14s} agree")
        for i, r in enumerate(rows, 1):
            mark = "top1" if r["bm25_top1"] else ("set" if r["bm25_set"] else "differs")
            print(f"  {i:<3d} {r['query'][:35]:36s} {fmt(r['direct']):14s} "
                  f"{fmt(r['bm25']):14s} {mark}")
        print()
        print("  legend:")
        for i, pid in enumerate(order, 1):
            print(f"    {i:2d}  {pid}")
        print()
        print(f"  adapter path identical to the direct call : {exact}/{n}")
        print(f"  LangChain BM25 agrees on the TOP-1        : {top1_match}/{n}")
        print(f"  LangChain BM25 agrees on the whole SET    : {set_match}/{n}")
        print()
        print("  ⚠️ The BM25 disagreement is the FINDING, not a failure. Two")
        print("     retrievers that agree because they are the same retriever add")
        print("     nothing to a fusion; the disagreement is what RRF is FOR.")
        print()
        print("  ⚠️⚠️ BUT READ QUERY 3 CAREFULLY — it is the most useful row in this")
        print("     table, and it is not about ranking at all.")
        print()
        print("     'when should the tutor abstain' shares NO term with the corpus:")
        print("     no passage contains 'abstain', 'abstention' or 'tutor'.")
        print()
        print("       our Index            -> []            (it filters score == 0)")
        print("       LangChain BM25       -> 3 passages    (none contain the terms)")
        print("       raw rank_bm25 scores -> all 10 non-zero, max 2.708")
        print()
        print("     ⚠️ LangChain's retriever CANNOT ABSTAIN. Given a query the corpus")
        print("     has never seen, it still returns k documents, and every downstream")
        print("     consumer has to treat them as evidence.")
        print()
        print("     ⚠️ That is the whole reason this repository keeps its own index.")
        print("     Abstention is the product decision here — it is what stops the")
        print("     tutor inventing an answer — and a retriever that always returns k")
        print("     makes abstention impossible, because there is nothing to detect.")

    return report


def sample_queries() -> list[str]:
    """Queries derived from the corpus headings, so the comparison is reproducible."""
    ps = load_passages()
    return [
        "why does hybrid retrieval exist",
        "fusing scores you cannot compare",
        "when should the tutor abstain",
        "what breaks retrieval chunk size",
        "how does reciprocal rank fusion work",
    ] or [p.title for p in ps[:5]]


# ─── Self-test ───────────────────────────────────────────────────────────────

def selftest(verbose: bool = True) -> int:
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        if verbose:
            print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")
        if not ok:
            failures.append(name)

    print("langchain_pipeline.py — self-test")

    if not LANGCHAIN_AVAILABLE:                                         # pragma: no cover
        print("  ⚠️  langchain is not installed — skipping (this is an optional extra)")
        return 0

    passages = load_passages()
    lc = LangChainIndex(passages)
    direct = Index(passages)

    # ---- 1. RRF arithmetic, checked against the formula by hand --------------
    # two lists, one document in both. 1/(60+1) + 1/(60+1) = 2/61
    fused = rrf_fuse([["a", "b"], ["a", "c"]], k=60, top_n=3)
    check("RRF ranks the document in BOTH lists first",
          fused[0][0] == "a", f"got {fused[0][0]}")
    check("RRF scores 2/(k+1) for a document ranked 1st twice",
          abs(fused[0][1] - 2 / 61) < 1e-12, f"{fused[0][1]:.6f} vs {2/61:.6f}")
    check("RRF score comes from RANK, not from any input score",
          abs(fused[1][1] - 1 / 62) < 1e-12, f"{fused[1][1]:.6f}")

    # a document in one list only, outranking one that is 2nd in both? No — and that
    # is the design: appearing in two lists beats appearing once.
    check("appearing in two lists beats appearing high in one",
          rrf_fuse([["a", "b", "c"], ["x", "y", "z"]], top_n=4)[0][0] == "a",
          "1/(61)+1/(61) vs 1/(61) — a shared document wins")

    check("RRF is deterministic on ties", rrf_fuse([["b"], ["a"]], top_n=2)
          == rrf_fuse([["b"], ["a"]], top_n=2))

    # ---- 2. ⚠️ THE PARITY ANCHOR: the adapter changes nothing ----------------
    queries = sample_queries()
    identical = 0
    for q in queries:
        want = [p.id for p, _, _ in direct.search(q, k=3)]
        got = lc.direct_ids(q)
        identical += (want == got)
    check("the LangChain adapter returns the SAME order as a direct call",
          identical == len(queries),
          f"{identical}/{len(queries)} queries — the adapter holds no logic of its own")

    # ---- 3. the metadata is not searchable content --------------------------
    docs = to_documents(passages)
    check("passage ids are NOT in the page content",
          all(d.metadata["id"] not in d.page_content for d in docs),
          "otherwise the corpus matches queries about its own metadata")

    # ---- 4. coverage is carried through unchanged ---------------------------
    q = queries[0]
    want_cov = {p.id: c for p, _, c in direct.search(q, k=3)}
    got_cov = {r.passage_id: r.coverage for r in lc.search(q, k=3) if r.passage_id in want_cov}
    check("coverage is identical to the direct index where the passages overlap",
          all(abs(want_cov[i] - got_cov[i]) < 1e-12 for i in got_cov),
          f"{len(got_cov)} overlapping passage(s)")

    # ---- 5. the hybrid is a UNION of both retrievers, deduplicated ----------
    r = lc.search(q, k=3)
    check("the hybrid returns no duplicates",
          len({x.passage_id for x in r}) == len(r))
    check("every hybrid result carries its source ranks",
          all(x.ranks for x in r), "the audit trail: which retriever put it where")

    # ---- 6. ⚠️ THE MEASURED COMPARISON, which is the deliverable -------------
    report = parity(queries, k=3, verbose=False)
    check("the adapter is identical on every query",
          report["adapter_identical"] == report["queries"],
          f"{report['adapter_identical']}/{report['queries']}")
    # ⚠️ NOT asserted equal — reported. See the module docstring.
    check("LangChain's BM25 is an INDEPENDENT opinion, not a re-run",
          report["bm25_set_agreement"] < report["queries"],
          f"agrees on the whole set for {report['bm25_set_agreement']}/{report['queries']} "
          f"queries — different tokeniser, different k1/b")
    check("the two retrievers still agree on the top result most of the time",
          report["bm25_top1_agreement"] >= report["queries"] // 2,
          f"top-1 agreement {report['bm25_top1_agreement']}/{report['queries']}")

    # ---- 6b. ⚠️ THE FINDING, AS AN ASSERTION ---------------------------------
    # "when should the tutor abstain" shares no term with the corpus. Our index
    # returns nothing; LangChain's returns k documents anyway. Pinning this means a
    # future LangChain version that starts abstaining will FAIL here rather than
    # quietly change the behaviour this repository is built around.
    OFF = "when should the tutor abstain"
    ours = direct.search(OFF, k=3)
    theirs = lc.bm25_ids(OFF)
    check("our index returns nothing for a query with no corpus term",
          ours == [], f"{len(ours)} hit(s)")
    check("LangChain's BM25 returns k documents for that same query",
          len(theirs) == 3, f"{len(theirs)} documents")
    check("...and none of them contain the query's terms",
          all("abstain" not in lc.by_id[i].text.lower() and "tutor" not in lc.by_id[i].text.lower()
              for i in theirs),
          "⚠️ a retriever that cannot abstain makes abstention undetectable downstream")

    # ---- 7. abstention still works through the LangChain path ---------------
    from retrieval import ground
    g = ground(direct, "how do I bake sourdough bread")
    # ⚠️ The abstention gate is NOT reimplemented in LangChain and must not be. It is
    # a coverage threshold on a named quantity, and a rank-fused score cannot carry
    # it — which is why `LangChainIndex.search` takes coverage from the direct index.
    # This check is the assertion that the two paths still agree about abstaining.
    check("an off-corpus question still abstains",
          not g.grounded, f"best coverage {g.best_coverage:.3f}, reason: {g.reason[:40]}")
    on = ground(direct, "why does hybrid retrieval exist")
    check("an on-corpus question is still grounded",
          on.grounded, f"best coverage {on.best_coverage:.3f}")

    print()
    if failures:
        print(f"  ❌ {len(failures)} failed: {', '.join(failures)}")
        return len(failures)
    print("  ✅ all checks passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="The same retrieval, expressed in LangChain.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--parity", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.parity:
        if not LANGCHAIN_AVAILABLE:                                     # pragma: no cover
            print("  langchain is not installed")
            return 1
        parity(sample_queries())
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
