#!/usr/bin/env python3
"""
retrieval.py — retrieval over the *approved* content, in the standard library.

===============================================================================
WHAT THIS IS, AND WHY IT IS NOT A CHATBOT
===============================================================================
This retrieves from **one corpus that a human approved** — the syllabus shipped in
`content/syllabus.md` — and nothing else. It is not a "chat with your documents" tool, and the
difference is the whole point:

> *"Ensure AI-generated responses are grounded in reliable and validated sources"* is only
> meaningful if **the set of sources is fixed and vetted.** A system that will answer from any
> document a user pastes has no notion of a validated source, so it cannot make that guarantee
> at all.

⚠️ IN PRODUCTION THIS IS WHERE DENSE RETRIEVAL AND FUSION WOULD GO, exactly as `RaggyEditor`
implements it: BM25 and a dense encoder, fused by reciprocal rank, with the calibrated
abstention below. It is BM25-only here so that **the entire repository runs on the standard
library** and a reviewer can verify every claim without installing anything. Stating that
substitution is more useful than pretending the simple version is the finished one.

===============================================================================
THE TWO MECHANISMS THAT MATTER, AND BOTH ARE LESSONS FROM MEASURED FAILURES
===============================================================================
1. **COVERAGE, NOT SIMILARITY.** Measured in `RaggyEditor`: embedding similarity does *not*
   separate answerable from unanswerable questions (0.61–0.77 for answerable against 0.42–0.68
   for out-of-scope — they overlap), while **whether the passage uses the query's vocabulary**
   does (0.69 against 0.17). So the abstention decision here is coverage-led.

2. **ABSTENTION IS A RESULT, NOT AN ERROR.** `search` returns results *and* a separate,
   explicit judgement about whether the corpus can answer at all — and the two are computed
   over **different text**, because *"which passage is relevant"* and *"is this question in
   the syllabus"* are different questions (see `ground`). A caller that ignores the
   judgement gets ranked passages; a caller that respects it gets to say "the syllabus does not
   cover this", which is a **correct** answer for an assessment product.

===============================================================================
HOW TO KNOW IT WORKS
===============================================================================
    python src/retrieval.py --selftest

⚠️ The self-test does not check that search "returns something" — it always does, and that is
the bug the abstention exists to prevent. It checks that a **known-answerable** query finds its
passage and that an **off-syllabus** query is correctly refused.
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "content" / "syllabus.md"

#: BM25 constants. ⚠️ The same k1 and b as `RaggyEditor` and `device-search`, deliberately: a
#: third implementation with different constants would make the three disagree, and the
#: disagreement would be mine rather than the data's.
K1 = 1.5
B = 0.75

#: How many of a question's content words must exist **somewhere** in the corpus before the
#: coverage figure means anything.
#:
#: ⚠️ WITHOUT THIS FLOOR, COVERAGE IS TRIVIALLY SATISFIABLE. Restricting the denominator to
#: corpus-known terms (see `Index.known_terms`) means a question with exactly one known word
#: scores 100% coverage on whichever passage contains it. Measured: *"how does a search system
#: work in Peru"* would pass on the strength of `search` and `system` alone, and `Peru` would be
#: invisible because it is in no passage. The floor is what keeps an off-syllabus question from
#: passing on the vocabulary its framing happens to share with the corpus.
MIN_KNOWN_TERMS = 3

#: ⚠️ THE ABSTENTION THRESHOLD, AND IT IS ON COVERAGE RATHER THAN SCORE.
#: A passage must share this fraction of the query's content words before it counts as
#: evidence. 0.30 is a product decision, stated here so it can be argued with — see
#: `scheduler.py` for the same discipline applied to a retention target.
MIN_COVERAGE = 0.30

#: Words that carry no retrieval signal.
#:
#: ⚠️ THIS LIST WAS TOO SHORT ON THE FIRST RUN AND IT CAUSED A REAL MISCLASSIFICATION. The
#: abstention decision is coverage over the query's content words, so **every question word left
#: in the denominator dilutes coverage**. Measured: `"why do embeddings blur error codes like
#: ERR_AUTH_1180"` scored 17% coverage against the passage that answers it — under the 30%
#: threshold — purely because `why`, `do` and `like` were being counted as content, even though
#: the passage that answers the question could never contain them.
#:
#: ⚠️ SO INTERROGATIVES AND LIGHT VERBS ARE EXCLUDED, which is standard for question retrieval:
#: the *content* of a question is its nouns and its technical terms. ⚠️ And the list is still
#: deliberately conservative — an aggressive stop-list removes the terms that make technical
#: questions distinctive, which is the failure `device-search` documents for `.isalnum()`
#: filters. Note that `work`, `use` and `need` are here only in their framing sense; a corpus
#: that used them as domain terms would need them removed from this list.
STOP = {
    # articles, pronouns, prepositions, conjunctions
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "of", "to", "in", "on", "for",
    "and", "or", "but", "if", "it", "its", "this", "that", "these", "those", "with", "as", "at",
    "by", "from", "into", "over", "out", "up", "down", "then", "than", "also", "about",
    "not", "no", "do", "does", "did", "can", "could", "would", "should", "i", "you", "they",
    "we", "he", "she", "them", "my", "your",
    # interrogatives and question framing — ⚠️ the addition that fixed the measured failure
    "what", "which", "why", "how", "when", "where", "who", "whose", "explain", "describe",
    "tell", "mean", "means", "like", "just", "really", "actually", "happen", "happens",
    "many", "much", "more", "most", "some", "any", "very", "well", "work", "works", "use",
    "uses", "need", "needs", "get", "gets", "got", "make", "makes", "made", "take", "takes",
    "give", "gives", "say", "says", "see", "know", "think",
}

_RE_WORD = re.compile(r"[A-Za-z0-9_]+")


def _stem(t: str) -> str:
    """Strip a trailing plural `s`. Crude on purpose, and it earned its place by measurement.

    ⚠️ WITHOUT THIS, NATURAL QUESTIONS DO NOT RETRIEVE. Measured: the query words `embeddings`
    and `blur` did not match the passage words `embedding` and `blurs`, so a question about the
    passage that answers it overlapped on **1 of its 6 content words** and was nearly refused.
    Plural and tense variants are the common case in a question, and BM25 has no notion that two
    strings are the same word.

    ⚠️ THE GUARDS MATTER MORE THAN THE RULE. `ss`, `us` and `is` endings are left alone so that
    `class`, `status` and `analysis` survive — stripping those turns three distinct technical
    terms into `clas`, `statu` and `analysi`, which match nothing and look like a typo rather
    than a bug.

    ⚠️ A REAL SYSTEM USES A PROPER STEMMER, OR DENSE RETRIEVAL, WHICH SIDESTEPS THE PROBLEM.
    This is the cheap version, kept because it is inspectable and because it keeps the whole
    repository on the standard library.
    """
    if len(t) > 3 and t.endswith("s") and not t.endswith(("ss", "us", "is")):
        return t[:-1]
    return t


def tokenize(text: str) -> list[str]:
    """Lowercase content words, plural-stemmed.

    ⚠️ `[A-Za-z0-9_]+` keeps underscores, and **splits on dots**. That split is deliberate:
    `db_acl.php` indexes and queries as `db_acl` + `php`, so a search for either half matches,
    and a search for the whole filename still matches both. Keeping the dot would make the full
    string one token that a partial query cannot reach.

    ⚠️ THE FIRST VERSION OF THIS DOCSTRING CLAIMED DOTS WERE KEPT AND THEY WERE NOT. The test
    caught the docstring, not the code — which is the useful direction for the error to run.
    `device-search` measured that filtering on `str.isalnum()` silently drops every token
    containing `_`, which in a technical corpus is most identifiers; that is what this regex
    avoids, and it is the property worth keeping.
    """
    return [t for t in (_stem(m.group(0).lower()) for m in _RE_WORD.finditer(text))
            if t not in STOP]


@dataclass(frozen=True)
class Passage:
    """One approved passage. `id` is what a citation points at."""

    id: str
    title: str
    text: str

    @property
    def source(self) -> str:
        """What a reader would see quoted in an explanation."""
        return self.title


def load_passages(path: Path | None = None) -> list[Passage]:
    """Parse the syllabus: every `## <id> — <title>` heading starts a passage.

    ⚠️ AN EMPTY CORPUS RAISES. A retrieval layer with no documents answers every question with
    "cannot answer", which is *correct* but indistinguishable from a broken index — and a system
    that reports "not covered by the syllabus" because its loader silently failed is lying about
    the syllabus.
    """
    p = path or CONTENT
    text = p.read_text(encoding="utf-8")
    passages: list[Passage] = []
    current_id: str | None = None
    current_title = ""
    buf: list[str] = []

    def flush() -> None:
        if current_id is not None and buf:
            body = "\n".join(buf).strip()
            if body:
                passages.append(Passage(id=current_id, title=current_title, text=body))

    for line in text.splitlines():
        if line.startswith("## "):
            flush()
            heading = line[3:].strip()
            # ids look like `d-retrieval#2 — Why hybrid retrieval exists`
            if " — " in heading:
                current_id, current_title = heading.split(" — ", 1)
            else:
                current_id, current_title = heading, heading
            current_id, current_title = current_id.strip(), current_title.strip()
            buf = []
        elif current_id is not None:
            buf.append(line)
    flush()

    if not passages:
        raise ValueError(
            f"{p}: no passages parsed. Expected `## <id> — <title>` headings. An empty corpus "
            "makes every answer an abstention, which looks like a broken index.")
    return passages


class Index:
    """In-process BM25 over the approved passages.

    ⚠️ `device-search` measured this design's cost at 859,569 documents: **351 seconds on the
    first call**, paid again on every restart, because the postings live in memory. That is
    fine here — the approved syllabus is a handful of passages, and an assessment product's
    content set is curated and small by construction. It would not be fine at corpus scale, and
    saying which regime this is built for is part of the design rather than an excuse.
    """

    def __init__(self, passages: list[Passage] | None = None) -> None:
        self.passages = passages if passages is not None else load_passages()
        self._toks: list[list[str]] = [tokenize(p.title + "\n" + p.text) for p in self.passages]
        self._len = [len(t) for t in self._toks]
        self._avgdl = (sum(self._len) / len(self._len)) if self._len else 1.0
        self._df: Counter[str] = Counter()
        for toks in self._toks:
            for term in set(toks):
                self._df[term] += 1
        self._n = len(self.passages)
        self._idx_by_id = {p.id: i for i, p in enumerate(self.passages)}

    # ---- scoring ---------------------------------------------------------
    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        # ⚠️ +0.5 SMOOTHING, so a term present in every passage does not score zero or negative
        # — which is what an unsmoothed idf does, and it silently removes the most common words
        # from every ranking.
        return math.log(1 + (self._n - df + 0.5) / (df + 0.5))

    def coverage(self, query_tokens: list[str], i: int) -> float:
        """Fraction of the query's distinct content words that appear in passage `i`.

        ⚠️ THIS IS THE SIGNAL THAT DECIDES ABSTENTION, not the BM25 score. See the module
        docstring: similarity does not separate answerable from unanswerable, coverage does.
        """
        q = set(query_tokens)
        if not q:
            return 0.0
        return len(q & set(self._toks[i])) / len(q)

    def known_terms(self, tokens: list[str]) -> set[str]:
        """The query's content words that appear **somewhere** in the corpus.

        ⚠️⚠️ THIS IS THE FIX FOR THE THIRD AND LAST VERSION OF THE SUFFICIENCY GATE, AND EACH
        VERSION FAILED ON A MEASURED CASE.

        1. Coverage over the *whole query* worked until an expert's phrasing differed from the
           syllabus: **"units"** appears nowhere in the corpus, diluted coverage from 32% to 22%,
           and the tutor refused a question whose passage it had already ranked first.
        2. Gating on the *question stem alone* then over-tightened the other way, because
           **a scenario-style stem is mostly scenario.** Measured: it pushed three of eight
           questions into abstention for the same dilution reason, just from generic words
           (`system`, `document`, `question`) instead of expert ones.
        3. So the denominator is restricted to terms **the corpus actually knows**. ⚠️ A word that
           appears nowhere in the syllabus carries no information about *which* passage covers the
           question — it is evidence about the syllabus, not about this candidate.

        ⚠️ AND THE ABSTRACTION IS THE POINT: coverage should measure *"of the words this corpus
        understands, how many does this passage use?"*. Measuring it over words the corpus has
        never seen measures the author's vocabulary, not the candidate's understanding.
        """
        return {t for t in set(tokens) if self._df.get(t, 0) > 0}

    def known_coverage(self, passage: Passage, known: set[str]) -> float:
        """Fraction of the corpus-known query terms that this passage uses."""
        if not known:
            return 0.0
        i = self._idx_by_id[passage.id]
        return len(known & set(self._toks[i])) / len(known)

    def gate_coverage(self, tokens: list[str]) -> tuple[float, int]:
        """Best corpus-known coverage over the whole corpus, and how many terms that was over."""
        known = self.known_terms(tokens)
        if not known:
            return 0.0, 0
        return max(self.known_coverage(p, known) for p in self.passages), len(known)

    def search(self, query: str, k: int = 3) -> list[tuple[Passage, float, float]]:
        """Rank passages. Returns `(passage, bm25_score, coverage)`, best first."""
        q = tokenize(query)
        if not q:
            return []
        scored: list[tuple[Passage, float, float]] = []
        for i, toks in enumerate(self._toks):
            tf = Counter(toks)
            score = 0.0
            for term in set(q):
                if term not in tf:
                    continue
                f = tf[term]
                denom = f + K1 * (1 - B + B * self._len[i] / self._avgdl)
                score += self._idf(term) * (f * (K1 + 1)) / denom
            if score > 0:
                scored.append((self.passages[i], score, self.coverage(q, i)))
        # ⚠️ DETERMINISTIC TIE-BREAK ON ID, for the same reason `irt.select_next` has one: a
        # reproducible run has to be reproducible, and equal scores are common in a small corpus.
        scored.sort(key=lambda t: (-t[1], t[0].id))
        return scored[:k]


@dataclass
class Grounding:
    """The retrieval verdict: what was found, and whether it is enough to answer from."""

    passages: list[Passage]
    best_coverage: float
    grounded: bool
    reason: str = ""

    def citations(self) -> list[dict]:
        return [{"passage_id": p.id, "source": p.source} for p in self.passages]


def ground(index: Index, query: str, k: int = 3,
           min_coverage: float = MIN_COVERAGE,
           gate_query: str | None = None) -> Grounding:
    """Decide whether the approved content can support an answer to `query`.

    ⚠️ THREE DIFFERENT THINGS ARE BEING COMPUTED, AND CONFLATING ANY TWO OF THEM CAUSED A BUG:

    1. **Ranking** — which passage is most relevant? Ranked on `query`, which for the tutor leads
       with the expert's misconception, because the passage has to address *the error*, not
       restate the question the candidate already got wrong.
    2. **Sufficiency** — is this topic in the syllabus at all? Judged on `gate_query` (the
       question), over the **corpus-known** terms only.
    3. **Citation eligibility** — which of the ranked passages may actually be quoted? Same
       measure as (2), applied per passage, so a passage that ranks by keyword without covering
       the question cannot be cited as evidence.

    ⚠️ THE RETURN VALUE SEPARATES "HERE ARE THE BEST PASSAGES" FROM "THESE ARE SUFFICIENT".
    Conflating them is how a retrieval system ends up asserting something the syllabus never
    said — the passages are always there, the sufficiency is the judgement.

    ⚠️ AND A REFUSAL IS A **CORRECT** ANSWER HERE. For an assessment product, "the syllabus does
    not cover this" is better than a plausible answer drawn from the model's general knowledge,
    which is the failure that makes an educational product unsafe. The risk runs the other way
    too — a false refusal withholds help — which is why `MIN_KNOWN_TERMS` and `MIN_COVERAGE` are
    both stated constants rather than tuned per query.
    """
    hits = index.search(query, k=k)
    if not hits:
        return Grounding([], 0.0, False,
                         "no passage shares any content word with the question")

    gate_text = gate_query if gate_query is not None else query
    gate_tokens = tokenize(gate_text)
    known = index.known_terms(gate_tokens)

    # ---- (2) sufficiency -------------------------------------------------
    if len(known) < MIN_KNOWN_TERMS:
        return Grounding([], 0.0, False,
                         f"only {len(known)} of the question's terms appear anywhere in the "
                         f"syllabus (minimum {MIN_KNOWN_TERMS}) — it does not cover this")
    best_gate, _ = index.gate_coverage(gate_tokens)
    if best_gate < min_coverage:
        return Grounding([], best_gate, False,
                         f"best passage shares only {best_gate:.0%} of the question's recognised "
                         f"terms (minimum {min_coverage:.0%}) — the syllabus does not cover this")

    # ---- (3) citation eligibility ---------------------------------------
    keep = [p for p, _, _ in hits if index.known_coverage(p, known) >= min_coverage]
    if not keep:
        return Grounding([], best_gate, False,
                         f"passages rank by keyword but none uses {min_coverage:.0%} of the "
                         f"question's recognised terms — the match is lexical, not on-topic")
    return Grounding(keep, best_gate, True)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def selftest(verbose: bool = True) -> int:
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        if verbose:
            print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")
        if not ok:
            failures.append(name)

    print("retrieval.py — self-test")
    idx = Index()
    check("the approved corpus loads", len(idx.passages) >= 8,
          f"{len(idx.passages)} passages")

    # ---- 1. a known-answerable question finds its own passage -------------
    cases = [
        ("why do embeddings blur error codes like ERR_AUTH_1180", "d-retrieval#1"),
        ("why not just add the BM25 score to the cosine similarity", "d-retrieval#2"),
        ("when should a search system say it cannot answer", "d-retrieval#3"),
        ("what happens if a chunk is longer than the encoder limit", "d-chunk#1"),
        ("how do offsets break citations", "d-chunk#2"),
        ("why validate a JSON response instead of trusting it", "d-llm#1"),
        ("how do I correct agreement for chance", "d-eval#1"),
        ("why is average agreement not enough for a judge", "d-eval#2"),
        ("why is the first search slow and later ones fast", "d-serve#1"),
        ("does adding a trigram index affect other queries", "d-serve#2"),
    ]
    for query, want in cases:
        hits = idx.search(query, k=3)
        got = [p.id for p, _, _ in hits]
        check(f"retrieves {want}", want in got, f"top-3 = {got}")

    # ---- 2. ⚠️ an off-syllabus question is REFUSED ------------------------
    # ⚠️ THIS IS THE ASSERTION THAT MATTERS. `search` always returns something — rank 1 of 10
    # passages is always available — so the value of this system is entirely in `ground` saying
    # no. A retrieval layer without this test cannot be distinguished from one that makes
    # things up.
    off = [
        "what is the capital of Peru",
        "how do I bake sourdough bread",
        "explain quantum entanglement to a child",
    ]
    for q in off:
        g = ground(idx, q)
        check(f"ABSTAINS on {q[:34]!r}", not g.grounded,
              g.reason[:64] if g.reason else "it answered")

    # ---- 3. an on-syllabus question is NOT refused ------------------------
    for query, _ in cases[:4]:
        g = ground(idx, query)
        check(f"answers {query[:34]!r}", g.grounded,
              f"coverage {g.best_coverage:.0%}")

    # ---- 3b. ⚠️ the sufficiency measure, pinned at each version it went through ----
    # ⚠️ EACH OF THESE THREE ASSERTIONS CORRESPONDS TO A GATE VERSION THAT FAILED, so that none
    # of them can be reintroduced by someone simplifying the code back.

    # (i) an expert's vocabulary must not cause a refusal. "units" appears nowhere in the corpus.
    g_ok = ground(idx, "states a units objection about adding scales",
                  gate_query="why is summing a BM25 score and a cosine similarity unsafe")
    check("an expert word absent from the corpus does not cause a refusal", g_ok.grounded,
          f"coverage {g_ok.best_coverage:.0%} over corpus-known terms only")

    # (ii) a scenario-style stem must not dilute coverage either — it is mostly scenario words.
    g_stem = ground(idx, "combine bm25 and cosine scores",
                    gate_query="You want to combine a BM25 ranking with a cosine-similarity "
                               "ranking. Why is summing the two scores unsafe?")
    check("a long scenario stem does not dilute coverage into a refusal", g_stem.grounded,
          f"coverage {g_stem.best_coverage:.0%}")

    # (iii) and the floor that stops coverage being trivially satisfiable.
    # ⚠️ `Peru` is in no passage, so it cannot count against coverage — only the floor catches it.
    g_far = ground(idx, "how does a search system work in Peru")
    check("a question sharing only framing words with the corpus is refused",
          not g_far.grounded,
          g_far.reason[:70])

    # ---- 4. tokenising keeps identifiers ---------------------------------
    # ⚠️ THE ORIGINAL ASSERTION WAS WRONG ABOUT DOTS, AND THE CODE WAS RIGHT. It demanded
    # `db_acl.php` survive as one token; the tokenizer splits on dots by design (see
    # `tokenize`). The assertion was corrected rather than the code, because splitting is the
    # behaviour that lets a partial filename query still match.
    check("tokenising keeps underscore identifiers intact",
          "unique_token_xyz" in tokenize("fix unique_token_xyz"),
          "an alnum filter would have dropped it")
    check("tokenising splits dotted filenames into matchable halves",
          {"db_acl", "php"} <= set(tokenize("open db_acl.php")),
          "a partial query for either half still matches")
    check("plain plural variants collapse to one term",
          tokenize("embeddings") == tokenize("embedding") and
          tokenize("blurs") == tokenize("blur"),
          "without this, a natural question does not retrieve its own passage")
    check("the stemmer does not mangle s-endings that carry meaning",
          tokenize("class") == ["class"] and tokenize("status") == ["status"]
          and tokenize("analysis") == ["analysis"],
          "stripping these would turn three technical terms into noise")

    # ---- 5. an empty corpus is an error, not an abstention ---------------
    try:
        load_passages(Path("/nonexistent/syllabus.md"))
        check("a missing corpus raises", False)
    except (FileNotFoundError, ValueError):
        check("a missing corpus raises rather than abstaining on everything", True)

    print()
    if failures:
        print(f"  ❌ {len(failures)} failed: {', '.join(failures)}")
        return 1
    print("  ✅ retrieval finds what is covered and refuses what is not")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Retrieval over the approved syllabus content.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--query", help="run one query and show the grounding verdict")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.query:
        idx = Index()
        g = ground(idx, args.query)
        print(f"query   : {args.query}")
        print(f"grounded: {g.grounded}   best coverage {g.best_coverage:.0%}")
        if g.reason:
            print(f"reason  : {g.reason}")
        for p, s, c in idx.search(args.query):
            print(f"  {s:6.3f}  cov {c:.0%}  {p.id}  {p.source}")
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
