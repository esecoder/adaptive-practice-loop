#!/usr/bin/env python3
"""
competency.py — the competency model: domain → competency → concept → question → performance.

===============================================================================
WHAT THE JOB DESCRIPTION ASKS FOR, AND WHAT THIS IS
===============================================================================
> *"Contribute to competency models linking exams, domains, competencies, concepts,
> questions, and candidate performance."*

That sentence is a **schema**, and it is worth reading as one. Five levels of entity, each
pointing at the next:

    Exam ──has──▶ Domain ──contains──▶ Competency ──assessed by──▶ Concept
                                                                     │
                                                          Question ──┘
                                                              │
                                              Performance ────┘   (one row per attempt)

⚠️ **THE REASON IT IS A GRAPH AND NOT A TAG.** A flat `question.topic = "rag"` column
answers "what is this question about". It cannot answer the three questions the product
actually needs:

1. **Which competencies are weak?** — a competency aggregates many concepts, and a concept is
   tested by many questions. Without the middle levels there is nothing to aggregate over,
   and readiness collapses to "percentage correct" — which IRT exists to replace.
2. **What should the candidate do next?** — selection has to trade a weak *competency*
   against the *exam weight* of the domain it belongs to. That needs the exam-level link.
3. **Can the content be trusted?** — the QC gate in the JD runs *before publication*, which
   means an item must be reachable from a competency to be reviewable as a coherent set.
   ⚠️ A bank of unlinked questions cannot be reviewed for coverage at all, because coverage
   is a property of the graph.

===============================================================================
⚠️ WHAT IS AUTHORED HERE AND WHAT WOULD BE REAL
===============================================================================
The competency graph below is **authored** for this track. The item parameters are
**assigned, not calibrated**.

⚠️ **THAT IS THE TRAP `irt.py` DOCUMENTS, AND IT IS DELIBERATELY REPRODUCED HERE SO IT CAN BE
MEASURED.** `irt.py` warns that author-perceived difficulty correlates poorly with measured
difficulty, and that a bank whose parameters were guessed makes the selector optimise against
fiction. So this module ships the guesses *and* `irt.calibrate()` so the difference can be
shown. `out/bank_calibration.json` reports how far the assigned parameters are from the fitted
ones, and that gap is the point rather than an embarrassment.

⚠️ **What is genuinely missing and cannot be faked:** real candidate response data. Calibration
needs thousands of real attempts. Nothing in this repository substitutes for that, and the
report says so rather than reporting a number as though it were calibrated.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from irt import Item

TRACK = Path(__file__).resolve().parent.parent
DATA = TRACK / "data"


# =============================================================================
# The graph
# =============================================================================

@dataclass(frozen=True)
class Concept:
    """The smallest unit of knowledge that a question can test."""

    id: str
    name: str
    competency: str


@dataclass(frozen=True)
class Competency:
    """A cluster of concepts a candidate either has or has not.

    ⚠️ `weight` IS THE EXAM WEIGHTING THE JD NAMES, AND IT LIVES AT THE DOMAIN LEVEL RATHER
    THAN THE COMPETENCY LEVEL. A competency is worth how much the exam says its domain is
    worth — which is why the graph has to reach up to `Domain` for the number.
    """

    id: str
    name: str
    domain: str


@dataclass(frozen=True)
class Domain:
    """A section of the exam, with the share of the paper it carries."""

    id: str
    name: str
    exam_weight: float          # 0–1, the share of the real exam

    def __post_init__(self) -> None:
        if not 0.0 < self.exam_weight <= 1.0:
            raise ValueError(
                f"domain {self.id!r}: exam_weight must be in (0, 1], got {self.exam_weight}. "
                "A weight of zero means the domain cannot appear on the exam, so questions "
                "in it should not be in the bank at all.")


@dataclass(frozen=True)
class Exam:
    id: str
    name: str
    domains: tuple[str, ...]


@dataclass
class Bank:
    """The whole graph, plus the items, plus an index from item to concept.

    ⚠️ VALIDATED ON CONSTRUCTION, AND THE VALIDATION IS THE FEATURE. A competency graph with a
    dangling reference is not a cosmetic problem: a question attached to a competency that
    does not exist becomes unreachable from any domain, so it can never be selected by a
    weighted selector and never appear in a coverage report. It is silently invisible, which
    is the failure mode this repository keeps finding.
    """

    exam: Exam
    domains: dict[str, Domain] = field(default_factory=dict)
    competencies: dict[str, Competency] = field(default_factory=dict)
    concepts: dict[str, Concept] = field(default_factory=dict)
    items: dict[str, Item] = field(default_factory=dict)
    item_concept: dict[str, str] = field(default_factory=dict)

    # ---- validation -------------------------------------------------------
    def validate(self) -> list[str]:
        """Return a list of problems. ⚠️ EMPTY MEANS VALID; it does not return "OK"."""
        problems: list[str] = []

        for did in self.exam.domains:
            if did not in self.domains:
                problems.append(f"exam references unknown domain {did!r}")

        total = sum(d.exam_weight for d in self.domains.values())
        # ⚠️ TOLERANCE IS NOT SLOPPINESS. Floating-point sums of 0.1 + 0.2 + 0.7 are not
        # exactly 1.0, and an equality test here would fail on a correct graph.
        if abs(total - 1.0) > 1e-9:
            problems.append(
                f"domain exam_weights sum to {total!r}, not 1.0 — a weighting that does not "
                "cover the paper makes readiness scores incomparable between candidates")

        for c in self.competencies.values():
            if c.domain not in self.domains:
                problems.append(f"competency {c.id!r} references unknown domain {c.domain!r}")

        for con in self.concepts.values():
            if con.competency not in self.competencies:
                problems.append(
                    f"concept {con.id!r} references unknown competency {con.competency!r}")

        for iid, item in self.items.items():
            cid = self.item_concept.get(iid)
            if cid is None:
                problems.append(f"item {iid!r} is not linked to any concept — unreachable")
            elif cid not in self.concepts:
                problems.append(f"item {iid!r} references unknown concept {cid!r}")
            elif item.concept != cid:
                # ⚠️ TWO PLACES RECORDING THE SAME FACT IS ONE PLACE TOO MANY. If the Item and
                # the item_concept index disagree, every downstream aggregate is
                # non-deterministic in a way that depends on which one you happened to read.
                problems.append(
                    f"item {iid!r}: Item.concept={item.concept!r} disagrees with "
                    f"item_concept={cid!r}")

        for cid in self.concepts:
            if not any(self.item_concept.get(i) == cid for i in self.items):
                problems.append(f"concept {cid!r} has no questions — it cannot be practised")
        for comp in self.competencies:
            if not any(c.competency == comp for c in self.concepts.values()):
                problems.append(f"competency {comp!r} has no concepts")

        return problems

    # ---- traversal --------------------------------------------------------
    def items_for_concept(self, concept_id: str) -> list[Item]:
        return [self.items[i] for i, c in sorted(self.item_concept.items()) if c == concept_id]

    def items_for_competency(self, competency_id: str) -> list[Item]:
        return [it for cid, c in sorted(self.concepts.items()) if c.competency == competency_id
                for it in self.items_for_concept(cid)]

    def concept_of(self, item_id: str) -> Concept:
        return self.concepts[self.item_concept[item_id]]

    def competency_of(self, item_id: str) -> Competency:
        """The competency an item belongs to — the level ability is estimated AT.

        ⚠️ WHY THIS LEVEL AND NOT THE CONCEPT, AND IT IS THE POINT OF HAVING A GRAPH.
        A concept holds two or three items, and an IRT ability estimate from two responses has
        a standard error of about 1.5 logits — see `irt`'s own selftest, which measures exactly
        that at n=2. **Estimating readiness per concept would produce a number that is mostly
        prior and looks like evidence.** A competency holds three to seven, and rolls up to a
        domain, which is the level the exam itself reports at.
        """
        return self.competencies[self.concept_of(item_id).competency]

    def domain_of(self, item_id: str) -> Domain:
        """Follow the chain item → concept → competency → domain.

        ⚠️ THIS IS THE FUNCTION THE WEIGHTING DEPENDS ON. It is three lookups deep, and every
        one of them is a place a dangling reference removes an item from the product without
        an error — which is why `validate` checks all three.
        """
        concept = self.concept_of(item_id)
        competency = self.competencies[concept.competency]
        return self.domains[competency.domain]

    def coverage(self) -> dict:
        """Questions per competency. ⚠️ A COVERAGE REPORT, NOT A QUALITY ONE.

        Ten questions on one competency and one on another is not "good coverage", and this
        function deliberately does not pretend to say so — it returns counts and the caller
        decides. Labelling a count as a quality score is how a thin bank looks healthy.
        """
        return {comp: len(self.items_for_competency(comp))
                for comp in sorted(self.competencies)}


# =============================================================================
# The bank that ships with this track
# =============================================================================

def _c(cid: str, name: str, competency: str) -> Concept:
    return Concept(id=cid, name=name, competency=competency)


# ⚠️ THE COMPETENCIES ARE THE MANUAL'S OWN TRACKS, AND THAT IS NOT A GIMMICK. It means the
# ground truth for each concept is a document that exists in this repository and can be read
# to check whether a question is fair. A demo bank invented from nothing has no such check.
_DOMAINS = [
    Domain(id="d-retrieval", name="Retrieval & RAG", exam_weight=0.30),
    Domain(id="d-training", name="Training & Fine-tuning", exam_weight=0.25),
    Domain(id="d-eval", name="Evaluation & Measurement", exam_weight=0.25),
    Domain(id="d-serving", name="Serving, Scale & Cost", exam_weight=0.20),
]

_COMPETENCIES = [
    Competency(id="c-rag", name="Build a retrieval system", domain="d-retrieval"),
    Competency(id="c-chunk", name="Chunk content for retrieval", domain="d-retrieval"),
    Competency(id="c-finetune", name="Fine-tune a model", domain="d-training"),
    Competency(id="c-data", name="Curate and clean training data", domain="d-training"),
    Competency(id="c-rubric", name="Write a scoring rubric", domain="d-eval"),
    Competency(id="c-reliab", name="Measure agreement between raters", domain="d-eval"),
    Competency(id="c-serve", name="Serve a model within a latency budget", domain="d-serving"),
    Competency(id="c-cache", name="Cut inference cost", domain="d-serving"),
]

_CONCEPTS = [
    _c("k-hybrid", "Hybrid lexical + dense retrieval", "c-rag"),
    _c("k-fusion", "Fusing ranks from incomparable scorers", "c-rag"),
    _c("k-abstain", "Calibrated abstention", "c-rag"),
    _c("k-chunksize", "Chunk size and encoder limits", "c-chunk"),
    _c("k-offsets", "Offset-preserving chunking", "c-chunk"),
    _c("k-lora", "LoRA and adapters", "c-finetune"),
    _c("k-lr", "Learning rate and warmup", "c-finetune"),
    _c("k-dedup", "Deduplication and contamination", "c-data"),
    _c("k-split", "Leakage-free splits", "c-data"),
    _c("k-anchors", "Behavioural anchors", "c-rubric"),
    _c("k-veto", "Blocking findings and vetoes", "c-rubric"),
    _c("k-kappa", "Chance-corrected agreement", "c-reliab"),
    _c("k-icc", "Variance components and ICC", "c-reliab"),
    _c("k-batch", "Batching and throughput", "c-serve"),
    _c("k-kv", "KV cache and memory", "c-serve"),
    _c("k-quant", "Quantisation trade-offs", "c-cache"),
    _c("k-caching", "Incremental and semantic caching", "c-cache"),
]


def _item(iid: str, concept: str, difficulty: float, discrimination: float = 1.0) -> Item:
    return Item(id=iid, difficulty=difficulty, discrimination=discrimination, concept=concept)


# ⚠️⚠️ READ THIS BEFORE TRUSTING ANY DIFFICULTY NUMBER BELOW.
#
# These `difficulty` values are **AUTHORED**. They are my estimate of how hard each question
# is, and `irt.py`'s own documentation says why that is a problem: author-perceived difficulty
# correlates poorly with measured difficulty. `irt.calibrate()` exists to fit them from real
# responses, and `out/bank_calibration.json` reports the gap between assigned and fitted.
#
# ⚠️ The bank ships with a spread of difficulties spanning roughly -1.5 to +1.5 ON PURPOSE,
# because a bank whose items are all the same difficulty makes the adaptive selector look
# better than it is — every item is equally informative, so "adaptive" and "random" become the
# same policy. The measurement script asserts the spread for that reason.
_ITEMS = [
    _item("q-001", "k-hybrid", -0.4), _item("q-002", "k-hybrid", 0.2),
    _item("q-003", "k-hybrid", 0.9, 1.3), _item("q-004", "k-fusion", 0.6),
    _item("q-005", "k-fusion", 1.4, 1.2), _item("q-006", "k-abstain", 0.1),
    _item("q-007", "k-abstain", 1.1), _item("q-008", "k-chunksize", -0.9),
    _item("q-009", "k-chunksize", 0.5), _item("q-010", "k-offsets", 0.8),
    _item("q-011", "k-offsets", -0.2), _item("q-012", "k-lora", -0.6),
    _item("q-013", "k-lora", 0.3), _item("q-014", "k-lr", 0.7),
    _item("q-015", "k-lr", -1.0), _item("q-016", "k-dedup", -0.3),
    _item("q-017", "k-dedup", 1.0, 1.1), _item("q-018", "k-split", 0.4),
    _item("q-019", "k-split", 1.2), _item("q-020", "k-anchors", -0.8),
    _item("q-021", "k-anchors", 0.0), _item("q-022", "k-veto", 0.9, 1.4),
    _item("q-023", "k-kappa", 0.2), _item("q-024", "k-kappa", 1.3, 1.2),
    _item("q-025", "k-icc", 1.5, 0.9), _item("q-026", "k-icc", 0.6),
    _item("q-027", "k-batch", -0.5), _item("q-028", "k-batch", 0.3),
    _item("q-029", "k-kv", 0.1), _item("q-030", "k-kv", -0.7),
    _item("q-031", "k-quant", 0.8), _item("q-032", "k-quant", -1.2),
    _item("q-033", "k-caching", -0.1), _item("q-034", "k-caching", 0.5),
]


def default_bank() -> Bank:
    bank = Bank(exam=Exam(id="x-aieng", name="Applied AI Engineering",
                          domains=tuple(d.id for d in _DOMAINS)))
    bank.domains = {d.id: d for d in _DOMAINS}
    bank.competencies = {c.id: c for c in _COMPETENCIES}
    bank.concepts = {c.id: c for c in _CONCEPTS}
    bank.items = {i.id: i for i in _ITEMS}
    bank.item_concept = {i.id: i.concept for i in _ITEMS}
    return bank


def main() -> int:
    bank = default_bank()
    problems = bank.validate()
    print("competency.py — the competency model")
    print(f"  exam          : {bank.exam.name}")
    print(f"  domains       : {len(bank.domains)}  "
          f"(weights sum to {sum(d.exam_weight for d in bank.domains.values()):.2f})")
    print(f"  competencies  : {len(bank.competencies)}")
    print(f"  concepts      : {len(bank.concepts)}")
    print(f"  items         : {len(bank.items)}")
    print()
    print("  coverage (questions per competency):")
    for comp, n in bank.coverage().items():
        flag = "  ⚠️ thin" if n < 3 else ""
        print(f"    {comp:12s} {n:2d}{flag}")
    print()
    if problems:
        print(f"  ❌ {len(problems)} problem(s):")
        for p in problems:
            print(f"     - {p}")
        return 1
    print("  ✅ the graph is valid: every item is reachable from a domain")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
