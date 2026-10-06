#!/usr/bin/env python3
"""
api.py — the FastAPI service layer.

===============================================================================
WHY THIS FILE EXISTS, AND WHY IT IS HERE RATHER THAN SOMEWHERE ELSE
===============================================================================
The job description names **Python / FastAPI** in the technology environment, alongside
PostgreSQL, AWS and a React/Next.js frontend. That is a description of a **product with an
API**, not a notebook and not a CLI. This track is the product-shaped thing in this
repository, so the service layer belongs here.

⚠️ `RaggyEditor/pyproject.toml` also declares FastAPI under an optional `serve` extra with the
comment *"the stdlib sidecar needs nothing, but this is the production swap"* — so a second
FastAPI layer over that engine is the natural follow-up. **This file is the reference
implementation**: typed request and response models, abstention as a first-class response, and
a test that the HTTP layer agrees with the library it wraps.

===============================================================================
THE FOUR DECISIONS THAT MAKE THIS A SERVICE RATHER THAN A WRAPPER
===============================================================================
1. **Pydantic models on every boundary, and validation that REJECTS.**
   A request that names a competency which does not exist returns 422 with the field, not a
   silently-empty recommendation. This is the same discipline as the JSON contract in the
   evaluation track: the schema is the interface, and a defaulted value is
   indistinguishable from a real one downstream.

2. **Abstention is a first-class response, not an error.**
   When a candidate has no history, `recommend` returns HTTP 200 with
   `{"item": {...}, "explanation": "...", "basis": "cold_start"}` and an explicit note that
   the pick is based on the prior. ⚠️ It does **not** 404 and it does **not** pretend to know.
   A 404 would say the endpoint is missing; a confident pick would be a measurement it does
   not have. `basis` is the field that makes the difference visible.

3. **Every response carries the uncertainty.**
   `readiness` returns θ **and its standard error**, and a `reliable: bool` that is false when
   the estimate rests on too few items. ⚠️ This is the *"uncertainty"* the JD asks for, and
   omitting it would let a client render "72% ready" from two answered questions.

4. **The tutor answers from a fixed, inspectable corpus, or refuses.**
   `POST /explain` submits a *wrong answer* and returns a named misconception, an explanation
   and the approved passage it came from — or an explicit abstention because the syllabus does
   not cover it. ⚠️ `GET /content/sources` exists so the approved set is enumerable, because
   *"grounded in validated sources"* is only meaningful if the sources are fixed.

5. **Nothing is computed in the handler.** Every endpoint delegates to `irt`, `scheduler`,
   `competency` and `selector`, which are independently tested. A service layer with logic in
   it is a second implementation, and the two drift.

===============================================================================
HOW TO RUN
===============================================================================
    pip install fastapi uvicorn pydantic
    uvicorn api:app --reload --port 8000      # from the src/ directory
    # interactive docs at http://127.0.0.1:8000/docs

⚠️ `fastapi`, `uvicorn` and `pydantic` are an OPTIONAL extra. The core modules (`irt.py`,
`scheduler.py`, `selector.py`, `competency.py`) run on the standard library alone, so this
track's maths is verifiable on a fresh clone before anything is installed — the same rule the
rest of the repository follows.

===============================================================================
HOW TO KNOW IT WORKS
===============================================================================
    python src/api.py --selftest

⚠️ The selftest uses `fastapi.testclient`, so it exercises the REAL routes: status codes,
response models, and the 422 paths. ⚠️ And the most important assertion is a **parity test** —
the HTTP response and the direct library call must produce the same recommendation. ⚠️ That is
the same move as the Inspect/DeepEval cross-check in the evaluation track: *two
implementations, one answer*. Without it, a service layer silently becomes a second
implementation of the thing it wraps.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Literal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from competency import Bank, default_bank
from retrieval import Index, Passage
from selector import (DEFAULT_WEIGHTS, CandidateState, score_item, select)
from scheduler import DEFAULT_TARGET, next_interval, retrievability
from tutor import Tutor, default_explainer, load_questions

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field, field_validator
    FASTAPI_AVAILABLE = True
except ImportError:                                            # pragma: no cover
    FASTAPI_AVAILABLE = False

BANK: Bank = default_bank()

#: The retrieval index and the tutor, built once at import. ⚠️ At this corpus size that is a few
#: milliseconds; `device-search` measures the same pattern at **351 seconds** over 859,569
#: documents, which is why it moved its index into the database. The approved syllabus is
#: curated and small by construction, so in-process is the right trade here and the wrong one at
#: scale — see `retrieval.Index`.
INDEX = Index()
TUTOR = Tutor(index=INDEX)


# =============================================================================
# Schemas
# =============================================================================

if FASTAPI_AVAILABLE:

    class Attempt(BaseModel):
        """One answer. ⚠️ `correct` is required and has no default — see the validator note."""
        item_id: str = Field(..., description="an item id from /bank/items")
        correct: bool
        day: float = Field(..., ge=0, description="days since the candidate started")

    class ReadinessRequest(BaseModel):
        candidate_id: str = "anonymous"
        attempts: list[Attempt] = Field(default_factory=list)

    class CompetencyReadiness(BaseModel):
        competency_id: str
        name: str
        domain_id: str
        exam_weight: float
        theta: float = Field(..., description="ability estimate in logits; NOT a percentage")
        standard_error: float
        n_items: int
        level: Literal["foundation", "developing", "proficient", "advanced", "unmeasured"]
        #: ⚠️ FALSE WHEN THE ESTIMATE RESTS ON TOO FEW ITEMS. A client must not render a
        #: readiness figure it cannot defend, and the server is the only place that knows.
        reliable: bool

    class ReadinessResponse(BaseModel):
        candidate_id: str
        competencies: list[CompetencyReadiness]
        exam_weighted_mean: float
        exam_weighted_bottleneck: float
        #: ⚠️ The weakest competency in each domain, because a certification is passed as a
        #: whole and a reassuring average hides a domain the candidate would fail.
        bottleneck_note: str

    class RecommendRequest(BaseModel):
        candidate_id: str
        attempts: list[Attempt] = Field(default_factory=list)
        day: float = Field(0.0, ge=0)
        weights: dict[str, float] | None = None

        @field_validator("weights")
        @classmethod
        def weights_sum_to_one(cls, v):
            """⚠️ REFUSED RATHER THAN RENORMALISED. Silently rescaling a caller's weights hides
            a units error in their code, and a score built from the wrong weights looks exactly
            like one built from the right ones."""
            if v is None:
                return v
            total = sum(v.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(
                    f"weights must sum to 1.0, got {total!r}. Refused rather than renormalised: "
                    "a silently-rescaled score is indistinguishable from a correct one.")
            unknown = set(v) - set(DEFAULT_WEIGHTS)
            if unknown:
                raise ValueError(f"unknown weight(s) {sorted(unknown)}; "
                                 f"known: {sorted(DEFAULT_WEIGHTS)}")
            return v

    class RecommendResponse(BaseModel):
        candidate_id: str
        item_id: str
        concept_id: str
        competency_id: str
        difficulty: float
        score: float
        terms: dict[str, float]
        probing: bool
        #: ⚠️ THE FIELD THAT MAKES ABSTENTION VISIBLE. "measured" means the pick is driven by
        #: this candidate's history; "cold_start" means it rests on the prior and should be
        #: presented as a starting point rather than a recommendation.
        basis: Literal["measured", "cold_start"]
        explanation: str
        note: str = ""

    class ExplainRequest(BaseModel):
        item_id: str = Field(..., description="an item id with authored content, e.g. q-002")
        chosen: str = Field(..., description="the option the candidate picked, e.g. B")

        @field_validator("chosen")
        @classmethod
        def single_option(cls, v):
            v = v.strip().upper()
            if len(v) != 1 or not v.isalpha():
                raise ValueError(f"chosen must be a single option letter, got {v!r}")
            return v

    class Citation(BaseModel):
        passage_id: str
        source: str

    class ExplainResponse(BaseModel):
        item_id: str
        chosen: str
        correct: str
        #: ⚠️ AUTHORED BY A SUBJECT EXPERT, NOT GENERATED. See `tutor.py` — a model that could
        #: invent the diagnosis would be telling candidates they misunderstood something they did
        #: not, which is the failure mode an educational product cannot ship.
        misconception: str
        explanation: str
        citations: list[Citation]
        grounded: bool
        abstained: bool
        #: ⚠️ "template" or "llm". A client must be able to tell whether a model was involved,
        #: and a reviewer with no API key must still be able to exercise the endpoint.
        generator: str
        reason: str = ""

    class ScheduleItem(BaseModel):
        item_id: str
        stability: float
        retrievability_now: float
        days_until_due: float
        target_retention: float

    class ScheduleResponse(BaseModel):
        candidate_id: str
        day: float
        items: list[ScheduleItem]
        note: str = ""

    app = FastAPI(
        title="Adaptive Practice API",
        version="1.0",
        description=(
            "Competency-based adaptive practice: readiness with uncertainty, next-item "
            "recommendation with its explanation, and spaced-repetition scheduling.\n\n"
            "⚠️ Every readiness figure carries a standard error and a `reliable` flag. "
            "θ is a logit, not a percentage, and the mapping to a label is a product decision."
        ),
    )

    # ---- helpers ----------------------------------------------------------

    def _validate_attempts(attempts: list[Attempt]) -> CandidateState:
        """Build state, refusing unknown items instead of ignoring them.

        ⚠️ AN UNKNOWN ITEM ID IS A 422, NOT A SKIP. Silently dropping it would make a client's
        typo look like a candidate who answered fewer questions — and the readiness number would
        be quietly wrong rather than loudly broken.
        """
        state = CandidateState(candidate_id="pending")
        for a in attempts:
            if a.item_id not in BANK.items:
                raise HTTPException(
                    status_code=422,
                    detail=f"unknown item_id {a.item_id!r}. "
                           f"GET /bank/items for the {len(BANK.items)} valid ids.")
            state.record(BANK, BANK.items[a.item_id], a.correct, a.day)
        return state

    def _readiness_rows(state: CandidateState) -> list[CompetencyReadiness]:
        rows: list[CompetencyReadiness] = []
        for cid, comp in sorted(BANK.competencies.items()):
            hist = state.history.get(cid, [])
            theta, se = state.ability(cid)
            rows.append(CompetencyReadiness(
                competency_id=cid,
                name=comp.name,
                domain_id=comp.domain,
                exam_weight=BANK.domains[comp.domain].exam_weight,
                theta=theta,
                standard_error=se,
                n_items=len(hist),
                level="unmeasured" if not hist
                      else _level(irt_estimate(hist)),
                # ⚠️ THE THRESHOLD IS STATED, NOT IMPLIED. 0.5 logits of standard error is
                # roughly 8–12 items; below that a readiness figure is mostly prior.
                reliable=bool(hist) and se < 0.5,
            ))
        return rows

    def irt_estimate(hist):
        from irt import estimate_ability
        return estimate_ability(hist)

    def _level(est) -> str:
        return est.competency_level()

    # ---- endpoints --------------------------------------------------------

    @app.get("/health")
    def health() -> dict:
        """⚠️ Reports what is LOADED, not merely that the process is up.

        The `device-search` track documents the same principle for its model store: a health
        endpoint that only says "ok" tells a client nothing about whether the thing it needs is
        present, and it is the reason a broken dependency looks like a slow service.
        """
        return {
            "status": "ok",
            "items": len(BANK.items),
            "competencies": len(BANK.competencies),
            "domains": len(BANK.domains),
            "graph_valid": not BANK.validate(),
            "weights": DEFAULT_WEIGHTS,
        }

    @app.get("/bank/items")
    def bank_items() -> list[dict]:
        return [{"item_id": it.id, "concept_id": it.concept,
                 "competency_id": BANK.competency_of(it.id).id,
                 "domain_id": BANK.domain_of(it.id).id,
                 "difficulty": it.difficulty, "discrimination": it.discrimination}
                for it in BANK.items.values()]

    @app.get("/bank/competencies")
    def bank_competencies() -> dict:
        return {"coverage": BANK.coverage(),
                "domains": [{"id": d.id, "name": d.name, "exam_weight": d.exam_weight}
                            for d in BANK.domains.values()]}

    @app.post("/readiness", response_model=ReadinessResponse)
    def readiness(req: ReadinessRequest) -> ReadinessResponse:
        """Readiness per competency, **with uncertainty**, plus the exam-weighted totals."""
        state = _validate_attempts(req.attempts)
        rows = _readiness_rows(state)
        theta_by_comp = {r.competency_id: r.theta for r in rows}

        from selector import exam_weighted_bottleneck, exam_weighted_readiness
        mean = exam_weighted_readiness(BANK, theta_by_comp)
        bottle = exam_weighted_bottleneck(BANK, theta_by_comp)

        # ⚠️ WHY BOTH TOTALS ARE RETURNED. The mean is the number people expect; the bottleneck
        # is the number that predicts passing. A response with only the mean lets a client show
        # a reassuring figure to a candidate who would fail a whole domain.
        weak = sorted((r for r in rows if r.exam_weight > 0),
                      key=lambda r: (r.theta, r.competency_id))[:2]
        note = (
            "Mean and bottleneck can disagree, and the bottleneck is the decision-relevant "
            "one: a certification is passed as a whole. Weakest now: "
            + ", ".join(f"{r.name} (θ={r.theta:+.2f}, SE={r.standard_error:.2f})" for r in weak)
        ) if weak else "No competencies to report."

        return ReadinessResponse(
            candidate_id=req.candidate_id,
            competencies=rows, exam_weighted_mean=mean,
            exam_weighted_bottleneck=bottle, bottleneck_note=note)

    @app.post("/recommend", response_model=RecommendResponse)
    def recommend(req: RecommendRequest) -> RecommendResponse:
        """The recommendation, with the terms that produced it and its evidential basis."""
        state = _validate_attempts(req.attempts)
        state.candidate_id = req.candidate_id
        picked = select(BANK, state, req.day, req.weights)
        if picked is None:
            # ⚠️ 409, NOT 404. The endpoint and the bank both exist; the request cannot be
            # satisfied with what is currently available. A 404 would blame the route.
            raise HTTPException(
                status_code=409,
                detail="no item available — every item has already been served today. "
                       "Advance `day` or remove the served-today exclusion.")

        comp = BANK.competency_of(picked.item.id)
        measured = bool(state.history.get(comp.id))
        return RecommendResponse(
            candidate_id=req.candidate_id,
            item_id=picked.item.id,
            concept_id=BANK.concept_of(picked.item.id).id,
            competency_id=comp.id,
            difficulty=picked.item.difficulty,
            score=picked.score,
            terms=picked.terms,
            probing=picked.probing,
            basis="measured" if measured else "cold_start",
            explanation=picked.explain(),
            note="" if measured else
                 ("Cold start: this candidate has no history, so the pick rests on the prior "
                  "and exam weighting rather than on evidence. Treat it as a starting point."))

    @app.get("/content/sources")
    def content_sources() -> dict:
        """⚠️ **The approved content, listed.** This endpoint is why the tutor is not a chatbot.

        *"Grounded in reliable and validated sources"* is only a meaningful claim if the set of
        sources is **fixed and inspectable** — a system that answers from whatever document it is
        handed cannot make it. So a reviewer can enumerate exactly what the tutor is permitted to
        teach from, and check that a citation resolves to something in this list.
        """
        return {
            "approved_passages": [
                {"passage_id": p.id, "source": p.source, "characters": len(p.text)}
                for p in INDEX.passages
            ],
            "authored_questions": sorted(TUTOR.questions),
            "note": (
                f"{len(INDEX.passages)} approved passages and "
                f"{len(TUTOR.questions)} authored questions. ⚠️ The engine holds "
                f"{len(BANK.items)} items; the rest have calibrated parameters but no authored "
                f"question text, because writing a distractor and naming the misconception it "
                f"encodes is subject-expert work. That gap is the real bottleneck in this "
                f"product, not the AI."
            ),
        }

    @app.post("/explain", response_model=ExplainResponse)
    def explain(req: ExplainRequest) -> ExplainResponse:
        """**Why the answer was wrong** — grounded in an approved passage, or a refusal.

        ⚠️ Submit a *wrong answer*, not a corpus. The corpus is fixed and pre-approved, because
        grounding is a promise about **which sources** an answer may come from.

        ⚠️ **The test that distinguishes this from a document chatbot: send the same `item_id`
        with two different `chosen` values and you get two different misconceptions.** A language
        model cannot do that, because it does not know which option the candidate ticked. This is
        distractor analysis, and it is what the role means by *"identify the reasons behind
        incorrect answers"*.
        """
        if req.item_id not in TUTOR.questions:
            raise HTTPException(
                status_code=404,
                detail=(f"no authored content for {req.item_id!r}. "
                        f"GET /content/sources for the {len(TUTOR.questions)} items that have it."))
        try:
            e = TUTOR.explain(req.item_id, req.chosen, explainer=default_explainer())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return ExplainResponse(**e.as_dict())

    @app.post("/schedule", response_model=ScheduleResponse)
    def schedule(req: RecommendRequest) -> ScheduleResponse:
        """When each attempted item is next due, and its current recall probability."""
        state = _validate_attempts(req.attempts)
        rows: list[ScheduleItem] = []
        for item_id, card in sorted(state.cards.items()):
            r = retrievability(card.since(req.day), card.stability) if card.reps else 0.0
            rows.append(ScheduleItem(
                item_id=item_id,
                stability=card.stability,
                retrievability_now=r,
                days_until_due=next_interval(card.stability, DEFAULT_TARGET),
                target_retention=DEFAULT_TARGET))
        return ScheduleResponse(
            candidate_id=req.candidate_id, day=req.day, items=rows,
            note=(f"Intervals target {DEFAULT_TARGET:.0%} recall. ⚠️ That target is not "
                  "achievable at a daily review cadence for items with stability below about "
                  "one day — see scheduler.py's selftest, which measures the gap."))


# =============================================================================
# Self-test
# =============================================================================

def selftest() -> int:
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")
        if not ok:
            failures.append(name)

    print("api.py — self-test")
    if not FASTAPI_AVAILABLE:
        print("  ⚠️  fastapi/pydantic not installed — the service layer cannot be exercised.")
        print("      Install: pip install fastapi pydantic httpx")
        print("      (the core modules still run on the standard library alone)")
        return 2

    from fastapi.testclient import TestClient
    client = TestClient(app)

    # ---- 1. liveness and content ------------------------------------------
    h = client.get("/health")
    check("GET /health is 200 and reports what is loaded", h.status_code == 200
          and h.json()["items"] == len(BANK.items), f"{h.json()['items']} items")
    check("the graph is valid at startup", h.json()["graph_valid"] is True)

    # ---- 2. the bank is browsable ----------------------------------------
    items = client.get("/bank/items")
    check("GET /bank/items lists every item", items.status_code == 200
          and len(items.json()) == len(BANK.items))

    # ---- 3. ⚠️ validation REJECTS an unknown item ------------------------
    bad = client.post("/readiness", json={"attempts": [{"item_id": "nope", "correct": True,
                                                        "day": 0}]})
    check("an unknown item_id is 422, not a silent skip", bad.status_code == 422,
          "a typo must not look like a candidate who answered fewer questions")

    # ---- 4. ⚠️ weights that do not sum to 1 are refused ------------------
    badw = client.post("/recommend", json={"candidate_id": "c", "day": 0,
                                           "weights": {"gap": 0.5}})
    check("weights that do not sum to 1.0 are 422", badw.status_code == 422,
          "refused rather than renormalised")

    # ---- 5. ⚠️ cold start is REPORTED as cold start ----------------------
    cold = client.post("/recommend", json={"candidate_id": "new", "attempts": [], "day": 0})
    body = cold.json()
    check("a cold start is 200 with basis=cold_start, not an error or a bare pick",
          cold.status_code == 200 and body["basis"] == "cold_start" and bool(body["note"]),
          "the client is told the pick is prior-driven")

    # ---- 6. readiness carries uncertainty --------------------------------
    r = client.post("/readiness", json={"attempts": [
        {"item_id": "q-001", "correct": True, "day": 0},
        {"item_id": "q-002", "correct": False, "day": 1}]})
    rr = r.json()
    row = next(c for c in rr["competencies"] if c["competency_id"] == "c-rag")
    check("readiness returns a standard error alongside theta",
          row["standard_error"] > 0 and row["n_items"] == 2)
    check("a 2-item estimate is flagged UNRELIABLE", row["reliable"] is False,
          "SE is wide after two items, and the client is told")
    check("unmeasured competencies are reported as unmeasured, not as zero-ability",
          any(c["level"] == "unmeasured" for c in rr["competencies"]))
    check("both the mean and the bottleneck are returned",
          "exam_weighted_mean" in rr and "exam_weighted_bottleneck" in rr)

    # ---- 7. ⚠️ PARITY: HTTP and the library must agree --------------------
    # ⚠️ THIS IS THE ASSERTION THAT MATTERS. A service layer that computes its own answer is a
    # second implementation, and the two drift silently. Same principle as the Inspect/DeepEval
    # cross-check in track 13: two implementations, one answer.
    attempts = [{"item_id": "q-008", "correct": False, "day": 0},
                {"item_id": "q-020", "correct": True, "day": 0}]
    http_body = client.post("/recommend",
                            json={"candidate_id": "p", "attempts": attempts, "day": 1}).json()

    lib_state = CandidateState(candidate_id="p")
    for a in attempts:
        lib_state.record(BANK, BANK.items[a["item_id"]], a["correct"], a["day"])
    lib_pick = select(BANK, lib_state, 1.0)

    check("HTTP recommendation == direct library call", http_body["item_id"] == lib_pick.item.id,
          f"both chose {http_body['item_id']}")
    check("the HTTP score equals the library score",
          abs(http_body["score"] - lib_pick.score) < 1e-12,
          f"{http_body['score']:.12f} vs {lib_pick.score:.12f}")
    check("the HTTP terms equal the library terms",
          http_body["terms"] == lib_pick.terms,
          "the explanation is the real decomposition, not a re-derivation")

    # ---- 8. ⚠️ THE TEST THAT PROVES IT IS NOT A CHATBOT -------------------
    src = client.get("/content/sources")
    check("GET /content/sources lists the approved corpus",
          src.status_code == 200 and len(src.json()["approved_passages"]) >= 8,
          f"{len(src.json()['approved_passages'])} approved passages")

    # ⚠️ SAME QUESTION, TWO WRONG ANSWERS, TWO DIAGNOSES. This is the assertion a generic
    # "chat with your docs" system cannot satisfy, and it is the reason the endpoint exists.
    b = client.post("/explain", json={"item_id": "q-002", "chosen": "B"}).json()
    c = client.post("/explain", json={"item_id": "q-002", "chosen": "C"}).json()
    check("the same question with two distractors yields two different diagnoses",
          b["misconception"] != c["misconception"] and b["explanation"] != c["explanation"],
          f"B → {b['misconception'][:34]!r}…")
    check("both explanations carry a resolvable citation",
          b["grounded"] and c["grounded"] and bool(b["citations"]) and bool(c["citations"]))

    approved = {p["passage_id"] for p in src.json()["approved_passages"]}
    check("every citation resolves to a passage in /content/sources",
          all(cit["passage_id"] in approved for cit in b["citations"] + c["citations"]),
          "a citation that does not resolve is not a citation")

    # ⚠️ and a wrong answer to a DIFFERENT question cites a DIFFERENT passage
    other = client.post("/explain", json={"item_id": "q-008", "chosen": "A"}).json()
    check("a different question cites a different passage",
          other["citations"][0]["passage_id"] != b["citations"][0]["passage_id"],
          f"{b['citations'][0]['passage_id']} vs {other['citations'][0]['passage_id']}")

    # ⚠️ A CORRECT ANSWER IS NOT AN ERROR TO EXPLAIN.
    right = client.post("/explain", json={"item_id": "q-002", "chosen": "A"}).json()
    check("answering correctly produces no misconception",
          right["misconception"] == "" and right["generator"] == "none")

    # ⚠️ VALIDATION REJECTS, AGAIN.
    check("an option letter that does not exist is 422",
          client.post("/explain", json={"item_id": "q-002", "chosen": "Z"}).status_code == 422)
    check("an item with no authored content is 404 with a pointer",
          client.post("/explain", json={"item_id": "q-033", "chosen": "A"}).status_code == 404,
          "q-033 is in the bank but has no authored question text")

    # ---- 9. scheduling reports recall probability ------------------------
    sch = client.post("/schedule", json={"candidate_id": "p", "attempts": attempts, "day": 1})
    check("POST /schedule returns an interval per attempted item",
          sch.status_code == 200 and len(sch.json()["items"]) == len(attempts))
    check("the schedule states its target retention",
          all(i["target_retention"] == DEFAULT_TARGET for i in sch.json()["items"]))

    print()
    if failures:
        print(f"  ❌ {len(failures)} failed: {', '.join(failures)}")
        return 1
    print("  ✅ the service validates, reports uncertainty, and agrees with its own library")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="FastAPI service for adaptive practice.")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    print("api.py — run with:  uvicorn api:app --reload --port 8000")
    print("                    then open http://127.0.0.1:8000/docs")
    print("                    or:  python src/api.py --selftest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
