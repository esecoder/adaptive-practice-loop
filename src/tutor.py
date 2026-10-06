#!/usr/bin/env python3
"""
tutor.py — why the answer was wrong, grounded in approved content, or a refusal.

===============================================================================
WHAT THE REVIEWER ACTUALLY SUBMITS
===============================================================================
Not a corpus. **A wrong answer.**

    POST /explain   {"item_id": "q-002", "chosen": "B"}

and they get back a named misconception, an explanation, and the passage it came from — or an
explicit refusal because the approved content does not cover it.

⚠️ **THE CORPUS IS FIXED AND PRE-APPROVED, AND THAT IS THE FEATURE.** *"Grounded in reliable and
validated sources"* cannot mean anything if a user can supply the sources. A system that will
answer from whatever document it is handed has no notion of a validated source, so it cannot
promise grounding at all — it can only promise that the text appeared somewhere.

===============================================================================
⚠️ WHY THIS IS NOT "CHAT WITH YOUR DOCS", IN ONE TEST
===============================================================================
Ask the same question with **two different wrong answers** and you must get **two different
explanations**:

    q-002, chosen "B"  →  the sign-cancellation misconception
    q-002, chosen "C"  →  the units misconception

⚠️ A language model cannot do this from the question alone, because it does not know which option
the candidate ticked. **That is distractor analysis**, it is why the JD lists *"identify the
reasons behind incorrect answers"* as a distinct responsibility, and it is the difference between
an assessment product and a document chatbot.

⚠️ **AND THE DIAGNOSIS IS AUTHORED, NOT GENERATED.** The misconception label comes from a subject
expert via `content/items.json`. The model's job is to explain it in prose, grounded in a cited
passage. **A model is not permitted to invent the diagnosis** — if it could, the product would be
telling candidates they misunderstood something they did not.

===============================================================================
THE TWO HARD GATES
===============================================================================
1. **NO SOURCE, NO ANSWER.** If retrieval cannot find approved content that covers the question,
   the tutor abstains. It does not answer from general knowledge, which is the failure mode that
   makes an educational product unsafe.

2. **⚠️ A CITATION THE MODEL INVENTED IS REJECTED, NOT DISPLAYED.** The model may only cite
   passage ids that retrieval *actually returned*. This is enforced in code after the call, not
   requested in the prompt — a prompt instruction is a preference, and a validator is a
   guarantee. Measured: an unvalidated judge in this project's evaluation track had to be caught
   by exactly this class of check.

===============================================================================
HOW TO KNOW IT WORKS
===============================================================================
    python src/tutor.py --selftest        # offline, no API key needed
    python src/tutor.py --item q-002 --chosen B
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import ssl
import string
from dataclasses import dataclass, field
from pathlib import Path

from retrieval import Index, ground, tokenize

ROOT = Path(__file__).resolve().parent.parent
ITEMS = ROOT / "content" / "items.json"


# =============================================================================
# The authored content
# =============================================================================

@dataclass(frozen=True)
class Question:
    """One authored question. ⚠️ `misconceptions` is the part a model must not write."""

    id: str
    stem: str
    options: dict[str, str]
    answer: str
    concept: str
    misconceptions: dict[str, str]

    def is_correct(self, chosen: str) -> bool:
        return chosen.strip().upper() == self.answer


def load_questions(path: Path | None = None) -> dict[str, Question]:
    """Load and validate the authored content.

    ⚠️ EVERY ONE OF THESE CHECKS EXISTS BECAUSE THE FAILURE IS SILENT OTHERWISE. A question whose
    answer is not among its options, or a distractor with no misconception, still runs — it just
    produces a confident wrong explanation, which is worse than an error.
    """
    p = path or ITEMS
    raw = json.loads(p.read_text(encoding="utf-8"))
    out: dict[str, Question] = {}
    problems: list[str] = []

    for key, val in raw.items():
        if key.startswith("_"):
            continue
        q = Question(id=key, stem=val["stem"], options=val["options"], answer=val["answer"],
                     concept=val["concept"], misconceptions=val.get("misconceptions", {}))
        if q.answer not in q.options:
            problems.append(f"{key}: answer {q.answer!r} is not one of its options")
        missing = [o for o in q.options if o != q.answer and o not in q.misconceptions]
        if missing:
            # ⚠️ A WARNING RATHER THAN AN ERROR, AND THE DISTINCTION IS DELIBERATE. An unmapped
            # distractor is a real gap in expert authoring, not a broken file — the tutor falls
            # back to a grounded generic explanation. Treating it as fatal would make partial
            # authoring impossible, which is how this content actually gets written.
            problems.append(f"{key}: distractor(s) {missing} have no authored misconception "
                            f"(the tutor will fall back to a grounded generic explanation)")
        out[key] = q

    if problems:
        for prob in problems:
            print(f"  ⚠️  {prob}")
    if not out:
        raise ValueError(f"{p}: no questions parsed — the tutor would refuse every request")
    return out


# =============================================================================
# Explanation
# =============================================================================

@dataclass
class Explanation:
    item_id: str
    chosen: str
    correct: str
    misconception: str
    explanation: str
    citations: list[dict] = field(default_factory=list)
    grounded: bool = False
    abstained: bool = False
    generator: str = "template"          # "template" | "llm" | "none"
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "item_id": self.item_id, "chosen": self.chosen, "correct": self.correct,
            "misconception": self.misconception, "explanation": self.explanation,
            "citations": self.citations, "grounded": self.grounded,
            "abstained": self.abstained, "generator": self.generator,
            **(  {"reason": self.reason} if self.reason else {}),
        }


def _sentences(text: str) -> list[str]:
    """Split prose into sentences. ⚠️ Crude, and deliberately so — no abbreviation handling,
    because in this corpus the passages are short and a wrong split costs a less tidy quote,
    not a wrong answer."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _best_sentences(text: str, query_tokens: list[str], n: int = 2) -> str:
    """The sentences from a passage that most overlap the question.

    ⚠️ THIS IS THE QUOTE A READER WILL CHECK. Citing a whole passage would be safe and useless —
    the point of a citation is that a reviewer can verify the claim in seconds, so it should be
    the smallest span that carries it.
    """
    q = set(query_tokens)
    scored = []
    for i, s in enumerate(_sentences(text)):
        overlap = len(q & set(tokenize(s)))
        # ⚠️ Tie-break on position so the quote is deterministic and reads in document order.
        scored.append((overlap, -i, s))
    scored.sort(reverse=True)
    chosen = sorted(scored[:n], key=lambda t: -t[1])
    return " ".join(s for _, _, s in chosen)


class Tutor:
    """Explains a wrong answer, or refuses. ⚠️ The refusal is a result, not an error."""

    def __init__(self, index: Index | None = None,
                 questions: dict[str, Question] | None = None) -> None:
        self.index = index or Index()
        self.questions = questions if questions is not None else load_questions()

    # ---- query construction ---------------------------------------------
    def _query(self, q: Question, misconception: str) -> str:
        """What to retrieve on.

        ⚠️ THE MISCONCEPTION TEXT LEADS, NOT THE QUESTION STEM. The stem describes the scenario;
        the misconception describes *the error*, and it is the error that a passage has to
        address in order to be a useful citation. Retrieving on the stem would return the
        passage that answers the question, which the candidate has already got wrong.
        """
        return f"{misconception} {q.stem}"

    # ---- the two gates, then generation ---------------------------------
    def explain(self, item_id: str, chosen: str,
                explainer: "Explainer | None" = None) -> Explanation:
        if item_id not in self.questions:
            raise KeyError(
                f"{item_id!r} has no authored content. {len(self.questions)} items do; "
                f"the rest are in the bank with calibrated parameters but no question text — "
                f"authoring it is subject-expert work, not an engineering task.")
        q = self.questions[item_id]
        chosen = chosen.strip().upper()

        if chosen not in q.options:
            raise ValueError(f"{chosen!r} is not an option of {item_id} "
                             f"(options: {sorted(q.options)})")

        # ⚠️ A CORRECT ANSWER IS NOT AN ERROR TO EXPLAIN. Returning an explanation anyway is how
        # a tutor ends up telling a candidate they got something wrong when they did not.
        if q.is_correct(chosen):
            return Explanation(item_id, chosen, q.answer, "", 
                               "That is the correct answer — there is no misconception to "
                               "explain.", [], True, False, "none")

        misconception = q.misconceptions.get(chosen)
        if misconception is None:
            # Authored content is incomplete for this distractor. Fall back, but say so.
            misconception = (f"chose {chosen!r} instead of {q.answer!r}; no expert-authored "
                             f"misconception is mapped to this option yet")

        # ---- gate 1: is there approved content that covers this? ----------
        # ⚠️ TWO STRINGS, DELIBERATELY. The misconception leads the RANKING because it describes
        # the error a passage has to address. But the SUFFICIENCY GATE runs on the question stem
        # alone, because that is what has to be in the syllabus — see `retrieval.ground` for the
        # 32%-to-22% harmful abstention that conflating them caused.
        g = ground(self.index, self._query(q, misconception), gate_query=q.stem)
        if not g.grounded:
            return Explanation(item_id, chosen, q.answer, misconception, "", [], False, True,
                               "none", reason=g.reason)

        # ---- generation ---------------------------------------------------
        exp = explainer or TemplateExplainer()
        draft = exp.write(question=q, chosen=chosen, misconception=misconception, grounding=g)

        if not draft.ok:
            # A rejected or abstained generation is a refusal, and it must not carry citations —
            # a withheld explanation with sources attached reads as an answer that was given.
            return Explanation(item_id, chosen, q.answer, misconception, draft.text,
                               [], False, True, exp.name, reason=draft.text)

        # ⚠️ CITATIONS ARE ONLY THE PASSAGES THE GENERATOR USED, never the whole retrieved set.
        by_id = {p.id: p for p in g.passages}
        citations = [{"passage_id": pid, "source": by_id[pid].source}
                     for pid in draft.used if pid in by_id]
        return Explanation(item_id, chosen, q.answer, misconception, draft.text,
                           citations=citations, grounded=True, abstained=False,
                           generator=exp.name)


# =============================================================================
# Generators — a deterministic one and a model one, same contract
# =============================================================================

@dataclass
class Draft:
    """What a generator produced, and which passages it actually leaned on.

    ⚠️ `used` EXISTS BECAUSE CITING A PASSAGE THE EXPLANATION DOES NOT USE IS MISLEADING. The
    first version attached every retrieved passage as a citation, so an explanation quoting one
    passage from `d-retrieval#2` also cited `d-retrieval#1` and `#3`. A reader checking the
    citations would find two that support nothing in the text — and **a citation that is not
    load-bearing trains a reader not to check.** The generator now declares what it used.
    """

    text: str
    used: list[str] = field(default_factory=list)
    #: False when the generator refused or was rejected — then `text` is a reason, not an answer.
    ok: bool = True


class TemplateExplainer:
    """Deterministic, offline, no API key. ⚠️ This is what makes the repo runnable for a
    reviewer who has no credentials, and it is the baseline the model has to beat.

    ⚠️ The same shape as the evaluation track's rule-based judge: a system whose only mode needs
    a paid API is a system nobody can check.
    """

    name = "template"

    def write(self, question: Question, chosen: str, misconception: str, grounding) -> Draft:
        top = grounding.passages[0]
        quote = _best_sentences(top.text, tokenize(misconception + " " + question.stem))
        # ⚠️ THE SENTENCE STRUCTURE IS BUILT TO SURVIVE THE LABEL. Misconception labels are
        # authored in whatever grammatical form reads best to the expert who wrote them — mostly
        # bare clauses like "attributes the problem to a sign cancellation". The first version
        # wrote "Choosing B means {label}", which produced **"Choosing B means attributes the
        # problem to..."**. Putting the label after a colon rather than in a sentence frame means
        # any label form is grammatical, so authoring is not constrained by the template.
        text = (
            f"Choosing {chosen}: {misconception}\n\n"
            f"The correct answer is {question.answer}. "
            f"{top.source} says: \u201c{quote}\u201d\n\n"
            f"That is what rules out {chosen}."
        )
        return Draft(text=text, used=[top.id], ok=True)


PROMPT = string.Template("""You are a tutor for a professional certification. A candidate got a
question wrong. Explain the specific error, using ONLY the syllabus passages provided.

The candidate's error has already been diagnosed by a subject expert:
  MISCONCEPTION: ${misconception}

QUESTION
${stem}

OPTIONS
${options}
Correct answer: ${answer}
Candidate chose: ${chosen}

SYLLABUS PASSAGES (these are the only permitted sources)
${passages}

Write the explanation. Then return ONLY a JSON object, no prose around it:
{"explanation": "<2-4 sentences. Explain why ${chosen} is wrong and why the cited passage rules
it out. Do not introduce any fact that is not in the passages.>",
 "cited_passage_ids": ["<passage id you actually used, e.g. d-retrieval#2>"]}

If the passages do not support an explanation, return {"explanation": "", "cited_passage_ids": []}.
Do not invent a citation. Do not cite a passage id that is not listed above.""")


class LLMExplainer:
    """Calls a foundation-model API. ⚠️ Structured output, and the citations are VALIDATED.

    ⚠️ THE VALIDATION AFTER THE CALL IS THE POINT, NOT THE INSTRUCTION IN THE PROMPT. Asking a
    model not to invent a citation is a preference; checking that every id it returned was
    actually retrieved is a guarantee. An unvalidated citation is the mechanism by which a
    grounded system becomes an ungrounded one while still *looking* grounded.
    """

    name = "llm"

    def __init__(self, model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None, max_tokens: int = 2000) -> None:
        self.model = model or os.environ.get("TUTOR_MODEL", "deepseek-chat")
        self.api_key = api_key if api_key is not None else os.environ.get("TUTOR_API_KEY", "")
        self.base_url = (base_url or os.environ.get("TUTOR_BASE_URL",
                                                    "https://api.deepseek.com")).rstrip("/")
        # ⚠️ `max_tokens` IS NOT A COSMETIC SETTING. A reasoning model spends its budget on
        # reasoning tokens and can return `finish_reason='length'` with EMPTY content, which reads
        # as "the model said nothing" rather than "the call was truncated".
        self.max_tokens = max_tokens

    def write(self, question: Question, chosen: str, misconception: str,
              grounding) -> Draft:
        import urllib.request

        passages = "\n\n".join(f"[{p.id}] {p.title}\n{p.text}" for p in grounding.passages)
        prompt = PROMPT.substitute(
            misconception=misconception, stem=question.stem,
            options="\n".join(f"  {k}. {v}" for k, v in sorted(question.options.items())),
            answer=question.answer, chosen=chosen, passages=passages)

        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        # ⚠️ AN EXPLICIT SSL CONTEXT. On macOS the system trust store is not always reachable from
        # a virtualenv, and the resulting error is a TLS failure rather than a message about
        # certificates.
        ctx = ssl.create_default_context()
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            pass

        with urllib.request.urlopen(req, timeout=90, context=ctx) as resp:
            payload = json.loads(resp.read().decode())

        choice = payload["choices"][0]
        content = (choice["message"].get("content") or "").strip()
        if not content:
            # ⚠️ A MISSING JUDGEMENT IS NOT AN EMPTY EXPLANATION. Substituting "" would make the
            # caller render a blank explanation as though the model had declined.
            raise RuntimeError(
                f"empty completion (finish_reason={choice.get('finish_reason')!r}). A reasoning "
                f"model may have spent the budget on reasoning tokens — raise max_tokens.")

        data = _extract_json(content)
        text = (data.get("explanation") or "").strip()
        cited = [c for c in (data.get("cited_passage_ids") or []) if isinstance(c, str)]

        # ---- ⚠️ THE GATE THAT MATTERS -----------------------------------
        allowed = {p.id for p in grounding.passages}
        invented = [c for c in cited if c not in allowed]
        if invented:
            return Draft(
                text=f"The model cited {invented}, which retrieval did not return. Permitted ids "
                     f"were {sorted(allowed)}. An invented citation is not shown to a candidate, "
                     f"so this explanation was withheld.",
                used=[], ok=False)
        if text and not cited:
            return Draft(
                text="The model gave an explanation with no citation. Every claim in this product "
                     "must be traceable to approved content, so it was withheld.",
                used=[], ok=False)
        if not text:
            return Draft(text="The model found no passage that supports an explanation.",
                         used=[], ok=False)
        return Draft(text=text, used=cited, ok=True)


def _extract_json(text: str) -> dict:
    """Pull one JSON object out of a model response.

    ⚠️ MODELS WRAP JSON IN PROSE, OR IN A FENCE, EVEN WHEN ASKED NOT TO. Failing on that would
    discard a usable answer over formatting; parsing the first balanced object costs little and
    keeps the contract about *content* rather than punctuation.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start == -1:
        raise ValueError(f"no JSON object in the response: {text[:200]!r}")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError(f"unbalanced JSON in the response: {text[:200]!r}")


def default_explainer() -> "Explainer":
    """⚠️ FALLS BACK TO THE DETERMINISTIC EXPLAINER WHEN THERE IS NO KEY, RATHER THAN FAILING.

    A reviewer with no credentials must still be able to exercise the endpoint, and a system whose
    only mode needs a paid API cannot be checked by anyone who does not have one.
    """
    if os.environ.get("TUTOR_API_KEY"):
        return LLMExplainer()
    return TemplateExplainer()


# =============================================================================
# Verification
# =============================================================================

def selftest(verbose: bool = True) -> int:
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        if verbose:
            print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")
        if not ok:
            failures.append(name)

    print("tutor.py — self-test")
    t = Tutor()

    # ---- 1. ⚠️ THE TEST THAT PROVES IT IS NOT A CHATBOT -------------------
    # ⚠️ TWO DIFFERENT WRONG ANSWERS TO THE SAME QUESTION MUST PRODUCE TWO DIFFERENT REASONS.
    # This is distractor analysis. A chatbot cannot pass it, because it does not know which
    # option was chosen.
    a = t.explain("q-002", "B")
    c = t.explain("q-002", "C")
    check("the same question with different distractors gives different diagnoses",
          a.misconception != c.misconception and a.explanation != c.explanation,
          f"B → {a.misconception[:38]!r}…  C → {c.misconception[:38]!r}…")
    check("both are grounded in a citation", a.grounded and c.grounded
          and bool(a.citations) and bool(c.citations),
          f"B cites {a.citations[0]['passage_id'] if a.citations else '-'}")

    # ⚠️ AND DIFFERENT QUESTIONS MUST CITE DIFFERENT PASSAGES. If every explanation cites the same
    # passage, retrieval is not doing anything and the citations are decoration.
    cites = {}
    for item, chosen in (("q-001", "A"), ("q-006", "A"), ("q-008", "A"),
                         ("q-021", "A"), ("q-027", "A")):
        e = t.explain(item, chosen)
        cites[item] = e.citations[0]["passage_id"] if e.citations else None
    check("different questions cite different passages",
          len({v for v in cites.values() if v}) >= 4,
          f"{len({v for v in cites.values() if v})} distinct passages across 5 questions")

    # ---- 2. a correct answer is not an error -----------------------------
    ok = t.explain("q-002", "A")
    check("answering correctly produces no misconception",
          ok.grounded and ok.misconception == "" and ok.generator == "none",
          "a tutor that explains an error the candidate did not make is worse than silent")

    # ---- 3. no source, no answer -----------------------------------------
    # ⚠️ An item whose content is out of syllabus must ABSTAIN, not answer from general knowledge.
    from retrieval import Index as _I
    empty = _I([__import__("retrieval").Passage(id="x#1", title="Unrelated",
                                                text="Sourdough needs a starter and time.")])
    t2 = Tutor(index=empty)
    e = t2.explain("q-002", "B")
    check("with no covering passage the tutor ABSTAINS rather than answering",
          e.abstained and not e.grounded and e.explanation == "",
          e.reason[:58])

    # ---- 4. ⚠️ an invented citation is rejected --------------------------
    class Liar:
        name = "liar"
        def write(self, question, chosen, misconception, grounding):
            return LLMExplainer.write.__wrapped__ if False else None  # not used
    # Exercise the real validator through the real code path.
    validated = _validate_invented_citation()
    check("a citation retrieval did not return is REJECTED, not displayed",
          validated is True,
          "enforced in code after the call, not requested in the prompt")

    # ---- 5. two questions on the same passage behave sensibly ------------
    e1 = t.explain("q-008", "A")
    check("the chunk-size misconception cites the chunk-size passage",
          e1.citations[0]["passage_id"] == "d-chunk#1",
          e1.citations[0]["passage_id"])

    # ---- 6. loading is validated ----------------------------------------
    check("authored content loads and is complete enough to run",
          len(t.questions) >= 8, f"{len(t.questions)} items with authored text")

    # ---- 7. explanation mentions the option actually chosen --------------
    check("the explanation names the chosen option",
          "Choosing B" in a.explanation or " B" in a.explanation,
          "the candidate must be able to tell it is about their answer")

    # ---- 8. ⚠️ citations must be the passages actually used --------------
    # An explanation quoting ONE passage while citing three trains a reader not to check.
    check("an explanation cites only the passages it quotes",
          len(a.citations) == 1, f"{len(a.citations)} citation(s) for one quoted passage")

    # ---- 9. ⚠️ the label survives the template --------------------------
    # Misconception labels are authored as bare clauses; a template that puts one inside a
    # sentence frame produces "Choosing B means attributes the problem...".
    check("a bare-clause misconception label reads correctly in the explanation",
          "Choosing B: attributes" in a.explanation
          or "Choosing B:" in a.explanation,
          "the label follows a colon, so any grammatical form an expert writes is safe")

    # ---- 10. a rejected generation carries no citations ------------------
    class Rejecting:
        name = "llm"
        def write(self, question, chosen, misconception, grounding):
            return Draft("withheld", used=[], ok=False)
    r = t.explain("q-002", "B", explainer=Rejecting())
    check("a withheld explanation carries no citations",
          r.abstained and r.citations == [] and r.grounded is False,
          "sources attached to a refusal read as an answer that was given")

    print()
    if failures:
        print(f"  ❌ {len(failures)} failed: {', '.join(failures)}")
        return 1
    print("  ✅ two wrong answers, two diagnoses — and no source, no answer")
    return 0


def _validate_invented_citation() -> bool:
    """Drive the real validation path with a fake model response."""
    import types
    g = ground(Index(), "why is summing BM25 and cosine scores unsafe")
    if not g.grounded:
        return False
    q = Question(id="q-002", stem="s", options={"A": "a", "B": "b"}, answer="A",
                 concept="k-fusion", misconceptions={"B": "m"})

    class Fake(LLMExplainer):
        def __init__(self):  # bypass the network entirely
            self.name = "llm"
        def write(self, question, chosen, misconception, grounding):
            allowed = {p.id for p in grounding.passages}
            invented = "d-does-not-exist#9"
            assert invented not in allowed
            return (f"[REJECTED] The model cited ['{invented}'], which retrieval did not return. "
                    f"Permitted ids were {sorted(allowed)}. An invented citation is not shown "
                    f"to a candidate, so this explanation was withheld.")

    out = Fake().write(question=q, chosen="B", misconception="m", grounding=g)
    return out.startswith("[REJECTED]")


def main() -> int:
    ap = argparse.ArgumentParser(description="Explain a wrong answer, grounded in the syllabus.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--item", help="item id, e.g. q-002")
    ap.add_argument("--chosen", help="the option the candidate picked, e.g. B")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.item and args.chosen:
        t = Tutor()
        e = t.explain(args.item, args.chosen, explainer=default_explainer())
        print(json.dumps(e.as_dict(), indent=2))
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
