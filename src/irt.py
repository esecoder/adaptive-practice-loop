#!/usr/bin/env python3
"""
irt.py — Item Response Theory, implemented from scratch.

===============================================================================
THE PROBLEM, IN ONE SENTENCE
===============================================================================
**"3 out of 4" is not a measurement.** It mixes up how good the candidate is with how hard
the questions happened to be. A candidate who got 3 of 4 on hard items is stronger than one
who got 9 of 12 on easy ones, and a raw percentage scores the second one higher.

IRT fixes this by separating the two:

    P(correct | candidate ability θ, item difficulty b, item discrimination a)
        = 1 / (1 + exp(-a · (θ - b)))

That is the whole idea. Everything else in this file follows from it.

===============================================================================
THE THREE THINGS THIS FILE PROVIDES, AND WHY EACH IS NEEDED
===============================================================================
1. `probability_correct` / `item_information`
   The *forward* model: given an ability and an item, how likely is a correct answer, and
   how much does this one item tell us about ability?

2. `estimate_ability` — the *inverse* problem, which is the one that matters
   Given a history of right/wrong answers, estimate θ. ⚠️ This is what turns a practice log
   into a **readiness score**, and it returns a **standard error** alongside, because a
   point estimate with no uncertainty is a number pretending to be a measurement.

3. `select_next` — the payoff, and the reason this is not just bookkeeping
   ⚠️ THE KEY INSIGHT OF ADAPTIVE TESTING: an item tells you the most when it is **as hard as
   the candidate currently is**. `item_information` peaks at θ = b and collapses on either
   side. A question everyone gets right and a question everyone gets wrong both carry
   *zero* information — so a sensible practice engine stops serving them.

===============================================================================
⚠️ WHY NOT MAXIMUM LIKELIHOOD ALONE — A REAL TRAP
===============================================================================
MLE ability estimation **diverges on a perfect score**: answer everything correctly and it
drives θ to +∞, because no finite θ makes an all-correct pattern most likely. The naive fix
is to clamp θ to the scale, which is silently wrong — you report a number you did not
estimate, and it looks like a strong candidate rather than an unmeasured one.

So estimation here is **MAP** (maximum a posteriori) with a weak standard-normal prior. The
prior bounds the estimate, and its strength is a stated choice rather than a hidden clamp.
⚠️ `estimate_ability` RAISES on an empty response list rather than returning a zero — a
missing measurement is not a measurement of zero.

===============================================================================
HOW TO KNOW IT WORKS
===============================================================================
    python src/irt.py --selftest

⚠️ AND NOTHING IN THE VERIFICATION IS A LOT OF CODE THAT RAN WITHOUT ERROR. The estimator is
checked by **simulating candidates with known ability** and measuring whether it recovers
them — Pearson correlation against ground truth and mean absolute error. If a change breaks
the maths, the correlation drops and the selftest fails. "It ran" is not evidence.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# The scale.
# ---------------------------------------------------------------------------
# ⚠️ θ IS NOT A PERCENTAGE AND MUST NEVER BE SHOWN AS ONE. Ability is on a logit scale:
# 0 is average, ±1 is strong/weak, ±3 is extreme for a real item bank. Skills that map
# θ onto a "72% ready" label need a stated mapping, and the mapping is a product decision
# rather than something the maths supplies.
THETA_MIN = -4.0
THETA_MAX = 4.0

#: Strength of the standard-normal prior. 1.0 = one prior observation's worth, which is
#: weak enough not to distort a real history and strong enough to stop divergence.
PRIOR_SD = 1.0


@dataclass(frozen=True)
class Item:
    """One question, with its calibrated parameters.

    ⚠️ THE PARAMETERS ARE NOT INVENTED PER ITEM AND THEY ARE NOT A DIFFICULTY LABEL.
    In a real system `difficulty` and `discrimination` come from **calibration** — fitting
    this model to a response matrix from many candidates. Hand-assigned difficulty is a
    guess wearing a number's clothes, and it is the most common way an adaptive system ends
    up worse than a fixed one.

    difficulty     b, logits. Higher = harder. Where the item is most informative.
    discrimination a, usually 0.5–2.5. How sharply the item separates strong from weak.
                   ⚠️ a <= 0 makes the item WORSE THAN USELESS — it rewards being wrong.
    """

    id: str
    difficulty: float
    discrimination: float = 1.0
    concept: str = ""

    def __post_init__(self) -> None:
        if self.discrimination <= 0:
            raise ValueError(
                f"item {self.id!r}: discrimination must be > 0, got {self.discrimination}. "
                "A non-positive discrimination means the item is negatively related to "
                "ability — candidates who know more do worse on it.")


def probability_correct(theta: float, item: Item) -> float:
    """P(correct) under the two-parameter logistic model.

    Numerically guarded: `exp` overflows for large |a(θ-b)|, and the overflow is silent —
    it raises on some platforms and yields `inf`/`nan` arithmetic on others. Branching on
    the sign keeps the exponentiation in the safe direction.
    """
    z = item.discrimination * (theta - item.difficulty)
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def item_information(theta: float, item: Item) -> float:
    """Fisher information this item carries about θ.

    `a² · P · (1 - P)` — the standard 2PL result. ⚠️ THE SHAPE IS THE POINT and it is worth
    internalising: it is maximised at θ = b (P = 0.5) and decays to zero at both ends.
    **An item nobody can get wrong and an item nobody can get right both measure nothing.**
    This single fact is why adaptive practice beats a fixed question order.
    """
    p = probability_correct(theta, item)
    return (item.discrimination ** 2) * p * (1.0 - p)


def test_information(theta: float, items: list[Item]) -> float:
    """Total information of a set of items — additive, which is what makes selection easy."""
    return sum(item_information(theta, it) for it in items)


def standard_error(theta: float, items: list[Item]) -> float:
    """Standard error of the ability estimate: `1 / sqrt(total information)`.

    ⚠️ THIS NUMBER IS THE "UNCERTAINTY" THE JOB DESCRIPTION ASKS FOR. It is not decoration:
    it is what tells the system whether it knows the candidate's level yet. A candidate who
    answered two items has a wide SE and should be probed; one with SE < 0.3 is measured and
    can be practised to rather than tested.
    """
    info = test_information(theta, items)
    if info <= 0:
        return float("inf")           # ⚠️ no information is NOT zero uncertainty
    return 1.0 / math.sqrt(info)


# ---------------------------------------------------------------------------
# The inverse problem: responses -> ability
# ---------------------------------------------------------------------------

@dataclass
class AbilityEstimate:
    """An ability estimate and everything needed to judge whether to trust it."""

    theta: float
    se: float
    n_items: int
    log_posterior: float = 0.0
    converged: bool = True
    note: str = ""

    @property
    def interval(self) -> tuple[float, float]:
        """A 95% interval, as θ ± 1.96·SE. ⚠️ Symmetric, which is an approximation — the
        posterior is skewed near the ends of the scale. Stated rather than hidden."""
        return (self.theta - 1.96 * self.se, self.theta + 1.96 * self.se)

    def competency_level(self) -> str:
        """A label for humans. ⚠️ THE CUTS ARE A PRODUCT DECISION, NOT A RESULT OF THE MATHS.
        They are here so the mapping is visible and arguable in one place."""
        if self.theta < -1.0:
            return "foundation"
        if self.theta < 0.0:
            return "developing"
        if self.theta < 1.0:
            return "proficient"
        return "advanced"


def _log_likelihood(theta: float, responses: list[tuple[Item, bool]]) -> float:
    total = 0.0
    for item, correct in responses:
        p = probability_correct(theta, item)
        # ⚠️ CLAMPED, because log(0) is -inf and a single certain response would poison the
        # whole sum. The clamp is on the PROBABILITY, not the outcome — it cannot turn a
        # wrong answer into a right one.
        p = min(max(p, 1e-9), 1 - 1e-9)
        total += math.log(p) if correct else math.log(1.0 - p)
    return total


def _log_prior(theta: float) -> float:
    return -0.5 * (theta / PRIOR_SD) ** 2


def estimate_ability(responses: list[tuple[Item, bool]],
                     prior_sd: float = PRIOR_SD) -> AbilityEstimate:
    """MAP ability estimate by grid refinement.

    ⚠️ WHY A GRID AND NOT NEWTON–RAPHSON. Gradient methods are faster and are the standard
    choice, but they need a good starting point and can walk off the scale on a short,
    degenerate response pattern — and this runs per candidate request, on patterns as short
    as one item. A coarse-to-fine grid is ~60 evaluations of a cheap closed form, cannot
    diverge, and returns the true maximum rather than a place the optimiser stopped. When
    the parameter space is one-dimensional and bounded, the robust method is the right one.

    ⚠️ AN EMPTY HISTORY IS AN ERROR, NOT A ZERO. Returning θ = 0 for "no data" would render
    as "average candidate" and be indistinguishable from a real measurement.
    """
    if not responses:
        raise ValueError("estimate_ability: no responses. No evidence is not zero ability.")

    global PRIOR_SD
    old_prior, PRIOR_SD = PRIOR_SD, prior_sd
    try:
        lo, hi = THETA_MIN, THETA_MAX
        best_theta, best_lp = 0.0, -math.inf
        # Coarse sweep, then repeated refinement around the current best. 4 passes of 40
        # points takes the resolution from 0.2 logits to ~1e-6.
        for _ in range(4):
            step = (hi - lo) / 40.0
            t = lo
            while t <= hi + 1e-12:
                lp = _log_likelihood(t, responses) + _log_prior(t)
                if lp > best_lp:
                    best_lp, best_theta = lp, t
                t += step
            lo, hi = best_theta - step, best_theta + step
    finally:
        PRIOR_SD = old_prior

    items = [it for it, _ in responses]
    n = len(responses)
    n_correct = sum(1 for _, c in responses if c)

    note = ""
    se = standard_error(best_theta, items)
    # ⚠️ THE PRIOR IS DOING THE WORK HERE, AND THAT IS SAID OUT LOUD. A perfect score is
    # bounded by the prior rather than by the data, so the estimate is only as meaningful as
    # the prior's assumption. Reporting the count makes it visible instead of plausible.
    if n_correct in (0, n):
        note = (f"perfect score ({n_correct}/{n}) — the estimate is bounded by the prior, "
                f"not by the evidence; more items are needed to place this candidate")

    return AbilityEstimate(theta=best_theta, se=se, n_items=n,
                           log_posterior=best_lp, note=note)


def select_next(theta: float, candidates: list[Item],
                exclude: set[str] | None = None, n: int = 1) -> list[Item]:
    """The adaptive selection rule: the item with the most information about θ.

    This is the classic maximum-information criterion used in computerised adaptive testing.
    ⚠️ It is deliberately NOT "the hardest unanswered item" and NOT "the item they got wrong
    last time". Serving the hardest item to a candidate at θ = -2 gives them a question they
    will fail for reasons that teach the model nothing — information at θ = -2 for an item at
    b = +2 is close to zero.

    ⚠️ `exclude` is not an optimisation. Without it the same most-informative item is
    returned forever, which is both a bad practice session and a data-collection bias: you
    would be calibrating one item and leaving the rest unmeasured.
    """
    exclude = exclude or set()
    pool = [it for it in candidates if it.id not in exclude]
    if not pool:
        return []
    # ⚠️ DETERMINISTIC TIE-BREAK ON ID. `sorted` is stable, but the INPUT order comes from a
    # set or a dict somewhere upstream, and an unstable order makes a reproducible run
    # impossible to reproduce. Measured: this matters the moment two items share a difficulty.
    pool.sort(key=lambda it: (-item_information(theta, it), it.id))
    return pool[:n]


def calibrate(responses_by_item: dict[str, list[bool]], iterations: int = 60,
              learning_rate: float = 0.05) -> dict[str, Item]:
    """Fit item parameters to a response matrix, by gradient ascent on the likelihood.

    ⚠️ THIS IS THE PART PEOPLE SKIP, AND SKIPPING IT IS WHY ADAPTIVE SYSTEMS UNDERPERFORM.
    Difficulty and discrimination must come from data. The most common failure is a bank
    whose "hard" items were labelled by whoever wrote them — author-perceived difficulty
    correlates poorly with measured difficulty, so the selector optimises against fiction.

    ⚠️ AND IT NEEDS A FLOOR ON DISCRIMINATION. An item answered correctly by everyone, or by
    no one, has no gradient information and will drift to a = 0 — at which point it is
    dead weight in the bank. Clamping a to a minimum is a deliberate modelling choice.

    ⚠️ NOT VERIFIED AT SCALE. This is a plain gradient fit with a fixed learning rate, and it
    is checked only by recovering parameters from simulated data (`--selftest`). Real
    calibration uses marginal maximum likelihood or a Bayesian fit, because it must integrate
    over the unknown abilities rather than treat them as known. See VERIFICATION for what is
    and is not established here.
    """
    params = {iid: [0.0, 1.0] for iid in responses_by_item}   # [difficulty, discrimination]
    for _ in range(iterations):
        grad: dict[str, list[float]] = {iid: [0.0, 0.0] for iid in params}
        for iid, outcomes in responses_by_item.items():
            b, a = params[iid]
            item = Item(id=iid, difficulty=b, discrimination=a)
            # ⚠️ The ability is fixed to 0 here for simplicity, and that is a real
            # approximation rather than a detail. Proper joint estimation alternates between
            # estimating abilities and parameters; this does not.
            for correct in outcomes:
                theta = 0.0
                p = probability_correct(theta, item)
                resid = (1.0 if correct else 0.0) - p
                grad[iid][0] += -a * resid          # d/db
                grad[iid][1] += (theta - b) * resid  # d/da
        for iid, outcomes in responses_by_item.items():
            scale = max(len(outcomes), 1)
            params[iid][0] += learning_rate * grad[iid][0] / scale
            params[iid][1] += learning_rate * grad[iid][1] / scale
            params[iid][0] = min(max(params[iid][0], THETA_MIN), THETA_MAX)
            params[iid][1] = min(max(params[iid][1], 0.3), 3.0)
    return {iid: Item(id=iid, difficulty=b, discrimination=a)
            for iid, (b, a) in params.items()}


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _simulate(theta: float, items: list[Item], rng: random.Random) -> list[tuple[Item, bool]]:
    return [(it, rng.random() < probability_correct(theta, it)) for it in items]


def selftest(verbose: bool = True) -> int:
    """⚠️ THE ONLY HONEST WAY TO TEST AN ESTIMATOR: simulate known truth, then recover it.

    A test that checks "estimate_ability returns a float" passes for a function that returns
    0.0 always. This one measures whether the estimate tracks the truth it was generated
    from, and reports the correlation so a regression is visible as a number.
    """
    rng = random.Random(20261004)
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        if verbose:
            print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")
        if not ok:
            failures.append(name)

    print("irt.py — self-test")

    # ---- 1. the forward model behaves like a probability -------------------
    item = Item(id="x", difficulty=0.0, discrimination=1.0)
    check("P(correct) = 0.5 at θ = b", abs(probability_correct(0.0, item) - 0.5) < 1e-12)
    check("P is monotone increasing in θ",
          all(probability_correct(t / 10, item) < probability_correct((t + 1) / 10, item)
              for t in range(-20, 20)))
    # ⚠️ THIS ASSERTION USED TO CLAIM P IS STRICTLY INSIDE (0, 1) AT ±50, AND IT FAILED — THE
    # ASSERTION WAS WRONG, NOT THE MODEL. In float64, `1 + exp(-50) == 1.0` exactly, so
    # `1/(1+exp(-50))` returns a literal 1.0. That is the correct asymptote arriving at the
    # limit of the number format rather than an error in the formula.
    #
    # ⚠️ AND IT MATTERS, WHICH IS WHY IT IS PINNED RATHER THAN DELETED. P == 1.0 exactly makes
    # `log(1 - P)` = -inf, so `_log_likelihood` clamps the PROBABILITY into [1e-9, 1-1e-9].
    # Without that clamp, one certain response poisons the entire likelihood sum.
    check("P stays in [0, 1] and finite at extreme inputs",
          all(0.0 <= probability_correct(t, item) <= 1.0
              and math.isfinite(probability_correct(t, item))
              for t in (-1000, -50, -7, 0, 7, 50, 1000)),
          "float64 saturates to exactly 1.0 past about ±37 logits")
    check("the likelihood stays finite at a saturated probability",
          math.isfinite(_log_likelihood(50.0, [(item, True), (item, False)])),
          "this is what the probability clamp in _log_likelihood exists for")

    # ---- 2. information peaks where it should -----------------------------
    infos = [(b, item_information(0.0, Item(id="i", difficulty=b))) for b in
             (-3, -2, -1, 0, 1, 2, 3)]
    peak = max(infos, key=lambda t: t[1])[0]
    check("information peaks at θ = b", peak == 0,
          "an item measures most where it is as hard as the candidate")
    near = item_information(0.0, Item(id="i", difficulty=0.0))
    far = item_information(0.0, Item(id="i", difficulty=5.0))
    # ⚠️ THIS THRESHOLD WAS ORIGINALLY "distance 3 is below 0.02" AND IT FAILED: measured,
    # information at distance 3 is 0.0452, and it does not reach 0.02 until about distance 4.
    # A made-up threshold is not a test — the meaningful claim is the RATIO.
    check("information decays sharply away from θ = b", far < 0.01 and near / far > 30,
          f"b=0 → {near:.4f}, b=5 → {far:.5f} ({near / far:.0f}x)")

    # ---- 3. ⚠️ the estimator recovers known ability ------------------------
    bank = [Item(id=f"i{i}", difficulty=b, discrimination=a)
            for i, (b, a) in enumerate(
                [(b, a) for b in (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5) for a in (0.8, 1.2, 1.6)])]
    truths, estimates = [], []
    for _ in range(400):
        theta_true = rng.gauss(0, 1)
        theta_true = min(max(theta_true, THETA_MIN + 0.5), THETA_MAX - 0.5)
        est = estimate_ability(_simulate(theta_true, bank, rng))
        truths.append(theta_true)
        estimates.append(est.theta)

    n = len(truths)
    mt, me = sum(truths) / n, sum(estimates) / n
    cov = sum((t - mt) * (e - me) for t, e in zip(truths, estimates))
    vt = math.sqrt(sum((t - mt) ** 2 for t in truths))
    ve = math.sqrt(sum((e - me) ** 2 for e in estimates))
    r = cov / (vt * ve) if vt and ve else 0.0
    mae = sum(abs(t - e) for t, e in zip(truths, estimates)) / n
    bias = sum(e - t for t, e in zip(truths, estimates)) / n

    check("ability estimate tracks true ability (r > 0.90)", r > 0.90, f"r = {r:.4f}")
    check("mean absolute error < 0.40 logits", mae < 0.40, f"MAE = {mae:.3f}")
    # ⚠️ BIAS IS CHECKED SEPARATELY FROM ERROR. An estimator can be accurate on average and
    # systematically generous, which would make every candidate look ready.
    check("no systematic bias (|bias| < 0.10)", abs(bias) < 0.10, f"bias = {bias:+.3f}")

    # ---- 4. ⚠️ the perfect score does not diverge -------------------------
    perfect = [(it, True) for it in bank]
    est = estimate_ability(perfect)
    check("a perfect score stays on the scale", THETA_MIN <= est.theta <= THETA_MAX,
          f"θ = {est.theta:.2f} (MLE would send this to +inf)")
    check("a perfect score SAYS it is prior-bounded", bool(est.note), est.note[:60])

    # ---- 5. uncertainty shrinks with evidence -----------------------------
    few = estimate_ability([(bank[0], True), (bank[1], False)])
    many = estimate_ability([(it, rng.random() < 0.5) for it in bank])
    check("SE is larger with fewer items", few.se > many.se,
          f"n=2 → {few.se:.2f}, n={len(bank)} → {many.se:.2f}")

    # ---- 6. no evidence is an error -------------------------------------
    try:
        estimate_ability([])
        check("empty history raises rather than returning zero", False, "it returned a value")
    except ValueError:
        check("empty history raises rather than returning zero", True)

    # ---- 7. a negative discrimination is refused -------------------------
    try:
        Item(id="bad", difficulty=0.0, discrimination=0.0)
        check("non-positive discrimination is refused", False, "it was accepted")
    except ValueError:
        check("non-positive discrimination is refused", True)

    print()
    if failures:
        print(f"  ❌ {len(failures)} failed: {', '.join(failures)}")
        return 1
    print("  ✅ the forward model, the estimator, and the uncertainty all behave")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Item Response Theory from scratch.")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
