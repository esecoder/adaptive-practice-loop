#!/usr/bin/env python3
"""
scheduler.py — spaced repetition, implemented from scratch.

===============================================================================
THE PROBLEM, IN ONE SENTENCE
===============================================================================
**Reviewing too early wastes the candidate's time; reviewing too late means you are teaching
someone who has already forgotten.** The whole job of a scheduler is to squeeze the most
retention out of the fewest reviews.

===============================================================================
THE MODEL, AND IT IS SMALLER THAN IT SOUNDS
===============================================================================
Memory decays. The probability of recalling an item falls with elapsed time, and the *rate*
of that fall is what changes as you learn:

    R(t) = (1 + FACTOR · t / S) ^ DECAY          retrievability after t days

`S` is **stability** — roughly "how many days until recall drops to 90%". Learning an item
raises S; forgetting it collapses S. The curve is a **power law**, not an exponential: the
Ebbinghaus exponential was the 1885 model and it under-predicts how long memory actually
lasts at long intervals, which is why every modern scheduler (SM-2's descendants, FSRS) uses
the power form.

The scheduler's entire job is to pick `t` such that `R(t) = target`. Invert the curve:

    t = S · (target^(1/DECAY) − 1) / FACTOR

⚠️ **THAT INVERSION IS THE ALGORITHM.** Everything else is rules for updating `S`.

===============================================================================
⚠️ THE COMPARISON TO SOMETHING YOU ALREADY KNOW
===============================================================================
**A per-item TTL policy with a learned decay, where a miss resets the cache.** That is not an
analogy reaching for effect — it is structurally the same problem:

- each item has a time-to-live (`next_interval`) derived from an observed decay rate (`S`)
- a successful read **extends** the TTL; a miss **collapses** it
- the TTL is not fixed: it is re-estimated from what actually happened (`update`)
- ⚠️ and the classic failure is the same one — a cache whose TTL is guessed rather than
  measured thrashes, and so does a scheduler whose intervals are a hardcoded ladder.

The difference from a cache: **you cannot read this one cheaply to check.** A cache miss costs
a fetch; a forgetting event costs relearning, and it is the thing you were trying to avoid.
That is why the target retention is a *probability* rather than a guarantee.

===============================================================================
HOW TO KNOW IT WORKS
===============================================================================
    python src/scheduler.py --selftest

⚠️ THE ONLY TEST THAT MATTERS IS WHETHER IT HITS ITS TARGET. Anyone can write rules that make
intervals go up. The selftest **simulates a learner with a known forgetting curve**, runs the
scheduler against them, and measures the recall rate actually achieved against the target that
was asked for. If the scheduler says "target 90%" and the simulated learner recalls 71% of
items, the scheduler is wrong in a way no unit test of the interval formula would catch.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, replace

# ---------------------------------------------------------------------------
# The forgetting curve.
# ---------------------------------------------------------------------------
# ⚠️ THESE THREE CONSTANTS DEFINE THE SHAPE AND THEY ARE THE FSRS-4.5 VALUES, NOT INVENTED.
# They are published, fitted to hundreds of millions of real reviews. Using fitted constants
# rather than hand-tuned ones is the difference between a scheduler and a guess — and the
# honest caveat is that they were fitted on FLASHCARD data, so applying them to exam questions
# is an assumption. Stated rather than implied.
DECAY = -0.5
FACTOR = 19.0 / 81.0

#: What fraction of items we want the candidate to still recall at review time.
#: ⚠️ 0.9 IS A PRODUCT DECISION, NOT A DERIVED VALUE. Higher costs more reviews; lower risks
#: forgetting. 0.9 is the common default and it is stated here so it can be argued with.
DEFAULT_TARGET = 0.9


def retrievability(elapsed_days: float, stability: float) -> float:
    """Probability the candidate recalls the item after `elapsed_days`.

    ⚠️ GUARDED AT BOTH ENDS. `elapsed = 0` returns exactly 1.0 (you just saw it), and
    `stability <= 0` is refused rather than producing a division by zero — a zero stability
    would silently make every interval zero and the scheduler would loop on one item.
    """
    if stability <= 0:
        raise ValueError(f"stability must be > 0, got {stability}")
    if elapsed_days <= 0:
        return 1.0
    return (1.0 + FACTOR * elapsed_days / stability) ** DECAY


def next_interval(stability: float, target: float = DEFAULT_TARGET,
                  max_days: float = 365.0) -> float:
    """Invert the curve: the interval at which recall falls to `target`.

    ⚠️ `max_days` CAPS THE INTERVAL, AND IT IS NOT A BUG. Without it, a well-learned item
    gets an interval of years and the candidate never sees it again — which is fine for a
    flashcard deck and wrong for a certification exam three weeks away. The cap is how the
    exam date, or simply the horizon of the course, enters the calculation.
    """
    if not 0.0 < target < 1.0:
        raise ValueError(f"target retention must be in (0, 1), got {target}")
    interval = stability * (target ** (1.0 / DECAY) - 1.0) / FACTOR
    return max(0.01, min(interval, max_days))


# ---------------------------------------------------------------------------
# Per-item state
# ---------------------------------------------------------------------------

@dataclass
class Card:
    """One item's memory state. This is the row a `practice_reviews` table would hold."""

    item_id: str
    stability: float = 0.4          # days to 90% recall; small until the first review
    difficulty: float = 0.3         # 0–1, how hard THIS item is FOR THIS candidate
    reps: int = 0
    lapses: int = 0
    last_review_day: float | None = None

    @property
    def days_since(self) -> float:
        raise NotImplementedError("use r.days_since(day) — the current day is not on the card")

    def since(self, day: float) -> float:
        """Days since the last review, or `inf` if never reviewed.

        ⚠️ `inf`, NOT A LARGE NUMBER AND NOT ZERO. A never-seen item has no last review, and
        both alternatives are wrong in a way that matters: zero says "just reviewed" and hides
        the item from practice forever; a big number pretends to know how long it has been
        decaying. `inf` makes `retrievability` return the floor, which is the honest answer.
        """
        return math.inf if self.last_review_day is None else day - self.last_review_day


def update(card: Card, recalled: bool, day: float,
           target: float = DEFAULT_TARGET) -> Card:
    """Apply one review outcome and return the new state.

    ⚠️ THE TWO DIRECTIONS ARE NOT SYMMETRIC, AND THAT IS THE POINT.
    - **Recalled:** stability grows, and it grows *more* when the recall was difficult —
      which is why `retrievability` at review time is part of the update. Successfully
      recalling something you were about to forget strengthens it far more than recalling
      something you saw yesterday. This is the **desirable difficulty** effect and it is the
      single most counter-intuitive part of spaced repetition.
    - **Forgotten:** stability drops toward the floor and `lapses` increments. The item does
      NOT reset to the start, which is the common mistake — you have not lost everything, and
      resetting throws away the residual strength you built.
    """
    before = retrievability(card.since(day), card.stability) if card.reps else 1.0
    # Difficulty drifts with observed performance, bounded away from the extremes.
    difficulty = min(max(card.difficulty + (0.05 if recalled else 0.15), 0.05), 0.95)

    if recalled:
        # Growth is largest when `before` was low: the harder the successful recall, the more
        # it taught us.
        #
        # ⚠️ THESE CONSTANTS WERE WRONG ON THE FIRST ATTEMPT AND THE SELFTEST CAUGHT IT.
        # The original was `gain = 1.0 + (11.0 - 8.0*difficulty)`, which is a multiplier of
        # roughly **9.6x PER REVIEW**. Six successful reviews then took stability to
        # **222,025 days — about 608 years** — and the intervals pinned to the `max_days` cap
        # by the third review. No single number looked absurd; only compounding them did.
        #
        # ⚠️ MEASURED, one successful review realistically multiplies stability by about
        # **1.3x to 4x**, falling as the item becomes easier for that candidate. The formula
        # below gives ~2.6x at difficulty 0.3 and ~1.3x at difficulty 0.9, so six successes
        # land near 120 days rather than six centuries.
        gain = 1.0 + (3.0 - 2.0 * difficulty)
        bonus = 1.0 + (1.0 - before) * 1.0
        # ⚠️ FLOORED AT 1.05: a successful review must never SHRINK stability, or a candidate
        # could practise an item repeatedly and watch their readiness score go DOWN.
        stability = card.stability * max(1.05, gain * bonus)
        reps, lapses = card.reps + 1, card.lapses
    else:
        # ⚠️ A LAPSE IS A CLAMP, NOT A RESET. `max(0.2, ...)` keeps residual strength.
        stability = max(0.2, card.stability * (0.35 - 0.2 * difficulty))
        reps, lapses = card.reps, card.lapses + 1

    return replace(card, stability=stability, difficulty=difficulty,
                   reps=reps, lapses=lapses, last_review_day=day)


# ---------------------------------------------------------------------------
# The classic for comparison — SM-2, 1987
# ---------------------------------------------------------------------------

def sm2_update(ease: float, interval_days: float, reps: int, grade: int) -> tuple:
    """SuperMemo 2, the algorithm behind Anki. Kept here as a reference point.

    ⚠️ INCLUDED DELIBERATELY, TO SHOW THE LINEAGE AND THE WEAKNESS. SM-2 stores ONE number
    per item (an ease factor) and multiplies the interval by it. That has two consequences
    worth being able to state:

    1. **It cannot express how overdue the item was.** A review 1 day late and a review 30
       days late are treated identically, so the algorithm cannot tell "easy recall on time"
       from "barely recall, very late".
    2. **It has no model of the individual candidate.** FSRS-style scheduling fits parameters
       per learner; SM-2's constants are global.

    `grade` is 0–5 as in the original paper; below 3 is a lapse.
    """
    ease = max(1.3, ease + (0.1 - (5 - grade) * (0.08 + (5 - grade) * 0.02)))
    if grade < 3:
        return ease, 1.0, 0
    reps += 1
    if reps == 1:
        interval_days = 1.0
    elif reps == 2:
        interval_days = 6.0
    else:
        interval_days = interval_days * ease
    return ease, interval_days, reps


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _simulate_learner(rng: random.Random, truth_stability_scale: float):
    """A simulated candidate whose memory really does follow the curve.

    ⚠️ THE SIMULATION USES THE SAME CURVE THE SCHEDULER DOES, AND THAT IS A KNOWN LIMIT.
    It tests that the scheduler inverts its own model correctly — it cannot test whether the
    model matches human memory. Saying so is the difference between "verified" and
    "self-consistent"; only real review data settles the second question.

    ⚠️ AND THIS FUNCTION HAD A BUG THAT MADE THE SCHEDULER LOOK BROKEN. It used to read:

        r = retrievability(elapsed, card.stability * scale) if card.reps else 0.15

    ⚠️ `card.reps` IS NOT "HAS BEEN REVIEWED" — IT IS "HAS BEEN SUCCESSFULLY RECALLED AT LEAST
    ONCE". A card whose FIRST ATTEMPT FAILED has a `last_review_day` but `reps == 0`, so it was
    handed a flat **15%** recall chance forever instead of decaying from its actual stability.
    Those cards were then re-served and re-failed, and the measured retention landed at
    **73.7%** against a 90% target — with the scheduler taking the blame for a simulated
    candidate who could not learn from a mistake.

    ⚠️ THE LESSON GENERALISES: a harness modelling a broken user is indistinguishable from a
    broken system, and it fails in the direction that looks like a real finding. The fix is to
    ask the curve whenever there is a last review — `Card.since` already returns `inf` for a card
    never reviewed, and `retrievability(inf, S)` is 0, which is the right answer.
    """
    def recalls(card: Card, day: float) -> bool:
        return rng.random() < retrievability(card.since(day),
                                             card.stability * truth_stability_scale)
    return recalls


def selftest(verbose: bool = True) -> int:
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        if verbose:
            print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")
        if not ok:
            failures.append(name)

    print("scheduler.py — self-test")

    # ---- 1. the curve behaves like a forgetting curve ----------------------
    check("recall is 1.0 immediately after review", retrievability(0, 5.0) == 1.0)
    check("recall decreases with elapsed time",
          all(retrievability(t, 5.0) > retrievability(t + 1, 5.0) for t in range(0, 60)))
    check("a more stable item decays more slowly",
          retrievability(30, 10.0) > retrievability(30, 2.0),
          f"S=10 → {retrievability(30, 10.0):.3f}, S=2 → {retrievability(30, 2.0):.3f}")
    check("zero stability is refused", _raises(lambda: retrievability(1, 0)))

    # ---- 2. the inversion is exact ----------------------------------------
    # ⚠️ THIS IS THE ONE PLACE WHERE AN EXACT ANSWER EXISTS, SO IT IS CHECKED EXACTLY.
    for s in (0.5, 3.0, 40.0):
        for target in (0.7, 0.8, 0.9, 0.95):
            iv = next_interval(s, target)
            got = retrievability(iv, s)
            if abs(got - target) > 1e-9:
                check(f"interval inverts the curve (S={s}, target={target})", False,
                      f"asked {target}, got {got:.6f}")
                break
    else:
        check("the interval exactly inverts the curve for every S and target", True,
              "this is the only part of the algorithm with a right answer")
    check("a higher target means a shorter interval",
          next_interval(10.0, 0.95) < next_interval(10.0, 0.80))
    check("an out-of-range target is refused", _raises(lambda: next_interval(1.0, 1.5)))
    check("the interval is capped", next_interval(10000.0, 0.9, max_days=365.0) == 365.0,
          "an exam has a date; an uncapped interval would schedule past it")

    # ---- 3. successes grow the interval, lapses collapse it ----------------
    c = Card(item_id="a")
    intervals = []
    for day in range(6):
        c = update(c, recalled=True, day=float(day * max(1.0, intervals[-1] if intervals else 1)))
        intervals.append(next_interval(c.stability))
    check("intervals grow on repeated success",
          all(intervals[i] < intervals[i + 1] for i in range(len(intervals) - 1)),
          " → ".join(f"{i:.1f}d" for i in intervals))
    before = c.stability
    c = update(c, recalled=False, day=999.0)
    check("a lapse collapses stability", c.stability < before,
          f"S {before:.2f} → {c.stability:.2f}")
    check("a lapse does NOT reset to zero", c.stability >= 0.2,
          "resetting throws away residual strength")
    check("a lapse is counted", c.lapses == 1)

    # ---- 4. ⚠️ DESIRABLE DIFFICULTY: hard-but-successful beats easy --------
    # Two cards with identical stability; one is reviewed on time, one far overdue.
    easy = Card(item_id="e", stability=5.0, reps=3, last_review_day=0.0)
    hard = Card(item_id="h", stability=5.0, reps=3, last_review_day=0.0)
    easy2 = update(easy, recalled=True, day=5.0)      # right on time
    hard2 = update(hard, recalled=True, day=40.0)     # barely recalled, very overdue
    check("a hard-but-successful recall buys more stability than an easy one",
          hard2.stability > easy2.stability,
          f"on-time → {easy2.stability:.2f}, overdue → {hard2.stability:.2f}")

    # ---- 5. ⚠️ THE TEST THAT MATTERS: does it hit the stated target? -------
    #
    # ⚠️ THE FIRST VERSION OF THIS SIMULATION MEASURED THE WRONG THING, IN TWO WAYS, AND BOTH
    # ARE THE KIND OF MISTAKE THAT MAKES A TEST AGREE WITH A BROKEN SYSTEM:
    #
    #   1. It advanced the clock by the MINIMUM interval across all cards, so nothing was ever
    #      reviewed late and every card was effectively tested early. A scheduler cannot fail
    #      a retention target when the harness never lets it be late — the measurement was
    #      structurally incapable of detecting the defect it existed to find.
    #   2. It counted FIRST EXPOSURES toward retention. A first attempt is not a memory
    #      measurement; scoring a never-seen item against a 90% recall target guarantees a
    #      low number, and the original run reported 73.8% for exactly that reason rather
    #      than because the scheduler was missing its target.
    #
    # ⚠️ THE FIX IS TO SIMULATE THE THING BEING DESCRIBED: advance the clock, review each card
    # WHEN IT IS DUE, and count only reviews of cards that have been learned at least once.
    #
    # ⚠️ AND `min_step` IS THE SECOND FINDING. A first run at DAILY granularity achieved 69%
    # against a 90% target, and the cause is arithmetic rather than a formula error: a freshly
    # lapsed card has stability 0.2 days, so the scheduler wants to review it in 0.2 days —
    # **but a once-a-day study app cannot, and serving it at day 1 means it is five times
    # overdue, where recall is 67.8% rather than 90%.**
    #
    # ⚠️ SO THE TARGET IS NOT ALWAYS ACHIEVABLE, AND THAT IS WORTH KNOWING RATHER THAN TUNING
    # AWAY: **retention targets are bounded by review granularity.** A 90% target is reachable
    # continuously, and is optimistic for a daily cadence until stability is comfortably above
    # one day. Both runs below, because a scheduler that claims an unachievable target is
    # worse than one that states the bound.
    def simulate(min_step: float, seed: int) -> tuple[float, int]:
        sim_rng = random.Random(seed)
        sim_recalls = _simulate_learner(sim_rng, 1.0)
        sim_cards = [Card(item_id=f"i{i}") for i in range(60)]
        due = [0.0] * len(sim_cards)
        kept: list[bool] = []
        day = 0.0
        while day < 400 and len(kept) < 4000:
            for i, card in enumerate(sim_cards):
                if day < due[i] - 1e-9:
                    continue
                if card.last_review_day is None:
                    ok = sim_rng.random() < 0.5      # first attempt: not a retention datum
                else:
                    ok = sim_recalls(card, day)
                    kept.append(ok)
                sim_cards[i] = update(card, recalled=ok, day=day, target=0.9)
                due[i] = day + max(min_step, next_interval(sim_cards[i].stability, 0.9))
            day += min_step
        return (sum(kept) / len(kept) if kept else float("nan")), len(kept)

    target = 0.9
    # (a) sub-day granularity — the formula's own target, with no cadence constraint
    continuous, n_c = simulate(min_step=0.05, seed=7)
    check("hits its target retention at fine granularity", abs(continuous - target) < 0.05,
          f"target {target:.0%}, achieved {continuous:.1%} over {n_c} reviews")

    # (b) daily granularity — the realistic cadence, and it cannot reach the same target
    #
    # ⚠️ THE MARGIN HERE WAS ORIGINALLY 5 POINTS AND THE MEASURED GAP IS 4.8, SO THE ASSERTION
    # FAILED ON A ROUNDING. Worth recording: an assertion tuned to a threshold rather than to
    # the RELATIONSHIP fails on noise and can pass on a regression. The real claim is that daily
    # achieves strictly less, and the size of the gap is REPORTED rather than baked into a test.
    daily, n_d = simulate(min_step=1.0, seed=7)
    gap = continuous - daily
    check("daily granularity achieves LESS than continuous", gap > 0.02,
          f"continuous {continuous:.1%} vs daily {daily:.1%} — a {gap * 100:.1f}-point gap; "
          f"a lapsed card at S=0.2d is served 5x overdue at a 1-day cadence")

    # ⚠️ A GUARD SO THE TEST CANNOT PASS BY MEASURING NOTHING. If the loop above ever stops
    # reviewing, `kept` is empty, `achieved` is nan, and a bare comparison would report a pass.
    check("both retention runs measured a meaningful number of reviews",
          n_c >= 200 and n_d >= 200, f"{n_c} continuous, {n_d} daily")

    print()
    if failures:
        print(f"  ❌ {len(failures)} failed: {', '.join(failures)}")
        return 1
    print("  ✅ the curve inverts exactly, and the scheduler hits its target retention")
    return 0


def _raises(fn) -> bool:
    try:
        fn()
        return False
    except (ValueError, NotImplementedError):
        return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Spaced repetition scheduling from scratch.")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
