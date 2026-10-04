# THEORY — the three ideas, in plain English

You asked three things: explain **item response theory**, explain **spaced-repetition
scheduling**, and explain the **adaptive practice loop** clearly — including whether it closes
the "recommendation / adaptive learning" gap or is something else entirely.

The third question has a precise answer and it is at the end. Read §1 and §2 first; they are
the two mechanisms, and the answer in §3 is that the job description's own sentence is a
formula built from them.

---

## 1. Item Response Theory — "3 out of 4" is not a measurement

### The problem it solves

Two candidates take a test.

- **Amara** got **3 of 4**.
- **Bola** got **9 of 12**.

Who is stronger? A percentage says Bola: 75% versus 83%. But that comparison is only valid if
the questions were equally hard, and they almost never are. If Amara's four questions were
genuinely difficult and Bola's twelve were easy, **Amara is the stronger candidate and the
percentage has it backwards.**

This is not a subtle statistical point; it is the reason professional certification exams cost
what they do. **A raw score mixes up two different things: how good the candidate is, and how
hard the questions happened to be.** IRT separates them.

### The model

One equation. For a candidate with ability **θ** ("theta") answering an item with difficulty
**b** and discrimination **a**:

```
P(correct) = 1 / (1 + exp(-a · (θ - b)))
```

- **θ — ability.** On a logit scale. 0 is average, +1 is strong, −1 is weak. **It is not a
  percentage and must never be displayed as one.**
- **b — difficulty.** Also in logits. The ability level at which a candidate has a 50% chance.
- **a — discrimination.** How sharply the question separates strong candidates from weak ones.
  Usually 0.5–2.5.

### The one fact worth remembering

**A question tells you the most when it is exactly as hard as the candidate is.** That is not a
rule of thumb; it is what the model says, and it is measurable:

```
information(θ, item) = a² · P · (1 − P)
```

That expression peaks at `θ = b` where P = 0.5, and collapses on both sides. Measured in
`irt.py`'s selftest: an item at b = 0 carries **0.2500** information at θ = 0, and an item at
b = 5 carries **0.00665** at θ = 0 — **38 times less.**

So: **a question everybody gets right and a question everybody gets wrong both measure nothing
at all.** Ask a weak candidate the hardest question on the paper and you learn one thing — that
they are weak — while telling them nothing useful. This one fact is why adaptive testing beats
a fixed question order, and it is why the practice selector in `selector.py` never serves an
item far from the candidate's level.

### The part that surprises people: uncertainty is an output

Estimating θ is the inverse problem — you have a string of right and wrong answers and you want
the ability that most likely produced them. `irt.py` does it by grid refinement (not
Newton–Raphson, because gradients walk off the scale on a one-item history).

The estimate comes with a **standard error**, and the SE is not decoration:

```
SE = 1 / sqrt(total information)
```

Measured in the selftest: **two items give SE = 1.55; twenty-one give SE = 0.42.** So a
readiness score from two answers is mostly the prior, and saying so is the honest response.
This SE *is* the "uncertainty" the job description asks for.

### Two traps, both demonstrated in the code

**Trap 1 — a perfect score diverges.** Maximum-likelihood estimation sends θ to +∞ when a
candidate answers everything correctly, because no finite θ makes that pattern most likely. The
sloppy fix is to clamp θ to the scale — which reports a number you did not estimate, and it
*looks* like a strong candidate rather than an unmeasured one. `irt.py` uses a weak prior (MAP)
instead, and **says so in the output**: `"perfect score (21/21) — the estimate is bounded by the
prior, not by the evidence"`.

**Trap 2 — item parameters must be calibrated, not assigned.** Difficulty is not something you
know because you wrote the question. Author-perceived difficulty correlates poorly with
measured difficulty, and `competency.py` ships **assigned** parameters with a loud warning,
because that is the trap reproducing itself in the code.

### The comparison to something you know

**IRT is load testing for a person.** You do not learn a system's capacity by throwing an
arbitrary load at it and seeing if it falls over — you probe near its limit, where the answer is
genuinely uncertain, and you report the capacity **with error bars**. A single "yes it survived
10k rps" is exactly as uninformative as "3 out of 4".

---

## 2. Spaced repetition — a TTL policy for human memory

### The problem it solves

Review too early and you waste the candidate's attention. Review too late and you are
re-teaching someone who has already forgotten. **Every minute of practice is a budget, and the
scheduler decides how to spend it.**

### The model

Memory decays. The probability of recall after `t` days is:

```
R(t) = (1 + FACTOR · t / S) ^ DECAY
```

**S — stability** — roughly "how many days until recall drops to 90%". Learning raises it;
forgetting collapses it.

⚠️ **The curve is a power law, not an exponential.** The exponential version was Ebbinghaus's
model from **1885**, and it under-predicts how long memory actually survives at long intervals.
That is why every modern scheduler uses the power form.

The scheduler's whole job is to pick the interval `t` where `R(t) = target`. Inverting the
curve:

```
t = S · (target^(1/DECAY) − 1) / FACTOR
```

**That inversion is the algorithm. Everything else is rules for updating S.**

### The counter-intuitive part: difficulty strengthens memory

Two items have identical stability. One candidate reviews theirs **right on time**; the other
reviews theirs **35 days late** and just barely recalls it. Which one is now stronger?

**The overdue one.** Measured in the selftest: on-time → stability **18.15**, overdue →
**23.27**. This is the *desirable difficulty* effect — recalling something at the edge of
forgetting strengthens it far more than recalling something you saw yesterday.

⚠️ It is also why "never let anything get overdue" is the wrong instinct.

### The comparison to something you know

**It is a per-item TTL policy with a learned decay, where a miss resets it.** Structurally the
same problem as a cache:

| cache | spaced repetition |
|---|---|
| time-to-live | `next_interval(stability)` |
| observed access decay | stability `S`, re-estimated from what happened |
| a read extends the TTL | a successful recall raises `S` |
| a miss collapses the TTL | a lapse drops `S` toward a floor |
| a guessed TTL thrashes | hand-set intervals waste the candidate's time |

⚠️ **One difference that changes everything:** you cannot read this cache cheaply to check. A
cache miss costs a fetch; forgetting costs relearning, and it is the thing you were avoiding.
That is why the target is a *probability* rather than a guarantee.

### ⚠️ And a finding the test produced

The target retention **is not always achievable**, and the reason is arithmetic rather than a
formula error. A freshly lapsed card has stability **0.2 days**, so the scheduler wants to review
it in 0.2 days — but a once-a-day study app cannot, and serving it at day 1 means it is **five
times overdue**, where recall is **67.8%** instead of 90%.

Measured:

| review cadence | achieved retention (target 90%) |
|---|---|
| sub-day (continuous) | **90.0%** — the formula hits its target exactly |
| once a day | **85.2%** — a 4.8-point shortfall, and structural |

**Retention targets are bounded by review granularity.** A scheduler that claims a target it
cannot deliver is worse than one that states the bound.

---

## 3. The adaptive practice loop — and the answer to your question

### ⚠️ Yes, it closes the recommendation gap. The job description says so itself.

Here is the requirement, quoted exactly:

> *"Develop mechanisms that determine what a candidate should practise next, taking into account
> **competency gaps**, **previous performance**, **exam weighting**, **uncertainty**, and
> **recency**."*

Read that as what it is: **a ranking function, written out factor by factor.**

```
score(item | candidate) = w₁·gap + w₂·performance + w₃·exam_weight + w₄·uncertainty + w₅·recency
```

**Every recommender system has this shape.** Given a user and a set of candidate items, estimate
each item's utility *for that user*, rank, present the top one, observe the outcome, update.
Only the vocabulary changes:

| | adaptive learning | e-commerce |
|---|---|---|
| user | candidate | shopper |
| item | question | product |
| utility | readiness gain | click / purchase probability |
| feedback | right or wrong | bought or not |
| scarce resource | **the candidate's attention** | attention and shelf space |

**The five terms the JD names are the five classic recommender signals, renamed:**

| JD's words | what it is in recommender language |
|---|---|
| competency gaps | relevance to the user's weakest area |
| previous performance | the user's interaction history |
| exam weighting | item importance / business value |
| uncertainty | **exploration bonus** — how little is known |
| recency | **novelty and decay** — do not re-show what they just saw |

### The hard problems transfer too, and naming them is the interview answer

- **Cold start.** A new candidate with no history is exactly a new user with no history. The
  API has a `basis: "cold_start"` field for this reason.
- **Exploration vs exploitation.** A question you never serve is a question you can never
  calibrate — **the item bank is a bandit.** ⚠️ This is the term my first attempt at the
  selector was missing, and the measurement caught it.
- **Selection bias.** You only observe outcomes for items you *chose* to serve, so the training
  data for your selector is generated by your selector.
- **Feedback loops.** Practise a weak competency and it stops looking weak — so a naive system
  drifts away from it just as it starts working.
- **Position bias.** The first item shown gets more engagement regardless of its quality.

### What is genuinely different, and you should say it rather than gloss it

The utility is **pedagogical, not commercial.** A shop can profit from a bad recommendation to
a returning customer. A practice platform cannot afford to demoralise someone three weeks before
their exam. **A difficulty spike has a real cost**, which is why the rubric in track 13 has a
veto rather than just a low weight, and the same instinct applies here.

### ⚠️ And the honest part: the selector does not beat the baselines

This is measured in `selector.py`, and it is recorded rather than hidden. On the simulated
candidate:

| policy | mean objective | bottleneck objective |
|---|---|---|
| **adaptive selector** | **+0.0416** | +0.0412 |
| random | +0.0483 | +0.0544 |
| least-recently-seen | +0.0477 | +0.0460 |
| **"practise your worst competency"** | **+0.1037** | +0.0857 |

**The trivial baseline wins, by more than double.** Three findings came out of diagnosing why,
and they are worth more than a win would have been:

1. **Four of the five terms the JD names are constants on a fresh candidate.** Measured spread
   across competencies: `gap` 0.000, `performance` 0.000, `uncertainty` 0.000, `recency` 0.000.
   **⚠️ A term with no spread is not a term.** Only `exam_weight` and item difficulty carried
   any signal.
2. **The objective is gameable, and that is the deeper finding.** Readiness within a domain was
   the *mean* of its competencies, and a mean responds strongly to raising one member — pumping
   a single competency in d-eval moves the whole domain from 0.250 to 0.586. **The winning
   baseline won by serving one competency 75 times out of 120.** A metric that a degenerate
   policy maximises is a broken metric, which is the same defect as an eval set containing only
   answerable questions.
3. **The correctness metric for this product is a bottleneck, not a mean.** A certification is
   passed as a whole: a candidate who is outstanding at retrieval and hopeless at evaluation
   *fails*, with a reassuring mean right up to that moment. So the API returns **both**
   `exam_weighted_mean` and `exam_weighted_bottleneck`, and says why.

⚠️ **And the ablation came back degenerate** — dropping any single term changed the outcome by
exactly 0.00000, while the served sequences genuinely differed. The mechanism is that the
simulated learning model only fires when `0.25 < p < 0.95`, which makes a weak competency almost
unteachable: at θ = −1.4 only **1 of 7** items falls in the band. **So the harness's learning
model, not the selector, decides the result — and a simplifying assumption quietly became the
whole model.**

### What would settle it

**Real candidate response data.** Every number above comes from a learner whose memory, learning
rate and competency structure I chose. The simulation can show a selector is *internally*
inconsistent; it cannot show any policy helps a person. That limitation is stated in the code
rather than buried.

---

## The three ideas in six lines

1. **IRT:** "3 of 4" conflates ability with item difficulty. Model them separately, and report
   the uncertainty — a score from two answers is mostly prior.
2. **The one fact:** a question measures most when it is as hard as the candidate is, and
   measures nothing at either extreme.
3. **Spaced repetition:** a TTL policy with a learned decay; a successful recall extends it, a
   lapse collapses it, and a *hard* successful recall strengthens memory more than an easy one.
4. **The target is bounded by cadence:** a 90% retention target is unreachable for once-a-day
   review of items with sub-day stability. Measured: 90.0% continuous, 85.2% daily.
5. **The adaptive loop *is* a recommender system**, and the JD's five factors are the five classic
   recommender signals renamed. Cold start, exploration, selection bias and feedback loops all
   transfer directly.
6. **The objective is the hard part.** Mean rewards concentrating; bottleneck rewards coverage;
   and picking which one is a statement about what the exam means, not a modelling choice.
