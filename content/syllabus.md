# Approved syllabus content

> ⚠️ **This file is the "approved content" the tutor is grounded in.** In the real product it is
> the vetted syllabus supplied by the awarding body, and it is the only thing the tutor is
> allowed to teach from. **It is not user-supplied**: the JD's *"grounded in reliable and
> validated sources"* and *"expert validation before publication"* both depend on a human
> having approved this text, so the corpus ships with the code and is inspectable at
> `GET /content/sources`.
>
> Each `##` heading below is one **passage**, and its heading id is what citations point at.

## d-retrieval#1 — Why hybrid retrieval exists

A purely lexical search cannot answer a paraphrase: a candidate asking about "finding things
that mean the same" shares no words with a passage about semantic similarity. A purely dense
search has the opposite failure. It blurs rare literals, because an embedding maps a string
toward its neighbourhood of meaning, so `ERR_AUTH_1180` drifts toward `ERR_CONN_4422`.
Hybrid retrieval runs both and fuses the rankings, which repairs recall on identifiers without
giving up paraphrase matching.

## d-retrieval#2 — Fusing scores you cannot compare

BM25 scores are unbounded and grow with term frequency and corpus size. Cosine similarity is
bounded to `[-1, 1]`. **Adding them lets whichever happens to have the larger scale dominate the
result**, and the dominance changes as the corpus grows. Reciprocal rank fusion avoids the
problem by discarding the scores and using only the ranks, which is why it is the default here.
The cost is that it ignores confidence — an item ranked first by a weak scorer counts the same
as one ranked first by a strong one.

## d-retrieval#3 — Abstention is not a failure

A retrieval system that always returns its top five results will return five results for a
question the corpus cannot answer. Cosine similarity is relative, so a control query — a string
definitely not in the corpus — still produces confident-looking neighbours. Measured on this
project: the best *garbage* query scored 0.649 and the worst *real* query scored 0.632, so the
boundary is negative and a similarity threshold cannot separate them. The fix is structural:
report lexical matches as evidence and semantic matches as leads, and return "cannot answer"
as a first-class result rather than a low-confidence one.

## d-chunk#1 — Chunk size is what breaks retrieval

An encoder truncates at a fixed token limit, commonly 512. A chunker that cannot split an
oversized unit silently produces a chunk far past that limit, and the encoder embeds only the
first part. Measured on this project: a chunker that could not split a 946 KB document produced
a single chunk of 237,315 characters, of which roughly 99% was never embedded. **The text was in
the index and invisible to retrieval**, and every evaluation passed because the test fixture was
small enough never to reach the broken path.

## d-chunk#2 — Offsets must survive chunking

A citation is only useful if it points at the right text. If chunking rewrites or normalises
whitespace, the stored offsets no longer map back to the source, and the highlighted passage is
wrong. The invariant worth asserting is `chunk.text == document[start:end]`, checked on every
chunk, because the failure is silent: the chunk text still looks plausible, it is just not the
text at those offsets. Non-ASCII input is where this breaks first, because a character offset is
not a byte offset.

## d-llm#1 — Structured output must be validated, not trusted

Asking a model for JSON is not a contract. A response can be truncated at the token limit and
return an empty string, which is easy to mistake for a score of zero; it can omit a required
field; or it can return prose with a JSON block inside it. The reliable pattern is an explicit
schema in the prompt, a JSON response format where the provider supports it, and a validator
that **rejects and names the missing field** instead of defaulting it. A silently defaulted
field is indistinguishable from a real one downstream.

## d-eval#1 — Chance-corrected agreement

Raw agreement is not a measurement. Two raters who both approve nearly everything will agree
most of the time by luck alone, so a high percentage can carry no information. Measured on this
project: a rater that always answers "3" agreed 88.9% of the time and scored a kappa of 0.000.
Chance-corrected measures — Cohen's kappa, Krippendorff's alpha — subtract the agreement
expected by accident, which is why they can be near zero when the raw figure looks reassuring.

## d-eval#2 — Agreement and tail sensitivity are different

A judge can agree with human raters on average and still be unable to identify the case that
must fail. Measured on this project: a judge had 0.90 adjacent agreement and **0.00 sensitivity**
to the one trajectory containing a safety violation, because it carried a rule never to score
below 2. Removing that floor raised sensitivity to 1.00 and raised agreement as well. **Average
agreement was never the number that decided whether the judge was usable.**

## d-serve#1 — Why the first query is the slow one

An in-process index has to be built before it can answer, and building it on the first request
means the first user waits. Measured on this project: the in-process BM25 took 351 seconds on
its first call over 859,569 documents, and because the postings live in memory that cost is paid
again on every process restart. Moving the index into the database and maintaining it with
triggers reduced the first query to 0.002 seconds, paid once for the life of the database rather
than once per start.

## d-serve#2 — Adding an index is not a local change

A trigram index is roughly twice the size of the text it indexes, and inside the same file it
interleaves with the content table, so unrelated range scans start reading scattered pages.
Measured on this project: a trigram index took fragment search from 10.7 s to 0.015 s **and made
path search 12 times slower**, 0.73 s to 9.16 s, because the table it shared was now fragmented.
The net effect on a live search was worse. Moving it to its own file recovered most of it. The
lesson is that an index has a cost to everything sharing its storage.
