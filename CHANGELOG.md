# Changelog

## v2.1: token budget and status labels

Found by the first real v2 run (2026-10-13).

**Raised the output budget from 2,048 to 4,096 tokens.** Every request
labelled `no_content` had stopped at exactly 2,048 tokens with only
reasoning, no answer. Reasoning models spend part of the budget thinking
before they write anything, and long-output prompts ran out first. That
measured the request settings, not the provider.

**`finish_reason=length` is now always `truncated`.** The status check
looked for missing content before checking the token budget, so a request
that ran out of tokens mid-reasoning was labelled `no_content` and counted
as a failure. Now any request that hits the budget is `truncated`, and
`no_content` only means a stream that ended normally with nothing in it.

**README findings rewritten** from the first v2 run.

## v2: measurement fixes

v1 produced numbers that looked plausible but were measured wrong. Every
finding in the README was withdrawn until it's rerun on this version.

### Fixed

**1. TPOT was time per chunk, not per token.**
v1 averaged the gaps between streamed chunks. Providers batch several tokens
into one chunk: Groq sends about one token per chunk, so its TPOT was roughly
right, but OpenRouter sends 1–5, so its TPOT was overstated by about 3×.
Now: `(last token time − first token time) / (completion_tokens − 1)`, using
the provider's billed token count. It's measured from the first token of any
kind, because `completion_tokens` includes reasoning tokens, which are
decoded before content.

**2. Token usage was never read.**
Providers send usage in a final chunk with an empty `choices` list, and the
stream loop skipped choice-less chunks before looking at them. The harness
now requests usage (`stream_options={"include_usage": True}`) and reads it
before that check. Fixes 1 and 4 depend on this.

**3. Concurrency 10 never had 10 requests in flight.**
There are only 8 prompts, so a 10-worker pool topped out at 8 and the
"concurrency 10" rows were really concurrency 8. The prompt list is now
cycled to at least 2 × concurrency requests, and each run records
`peak_in_flight` so this can be checked. It warns if the pool never fills.

**4. Retry waits were hidden inside TTFT.**
The OpenAI SDK retries 429s and 5xx errors by itself (`max_retries=2` by
default), inside the same call the timer wraps, so rate-limit sleeps showed
up as latency. Groq's ~29 s TTFT outliers were partly this. The client now
uses `max_retries=0`, the harness retries on its own, each attempt is timed
separately, and `retries` and `retry_wait_ms` are stored as their own columns.

**5. Cost counted words, not tokens.**
`len(text.split())` under-counts tokens and ignores reasoning tokens
completely, even though they're billed. On OpenRouter the reasoning trace
was often longer than the answer. Cost now uses `usage.completion_tokens`.

**6. Failed requests were dropped.**
An exception from `future.result()` ended the run, and requests that
returned no content were silently left out of the stats, which made the
results look better than they were. Every request now saves a row with a
`status` (`ok`, `truncated`, `no_content`, `error`), and the summary
reports the fail rate.

**7. Averages hid the distribution, and the first requests paid for setup.**
Chunk arrival is bursty, with multi-second stalls on free tiers, so a mean
isn't what a typical request sees. `compare_providers.py` and the dashboard
now report p50/p95. Each run starts with a discarded warmup of
`concurrency` requests, so connection setup isn't counted in TTFT.

### Added

- `runs` table: wall-clock time, total output tokens, throughput (tok/s)
  and peak in-flight requests per run
- Columns: `run_id`, `status`, `error`, `finish_reason`, `prompt_tokens`,
  `completion_tokens`, `reasoning_tokens`, `retries`, `retry_wait_ms`
- `requirements.txt`

### Removed

- `approx_tokens` column (replaced by `completion_tokens`)
- `get-pip.py` (28k lines that didn't belong in the repo), `test_qwen.py`
  (leftover Cerebras test), committed `__pycache__/`

### Breaking

The `results` schema changed. `benchmark.py` refuses to write into a v1
`results.db`; delete it before rerunning.

### Verified

Tested against a fake streaming server that sends 4 tokens per chunk every
4 ms (true TPOT 1.0 ms) and returns a 429 on every 7th request:

| | v1 would report | v2 reports |
|---|---|---|
| TPOT | ~4 ms (per chunk) | 0.96 ms p50 |
| Peak in flight at concurrency 10 | 8 | 10 |
| TTFT on a request that waited 0.7 s for a 429 | includes the 0.7 s wait | 61 ms, with the wait in `retry_wait_ms` (700) |
