# Inference Provider Benchmark

Benchmarks LLM inference performance (TTFT, TPOT, reasoning overhead) across
providers under varying concurrency — Groq (GPT-OSS 120B) vs OpenRouter
(Nemotron-3-Super-120B).

## Method

8 fixed prompts (short/long input × short/long output), streamed and run
at concurrency 1, 5, and 10. Measures:
- **TTFT** — time to first content token (and, separately, to the first token of any kind)
- **TPOT** — decode time per output token, from billed token counts
- **Throughput** — output tokens per second across the whole run
- **Reasoning vs. content tokens** — thinking time isolated from answer generation
- Cost per request (billed token usage × published rates)
- Retries, retry wait, and failed or truncated requests, reported separately

Stats are p50/p95, not averages.

## Findings

From the v2 harness, run 2026-10-13. Both providers on their free tier;
p50/p95 over successful requests; one run per concurrency level.

| Provider | Conc. | n | Failed | TTFT p50 / p95 | TPOT p50 / p95 | Retries | Throughput |
|---|---|---|---|---|---|---|---|
| Groq | 1 | 8 | 0% | 418 / 546 ms | 2.04 / 2.13 ms | 1 | 293 tok/s |
| Groq | 5 | 10 | 10% | 470 / 674 ms | 2.09 / 2.38 ms | 5 | 129 tok/s |
| Groq | 10 | 20 | 25% | 380 / 935 ms | 2.05 / 3.15 ms | 33 | 153 tok/s |
| OpenRouter | 1 | 8 | 0% | 2,763 / 8,816 ms | 10.0 / 13.3 ms | 0 | 84 tok/s |
| OpenRouter | 5 | 10 | 10% | 2,849 / 7,797 ms | 12.4 / 18.0 ms | 0 | 277 tok/s |
| OpenRouter | 10 | 20 | 10% | 1,602 / 13,660 ms | 11.0 / 18.7 ms | 0 | 488 tok/s |

**1. Groq decodes about 5× faster per token, and that holds under load.**
Groq's TPOT stays at 2.0–2.1 ms p50 at every concurrency level. OpenRouter's
Nemotron is 10–12 ms. v1 reported 26.5–32 ms for OpenRouter "rising with
load"; that was the per-chunk bug, and with per-token TPOT the rise mostly
disappears.

**2. Groq is limited by its rate limit, not its hardware.** Throughput
*falls* as concurrency rises (293 → 129–153 tok/s), retries climb from 1 to
33, and at concurrency 10 a quarter of requests fail after exhausting their
retries. Each request is still fast; the run spends its time waiting out
the free tier's tokens-per-minute cap. v1 reported this as TTFT spikes of up
to ~29 s, but those seconds were retry waits the SDK hid inside the timer.
With retries separated, Groq's TTFT p95 stays under 1 s.

**3. OpenRouter batches: throughput scales almost linearly with load.**
84 → 277 → 488 tok/s from concurrency 1 to 10, while per-request TPOT barely
moves. The server runs concurrent requests in the same decode steps
(continuous batching), so serving more users costs each user very little.

**4. On OpenRouter, reasoning dominates time to first answer.** The first
token of any kind arrives in about 0.3–0.6 s, but the first *answer* token
takes 1.6–2.8 s at p50 and up to 13.7 s at p95. The gap is the model
thinking. A TTFT that only watched `content` would blame this on the server.

**5. Reasoning can use up the whole output budget.** Every "failed" request
apart from Groq's rate-limit errors stopped at exactly 2,048 tokens, all
reasoning, no answer. That's the request budget, not the provider: the
budget is now 4,096 and these are labelled `truncated` (see `CHANGELOG.md`).
The failure rates above still include them.

## Limitations

- Small samples: 8–20 requests per cell and one run per level. Trust the
  direction of a result more than the exact numbers. OpenRouter's lower TTFT
  p50 at concurrency 10 is probably noise.
- Free tiers: results reflect each provider's free-tier limits and shared
  capacity, not what a paid account would get. OpenRouter's free models also
  have a daily request cap, which a full rerun can exceed.
- Different model families (~120B params each), not identical models
- Self-hosted/quantized comparison out of scope (avoided GPU cost)

## Structure

- `prompts.py` — frozen test set
- `benchmark.py` — harness: streaming, instrumentation, retry logic, SQLite logging
- `compare_providers.py` — p50/p95, fail rate and throughput by provider/concurrency
- `build_dashboard.py` — generates `dashboard.html`
- `visualizer.html` — standalone KV cache explainer (not benchmark output)
- `CHANGELOG.md` — what was fixed in the harness, and why

## Run

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# .env: GROQ_API_KEY=..., OPENROUTER_API_KEY=...

rm -f results.db        # v1 databases have an incompatible schema
python benchmark.py      # concurrency 1, 5, 10
python compare_providers.py
python build_dashboard.py
```
