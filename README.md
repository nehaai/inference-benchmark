# Inference Provider Benchmark

Benchmarks LLM inference performance (TTFT, TPOT, reasoning overhead) across
providers under varying concurrency — Groq (GPT-OSS 120B) vs OpenRouter
(Nemotron-3-Super-120B).

## Method

8 fixed prompts (short/long input × short/long output), streamed and run
at concurrency 1, 5, and 10. Measures:
- **TTFT** — time to first content token
- **TPOT** — decode time per output token, from billed token counts
- **Throughput** — output tokens per second across the whole run
- **Reasoning vs. content tokens** — thinking time isolated from answer generation
- Cost per request (billed token usage × published rates)
- Retries, retry wait, and failed or truncated requests, reported separately

Stats are p50/p95, not averages.

## Findings

> **Withdrawn until rerun.** The numbers below came from the v1 harness,
> which had measurement bugs (see `CHANGELOG.md`). The TPOT figures in
> particular are wrong: OpenRouter's were overstated by about 3×.

| Finding | Data |
|---|---|
| Groq's decode speed is stable and fast | TPOT flat at ~2.1ms across all concurrency levels vs. OpenRouter's 26.5–32.0ms, rising with load |
| Reasoning time can dominate latency, invisible by default | One request: 6,808 reasoning chars vs. 986 content tokens, 42s total |
| Reasoning models can truncate under load, not just slow down | Concurrency 10: a request hit its token budget mid-reasoning, 0 output tokens |
| Free-tier rate limits produce real, explainable latency spikes | Groq TTFT outliers (up to ~29s) correlate with observed 429s tied to its TPM cap |

## Limitations

- n=8 per concurrency level — OpenRouter's apparent TTFT *decrease* at higher
  concurrency is likely outlier noise, not a real trend
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
