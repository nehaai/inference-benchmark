# Inference Provider Benchmark

Benchmarks LLM inference performance (TTFT, TPOT, reasoning overhead) across
providers under varying concurrency — Groq (GPT-OSS 120B) vs OpenRouter
(Nemotron-3-Super-120B).

## Method

8 fixed prompts (short/long input × short/long output), streamed and run
at concurrency 1, 5, and 10. Measures:
- **TTFT** — time to first content token
- **TPOT** — avg time between output tokens (decode speed)
- **Reasoning vs. content tokens** — thinking time isolated from answer generation
- Cost per request (published per-token rates)

## Findings

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
- Cost figures use word-count as a token proxy, not a real tokenizer

## Structure

- `prompts.py` — frozen test set
- `benchmark.py` — harness: streaming, instrumentation, retry logic, SQLite logging
- `compare_providers.py` — aggregate comparison by provider/concurrency
- `build_dashboard.py` — generates `dashboard.html`

## Run

```bash
python -m venv venv && source venv/bin/activate
pip install openai python-dotenv plotly pandas
# .env: GROQ_API_KEY=..., OPENROUTER_API_KEY=...

python -c "
from benchmark import run_all_prompts
for c in [1, 5, 10]:
    run_all_prompts(concurrency=c)
"
python compare_providers.py
python build_dashboard.py
```