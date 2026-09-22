import os
import time
import sqlite3
from datetime import datetime, UTC
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError

from prompts import PROMPTS

load_dotenv()

# One shared comparison basis: best free-tier model per provider, matched
# roughly by parameter scale (~120B) since exact model parity isn't
# available for free across providers. Cerebras dropped -- paywalled
# for this account despite advertised no-card signup.
PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key": os.environ.get("GROQ_API_KEY"),
        "model": "openai/gpt-oss-120b",
        "cost_per_1m_output": 0.60,  # verified Groq rate
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.environ.get("OPENROUTER_API_KEY"),
        "model": "nvidia/nemotron-3-super-120b-a12b:free",
        "cost_per_1m_output": 0.0,  # free-tier model, no metered cost
    },
}


def run_single_request(client, model: str, cost_per_1m_output: float, prompt_text: str, max_retries: int = 4) -> dict:
    """
    Wraps _run_single_request_inner with retry logic for two distinct
    failure modes:
      - RateLimitError (429): Groq includes a "try again in X.Xs"
        message -- we parse and respect that instead of guessing.
      - Other transient APIError (500/502/503, "temporarily overloaded"):
        common on free-tier/community-hosted models -- exponential backoff.
    Non-transient errors (bad request, auth failure, etc.) are not
    retried and will raise immediately.
    """
    for attempt in range(max_retries):
        try:
            return _run_single_request_inner(client, model, cost_per_1m_output, prompt_text)
        except RateLimitError as e:
            wait_s = _parse_retry_after(str(e)) or (2 ** attempt)
            print(f"  [RATE LIMIT] waiting {wait_s:.1f}s before retry {attempt + 1}/{max_retries}")
            time.sleep(wait_s)
        except APIError as e:
            # transient server-side error (overloaded, temporary outage) --
            # back off and retry, since free-tier models are best-effort
            wait_s = 2 ** attempt
            print(f"  [TRANSIENT ERROR] {str(e)[:100]} -- waiting {wait_s}s before retry {attempt + 1}/{max_retries}")
            time.sleep(wait_s)
    # final attempt, let it raise if it still fails
    return _run_single_request_inner(client, model, cost_per_1m_output, prompt_text)


def _parse_retry_after(error_message: str) -> float | None:
    import re
    match = re.search(r"try again in ([\d.]+)s", error_message)
    return float(match.group(1)) + 0.5 if match else None  # small buffer


def _run_single_request_inner(client, model: str, cost_per_1m_output: float, prompt_text: str) -> dict:
    """
    Send one streaming request and measure:
      - TTFT: time from request start to first token chunk
      - TPOT: average time between subsequent chunks
      - total tokens and rough cost
    Returns a dict of metrics for this one request.
    """
    start_time = time.perf_counter()
    first_token_time = None
    first_any_token_time = None
    chunk_times = []
    full_response = ""

    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt_text}],
        max_completion_tokens=2048,
        stream=True,
    )

    reasoning_chars = 0
    finish_reason = None

    for chunk in stream:
        now = time.perf_counter()
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        if choice.finish_reason:
            finish_reason = choice.finish_reason
        delta_obj = choice.delta
        content_delta = getattr(delta_obj, "content", None)
        # different providers/SDK versions name this field differently
        reasoning_delta = getattr(delta_obj, "reasoning", None) or getattr(
            delta_obj, "reasoning_content", None
        )

        if reasoning_delta:
            # thinking tokens -- counted separately, don't count toward
            # "answer" TTFT/TPOT, but track so you know reasoning happened
            if first_any_token_time is None:
                first_any_token_time = now
            reasoning_chars += len(reasoning_delta)

        if content_delta:
            if first_any_token_time is None:
                first_any_token_time = now
            if first_token_time is None:
                first_token_time = now
            chunk_times.append(now)
            full_response += content_delta

    if first_token_time is None:
        # nothing landed in `content` at all -- surface why, instead of
        # silently returning None and crashing the caller's .format() call
        print(
            f"  [WARN] no content tokens received. "
            f"finish_reason={finish_reason}  reasoning_chars={reasoning_chars}"
        )

    end_time = time.perf_counter()

    ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else None
    queue_to_first_token_ms = (
        (first_any_token_time - start_time) * 1000 if first_any_token_time else None
    )
    # time spent reasoning specifically, isolated from queueing/prefill delay
    reasoning_ms = (
        ttft_ms - queue_to_first_token_ms
        if ttft_ms is not None and queue_to_first_token_ms is not None
        else None
    )

    # inter-token latency: average gap between consecutive chunk arrivals
    if len(chunk_times) > 1:
        gaps = [chunk_times[i] - chunk_times[i - 1] for i in range(1, len(chunk_times))]
        tpot_ms = (sum(gaps) / len(gaps)) * 1000
    else:
        tpot_ms = None

    total_ms = (end_time - start_time) * 1000

    # crude token count estimate (word count) -- swap for a real tokenizer later if needed
    approx_tokens = len(full_response.split())
    cost_usd = (approx_tokens / 1_000_000) * cost_per_1m_output

    return {
        "ttft_ms": ttft_ms,
        "queue_to_first_token_ms": queue_to_first_token_ms,
        "reasoning_ms": reasoning_ms,
        "tpot_ms": tpot_ms,
        "total_ms": total_ms,
        "approx_tokens": approx_tokens,
        "reasoning_chars": reasoning_chars,
        "cost_usd": cost_usd,
        "response_preview": full_response[:80],
    }


def init_db(db_path="results.db"):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS results (
            provider TEXT,
            model TEXT,
            prompt_id TEXT,
            category TEXT,
            concurrency INTEGER,
            ttft_ms REAL,
            queue_to_first_token_ms REAL,
            reasoning_ms REAL,
            tpot_ms REAL,
            total_ms REAL,
            approx_tokens INTEGER,
            reasoning_chars INTEGER,
            cost_usd REAL,
            timestamp TEXT
        )
    """)
    conn.commit()
    return conn


def run_all_prompts(concurrency=1, providers=None):
    """
    Run the full frozen prompt set against each provider in `providers`
    (defaults to all of PROVIDERS), with up to `concurrency` requests
    in flight at once per provider.
    """
    if providers is None:
        providers = list(PROVIDERS.keys())

    conn = init_db()

    def fmt(val, suffix="ms", decimals=0):
        return f"{val:.{decimals}f}{suffix}" if val is not None else "N/A"

    for provider_name in providers:
        cfg = PROVIDERS[provider_name]
        if not cfg["api_key"]:
            print(f"[SKIP] {provider_name}: no API key found in .env")
            continue

        client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
        model = cfg["model"]
        cost_per_1m_output = cfg["cost_per_1m_output"]

        print(f"\n=== provider={provider_name}  model={model}  concurrency={concurrency} ===")

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_prompt = {
                executor.submit(run_single_request, client, model, cost_per_1m_output, p["prompt"]): p
                for p in PROMPTS
            }

            for future in as_completed(future_to_prompt):
                p = future_to_prompt[future]
                metrics = future.result()

                conn.execute(
                    """INSERT INTO results
                       (provider, model, prompt_id, category, concurrency,
                        ttft_ms, queue_to_first_token_ms, reasoning_ms, tpot_ms, total_ms,
                        approx_tokens, reasoning_chars, cost_usd, timestamp)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        provider_name, model, p["id"], p["category"], concurrency,
                        metrics["ttft_ms"], metrics["queue_to_first_token_ms"],
                        metrics["reasoning_ms"], metrics["tpot_ms"], metrics["total_ms"],
                        metrics["approx_tokens"], metrics["reasoning_chars"],
                        metrics["cost_usd"], datetime.now(UTC).isoformat(),
                    ),
                )
                conn.commit()

                print(
                    f"  {p['id']:15s} "
                    f"Queue+Reason: {fmt(metrics['queue_to_first_token_ms'])}  "
                    f"TTFT(content): {fmt(metrics['ttft_ms'])}  "
                    f"TPOT: {fmt(metrics['tpot_ms'], decimals=1)}  "
                    f"Total: {fmt(metrics['total_ms'])}  "
                    f"Tokens: {metrics['approx_tokens']}"
                )

    conn.close()
    print("\nDone. Results saved to results.db")


if __name__ == "__main__":
    run_all_prompts(concurrency=1)