import os
import re
import sys
import time
import uuid
import sqlite3
import threading
from itertools import cycle, islice
from datetime import datetime, UTC
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError, APIConnectionError, APITimeoutError

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

MAX_COMPLETION_TOKENS = 2048
WARMUP_PROMPT = "Reply with the single word: ok"


# ---------------------------------------------------------------------------
# One request
# ---------------------------------------------------------------------------

def run_single_request(client, model: str, cost_per_1m_output: float,
                       prompt_text: str, max_retries: int = 4,
                       max_tokens: int = MAX_COMPLETION_TOKENS) -> dict:
    """
    Run one request with our own retry loop.

    The OpenAI client is created with max_retries=0, so the SDK never
    retries silently. Each attempt is timed on its own, so TTFT/TPOT only
    ever describe the attempt that succeeded. Time spent waiting between
    attempts is reported separately as retry_wait_ms.

    Never raises: a request that still fails after all retries comes back
    as a row with status="error", so failures show up in the stats.
    """
    retries = 0
    retry_wait_s = 0.0
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            metrics = _run_single_request_inner(
                client, model, cost_per_1m_output, prompt_text, max_tokens
            )
            metrics["retries"] = retries
            metrics["retry_wait_ms"] = retry_wait_s * 1000
            return metrics
        except RateLimitError as e:
            last_error = e
            wait_s = _parse_retry_after(str(e)) or (2 ** attempt)
            kind = "RATE LIMIT"
        except (APIConnectionError, APITimeoutError) as e:
            last_error = e
            wait_s = 2 ** attempt
            kind = "CONNECTION"
        except APIError as e:
            # 4xx other than 429 won't get better by retrying
            status_code = getattr(e, "status_code", None)
            if status_code is not None and 400 <= status_code < 500:
                return _error_row(e, retries, retry_wait_s)
            last_error = e
            wait_s = 2 ** attempt
            kind = "TRANSIENT ERROR"

        if attempt == max_retries:
            break
        print(f"  [{kind}] {str(last_error)[:80]} -- waiting {wait_s:.1f}s "
              f"(retry {attempt + 1}/{max_retries})")
        time.sleep(wait_s)
        retries += 1
        retry_wait_s += wait_s

    return _error_row(last_error, retries, retry_wait_s)


def _parse_retry_after(error_message: str) -> float | None:
    match = re.search(r"try again in ([\d.]+)s", error_message)
    return float(match.group(1)) + 0.5 if match else None  # small buffer


def _error_row(error, retries: int, retry_wait_s: float) -> dict:
    return {
        "status": "error",
        "error": f"{type(error).__name__}: {str(error)[:200]}",
        "finish_reason": None,
        "ttft_ms": None,
        "queue_to_first_token_ms": None,
        "reasoning_ms": None,
        "tpot_ms": None,
        "total_ms": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "reasoning_tokens": None,
        "reasoning_chars": None,
        "cost_usd": None,
        "retries": retries,
        "retry_wait_ms": retry_wait_s * 1000,
        "response_preview": "",
    }


def _run_single_request_inner(client, model: str, cost_per_1m_output: float,
                              prompt_text: str, max_tokens: int) -> dict:
    """
    Send one streaming request and measure:
      - queue_to_first_token_ms: start -> first token of any kind
        (reasoning or content). This is queueing + prefill.
      - ttft_ms: start -> first *content* token (what a user sees)
      - reasoning_ms: the gap between the two
      - tpot_ms: decode time per token, from real token counts
      - cost, from the provider's billed token usage
    """
    start_time = time.perf_counter()
    first_any_token_time = None
    first_content_time = None
    last_token_time = None
    full_response = ""
    reasoning_chars = 0
    finish_reason = None
    usage = None

    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt_text}],
        max_completion_tokens=max_tokens,
        stream=True,
        # ask for token counts in the final chunk
        stream_options={"include_usage": True},
    )

    for chunk in stream:
        now = time.perf_counter()

        # The usage chunk has an empty `choices` list, so read it
        # BEFORE skipping choice-less chunks.
        if getattr(chunk, "usage", None):
            usage = chunk.usage
        if not chunk.choices:
            continue

        choice = chunk.choices[0]
        if choice.finish_reason:
            finish_reason = choice.finish_reason
        delta = choice.delta
        content_delta = getattr(delta, "content", None)
        # different providers/SDK versions name this field differently
        reasoning_delta = getattr(delta, "reasoning", None) or getattr(
            delta, "reasoning_content", None
        )

        if reasoning_delta or content_delta:
            if first_any_token_time is None:
                first_any_token_time = now
            last_token_time = now
        if reasoning_delta:
            reasoning_chars += len(reasoning_delta)
        if content_delta:
            if first_content_time is None:
                first_content_time = now
            full_response += content_delta

    end_time = time.perf_counter()

    def ms_since_start(t):
        return (t - start_time) * 1000 if t is not None else None

    ttft_ms = ms_since_start(first_content_time)
    queue_to_first_token_ms = ms_since_start(first_any_token_time)
    reasoning_ms = (
        ttft_ms - queue_to_first_token_ms
        if ttft_ms is not None and queue_to_first_token_ms is not None
        else None
    )

    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
    completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
    details = getattr(usage, "completion_tokens_details", None) if usage else None
    reasoning_tokens = getattr(details, "reasoning_tokens", None) if details else None

    # TPOT per TOKEN, not per chunk. completion_tokens includes reasoning
    # tokens, which are decoded before content, so measure from the first
    # token of any kind to the last one.
    if completion_tokens and completion_tokens > 1 and first_any_token_time is not None:
        decode_s = last_token_time - first_any_token_time
        tpot_ms = decode_s / (completion_tokens - 1) * 1000
    else:
        tpot_ms = None

    cost_usd = (
        completion_tokens / 1_000_000 * cost_per_1m_output
        if completion_tokens is not None else None
    )

    if first_content_time is None:
        status = "no_content"
    elif finish_reason == "length":
        status = "truncated"
    else:
        status = "ok"

    return {
        "status": status,
        "error": None if usage else "no usage in stream",
        "finish_reason": finish_reason,
        "ttft_ms": ttft_ms,
        "queue_to_first_token_ms": queue_to_first_token_ms,
        "reasoning_ms": reasoning_ms,
        "tpot_ms": tpot_ms,
        "total_ms": (end_time - start_time) * 1000,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "reasoning_chars": reasoning_chars,
        "cost_usd": cost_usd,
        "response_preview": full_response[:80],
    }


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

RESULT_COLUMNS = [
    "run_id", "provider", "model", "prompt_id", "category", "concurrency",
    "status", "error", "finish_reason",
    "ttft_ms", "queue_to_first_token_ms", "reasoning_ms", "tpot_ms", "total_ms",
    "prompt_tokens", "completion_tokens", "reasoning_tokens", "reasoning_chars",
    "cost_usd", "retries", "retry_wait_ms", "timestamp",
]


def init_db(db_path="results.db"):
    conn = sqlite3.connect(db_path)
    existing = {row[1] for row in conn.execute("PRAGMA table_info(results)")}
    if existing and "status" not in existing:
        sys.exit(
            f"{db_path} was written by the old (v1) harness and has a different "
            f"schema. Delete or rename it, then rerun."
        )
    conn.execute("""
        CREATE TABLE IF NOT EXISTS results (
            run_id TEXT, provider TEXT, model TEXT, prompt_id TEXT,
            category TEXT, concurrency INTEGER,
            status TEXT, error TEXT, finish_reason TEXT,
            ttft_ms REAL, queue_to_first_token_ms REAL, reasoning_ms REAL,
            tpot_ms REAL, total_ms REAL,
            prompt_tokens INTEGER, completion_tokens INTEGER,
            reasoning_tokens INTEGER, reasoning_chars INTEGER,
            cost_usd REAL, retries INTEGER, retry_wait_ms REAL,
            timestamp TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT, provider TEXT, model TEXT, concurrency INTEGER,
            n_requests INTEGER, peak_in_flight INTEGER, wall_s REAL,
            output_tokens INTEGER, throughput_tok_s REAL, timestamp TEXT
        )
    """)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# A full run
# ---------------------------------------------------------------------------

def warmup(client, model: str, concurrency: int):
    """
    Open `concurrency` connections before measuring, so the first measured
    requests don't pay for TCP/TLS setup. Results are discarded.
    """
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        list(ex.map(
            lambda _: run_single_request(client, model, 0.0, WARMUP_PROMPT,
                                         max_retries=1, max_tokens=16),
            range(concurrency),
        ))


def run_all_prompts(concurrency=1, providers=None, rounds=2, db_path="results.db"):
    """
    Run the prompt set against each provider with exactly `concurrency`
    requests in flight.

    The prompt list is cycled so there are at least `rounds * concurrency`
    requests (and at least one pass over every prompt). With only 8 prompts,
    concurrency 10 used to top out at 8 in flight.
    """
    if providers is None:
        providers = list(PROVIDERS.keys())

    conn = init_db(db_path)
    n_requests = max(len(PROMPTS), rounds * concurrency)
    requests = list(islice(cycle(PROMPTS), n_requests))

    def fmt(val, suffix="ms", decimals=0):
        return f"{val:.{decimals}f}{suffix}" if val is not None else "N/A"

    for provider_name in providers:
        cfg = PROVIDERS[provider_name]
        if not cfg["api_key"]:
            print(f"[SKIP] {provider_name}: no API key found in .env")
            continue

        # max_retries=0: the SDK's own retries would hide 429 waits inside
        # our timers. We retry ourselves and record the wait separately.
        client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"], max_retries=0)
        model = cfg["model"]
        run_id = uuid.uuid4().hex[:8]

        print(f"\n=== provider={provider_name}  model={model}  "
              f"concurrency={concurrency}  requests={n_requests}  run={run_id} ===")
        warmup(client, model, concurrency)

        in_flight = 0
        peak_in_flight = 0
        lock = threading.Lock()

        def tracked(prompt_text):
            nonlocal in_flight, peak_in_flight
            with lock:
                in_flight += 1
                peak_in_flight = max(peak_in_flight, in_flight)
            try:
                return run_single_request(client, model, cfg["cost_per_1m_output"], prompt_text)
            finally:
                with lock:
                    in_flight -= 1

        output_tokens = 0
        run_start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_prompt = {executor.submit(tracked, p["prompt"]): p for p in requests}

            for future in as_completed(future_to_prompt):
                p = future_to_prompt[future]
                m = future.result()  # never raises: errors come back as rows
                output_tokens += m["completion_tokens"] or 0

                row = {
                    "run_id": run_id, "provider": provider_name, "model": model,
                    "prompt_id": p["id"], "category": p["category"],
                    "concurrency": concurrency,
                    "timestamp": datetime.now(UTC).isoformat(),
                    **{k: m[k] for k in RESULT_COLUMNS if k in m},
                }
                conn.execute(
                    f"INSERT INTO results ({', '.join(RESULT_COLUMNS)}) "
                    f"VALUES ({', '.join('?' * len(RESULT_COLUMNS))})",
                    [row.get(c) for c in RESULT_COLUMNS],
                )
                conn.commit()

                print(
                    f"  {p['id']:15s} {m['status']:10s} "
                    f"Queue: {fmt(m['queue_to_first_token_ms'])}  "
                    f"TTFT: {fmt(m['ttft_ms'])}  "
                    f"TPOT: {fmt(m['tpot_ms'], decimals=1)}  "
                    f"Tokens: {m['completion_tokens'] if m['completion_tokens'] is not None else 'N/A'}  "
                    f"Retries: {m['retries']}"
                )

        wall_s = time.perf_counter() - run_start
        throughput = output_tokens / wall_s if wall_s > 0 else 0.0
        conn.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, provider_name, model, concurrency, n_requests, peak_in_flight,
             wall_s, output_tokens, throughput, datetime.now(UTC).isoformat()),
        )
        conn.commit()
        print(f"  -> peak in flight {peak_in_flight}/{concurrency}, "
              f"{output_tokens} tokens in {wall_s:.1f}s = {throughput:.1f} tok/s")
        if peak_in_flight < concurrency:
            print(f"  [WARN] pool never filled: peak {peak_in_flight} < {concurrency}")

    conn.close()
    print(f"\nDone. Results saved to {db_path}")


if __name__ == "__main__":
    for c in [1, 5, 10]:
        run_all_prompts(concurrency=c)
