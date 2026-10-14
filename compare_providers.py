"""
Summarize results.db by provider and concurrency.

Reports p50/p95 instead of averages: chunk arrival is bursty and a few
multi-second stalls drag an average far from what a typical request sees.
Failed and truncated requests are counted, not hidden.
"""
import sqlite3
from collections import defaultdict


def pct(values, q):
    """Nearest-rank percentile; None if there's no data."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    idx = max(0, min(len(vals) - 1, round(q / 100 * len(vals) + 0.5) - 1))
    return vals[idx]


def f(v, decimals=0):
    return f"{v:.{decimals}f}" if v is not None else "N/A"


conn = sqlite3.connect("results.db")
rows = conn.execute(
    "SELECT provider, concurrency, status, ttft_ms, tpot_ms, retries FROM results"
).fetchall()
runs = conn.execute(
    "SELECT provider, concurrency, peak_in_flight, throughput_tok_s FROM runs"
).fetchall()
conn.close()

groups = defaultdict(list)
for provider, conc, status, ttft, tpot, retries in rows:
    groups[(provider, conc)].append((status, ttft, tpot, retries))

throughput = defaultdict(list)
peak = defaultdict(int)
for provider, conc, peak_in_flight, tps in runs:
    throughput[(provider, conc)].append(tps)
    peak[(provider, conc)] = max(peak[(provider, conc)], peak_in_flight)

header = (f"{'provider':12s} {'conc':>4s} {'peak':>4s} {'n':>4s} {'fail%':>6s} {'trunc':>5s} "
          f"{'ttft p50':>9s} {'ttft p95':>9s} {'tpot p50':>9s} {'tpot p95':>9s} "
          f"{'retries':>7s} {'tok/s':>7s}")
print(header)
print("-" * len(header))

for key in sorted(groups):
    provider, conc = key
    g = groups[key]
    ok = [r for r in g if r[0] in ("ok", "truncated")]
    n = len(g)
    fail_pct = 100 * sum(r[0] in ("error", "no_content") for r in g) / n
    truncated = sum(r[0] == "truncated" for r in g)
    tps = throughput.get(key)
    print(
        f"{provider:12s} {conc:4d} {peak.get(key, 0):4d} {n:4d} {fail_pct:5.1f}% {truncated:5d} "
        f"{f(pct([r[1] for r in ok], 50)):>9s} {f(pct([r[1] for r in ok], 95)):>9s} "
        f"{f(pct([r[2] for r in ok], 50), 2):>9s} {f(pct([r[2] for r in ok], 95), 2):>9s} "
        f"{sum(r[3] or 0 for r in g):7d} "
        f"{f(sum(tps) / len(tps), 1) if tps else 'N/A':>7s}"
    )

print("\nttft/tpot in ms, over successful requests. peak = max requests in flight.")
