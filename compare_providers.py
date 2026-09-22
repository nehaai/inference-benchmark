import sqlite3

conn = sqlite3.connect("results.db")
rows = conn.execute("""
    SELECT
        provider,
        concurrency,
        AVG(ttft_ms) AS avg_ttft,
        MIN(ttft_ms) AS min_ttft,
        MAX(ttft_ms) AS max_ttft,
        AVG(tpot_ms) AS avg_tpot,
        AVG(total_ms) AS avg_total,
        SUM(reasoning_chars) AS total_reasoning_chars,
        COUNT(*) AS n
    FROM results
    WHERE ttft_ms IS NOT NULL
    GROUP BY provider, concurrency
    ORDER BY provider, concurrency
""").fetchall()

print(f"{'provider':12s} {'conc':>5s} {'avg_ttft':>10s} {'min_ttft':>10s} {'max_ttft':>10s} {'avg_tpot':>10s} {'avg_total':>10s} {'reasoning':>10s} {'n':>3s}")
for r in rows:
    provider, concurrency, avg_ttft, min_ttft, max_ttft, avg_tpot, avg_total, reasoning_chars, n = r
    avg_tpot_str = f"{avg_tpot:9.2f}ms" if avg_tpot is not None else "N/A".rjust(11)
    print(f"{provider:12s} {concurrency:5d} {avg_ttft:9.0f}ms {min_ttft:9.0f}ms {max_ttft:9.0f}ms {avg_tpot_str} {avg_total:9.0f}ms {reasoning_chars:10d} {n:3d}")