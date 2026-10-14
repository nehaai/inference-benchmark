"""
Build an interactive HTML dashboard from results.db.
Run this after your benchmark runs are complete -- it reads all
existing data and produces a single self-contained dashboard.html
you can open in any browser or embed in a writeup.

Requires: pip install plotly pandas
"""
import sqlite3
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

conn = sqlite3.connect("results.db")
df = pd.read_sql_query(
    "SELECT * FROM results WHERE status IN ('ok', 'truncated') AND ttft_ms IS NOT NULL", conn
)
conn.close()

providers = sorted(df["provider"].unique())
colors = {"groq": "#F55036", "openrouter": "#6467F2", "cerebras": "#00A87A"}

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "TPOT p50 by concurrency (decode speed -- lower is better)",
        "TTFT by concurrency (p50, with p5-p95 range)",
        "TTFT distribution per request (all runs)",
        "Reasoning vs total output tokens (per request)",
    ),
    specs=[[{}, {}], [{}, {}]],
)

# --- Panel 1: TPOT by concurrency, line per provider ---
for p in providers:
    sub = df[df["provider"] == p].groupby("concurrency")["tpot_ms"].median().reset_index()
    fig.add_trace(
        go.Scatter(x=sub["concurrency"], y=sub["tpot_ms"], mode="lines+markers",
                   name=f"{p} TPOT", line=dict(color=colors.get(p, "#888")), legendgroup=p),
        row=1, col=1,
    )

# --- Panel 2: TTFT avg with min/max band, per provider ---
for p in providers:
    sub = df[df["provider"] == p].groupby("concurrency")["ttft_ms"].agg(
        mean="median", min=lambda x: x.quantile(0.05), max=lambda x: x.quantile(0.95)
    ).reset_index()
    fig.add_trace(
        go.Scatter(x=sub["concurrency"], y=sub["mean"], mode="lines+markers",
                   name=f"{p} p50 TTFT", line=dict(color=colors.get(p, "#888")),
                   legendgroup=p, showlegend=False),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=list(sub["concurrency"]) + list(sub["concurrency"])[::-1],
            y=list(sub["max"]) + list(sub["min"])[::-1],
            fill="toself", fillcolor=colors.get(p, "#888"), opacity=0.15,
            line=dict(width=0), name=f"{p} p5-p95 range", legendgroup=p, showlegend=False,
        ),
        row=1, col=2,
    )

# --- Panel 3: raw TTFT scatter, jittered by provider, colored ---
for p in providers:
    sub = df[df["provider"] == p]
    fig.add_trace(
        go.Box(y=sub["ttft_ms"], name=p, marker_color=colors.get(p, "#888"),
               boxpoints="all", jitter=0.4, legendgroup=p, showlegend=False),
        row=2, col=1,
    )

# --- Panel 4: reasoning chars vs content tokens, scatter ---
for p in providers:
    sub = df[df["provider"] == p]
    fig.add_trace(
        go.Scatter(x=sub["completion_tokens"], y=sub["reasoning_tokens"], mode="markers",
                   name=f"{p} reasoning", marker=dict(color=colors.get(p, "#888"), size=9),
                   legendgroup=p, showlegend=False),
        row=2, col=2,
    )

fig.update_xaxes(title_text="Concurrency", row=1, col=1)
fig.update_yaxes(title_text="TPOT (ms)", row=1, col=1)
fig.update_xaxes(title_text="Concurrency", row=1, col=2)
fig.update_yaxes(title_text="TTFT (ms)", row=1, col=2)
fig.update_yaxes(title_text="TTFT (ms)", row=2, col=1)
fig.update_xaxes(title_text="Output tokens (incl. reasoning)", row=2, col=2)
fig.update_yaxes(title_text="Reasoning tokens", row=2, col=2)

fig.update_layout(
    title_text="Inference Provider Benchmark: Groq vs OpenRouter (gpt-oss-120b vs Nemotron-3-Super-120B)",
    title_y=0.98,
    height=850,
    margin=dict(t=100, b=100),
    template="plotly_white",
    legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="center", x=0.5),
)

fig.write_html("dashboard.html", include_plotlyjs="cdn")
print("Dashboard written to dashboard.html -- open it in a browser.")