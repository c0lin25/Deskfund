"""deskfund dashboard.   streamlit run app/dashboard.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deskfund.backtest.metrics import drawdown_series  # noqa: E402

# Validated categorical slots (dataviz reference palette); benchmark is a neutral reference line.
PORT = "#2a78d6"
BENCH = "#8a8984"
SLEEVE_COLORS = {"US": "#2a78d6", "EAFE": "#eb6834", "EM": "#1baf7a", "CASH": "#8a8984"}
FALLBACK = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

st.set_page_config(page_title="deskfund", layout="wide")


def find_runs() -> dict[str, Path]:
    runs = {}
    for base in [ROOT / "data" / "runs", ROOT / "data" / "demo" / "runs"]:
        if base.exists():
            for d in sorted(base.iterdir()):
                if (d / "meta.json").exists():
                    tag = " (demo data)" if "demo" in base.parts else ""
                    runs[f"{d.name}{tag}"] = d
    return runs


def color_for(name: str, i: int) -> str:
    return SLEEVE_COLORS.get(name, FALLBACK[i % len(FALLBACK)])


def layout(fig: go.Figure, height: int = 340, pct: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(showgrid=False), yaxis=dict(gridcolor="rgba(128,128,128,0.15)", zeroline=False),
    )
    if pct:
        fig.update_yaxes(tickformat=".0%")
    return fig


runs = find_runs()
st.sidebar.title("deskfund")
if not runs:
    st.info("No runs yet. Run `python scripts/demo.py` (offline) or `python scripts/ingest.py && python scripts/backtest.py`.")
    st.stop()

choice = st.sidebar.selectbox("Run", list(runs), index=0)
rd = runs[choice]
meta = json.loads((rd / "meta.json").read_text())
ret = pd.read_parquet(rd / "returns.parquet")
targets = pd.read_parquet(rd / "targets.parquet")
stats = pd.read_parquet(rd / "stats.parquet")
ic = pd.read_parquet(rd / "ic.parquet")
decay = pd.read_parquet(rd / "ic_decay.parquet")
attr = pd.read_parquet(rd / "attribution_linked.parquet")

min_d, max_d = ret.index.min().date(), ret.index.max().date()
start, end = st.sidebar.slider("Window", min_value=min_d, max_value=max_d, value=(min_d, max_d))
r = ret.loc[str(start):str(end)]
st.sidebar.caption(f"Provider: {meta['provider']} · created {meta['created']}")
if "demo" in choice:
    st.sidebar.warning("Synthetic data — plumbing check only, not a result.")

# --- headline tiles --------------------------------------------------------
st.subheader(meta["name"])
a = stats.loc["active"]
c = st.columns(5)
c[0].metric("Portfolio CAGR", f"{stats.loc['port','CAGR']:.1%}", f"{a['CAGR']:+.1%} vs bench")
c[1].metric("Sharpe", f"{stats.loc['port','Sharpe']:.2f}", f"bench {stats.loc['bench','Sharpe']:.2f}", delta_color="off")
c[2].metric("Information ratio", f"{a['Sharpe']:.2f}", f"TE {a['Vol']:.1%}", delta_color="off")
c[3].metric("Max drawdown", f"{stats.loc['port','MaxDD']:.1%}", f"bench {stats.loc['bench','MaxDD']:.1%}", delta_color="off")
c[4].metric("Turnover / yr", f"{meta['stats_attrs']['annual_turnover']:.0%}",
            f"hit rate {meta['stats_attrs']['monthly_hit_rate']:.0%}", delta_color="off")

# --- performance -------------------------------------------------------------
left, right = st.columns(2)
with left:
    eq = (1 + r[["port", "bench"]]).cumprod()
    fig = go.Figure()
    fig.add_scatter(x=eq.index, y=eq["port"], name="Portfolio", line=dict(color=PORT, width=2))
    fig.add_scatter(x=eq.index, y=eq["bench"], name="Benchmark", line=dict(color=BENCH, width=2, dash="dot"))
    st.markdown("**Growth of $1**")
    st.plotly_chart(layout(fig, pct=False), width="stretch")
with right:
    act = (1 + r["port"]).cumprod() / (1 + r["bench"]).cumprod() - 1
    fig = go.Figure()
    fig.add_scatter(x=act.index, y=act, name="Cumulative active", line=dict(color=PORT, width=2), showlegend=False)
    fig.add_hline(y=0, line=dict(color=BENCH, width=1))
    st.markdown("**Cumulative active return (relative wealth)**")
    st.plotly_chart(layout(fig), width="stretch")

left, right = st.columns(2)
with left:
    fig = go.Figure()
    fig.add_scatter(x=r.index, y=drawdown_series(r["port"]), name="Portfolio", line=dict(color=PORT, width=2))
    fig.add_scatter(x=r.index, y=drawdown_series(r["bench"]), name="Benchmark", line=dict(color=BENCH, width=2, dash="dot"))
    st.markdown("**Drawdown**")
    st.plotly_chart(layout(fig), width="stretch")
with right:
    t = targets.loc[str(start):str(end)]
    t = t.loc[:, (t != 0).any()]
    fig = go.Figure()
    for i, col in enumerate(t.columns):
        fig.add_scatter(x=t.index, y=t[col], name=col, stackgroup="w", mode="lines",
                        line=dict(width=0.5, color=color_for(col, i)))
    st.markdown("**Target weights at each rebalance**")
    st.plotly_chart(layout(fig), width="stretch")

# --- current book ----------------------------------------------------------
st.markdown(f"**Latest target vs benchmark** (signal date {meta['signals_asof']})")
lt = pd.Series(meta["latest_target"])
bw = pd.Series(meta["benchmark_weights"]).reindex(lt.index).fillna(0.0)
book = pd.DataFrame({"Target": lt, "Benchmark": bw, "Active": lt - bw})
st.dataframe(book.style.format("{:.1%}"), width="stretch")

# --- research ---------------------------------------------------------------
left, right = st.columns(2)
with left:
    st.markdown("**Signal IC** (rank IC vs next-period return)")
    st.dataframe(ic[["n", "mean_ic", "icir_annual", "t_stat", "hit_rate"]].astype(float).style.format(
        {"n": "{:.0f}", "mean_ic": "{:.3f}", "icir_annual": "{:.2f}", "t_stat": "{:.2f}", "hit_rate": "{:.0%}"}),
        width="stretch")
    st.caption("3 sleeves means each month's IC takes only a few values — judge the series, not single months.")
with right:
    fig = go.Figure()
    fig.add_bar(x=[f"{h}m" for h in decay.index], y=decay["mean_ic"], marker_color=PORT, name="Mean IC",
                hovertemplate="%{x}: IC %{y:.3f}<extra></extra>")
    st.markdown("**Composite IC decay by horizon**")
    st.plotly_chart(layout(fig, pct=False, height=300), width="stretch")

# --- attribution -----------------------------------------------------------
st.markdown("**Brinson-Fachler attribution, Carino-linked over the full run**")
a2 = attr.copy()
fig = go.Figure()
fig.add_bar(x=a2.index, y=a2["total"], marker_color=PORT, hovertemplate="%{x}: %{y:.2%}<extra></extra>")
fig.add_hline(y=0, line=dict(color=BENCH, width=1))
left, right = st.columns([3, 2])
left.plotly_chart(layout(fig, height=300), width="stretch")
right.dataframe(a2.style.format("{:.2%}"), width="stretch")
chk = meta["attribution_check"]
st.caption(f"Active {chk['active']:.2%} = effects {chk['sum_of_effects']:.2%} (residual {chk['residual']:.1e}). "
           "Attribution covers the full run, not the window slider.")

with st.expander("Strategy config for this run"):
    st.json(meta["strategy"])
