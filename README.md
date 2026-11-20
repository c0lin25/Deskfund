# deskfund

A desktop quant platform that runs on a laptop for $0–30/month in data. It runs one strategy end to end, from data through signals, portfolio, backtest, execution and attribution, and every layer is swappable.

```
Data (Tiingo / FMP / Yahoo + FRED)
  → Store (Parquet + DuckDB)
  → Signals + IC analysis
  → Portfolio construction (active tilt / inverse vol / Black-Litterman)
  → Backtest (daily drifting weights, costs, 1-day execution lag)
  → Execution (Alpaca, paper + dry-run by default)
  → Dashboard + Brinson-Fachler attribution (Carino-linked) ↻ back to research
```

**Strategy 1: regional rotation.** It rotates US / EAFE / EM (SPY / EFA / EEM) against static ACWI-like policy weights. The signals are 12-1 momentum, 3-month momentum and trend vs. the 200-day moving average, z-scored across the three sleeves and blended. Active weights are capped at ±15%. It rebalances monthly.

---

## Setup (Mac, Anaconda)

```bash
git clone https://github.com/<you>/deskfund.git && cd deskfund
conda env create -f environment.yml      # creates the 'deskfund' env + installs this package
conda activate deskfund
cp .env.example .env                     # add keys (see below)
pytest -q                                # 12 tests, should all pass
python scripts/demo.py                   # offline check on synthetic data
streamlit run app/dashboard.py           # opens the dashboard in your browser
```

### Keys (`.env`)

| Key | Needed for | Cost |
|---|---|---|
| `TIINGO_API_KEY` | default price provider | free (500 symbols/mo); $30/mo Power |
| `FMP_API_KEY` | `price_provider: fmp` | ~$19–22/mo |
| `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` | paper / live execution | free (paper) |

FRED macro series need no key. To use Yahoo with no key at all, run `pip install yfinance` and set `price_provider: yahoo`. That's fine for prototyping, not for anything you'll trade.

## Daily use

```bash
python scripts/ingest.py        # incremental pull into data/prices.parquet (+ macro)
python scripts/backtest.py      # run strategy, print stats/IC/attribution, save to data/runs/<name>/
streamlit run app/dashboard.py  # view any saved run
python scripts/rebalance.py --force   # preview today's orders on Alpaca paper (dry run)
```

Ad-hoc SQL over the store:
```python
from deskfund.config import load_settings; from deskfund.data.store import Store
s = load_settings(); Store(s.store_dir).query("SELECT ticker, max(date) FROM prices GROUP BY 1")
```

### Automate (macOS launchd)
```bash
bash schedule/install_schedule.sh       # weekdays 9:45: ingest, then rebalance on the 1st trading day
bash schedule/install_schedule.sh --uninstall
```
`rebalance.py` respects `execution.paper` and `execution.dry_run` in `config/settings.yaml`. Trading a live account needs both `paper: false` **and** the `--live` flag.

> Compliance: as a registered-firm employee, check LPL's personal-trading / outside-activity rules before connecting a funded account.

## Experimenting

All strategy knobs live in `config/settings.yaml`: sleeves, benchmark weights, signal weights, construction method, caps, risk-off overlay and costs. To compare variants, copy the file, edit it, and run:
```bash
python scripts/backtest.py --settings config/variant.yaml --name variant_bl
```
Both runs then show up in the dashboard's run picker.

**To add a signal**, write `f(prices) -> DataFrame` in `deskfund/research/signals.py`, register it in `SIGNALS`, and give it a weight in settings. IC stats show up automatically.

**To add a data vendor**, subclass `PriceProvider` in `deskfund/data/providers/`, return the standard long format, and add it to `get_provider`.

## How the backtest avoids lying

- The signal is computed at the close of month-end day *d*. The trade fills at the close of *d+1*, and returns accrue from *d+2*.
- Weights drift with prices between rebalances, so turnover and costs are what a real book would incur.
- Costs (`costs_bps` per side) are charged against capital at the start of each holding period.
- Attribution reconciles exactly. Brinson-Fachler effects plus costs equal the active return, with a residual of about 1e-15 (tested).
- `tests/test_backtest.py::test_signals_have_no_lookahead` shocks future prices and asserts past signals don't change.

**What it doesn't do yet:** survivorship-free universes. It isn't needed for three ETFs, but you'll need them before any single-stock work.

## Layout

```
config/settings.yaml          all knobs
deskfund/data/                providers (tiingo, fmp, yahoo, synthetic), FRED, Parquet/DuckDB store, ingest
deskfund/research/            signals, IC / ICIR / decay
deskfund/portfolio/           active tilt, inverse vol, Black-Litterman (numpy), risk-off overlay
deskfund/backtest/            engine + metrics
deskfund/attribution/         Brinson-Fachler + Carino linking
deskfund/execution/           order planning + Alpaca broker
deskfund/pipeline.py          wires it all together
scripts/                      ingest, backtest, rebalance, demo
app/dashboard.py              Streamlit
schedule/                     launchd job
tests/                        pytest
```

## Roadmap (phase 2+)

| When you need… | Add | Why |
|---|---|---|
| Fast parameter sweeps | `vectorbt` | thousands of variants in seconds |
| Deeper factor tearsheets | `alphalens-reloaded` | quantile spreads and turnover per factor |
| Full HTML tearsheets | `quantstats` | `qs.reports.html(returns.port, returns.bench)` |
| Constrained optimizers | `riskfolio-lib` or `skfolio` | CVaR, risk budgeting, turnover constraints |
| Intraday / event-driven / multi-venue | `nautilus_trader` | the same strategy code for backtest and live |
| ML alpha research | `qlib` | research layer only, not execution |
| A Bloomberg-style front end | OpenBB (ODP + Workspace) | point it at `data/` |
| Single-stock factors | FMP fundamentals + SEC EDGAR XBRL (point-in-time by filing date) | stays inside the $25 budget |
