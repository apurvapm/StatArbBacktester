# StatArb Backtester — V/MA Pairs Trading

A from-scratch statistical arbitrage / pairs trading backtester, built in
Python without any backtesting frameworks (no Backtrader/Zipline/vectorbt).
Everything — data loading, cointegration testing, signal generation, the
event-driven backtest loop, transaction cost modeling, and performance
metrics — is implemented directly.

Currently trades **V (Visa) / MA (Mastercard)** — two payment-processing
duopoly stocks with closely related business models.

## Tech stack

- Python 3.11+
- pandas, numpy, statsmodels, scipy
- yfinance (data, with local CSV caching)
- matplotlib (plots)
- pytest (tests)

## Project structure

| File | Purpose |
|---|---|
| `config.py` | All tunable constants (tickers, date ranges, thresholds, costs) |
| `data_loader.py` | Downloads/caches OHLCV data, aligns and cleans both tickers, splits in-sample/out-of-sample |
| `pair_selection.py` | Engle-Granger cointegration test (OLS hedge ratio + ADF test), half-life estimation |
| `signal_generator.py` | Rolling z-score computation, entry/exit state machine, standalone (vectorized) Kalman filter reference implementations (1D beta-only and 2D beta+alpha) |
| `KalmanFilter.py` | `AssetDataLoader` and `Strategy` classes — bar-by-bar signal generation with either a static or Kalman-filtered (dynamic) hedge ratio |
| `costs.py` | Transaction cost, slippage, and daily borrow cost functions |
| `backtester.py` | `ExecutionEngine` — the event-driven, bar-by-bar backtest loop; applies costs, tracks cash/positions, logs every trade with entry/exit z-scores and exit reason |
| `metrics.py` | `PerformanceVisualizer` — Sharpe/Sortino/drawdown/win-rate/profit-factor, win-rate & P&L breakdown by exit reason, equity/drawdown/spread/beta plots |
| `main.py` | Runs the full pipeline end to end and produces all reports/plots |
| `test_*.py` | Unit tests (pytest) |

## Running it

```bash
pip install pandas numpy statsmodels scipy yfinance matplotlib pytest
python3 main.py     # runs the full pipeline: fit, backtest, plots, metrics
pytest -v      
      # runs the test suite
```
Methodology

Data split: hedge ratio and cointegration relationship are fit on
in-sample data (2015-2021) only; the backtest trades out-of-sample
(2022-2026) — the hedge ratio is never re-fit using data from the trading
period itself, to avoid look-ahead bias.
Signal: a rolling 20-day z-score of the spread (price_A - beta * price_B),
using only trailing data (.shift(1) convention -- today's own value never
contributes to its own normalization baseline). Entry at |z| > 2.0, exit
at |z| < 0.5, stop-loss at |z| > 3.5.

Hedge ratio: can be static (fit once, in-sample) or dynamic - a from-scratch Kalman filter that re-estimates the hedge ratio (and
optionally the intercept) one bar at a time, using only data up to and
including that bar. Toggle via Strategy(use_kalman=..., track_alpha=...).

Execution: signals are computed using data through bar t, but trades
execute at bar t+1's price - never the same bar the signal was generated
on. Transaction costs (5 bps/leg), slippage (3 bps), and daily borrow cost
(3% annualized on the short leg) are all applied.

Testing: the most important test (test_kalman_filter.py) asserts that
changing a future price never changes a past value of the estimated
hedge ratio â the concrete, checkable definition of "no look-ahead bias"
used throughout this project.

Current status

The backtester is complete and tested end to end. config.py's
STOP_LOSS_Z is currently 3.5 — the original planned value, deliberately
left untuned. A significant diagnostic investigation (see report.md)
found that the V/MA cointegration relationship shifts meaningfully between
the in-sample fitting period and the out-of-sample trading period (the OLS
intercept moves from +22 to -5), and that tuning parameters like
STOP_LOSS_Z by watching out-of-sample performance directly is itself a
form of look-ahead bias / data snooping. config.py reserves
VALIDATION_END / TEST_START for a proper three-way split (fit /
validate / test-once) to be used when actual strategy development —
hyperparameter tuning, model selection — begins as a separate, later phase.