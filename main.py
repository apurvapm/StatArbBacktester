import pandas as pd
import matplotlib.pyplot as plt

from config import TICKER_A, TICKER_B, OUT_OF_SAMPLE_START, ENTRY_Z, EXIT_Z, STOP_LOSS_Z
from KalmanFilter import AssetDataLoader, Strategy
from backtester import ExecutionEngine
from metrics import PerformanceVisualizer


def compute_buy_and_hold(price_data, initial_capital=100_000):
    out_of_sample = price_data[price_data.index >= pd.Timestamp(OUT_OF_SAMPLE_START)]
    price_a0 = out_of_sample["A_close"].iloc[0]
    price_b0 = out_of_sample["B_close"].iloc[0]

    shares_a = (initial_capital / 2) / price_a0
    shares_b = (initial_capital / 2) / price_b0

    benchmark_value = shares_a * out_of_sample["A_close"] + shares_b * out_of_sample["B_close"]
    benchmark_value.name = "buy_and_hold"
    return benchmark_value

def main():
    loader = AssetDataLoader()
    data = loader.load()
    in_sample = loader.get_in_sample()

    strategy = Strategy()
    strategy.fit(in_sample)

    engine = ExecutionEngine(initial_capital=100_000)
    equity_curve, trade_log, cost_summary, signal_history = engine.run(data, strategy)

    net_pnl = equity_curve.iloc[-1] - equity_curve.iloc[0]
    total_costs = sum(cost_summary.values())
    gross_pnl = net_pnl + total_costs

    print(f"=== {TICKER_A}/{TICKER_B} Pairs Trading Backtest ===")
    print(f"Start: {equity_curve.iloc[0]:.2f}, End: {equity_curve.iloc[-1]:.2f}")
    print(f"Net P&L: {net_pnl:.2f}")
    print(f"Costs -> transaction: {cost_summary['transaction']:.2f}, "
          f"slippage: {cost_summary['slippage']:.2f}, borrow: {cost_summary['borrow']:.2f}")
    print(f"Gross P&L (before costs): {gross_pnl:.2f}")
    print(f"Number of trades: {len(trade_log)}")
    print()

    visualizer = PerformanceVisualizer(equity_curve, trade_log, signal_history, strategy.static_beta)
    visualizer.summary()

    benchmark = compute_buy_and_hold(data, initial_capital=100_000)

    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve.index, equity_curve.values, label="Strategy")
    plt.plot(benchmark.index, benchmark.values, label=f"Buy & hold ({TICKER_A}/{TICKER_B} 50/50)")
    plt.title("Equity curve: strategy vs buy-and-hold benchmark")
    plt.xlabel("Date")
    plt.ylabel("Portfolio value ($)")
    plt.legend()
    plt.show()

    visualizer.plot_spread_and_signals()

    plt.figure(figsize=(12, 6))
    plt.plot(signal_history.index, signal_history["zscore"], label="z-score", color="steelblue")
    plt.axhline(ENTRY_Z, color="green", linestyle="--", label=f"entry (+/-{ENTRY_Z})")
    plt.axhline(-ENTRY_Z, color="green", linestyle="--")
    plt.axhline(EXIT_Z, color="orange", linestyle="--", label=f"exit (+/-{EXIT_Z})")
    plt.axhline(-EXIT_Z, color="orange", linestyle="--")
    plt.axhline(STOP_LOSS_Z, color="red", linestyle="--", label=f"stop-loss (+/-{STOP_LOSS_Z})")
    plt.axhline(-STOP_LOSS_Z, color="red", linestyle="--")
    plt.title("Rolling z-score with entry/exit/stop-loss thresholds")
    plt.xlabel("Date")
    plt.ylabel("z-score")
    plt.legend()
    plt.show()

    visualizer.plot_drawdown()
    visualizer.plot_rolling_beta()

if __name__ == "__main__":
    main()