import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PerformanceVisualizer:
    def __init__(self, equity_curve, trade_log, signal_history=None, static_beta=None):
        self.equity_curve = equity_curve
        self.trade_log = trade_log
        self.signal_history = signal_history
        self.static_beta = static_beta

    def _daily_returns(self):
        return self.equity_curve.pct_change().dropna()

    def sharpe_ratio(self):
        returns = self._daily_returns()
        return returns.mean() / returns.std()*np.sqrt(252)

    def sortino_ratio(self):
        returns = self._daily_returns()
        downside = returns[returns<0]
        return returns.mean() / downside.std()*np.sqrt(252)

    def max_drawdown(self):
        cummax = self.equity_curve.cummax()
        drawdown = self.equity_curve / cummax - 1
        return drawdown.min()

    def max_drawdown_duration(self):
        cummax = self.equity_curve.cummax()
        in_drawdown = self.equity_curve < cummax

        longest = 0
        current =0
        for flag in in_drawdown:
            if flag :
                current+= 1
                longest = max(longest, current)
            else: 
                current =0
        return longest

    def win_rate(self, trades=None):
        trades = self.trade_log if trades is None else trades
        if(len(trades)==0):
            return np.nan
        return (trades["pnl"] > 0).mean()

    def profit_factor(self, trades=None):
        trades = self.trade_log if trades is None else trades
        gains = trades.loc[trades["pnl"] > 0 , "pnl"].sum()
        losses = trades.loc[trades["pnl"] < 0, "pnl"].sum()
        if losses == 0:
            return np.nan
        return gains/abs(losses)

    def exit_reason_breakdown(self):
        rows =[]
        for reason, group in self.trade_log.groupby("exit_reason"):
            rows.append({
                "exit_reason" : reason,
                "count" : len(group), 
                "win_rate" : self.win_rate(group),
                "avg_pnl" : group["pnl"].mean(),
                "totl_pnl" : group["pnl"].sum()
            })
        return pd.DataFrame(rows)

    def summary(self): 
        print(f"Sharpe ratio:        {self.sharpe_ratio():.3f}")
        print(f"Sortino ratio:       {self.sortino_ratio():.3f}")
        print(f"Max drawdown:        {self.max_drawdown()*100:.2f}%")
        print(f"Max DD duration:     {self.max_drawdown_duration()} trading days")
        print(f"Win rate:            {self.win_rate()*100:.2f}%")
        print(f"Profit factor:       {self.profit_factor():.3f}")
        print()
        print("Breakdown by exit reason:")
        print(self.exit_reason_breakdown().to_string(index=False))

    def plot_equity_curve(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.equity_curve.index, self.equity_curve.values)
        plt.title("Equity curve")
        plt.xlabel("Date")
        plt.ylabel("Portfolio value ($)")
        plt.show()

    def plot_drawdown(self):
        cummax = self.equity_curve.cummax()
        drawdown = (self.equity_curve / cummax - 1) * 100

        plt.figure(figsize=(12, 4))
        plt.fill_between(drawdown.index, drawdown.values, 0, color="red", alpha=0.4)
        plt.title("Drawdown (%)")
        plt.xlabel("Date")
        plt.ylabel("Drawdown (%)")
        plt.show()

    def plot_rolling_beta(self):
        if self.signal_history is None:
            raise ValueError("signal history was not provided to PerformanceVisualizer")

        plt.figure(figsize = (12, 6))
        plt.plot(self.signal_history.index,self.signal_history["beta"], label ="beta_t")

        if self.static_beta is not None:
            plt.axhline(self.static_beta, color = "black", linestyle="--", label = "static_beta")

        plt.legend()
        plt.title("Hedge ratio over time")
        plt.xlabel("Date")
        plt.ylabel("beta_t")
        plt.show()

    def plot_spread_and_signals(self):
        if self.signal_history is None:
            raise ValueError("signal history was not provided to PerformanceVisualizer")

        plt.figure(figsize=(14, 6))
        plt.plot(self.signal_history.index, self.signal_history["spread"], label = "spread", color = "steelblue")

        for _ , trade in self.trade_log.iterrows():
            entry_color = "green" if trade["side"]=="long" else "red"
            exit_color = "black" if trade["exit_reason"]=="stop_loss" else "gray"
            plt.scatter(trade["entry_date"], self.signal_history.loc[trade["entry_date"], "spread"], color = entry_color, marker="^", zorder = 5)

            plt.scatter(trade["exit_date"], self.signal_history.loc[trade["exit_date"], "spread"],
                        color=exit_color, marker="v", zorder=5)

        plt.title("Spread with entry/exit markers (^ entry, v exit; green=long/red=short entry; black=stop_loss exit)")
        plt.xlabel("Date")
        plt.ylabel("Spread")
        plt.show()

            


