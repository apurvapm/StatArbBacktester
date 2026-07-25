import pytest
import pandas as pd

import config
from backtester import ExecutionEngine


class FakeStrategy:
    """Minimal stand-in for Strategy: scripted positions/z-scores, fixed beta."""
    def __init__(self, positions, zscores, beta):
        self.positions = positions
        self.zscores = zscores
        self.beta = beta
        self._i = 0

    def generate_signal(self, price_a_t, price_b_t):
        i = self._i
        self._i += 1
        return {
            "beta": self.beta,
            "spread": price_a_t - self.beta * price_b_t,
            "zscore": self.zscores[i],
            "position": self.positions[i],
        }


def _make_price_data():
    start = pd.Timestamp(config.OUT_OF_SAMPLE_START)
    dates = pd.date_range(start, periods=5, freq="D")
    return pd.DataFrame({"A_close": [100.0] * 5, "B_close": [50.0] * 5}, index=dates)


def test_stop_loss_trade_is_logged_correctly():
    extreme_z = config.STOP_LOSS_Z + 1.0
    positions = ["flat", "short", "short", "flat", "flat"]
    zscores = [0.0, 2.5, 2.5, extreme_z, 0.0]

    strategy = FakeStrategy(positions, zscores, beta=1.0)
    engine = ExecutionEngine(initial_capital=100_000)
    _, trade_log, cost_summary, _ = engine.run(_make_price_data(), strategy)

    assert len(trade_log) == 1
    trade = trade_log.iloc[0]
    assert trade["side"] == "short"
    assert trade["exit_reason"] == "stop_loss"

    # prices never move, so a frictionless trade would have exactly zero P&L --
    # any realized loss is entirely attributable to costs
    total_costs = sum(cost_summary.values())
    assert total_costs > 0
    assert trade["pnl"] == pytest.approx(-total_costs, rel=1e-9)

def test_normal_exit_is_logged_as_mean_reversion():
    positions = ["flat", "short", "short", "flat", "flat"]
    zscores = [0.0, 2.5, 2.5, 0.3, 0.0]  # well under STOP_LOSS_Z

    strategy = FakeStrategy(positions, zscores, beta=1.0)
    engine = ExecutionEngine(initial_capital=100_000)
    _, trade_log, _, _ = engine.run(_make_price_data(), strategy)

    assert len(trade_log) == 1
    assert trade_log.iloc[0]["exit_reason"] == "mean_reversion"

#Uses a minimal fake Strategy (scripted positions/z-scores) so the test isolates ExecutionEngine's behavior from the Kalman filter/z-score logic already tested above. Prices are held constant, which makes the cost check exact: with zero price movement, a trade's entire realized P&L must come from costs alone.