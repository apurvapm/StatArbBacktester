import numpy as np
import pandas as pd

from signal_generator import kalman_filter_beta


def _make_prices(n=50, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    price_b = pd.Series(100 + rng.normal(0, 1, n).cumsum(), index=dates)
    price_a = 0.5 * price_b + 10 + rng.normal(0, 0.5, n)
    return pd.Series(price_a, index=dates), price_b


def test_beta_t_unaffected_by_future_price_changes():
    price_a, price_b = _make_prices()

    beta_original = kalman_filter_beta(
        price_a, price_b, beta_init=0.5, P_init=1.0, Q=1e-6, R=1.0, alpha_fixed=10.0
    )

    price_a_modified = price_a.copy()
    modify_from = 30
    price_a_modified.iloc[modify_from:] = price_a_modified.iloc[modify_from:] * 3 + 100

    beta_modified = kalman_filter_beta(
        price_a_modified, price_b, beta_init=0.5, P_init=1.0, Q=1e-6, R=1.0, alpha_fixed=10.0
    )

    pd.testing.assert_series_equal(
        beta_original.iloc[:modify_from], beta_modified.iloc[:modify_from]
    )

#The look-ahead test the plan calls out as most important: prove beta_t at an early time is unaffected by later price changes.