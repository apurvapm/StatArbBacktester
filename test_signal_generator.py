import numpy as np
import pandas as pd
import pytest

from signal_generator import compute_z_score


def test_zscore_matches_hand_calculation():
    spread = pd.Series(range(1, 22), index=pd.date_range("2024-01-01", periods=21), dtype=float)
    zscore = compute_z_score(spread)

    # first 20 entries: rolling window not yet full (plus the .shift(1) delay)
    assert zscore.iloc[:20].isna().all()

    # day 21 (value=21): mean/std computed over values 1..20 only (today excluded)
    # mean(1..20) = 10.5; sample variance of 1..20 = 35.0 exactly
    expected_z = (21 - 10.5) / np.sqrt(35.0)
    assert zscore.iloc[20] == pytest.approx(expected_z, rel=1e-9)


def test_zscore_does_not_use_todays_value():
    spread_normal = pd.Series(range(1, 22), index=pd.date_range("2024-01-01", periods=21), dtype=float)
    spread_outlier = spread_normal.copy()
    spread_outlier.iloc[20] = 10_000.0  # extreme change to TODAY's value only

    mean_used = spread_normal.iloc[:20].mean()
    std_used = spread_normal.iloc[:20].std()
    expected_outlier_z = (10_000.0 - mean_used) / std_used

    z_outlier = compute_z_score(spread_outlier).iloc[20]
    assert z_outlier == pytest.approx(expected_outlier_z, rel=1e-9)

#Hand-calculated toy example, plus an explicit leakage check (today's own value shouldn't affect today's normalization baseline).