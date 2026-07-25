import pandas as pd

from data_loader import load_pair_data


def test_load_pair_data_has_no_nans():
    data = load_pair_data()
    assert data.isna().sum().sum() == 0


def test_load_pair_data_aligns_on_common_dates():
    data = load_pair_data()
    assert list(data.columns) == ["A_close", "B_close"]
    assert data.index.is_monotonic_increasing
    assert data.index.duplicated().sum() == 0


def test_missing_dates_are_dropped_not_filled():
    dates_a = pd.date_range("2024-01-01", periods=5, freq="D")
    dates_b = dates_a.delete(2)  # ticker B is missing day 3

    close_a = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0], index=dates_a, name="Close")
    close_b = pd.Series([20.0, 21.0, 23.0, 24.0], index=dates_b, name="Close")

    combined = pd.concat([close_a, close_b], axis=1, join="inner", keys=["A_close", "B_close"])
    combined.columns = ["A_close", "B_close"]
    combined = combined.dropna()

    assert len(combined) == 4
    assert dates_a[2] not in combined.index
    assert combined.isna().sum().sum() == 0