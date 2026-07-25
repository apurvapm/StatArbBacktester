import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from data_loader import load_pair_data, split_in_out_sample


def compute_hedge_ratio(in_sample_df):
    X = sm.add_constant(in_sample_df["B_close"])
    model = sm.OLS(in_sample_df["A_close"], X).fit()
    alpha = model.params["const"]
    beta = model.params["B_close"]
    return alpha, beta

def compute_spread(df, beta):
    spread = df["A_close"] - beta*df["B_close"]
    spread.name = "spread"
    return spread

def adf_test(spread):
    result = adfuller(spread, regression="c")
    adf_stat = result[0]
    p_value = result[1]
    return adf_stat, p_value

def compute_half_life(spread):
    spread_lag = spread.shift(1).iloc[1:]
    delta = (spread - spread.shift(1)).iloc[1:]

    X= sm.add_constant(spread_lag)
    model = sm.OLS(delta, X).fit()
    theta = model.params["spread"]

    half_life = -np.log(2)/theta
    return half_life, theta

if __name__ =="__main__":
    data = load_pair_data()
    in_sample, _ = split_in_out_sample(data)

    alpha, beta = compute_hedge_ratio(in_sample)
    print(f"Hedge ratio (beta): {beta:.4f}")
    print(f"Intercept (alpha):  {alpha:.4f}")

    spread = compute_spread(in_sample, beta)

    adf_stat, p_value = adf_test(spread)
    print(f"ADF statistic: {adf_stat:.4f}")
    print(f"ADF p-value:   {p_value:.4f}")

    half_life, theta = compute_half_life(spread)
    print(f"Theta (mean reversion coefficient): {theta:.4f}")
    print(f"Half-life: {half_life:.2f} days")
    

