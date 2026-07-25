import pandas as pd
import numpy as np
from config import ROLLING_WINDOW, ENTRY_Z, EXIT_Z, STOP_LOSS_Z
from data_loader import load_pair_data
from pair_selection import compute_hedge_ratio, compute_spread, split_in_out_sample

def compute_z_score(spread: pd.Series)->pd.Series:
    rolling_mean = spread.rolling(window=ROLLING_WINDOW).mean().shift(1)
    rolling_std = spread.rolling(window = ROLLING_WINDOW).std().shift(1)
    zscore = (spread-rolling_mean) / rolling_std
    zscore.name = "zscore"
    return zscore

def generate_position(z: float, cur_position : str)->str:
    """current position /return value: one of 'flat' , 'long', 'short'
        long = long the spread (long A, short beta*B) -- bets the spread rises back up
        short = short the spread (short A, long beta*B) --  bets the spread falls back down
    """

    if cur_position == "short":
        if z>= STOP_LOSS_Z:
            return "flat"
        if z<= EXIT_Z:
            return "flat"
        return "short"
    
    if cur_position =="long":
        if z <= -STOP_LOSS_Z:
            return "flat"
        if z>= -EXIT_Z:
            return "flat"
        return "long"

    #flat 
    if z>=ENTRY_Z:
        return "short"
    if z<= -ENTRY_Z:
        return "long"
    return "flat"

def kalman_filter_beta(price_a : pd.Series, price_b: pd.Series, beta_init:float, P_init: float, Q:float, R:float, alpha_fixed: float)->pd.Series:
    """Recursively estimates beta_t using only data upto and including t"""
    n =len(price_a)
    betas = np.empty(n)
    beta = beta_init
    P = P_init

    for t in range(n):
        x_t = price_b.iloc[t]
        y_t = price_a.iloc[t]

        #predict (random walk state model)
        beta_pred = beta
        P_pred = P+Q
        #update(incorporate today's obsevation)
        y_pred = alpha_fixed + beta_pred * x_t
        innovation = y_t - y_pred
        S = (x_t**2 *P_pred) +R
        K =P_pred*x_t /S

        beta = beta_pred + K*innovation
        P= (1-(K*x_t))*P_pred

        betas[t] = beta
    return pd.Series(betas, index = price_a.index, name="beta_t")




    
if __name__=="__main__":
    import matplotlib.pyplot as plt

    data = load_pair_data()
    in_sample, _ = split_in_out_sample(data)

    alpha, static_beta = compute_hedge_ratio(in_sample)
    # spread = compute_spread(data, static_beta)
    # zscore = compute_z_score(spread)
    # combined = pd.concat([spread, zscore], axis =1)
    # print(combined.dropna().head(10))
    # print(combined.dropna().tail(10))
    # print(f"\nNan count in z-score: {zscore.isna().sum()}")

    residuals = in_sample["A_close"] - (alpha + static_beta*in_sample["B_close"])
    R = residuals.var()
    P_init = 1.0

    # Tuning Q 
    # q_multipliers = [1e-11, 1e-10, 1e-9, 1e-8,1e-7, 1e-6]

    # plt.figure(figsize = (12, 6))
    # for mult in q_multipliers:
    #     Q = R*mult
    #     beta_t = kalman_filter_beta(data["A_close"], data["B_close"], static_beta, P_init, Q, R, alpha)
    #     plt.plot(beta_t.index, beta_t.values, label = f"Q=R*{mult:.0e}(std={beta_t.std():.4f})")
    #     diffs = beta_t.diff().dropna()
    #     autocorr = diffs.autocorr(lag=1)
    #     print(f"Q=R*{mult:.0e}: std= {beta_t.std():.4f}, diff_std={diffs.std():.5f}, lag_autocorr={autocorr:.3f}")

    # Fixing Q = R*1e-9
    Q = R*1e-11
    beta_t = kalman_filter_beta(data["A_close"], data["B_close"], static_beta, P_init, Q, R, alpha)
    dynamic_spread = data["A_close"] - beta_t*data["B_close"]
    dynamic_spread.name = "spread"
    zscore = compute_z_score(dynamic_spread)

    combined = pd.concat([dynamic_spread, zscore], axis =1)
    print(combined.dropna().head(10))
    print(combined.dropna().tail(10))
    print(f"\n Nan count in zscore : {zscore.isna().sum()}")

    plt.figure(figsize= (12, 6))
    plt.plot(beta_t.index, beta_t.values, label = f"beta_t Kalman, Q=R*1e-11")

    plt.axhline(static_beta, color = "black", linestyle = "--", label =f"static_beta({static_beta})")
    plt.legend()
    plt.title("kalman filter beta_t for different Q values (R fixed from OLS residual variance)")
    plt.xlabel("Data")
    plt.ylabel("beta_t")
    plt.show()

    print(f"R (observation variance, from OLS residuals) :{R:.4f}")
    


