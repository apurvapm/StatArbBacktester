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

def kalman_filter_2d(price_a: pd.Series, price_b: pd.Series,
                      beta_init: float, alpha_init: float,
                      P_init: np.ndarray, Q: np.ndarray, R: float):
    """2D Kalman filter tracking state [beta_t, alpha_t]."""
    n = len(price_a)
    betas = np.empty(n)
    alphas = np.empty(n)

    x = np.array([beta_init, alpha_init])
    P = P_init.copy()

    for t in range(n):
        x_t_price_b = price_b.iloc[t]
        y_t = price_a.iloc[t]

        x_pred = x
        P_pred = P + Q

        H = np.array([x_t_price_b, 1.0])
        y_pred = H @ x_pred
        innovation = y_t - y_pred
        S = H @ P_pred @ H + R
        K = P_pred @ H / S

        x = x_pred + K * innovation
        P = P_pred - np.outer(K, H) @ P_pred

        betas[t] = x[0]
        alphas[t] = x[1]

    return (pd.Series(betas, index=price_a.index, name="beta_t"),
            pd.Series(alphas, index=price_a.index, name="alpha_t"))

"""We now have something better than guesswork: we know from the out-of-sample ADF re-fit that alpha needs to plausibly move from +22.07 to around -5.06 over the ~1,150 out-of-sample trading days. For a random-walk state, cumulative drift over N steps scales as sqrt(N * Q_alpha). Solving for a ~$27 total drift over ~1,150 days:
Q_alpha ≈ 27² / 1150 ≈ 0.63
That's a starting point to validate, not a final answer. For Q_beta, reuse the earlier validated result (R * 1e-9) as a reasonable prior, acknowledging the model has changed slightly so it may need re-checking."""
    
if __name__=="__main__":

        # Initail tests#1
    #     import matplotlib.pyplot as plt

    #     data = load_pair_data()
    #     in_sample, _ = split_in_out_sample(data)

    #     alpha, static_beta = compute_hedge_ratio(in_sample)
    #     # spread = compute_spread(data, static_beta)
    #     # zscore = compute_z_score(spread)
    #     # combined = pd.concat([spread, zscore], axis =1)
    #     # print(combined.dropna().head(10))
    #     # print(combined.dropna().tail(10))
    #     # print(f"\nNan count in z-score: {zscore.isna().sum()}")

    # residuals = in_sample["A_close"] - (alpha + static_beta*in_sample["B_close"])
    # R = residuals.var()
    # P_init = 1.0

    #tests #2
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
    #tests#3
    # Q = R*1e-11
    # beta_t = kalman_filter_beta(data["A_close"], data["B_close"], static_beta, P_init, Q, R, alpha)
    # dynamic_spread = data["A_close"] - beta_t*data["B_close"]
    # dynamic_spread.name = "spread"
    # zscore = compute_z_score(dynamic_spread)

    # combined = pd.concat([dynamic_spread, zscore], axis =1)
    # print(combined.dropna().head(10))
    # print(combined.dropna().tail(10))
    # print(f"\n Nan count in zscore : {zscore.isna().sum()}")

    # plt.figure(figsize= (12, 6))
    # plt.plot(beta_t.index, beta_t.values, label = f"beta_t Kalman, Q=R*1e-11")

    # plt.axhline(static_beta, color = "black", linestyle = "--", label =f"static_beta({static_beta})")
    # plt.legend()
    # plt.title("kalman filter beta_t for different Q values (R fixed from OLS residual variance)")
    # plt.xlabel("Data")
    # plt.ylabel("beta_t")
    # plt.show()

    # print(f"R (observation variance, from OLS residuals) :{R:.4f}")
    
    #tests#4
    data = load_pair_data()
    in_sample, _ = split_in_out_sample(data)

    alpha, static_beta = compute_hedge_ratio(in_sample)
    residuals = in_sample["A_close"] - (alpha + static_beta * in_sample["B_close"])
    R = residuals.var()

    Q = np.array([[1e-4, 0.0],
                  [0.0, 0.05]])
    P_init = np.diag([1.0, 1.0])

    beta_t, alpha_t = kalman_filter_2d(data["A_close"], data["B_close"],
                                        static_beta, alpha, P_init, Q, R)

    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1.plot(beta_t.index, beta_t.values)
    ax1.axhline(static_beta, color="black", linestyle="--", label="in-sample static beta")
    ax1.set_ylabel("beta_t")
    ax1.legend()

    ax2.plot(alpha_t.index, alpha_t.values)
    ax2.axhline(22.07, color="black", linestyle="--", label="in-sample alpha (22.07)")
    ax2.axhline(-5.06, color="red", linestyle="--", label="out-of-sample alpha (-5.06)")
    ax2.set_ylabel("alpha_t")
    ax2.legend()
    plt.show()
