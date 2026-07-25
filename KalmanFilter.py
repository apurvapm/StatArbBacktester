from collections import deque

import numpy as np

from config import ROLLING_WINDOW
from data_loader import load_pair_data, split_in_out_sample
from pair_selection import compute_hedge_ratio, adf_test, compute_half_life, compute_spread
from signal_generator import kalman_filter_beta, generate_position, compute_z_score

class AssetDataLoader:
    def __init__(self):
        self.data = None

    def load(self):
        self._data = load_pair_data()
        return self._data

    def get_in_sample(self):
        in_sample, _ = split_in_out_sample(self._data)
        return in_sample
    def get_out_of_sample(self):
        _, out_of_sample = split_in_out_sample(self._data)
        return out_of_sample

class Strategy:
    def __init__(self, q_multiplier=1e-9, use_kalman=True, track_alpha=False,
                 q_beta=1e-4, q_alpha=0.05):
        self.q_multiplier = q_multiplier
        self.use_kalman = use_kalman
        self.track_alpha = track_alpha
        self.q_beta = q_beta
        self.q_alpha = q_alpha
        self.alpha = None
        self.static_beta = None
        self.R = None
        self.Q = None
        self.beta = None
        self.P = None
        self.state = None
        self.spread_buffer = deque(maxlen=ROLLING_WINDOW)
        self.position = "flat"

    def fit(self, in_sample_data):
        self.alpha, self.static_beta = compute_hedge_ratio(in_sample_data)

        static_spread = compute_spread(in_sample_data, self.static_beta)
        adf_stat, p_value = adf_test(static_spread)
        half_life, theta = compute_half_life(static_spread)
        print(f"[Strategy.fit] ADF p-value={p_value:.4f}, half-life  = {half_life:.2f} days")

        residuals = in_sample_data["A_close"] - (self.alpha +(self.static_beta*in_sample_data["B_close"]))

        self.R = residuals.var()
        self.beta = self.static_beta

        if self.track_alpha:
            self.Q = np.array([[self.q_beta, 0.0], [0.0, self.q_alpha]])
            self.state = np.array([self.static_beta, self.alpha])
            self.P = np.diag([1.0, 1.0])
        else:
            self.Q = self.R * self.q_multiplier
            self.P = 1.0

    def _step_kalman(self, price_a_t, price_b_t):
        if self.track_alpha:
            x_pred = self.state
            P_pred = self.P + self.Q

            H = np.array([price_b_t, 1.0])
            y_pred = H @ x_pred
            innovation = price_a_t - y_pred
            S = H @ P_pred @ H + self.R
            K = P_pred @ H / S

            self.state = x_pred + K * innovation
            self.P = P_pred - np.outer(K, H) @ P_pred

            self.beta = self.state[0]
            self.alpha = self.state[1]
        else:
            beta_pred = self.beta
            P_pred = self.P + self.Q

            y_pred = self.alpha + beta_pred*price_b_t

            innovation= price_a_t - y_pred
            S = price_b_t**2 * P_pred + self.R
            K = P_pred*price_b_t/S

            self.beta = beta_pred + K*innovation
            self.P = (1-K*price_b_t)*P_pred

    def generate_signal(self, price_a_t, price_b_t):
        if self.use_kalman:
            self._step_kalman(price_a_t, price_b_t)

        if self.track_alpha:
            spread_t = price_a_t - self.beta*price_b_t - self.alpha
        else:
            spread_t = price_a_t - self.beta*price_b_t

        if len(self.spread_buffer)==ROLLING_WINDOW:
            mean = np.mean(self.spread_buffer)
            std = np.std(self.spread_buffer, ddof=1)

            z_t = (spread_t - mean)/std

        else:
            z_t = np.nan

        self.spread_buffer.append(spread_t)

        if np.isnan(z_t):
            new_position = "flat"

        else :
            new_position = generate_position(z_t, self.position)

        self.position = new_position

        return{
            "beta" : self.beta,
            "alpha" : self.alpha,
            "spread": spread_t,
            "zscore" : z_t,
            "position": new_position
        }

import pandas as pd


## tesing whether the class functions perform same as dispered functions in signal_generator.py
if __name__ == "__main__":
    loader = AssetDataLoader()
    data = loader.load()
    in_sample_data = loader.get_in_sample()
    out_of_sample_data = loader.get_out_of_sample()

    strategy = Strategy()
    strategy.fit(in_sample_data)

    incremental_betas = []
    incremental_zscores = []

    for date, row in data.iterrows():
        res = strategy.generate_signal(row["A_close"], row["B_close"])
        incremental_betas.append(res["beta"])
        incremental_zscores.append(res["zscore"])

    incremental_beta = pd.Series(incremental_betas, index =data.index)
    incremental_zscore = pd.Series(incremental_zscores, index = data.index)

    # for i in range(30):
    #     print(incremental_betas[i], incremental_zscores[i])

    
    # cross check :same computed earlier in a different function
    reference_beta = kalman_filter_beta(data["A_close"],data["B_close"], strategy.static_beta, 1.0, strategy.Q, strategy.R, strategy.alpha)
    reference_spread = data["A_close"] - reference_beta*data["B_close"]
    reference_zscore = compute_z_score(reference_spread)

    beta_diff = (incremental_beta - reference_beta).abs().max()
    zscore_diff = (incremental_zscore - reference_zscore).abs().max()
    print(beta_diff, zscore_diff)
    print(incremental_zscore.tail(10))
    # print(in_sample_data.head(10))


        