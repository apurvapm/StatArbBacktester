import pandas as pd

from config import STOP_LOSS_Z, OUT_OF_SAMPLE_START
from costs import apply_slippage, transaction_cost, daily_borrow_cost

class ExecutionEngine: 
    def __init__(self, initial_capital =100_000):
        self.initial_capital = initial_capital
        # self.capital_per_trade = initial_capital
    def _trade_leg(self, current_shares, target_shares, price):
        delta = target_shares - current_shares
        if delta==0:
            return current_shares, 0.0, 0.0, 0.0
        buying = delta >0 
        exec_price = apply_slippage(price, buying)
        cash_delta = -delta*exec_price
        tc = transaction_cost(delta, exec_price)

    
        slip_cost = abs(delta) * abs(exec_price-price)
        return target_shares, cash_delta - tc, tc, slip_cost

    def run(self, price_data, strategy):
        dates = price_data.index
        out_of_sample_start = pd.Timestamp(OUT_OF_SAMPLE_START)

        cash = self.initial_capital
        shares_a = 0.0
        shares_b = 0.0

        pending_action = None
        open_trade = None

        equity_records = []
        trade_log = []
        signal_records= []
        total_transaction_cost =0.0
        total_slippage_cost = 0.0
        total_borrow_cost = 0.0

        for i in range(len(dates)):

            date = dates[i]
            price_a = price_data["A_close"].iloc[i]
            price_b = price_data["B_close"].iloc[i]

            trading_active = date >= out_of_sample_start

            if trading_active:
                #1. acrrue overnight borrow on the position held coming into today

                borrow_a = daily_borrow_cost(shares_a, price_a)
                borrow_b = daily_borrow_cost(shares_b, price_b)
                 
                cash -= borrow_a + borrow_b
                total_borrow_cost += borrow_a + borrow_b

                #2. execute yesterday's queued decision at today's price
                if pending_action is not None :
                    action, z_at_signal = pending_action

                    if action in ("enter_long", "enter_short"):
                        cash_before_entry = cash
                        if action =="enter_long":
                            side = "long"
                        else :
                            side = "short"

                        buying_a = (side == "long")
                        exec_price_a = apply_slippage(price_a, buying_a)

                        #changed here
                        target_shares_a = cash_before_entry /exec_price_a

                        if side == "short":
                            target_shares_a = - target_shares_a
                        target_shares_b = -strategy.beta*target_shares_a
                        shares_a, cash_delta_a, tc_a, slip_a = self._trade_leg(shares_a, target_shares_a, price_a)
                        shares_b, cash_delta_b ,tc_b, slip_b= self._trade_leg(shares_b, target_shares_b, price_b)

                        cash+= cash_delta_a + cash_delta_b
                        total_transaction_cost += tc_a+ tc_b
                        total_slippage_cost += slip_a + slip_b

                        open_trade = {
                            "entry_date" : date, 
                            "side" : side,
                            "entry_z" : z_at_signal, 
                            "cash_before_entry" : cash_before_entry,
                        }

                    elif action == "exit":
                        if abs(z_at_signal) >= STOP_LOSS_Z:
                            exit_reason = "stop_loss"
                        else : exit_reason = "mean_reversion"

                        shares_a, cash_delta_a, tc_a, slip_a = self._trade_leg(shares_a, 0.0, price_a)
                        shares_b, cash_delta_b,tc_b, slip_b = self._trade_leg(shares_b, 0.0, price_b)

                        cash+= cash_delta_a+ cash_delta_b
                        total_transaction_cost += tc_a+ tc_b
                        total_slippage_cost += slip_a + slip_b
                        trade_log.append({
                            "entry_date" : open_trade["entry_date"],
                            "exit_date" : date,
                            "side" : open_trade["side"],
                            "entry_z": open_trade["entry_z"],
                            "exit_z" : z_at_signal,
                            "exit_reason" : exit_reason,
                            "pnl" : cash - open_trade["cash_before_entry"]
                        })

                        open_trade = None

                    pending_action = None
            #3 .Generate today's signal (always advance strategy's internal state)

            signal = strategy.generate_signal(price_a, price_b)

            if trading_active :
                #4. queue tomorrows action if todays signal implies a state change
                if shares_a ==0: 
                    current_engine_position = "flat"
                elif shares_a >0 : 
                    current_engine_position = "long"
                else : current_engine_position=  "short"

                new_position = signal["position"]
                if new_position != current_engine_position:
                    if current_engine_position == "flat" and new_position =="long":
                        pending_action = ("enter_long", signal["zscore"])
                    elif current_engine_position=="flat" and new_position == "short":
                        pending_action = ("enter_short", signal["zscore"])
                    elif current_engine_position != "flat" and new_position=="flat":
                        pending_action = ("exit", signal["zscore"])
                # 5. mark to market
                portfolio_value = cash + shares_a*price_a + shares_b*price_b

                equity_records.append({"date" : date, "portfolio_value": portfolio_value})
                signal_records.append({
                    "date": date,
                    "beta": signal["beta"],
                    "spread": signal["spread"],
                    "zscore": signal["zscore"],
                })


        equity_curve = pd.DataFrame(equity_records).set_index("date")["portfolio_value"]
        trade_log_df = pd.DataFrame(trade_log)
        signal_history = pd.DataFrame(signal_records).set_index("date")
        cost_summary = {
            "transaction": total_transaction_cost,
            "slippage" : total_slippage_cost,
            "borrow" : total_borrow_cost
        }
        return equity_curve, trade_log_df, cost_summary, signal_history

# smoke test

if __name__ == "__main__":
    from KalmanFilter import AssetDataLoader, Strategy
    loader = AssetDataLoader()
    data = loader.load()
    in_sample = loader.get_in_sample()

    strategy = Strategy(q_multiplier= 1e-9, use_kalman=False)
    strategy.fit(in_sample)

    engine = ExecutionEngine(initial_capital=100_000)
    equity_curve, trade_log, cost_summary, signal_history = engine.run(data, strategy)
    net_pnl = equity_curve.iloc[-1] - equity_curve.iloc[0]
    total_costs =sum(cost_summary.values())
    gross_pnl = net_pnl + total_costs

    print(f"Start: {equity_curve.iloc[0]:.2f}, End: {equity_curve.iloc[-1]:.2f}")
    print(f"Net PnL: {net_pnl:.2f}")
    print(f"Costs : transaction : {cost_summary['transaction']: .2f}, "
          f"slippage: {cost_summary['slippage']:.2f}, borrow: {cost_summary['borrow']:.2f}")
    print(f"Total costs : {total_costs:.2f}")
    print(f"GrossPnl (before costs) : {gross_pnl:.2f}")
    print(f"Number of trades: {len(trade_log)}")
    print(trade_log["exit_reason"].value_counts())
    print(trade_log.head())

    from metrics import PerformanceVisualizer
    visualizer = PerformanceVisualizer(equity_curve, trade_log, signal_history, strategy.static_beta)

    visualizer.summary()
    visualizer.plot_equity_curve()
    visualizer.plot_drawdown()
    visualizer.plot_spread_and_signals()
    visualizer.plot_rolling_beta()





        