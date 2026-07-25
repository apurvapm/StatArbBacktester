from config import TRANSACTION_COST_BPS,  SLIPPAGE_BPS, BORROW_RATE_ANNUAL

def apply_slippage(price, buying):
    """Effective execution price after slippage - worse than the quoted price in the direction of the trade
       higher if buying, 
       lower if selling 
    """
    slip = SLIPPAGE_BPS/10000
    return price*(1+slip) if buying else price*(1-slip)

def transaction_cost(shares_traded, exec_price):
    """Flat per-leg, per-trade fee on the shares actually traded"""

    return abs(shares_traded)*exec_price*TRANSACTION_COST_BPS/10000

def daily_borrow_cost(shares, price):
    """Daily accrual of annualized borrow cost on a short leg's notional
    Returns 0 if the leg is'nt currently short"""
    if(shares >=0):
        return 0.0

    return abs(shares)*price*BORROW_RATE_ANNUAL/365

