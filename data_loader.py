import os
import pandas as pd 
import yfinance as yf 

from config import TICKER_A, TICKER_B, START_DATE, END_DATE, IN_SAMPLE_END, OUT_OF_SAMPLE_START

CACHE_DIR = "data_cache"

def _download_ticker(ticker: str)->pd.DataFrame:
    cache_path = os.path.join(CACHE_DIR, f"{ticker}.csv")

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True, 
                         #, date_format="%Y-%m-%d"
                         )
        # df.index = pd.to_datetime(df.index)
        return df

    df = yf.download(
        ticker,
        start = START_DATE, 
        end = END_DATE,
        auto_adjust=True,
        progress=False,
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_csv(cache_path)
    return df

def load_pair_data()->pd.DataFrame:
    """returns df : columns [ticker_A, ticker_B]"""
    df_a = _download_ticker(TICKER_A)
    df_b = _download_ticker(TICKER_B)

    close_a = df_a["Close"]
    close_b = df_b["Close"]

    combined = pd.concat(
        [close_a, close_b], axis=1, join="inner", keys = ["A_close", "B_close"]
    )
    combined.columns = ["A_close", "B_close"]

    return combined

def split_in_out_sample(df: pd.DataFrame):
    in_sample = df.loc[:IN_SAMPLE_END]
    out_of_sample = df.loc[OUT_OF_SAMPLE_START:]
    return in_sample, out_of_sample

if __name__ == "__main__":
    data = load_pair_data()
    in_sample, out_of_sample = split_in_out_sample(data)

    print(f"Full range: {data.index.min()} to {data.index.max()}, {len(data)} rows")
    print((f"In-sample: {in_sample.index.min()} to {in_sample.index.max()}, {len(in_sample)} rows"))
    print(f"Out-of-sample: {out_of_sample.index.min()} to {out_of_sample.index.max()}, {len(out_of_sample)} rows")
    print(data.head())

