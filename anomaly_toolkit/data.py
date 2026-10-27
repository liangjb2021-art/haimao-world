"""行情数据下载：yfinance（美股/港股/全球）与 akshare（A 股指数）。"""

from __future__ import annotations

import pandas as pd
import yfinance as yf


def download_close(ticker, start, end):
    """下载单个标的的调整后收盘价序列（yfinance）。

    兼容新版 yfinance 返回多层列名的情况。
    """
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.dropna().rename(ticker)


def download_closes(tickers, start, end):
    """下载多个标的，内连接对齐到共同交易日。"""
    series = [download_close(t, start, end) for t in tickers]
    return pd.concat(series, axis=1, join="inner").dropna()


def download_ohlc(ticker, start, end):
    """下载单个标的的调整后 OHLC 全量（yfinance），列名拍平为 Open/High/Low/Close。"""
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


def download_akshare_index_close(symbol):
    """下载 A 股指数日线收盘价（akshare），如 sh000300（沪深 300）。"""
    import akshare as ak

    idx = ak.stock_zh_index_daily(symbol=symbol)
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.sort_values("date").reset_index(drop=True)
    return idx.set_index("date")["close"].astype(float)
