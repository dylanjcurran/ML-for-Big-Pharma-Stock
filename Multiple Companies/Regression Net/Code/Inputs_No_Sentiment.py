import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

# --- Helper Functions ---


def compute_sma(series, window):
    """Compute the Simple Moving Average (SMA)"""
    return series.rolling(window=window).mean()

def compute_rsi(series, window=14):
    """Compute Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    """When the code runs, this print statement will indicate that this function has run correctly """
    print("\n RSI Calculated Successfully \n") 
    
    return rsi

def compute_macd(series, span_short=12, span_long=26, span_signal=9):
    """Compute MACD and its Signal Line."""
    ema_short = series.ewm(span=span_short, adjust=False).mean()
    ema_long = series.ewm(span=span_long, adjust=False).mean()
    macd_line = ema_short - ema_long
    macd_signal = macd_line.ewm(span=span_signal, adjust=False).mean()

    """When the code runs, this print statement will indicate that this function has run correctly """
    print("\n MACD Calculated Successfully \n")
    
    return macd_line, macd_signal

def compute_bollinger_bands(series, window=20, window_dev=2):
    """Compute Bollinger Bands (upper and lower)."""
    sma = compute_sma(series, window)
    std = series.rolling(window=window, min_periods=1).std()
    hband = sma + window_dev * std
    lband = sma - window_dev * std

    """When the code runs, this print statement will indicate that this function has run correctly """
    print("\n Bollinger Bands Calculated Successfully \n")
    
    return hband, lband

def compute_obv(close, volume):
    """Compute On Balance Volume (OBV)."""
    delta = close.diff()
    direction = np.sign(delta)
    obv = (volume * direction).fillna(0).cumsum()
    
    """When the code runs, this print statement will indicate that this function has run correctly """
    print("\n OBV Calculated Successfully \n")
    
    return obv

def compute_atr(high, low, close, window=14):
    """Compute Average True Range (ATR)."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=window, min_periods=1).mean()

    """When the code runs, this print statement will indicate that this function has run correctly """
    print("\n ATR Calculated Successfully \n")
    
    return atr

def compute_adx(high, low, close, window=14):
    """Compute Average Directional Index (ADX) using pure pandas."""
    plus_dm = high.diff()
    minus_dm = low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(window=window, min_periods=window).mean()

    plus_di = 100 * (plus_dm.rolling(window=window, min_periods=window).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(window=window, min_periods=window).sum() / atr)

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    adx = dx.rolling(window=window, min_periods=window).mean()

    """When the code runs, this print statement will indicate that this function has run correctly """
    print("\n ADX Calculated Successfully \n")

    return adx

def compute_indicators(df):
    """Compute a single *average* value for each indicator over the window."""
    df = df.sort_index()

    # Compute each indicator (full Series)
    sma = compute_sma(df['Close'], window=20)
    rsi = compute_rsi(df['Close'], window=14)
    macd_line, macd_signal = compute_macd(df['Close'], span_short=12, span_long=26, span_signal=9)
    hband, lband = compute_bollinger_bands(df['Close'], window=20, window_dev=2)
    obv = compute_obv(df['Close'], df['Volume'])
    atr = compute_atr(df['High'], df['Low'], df['Close'], window=14)
    adx = compute_adx(df['High'], df['Low'], df['Close'], window=14)

    # Return a one-row DataFrame with the *mean* of each indicator
    # Note: Using `.dropna().mean()` ensures we skip NaNs at the start.
    return pd.DataFrame([{
        'Simple Moving Average': sma.dropna().mean(),
        'Relative Strength Index': rsi.dropna().mean(),
        'Moving Average Convergence Divergence': macd_line.dropna().mean(),
        'Moving Average Convergence Divergence Signal': macd_signal.dropna().mean(),
        'Bollinger Upper Band': hband.dropna().mean(),
        'Bollinger Lower Band': lband.dropna().mean(),
        'On-Balance Volume': obv.dropna().mean(),
        'Average True Range': atr.dropna().mean(),
        'Average Directional Index': adx.dropna().mean()
    }])
