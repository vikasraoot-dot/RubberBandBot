# === RubberBand/src/indicators.py ===
from __future__ import annotations
import numpy as np
import pandas as pd

# -------- helpers --------
def _ensure_df(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("expected DataFrame")
    if not {"high", "low", "close"}.issubset(df.columns):
        missing = {"high", "low", "close"} - set(df.columns)
        raise KeyError(f"DataFrame is missing columns: {sorted(missing)}")
    return df

# -------- RSI (Wilder) --------
def ta_add_rsi(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    df = _ensure_df(df).copy()
    length = max(1, int(length))
    delta = df["close"].diff()

    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's RMA via EWM(alpha=1/n)
    avg_gain = gain.ewm(alpha=1.0/length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1.0/length, adjust=False, min_periods=length).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    df["rsi"] = rsi.bfill().ffill()
    return df

# -------- ADX / +DI / -DI (Wilder) --------
def ta_add_adx_di(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = _ensure_df(df).copy()
    period = max(1, int(period))

    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    up_move = high - prev_high.fillna(0)
    down_move = prev_low.fillna(0) - low

    pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1.0/period, adjust=False, min_periods=period).mean()
    pos_dm_sm = pd.Series(pos_dm, index=df.index).ewm(alpha=1.0/period, adjust=False, min_periods=period).mean()
    neg_dm_sm = pd.Series(neg_dm, index=df.index).ewm(alpha=1.0/period, adjust=False, min_periods=period).mean()

    pdi = 100.0 * (pos_dm_sm / atr.replace(0.0, np.nan))
    mdi = 100.0 * (neg_dm_sm / atr.replace(0.0, np.nan))

    dx = 100.0 * ( (pdi - mdi).abs() / (pdi + mdi).replace(0.0, np.nan) )
    adx = dx.ewm(alpha=1.0/period, adjust=False, min_periods=period).mean()

    df["+DI"] = pdi.bfill().ffill().astype(float)
    df["-DI"] = mdi.bfill().ffill().astype(float)
    df["ADX"] = adx.bfill().ffill().astype(float)
    
    # Aliases
    df["pdi"] = df["+DI"]
    df["mdi"] = df["-DI"]
    df["adx"] = df["ADX"]

    return df

# -------- Dollar-volume SMA (utility) --------
def ta_add_vol_dollar(df: pd.DataFrame, window: int = 10, min_periods: int = None) -> pd.DataFrame:
    df = _ensure_df(df).copy()
    window = max(1, int(window))
    if min_periods is None:
        min_periods = max(1, window // 2)
    dv = (df["close"] * df["volume"]).rolling(window=window, min_periods=min_periods).mean()
    df["dollar_vol_avg"] = dv.ffill().fillna(0.0)
    return df

# -------- ATR (Wilder) --------
def ta_add_atr(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    df = _ensure_df(df).copy()
    length = max(1, int(length))
    
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    
    df["atr"] = tr.ewm(alpha=1.0/length, adjust=False, min_periods=length).mean()
    return df

# -------- SMA (Simple Moving Average) --------
def ta_add_sma(df: pd.DataFrame, length: int = 200) -> pd.DataFrame:
    df = _ensure_df(df).copy()
    length = max(1, int(length))
    df[f"sma_{length}"] = df["close"].rolling(window=length, min_periods=length).mean()
    return df

# -------- Keltner Channels --------
def ta_add_keltner(df: pd.DataFrame, length: int = 20, mult: float = 2.0, atr_length: int = 10) -> pd.DataFrame:
    df = _ensure_df(df).copy()
    length = max(1, int(length))
    atr_length = max(1, int(atr_length))
    
    # Middle Line = EMA(length)
    df["kc_middle"] = df["close"].ewm(span=length, adjust=False, min_periods=length).mean()
    
    # ATR for width
    tr = pd.concat(
        [
            (df["high"] - df["low"]),
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0/atr_length, adjust=False, min_periods=atr_length).mean()
    
    df["kc_upper"] = df["kc_middle"] + (atr * mult)
    df["kc_lower"] = df["kc_middle"] - (atr * mult)
    return df

# -------- Bollinger Bands --------
def ta_add_bollinger(df: pd.DataFrame, length: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    df = _ensure_df(df).copy()
    length = max(1, int(length))
    
    # Middle Line = SMA(length)
    df["bb_middle"] = df["close"].rolling(window=length, min_periods=length).mean()
    std = df["close"].rolling(window=length, min_periods=length).std()
    
    df["bb_upper"] = df["bb_middle"] + (std * std_dev)
    df["bb_lower"] = df["bb_middle"] - (std * std_dev)
    return df