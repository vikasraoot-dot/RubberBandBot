"""
Technical indicators for scalping strategies.
Extends the RubberBand indicators with VWAP, EMA crossovers, volume profile, etc.
"""
import numpy as np
import pandas as pd


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add intraday VWAP (resets each day).
    Requires: high, low, close, volume columns.
    """
    df = df.copy()
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    tp_vol = typical_price * df["volume"]

    # Group by date for daily reset
    dates = df.index.tz_convert("US/Eastern").date
    df["_date"] = dates
    df["_tp_vol_cumsum"] = tp_vol.groupby(df["_date"]).cumsum()
    df["_vol_cumsum"] = df["volume"].groupby(df["_date"]).cumsum()
    df["vwap"] = df["_tp_vol_cumsum"] / df["_vol_cumsum"].replace(0, np.nan)

    # VWAP bands (1 and 2 std dev)
    df["_sq_dev"] = ((typical_price - df["vwap"]) ** 2) * df["volume"]
    df["_sq_dev_cumsum"] = df["_sq_dev"].groupby(df["_date"]).cumsum()
    vwap_std = np.sqrt(df["_sq_dev_cumsum"] / df["_vol_cumsum"].replace(0, np.nan))
    df["vwap_upper1"] = df["vwap"] + vwap_std
    df["vwap_lower1"] = df["vwap"] - vwap_std
    df["vwap_upper2"] = df["vwap"] + 2 * vwap_std
    df["vwap_lower2"] = df["vwap"] - 2 * vwap_std

    df.drop(columns=["_date", "_tp_vol_cumsum", "_vol_cumsum", "_sq_dev", "_sq_dev_cumsum"], inplace=True)
    return df


def add_ema(df: pd.DataFrame, length: int, col: str = "close", name: str = None) -> pd.DataFrame:
    """Add EMA of specified length."""
    df = df.copy()
    name = name or f"ema_{length}"
    df[name] = df[col].ewm(span=length, adjust=False, min_periods=length).mean()
    return df


def add_rsi(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """Add Wilder RSI."""
    df = df.copy()
    delta = df["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    df["rsi"] = (100.0 - (100.0 / (1.0 + rs))).bfill().ffill()
    return df


def add_atr(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """Add ATR (Average True Range)."""
    df = df.copy()
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    return df


def add_bollinger(df: pd.DataFrame, length: int = 20, std_mult: float = 2.0) -> pd.DataFrame:
    """Add Bollinger Bands."""
    df = df.copy()
    df["bb_mid"] = df["close"].rolling(window=length, min_periods=length).mean()
    std = df["close"].rolling(window=length, min_periods=length).std()
    df["bb_upper"] = df["bb_mid"] + std * std_mult
    df["bb_lower"] = df["bb_mid"] - std * std_mult
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"].replace(0, np.nan)
    return df


def add_keltner(df: pd.DataFrame, length: int = 20, mult: float = 1.5, atr_length: int = 14) -> pd.DataFrame:
    """Add Keltner Channels."""
    df = df.copy()
    df["kc_mid"] = df["close"].ewm(span=length, adjust=False, min_periods=length).mean()
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / atr_length, adjust=False, min_periods=atr_length).mean()
    df["kc_upper"] = df["kc_mid"] + atr * mult
    df["kc_lower"] = df["kc_mid"] - atr * mult
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Add MACD, Signal, and Histogram."""
    df = df.copy()
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def add_volume_profile(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Add volume-weighted price levels (POC, VAH, VAL).
    Simple implementation: rolling volume-weighted average price and std.
    """
    df = df.copy()
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    vol_price = typical * df["volume"]

    rolling_vp = vol_price.rolling(lookback, min_periods=5).sum()
    rolling_vol = df["volume"].rolling(lookback, min_periods=5).sum()
    df["poc"] = rolling_vp / rolling_vol.replace(0, np.nan)

    # Value area using rolling std of typical price weighted by volume
    rolling_std = typical.rolling(lookback, min_periods=5).std()
    df["vah"] = df["poc"] + rolling_std  # Value area high
    df["val"] = df["poc"] - rolling_std  # Value area low
    return df


def add_rvol(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Add Relative Volume (current volume vs rolling average)."""
    df = df.copy()
    avg_vol = df["volume"].rolling(lookback, min_periods=5).mean()
    df["rvol"] = df["volume"] / avg_vol.replace(0, np.nan)
    return df


def add_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """Add Stochastic %K and %D."""
    df = df.copy()
    lowest_low = df["low"].rolling(k_period, min_periods=k_period).min()
    highest_high = df["high"].rolling(k_period, min_periods=k_period).max()
    df["stoch_k"] = 100 * (df["close"] - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    df["stoch_d"] = df["stoch_k"].rolling(d_period).mean()
    return df


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add ADX, +DI, -DI."""
    df = df.copy()
    prev_high = df["high"].shift(1)
    prev_low = df["low"].shift(1)
    prev_close = df["close"].shift(1)

    up_move = df["high"] - prev_high.fillna(0)
    down_move = prev_low.fillna(0) - df["low"]

    pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    pos_dm_sm = pd.Series(pos_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    neg_dm_sm = pd.Series(neg_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    pdi = 100.0 * pos_dm_sm / atr.replace(0.0, np.nan)
    mdi = 100.0 * neg_dm_sm / atr.replace(0.0, np.nan)
    dx = 100.0 * (pdi - mdi).abs() / (pdi + mdi).replace(0.0, np.nan)
    adx = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    df["pdi"] = pdi.bfill().ffill()
    df["mdi"] = mdi.bfill().ffill()
    df["adx"] = adx.bfill().ffill()
    return df


def add_squeeze(df: pd.DataFrame, bb_length: int = 20, bb_mult: float = 2.0,
                kc_length: int = 20, kc_mult: float = 1.5) -> pd.DataFrame:
    """
    Add Bollinger-Keltner squeeze indicator.
    squeeze_on = True when BB is inside KC (low volatility).
    """
    df = df.copy()
    if "bb_upper" not in df.columns:
        df = add_bollinger(df, bb_length, bb_mult)
    if "kc_upper" not in df.columns:
        df = add_keltner(df, kc_length, kc_mult)

    df["squeeze_on"] = (df["bb_lower"] > df["kc_lower"]) & (df["bb_upper"] < df["kc_upper"])
    df["squeeze_off"] = ~df["squeeze_on"]
    return df


def add_opening_range(df: pd.DataFrame, minutes: int = 15) -> pd.DataFrame:
    """
    Add Opening Range (first N minutes) high/low for each day.
    """
    df = df.copy()
    eastern = df.index.tz_convert("US/Eastern")
    dates = eastern.date
    df["_date"] = dates
    df["_minutes_from_open"] = (eastern.hour * 60 + eastern.minute) - (9 * 60 + 30)

    # Determine opening range bars (first N minutes)
    or_mask = (df["_minutes_from_open"] >= 0) & (df["_minutes_from_open"] < minutes)

    # Calculate OR high/low per day
    or_high = df[or_mask].groupby("_date")["high"].max()
    or_low = df[or_mask].groupby("_date")["low"].min()
    or_close = df[or_mask].groupby("_date")["close"].last()
    or_vol = df[or_mask].groupby("_date")["volume"].sum()

    df["or_high"] = df["_date"].map(or_high)
    df["or_low"] = df["_date"].map(or_low)
    df["or_close"] = df["_date"].map(or_close)
    df["or_volume"] = df["_date"].map(or_vol)
    df["or_range"] = df["or_high"] - df["or_low"]

    df.drop(columns=["_date", "_minutes_from_open"], inplace=True)
    return df


def add_gap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add overnight gap size (today's open vs yesterday's close).
    Works on intraday data by detecting day boundaries.
    """
    df = df.copy()
    eastern = df.index.tz_convert("US/Eastern")
    dates = eastern.date
    df["_date"] = dates

    # Get first bar of each day (open) and last bar of each day (close)
    day_open = df.groupby("_date")["open"].first()
    day_close = df.groupby("_date")["close"].last()

    # Gap = today's open - yesterday's close
    prev_close = day_close.shift(1)
    gap = day_open - prev_close
    gap_pct = gap / prev_close.replace(0, np.nan) * 100

    df["gap"] = df["_date"].map(gap)
    df["gap_pct"] = df["_date"].map(gap_pct)
    df["prev_close"] = df["_date"].map(prev_close)

    df.drop(columns=["_date"], inplace=True)
    return df
