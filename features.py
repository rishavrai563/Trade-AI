"""
TradeAI Feature Engineering - Technical Indicators
"""

import pandas as pd
import numpy as np
from config import FEATURES, PREDICTION_DAYS


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index - momentum oscillator."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_macd(prices: pd.Series):
    """MACD - trend following momentum indicator."""
    ema_fast = prices.ewm(span=12, adjust=False).mean()
    ema_slow = prices.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram


def calculate_bollinger(prices: pd.Series, period: int = 20, std: float = 2.0):
    """Bollinger Bands - volatility indicator."""
    sma = prices.rolling(window=period).mean()
    rolling_std = prices.rolling(window=period).std()
    upper = sma + (rolling_std * std)
    lower = sma - (rolling_std * std)
    width = (upper - lower) / sma
    return upper, lower, width


def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series):
    """Stochastic Oscillator - momentum indicator."""
    lowest = low.rolling(window=14).min()
    highest = high.rolling(window=14).max()
    k = 100 * (close - lowest) / (highest - lowest)
    d = k.rolling(window=3).mean()
    return k, d


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    """Average True Range - volatility measure."""
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume - volume-based indicator."""
    obv = pd.Series(index=close.index, dtype=float)
    obv.iloc[0] = volume.iloc[0]
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i - 1]
    return obv


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to dataframe."""
    data = df.copy()

    # Simple Moving Averages
    data['SMA_5'] = data['Close'].rolling(5).mean()
    data['SMA_10'] = data['Close'].rolling(10).mean()
    data['SMA_20'] = data['Close'].rolling(20).mean()
    data['SMA_50'] = data['Close'].rolling(50).mean()

    # Exponential Moving Averages
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()

    # MACD
    data['MACD'], data['MACD_Signal'], data['MACD_Hist'] = calculate_macd(data['Close'])

    # RSI
    data['RSI'] = calculate_rsi(data['Close'])

    # Stochastic
    data['Stochastic_K'], data['Stochastic_D'] = calculate_stochastic(
        data['High'], data['Low'], data['Close']
    )

    # Williams %R
    highest = data['High'].rolling(14).max()
    lowest = data['Low'].rolling(14).min()
    data['Williams_R'] = -100 * (highest - data['Close']) / (highest - lowest)

    # Rate of Change
    data['ROC'] = ((data['Close'] - data['Close'].shift(10)) / data['Close'].shift(10)) * 100

    # ATR
    data['ATR'] = calculate_atr(data['High'], data['Low'], data['Close'])

    # Bollinger Bands
    data['Bollinger_Upper'], data['Bollinger_Lower'], data['Bollinger_Width'] = calculate_bollinger(data['Close'])

    # OBV
    data['OBV'] = calculate_obv(data['Close'], data['Volume'])

    # Volume indicators
    data['Volume_SMA'] = data['Volume'].rolling(20).mean()
    data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']

    # Price patterns
    data['Price_Change'] = data['Close'].pct_change()
    data['High_Low_Ratio'] = (data['High'] - data['Low']) / data['Close']
    data['Close_Open_Ratio'] = (data['Close'] - data['Open']) / data['Open']

    return data


def add_target(df: pd.DataFrame, prediction_days: int = PREDICTION_DAYS) -> pd.DataFrame:
    """Add binary target: 1 if price goes up by >0.5% in N days."""
    data = df.copy()
    future_return = (data['Close'].shift(-prediction_days) - data['Close']) / data['Close']
    data['Target'] = (future_return > 0.005).astype(int)
    return data


def prepare_data(df: pd.DataFrame, prediction_days: int = PREDICTION_DAYS) -> pd.DataFrame:
    """Complete data preparation pipeline."""
    data = add_features(df)
    data = add_target(data, prediction_days=prediction_days)
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()

    if len(data) == 0:
        raise ValueError("No data remaining after preprocessing")

    # Print class distribution
    up = (data['Target'] == 1).sum()
    down = (data['Target'] == 0).sum()
    print(f"✓ Prepared {len(data)} samples with {len(FEATURES)} features")
    print(f"  Class distribution: UP={up} ({up/len(data):.1%}), DOWN={down} ({down/len(data):.1%})")

    return data
