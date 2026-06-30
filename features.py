"""
QuantAI Feature Engineering — Alpha Factors for ML Filter
"""

import pandas as pd
import numpy as np
from config import (
    RETURN_WINDOWS, SMA_WINDOWS, EMA_WINDOWS, RSI_PERIODS, VOL_WINDOWS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Alpha Factors for ML Filter
# ─────────────────────────────────────────────────────────────────────────────

def build_ml_features(df):
    """
    Build features for the ML filter model.
    These capture market conditions to predict if a rule signal will work.
    """
    data = df.copy()

    # Returns
    for w in RETURN_WINDOWS:
        data[f'ret_{w}d'] = np.log(data['Close'] / data['Close'].shift(w))

    # RSI
    for p in RSI_PERIODS:
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_g = gain.ewm(com=p-1, min_periods=p).mean()
        avg_l = loss.ewm(com=p-1, min_periods=p).mean()
        data[f'rsi_{p}'] = 100 - (100 / (1 + avg_g / avg_l))

    # MACD histogram
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    data['macd_hist'] = macd - macd.ewm(span=9, adjust=False).mean()

    # SMA crossover distances
    for w in SMA_WINDOWS:
        data[f'_sma_{w}'] = data['Close'].rolling(w).mean()

    data['sma_x_10_50'] = (data['_sma_10'] - data['_sma_50']) / data['Close']
    data['sma_x_50_200'] = (data['_sma_50'] - data['_sma_200']) / data['Close']
    data['price_vs_sma_20'] = (data['Close'] - data['_sma_20']) / data['_sma_20']
    data['price_vs_sma_50'] = (data['Close'] - data['_sma_50']) / data['_sma_50']

    # ADX (if not already present)
    if 'adx' not in data.columns:
        hdiff = data['High'].diff()
        ldiff = -data['Low'].diff()
        pdm = hdiff.where((hdiff > ldiff) & (hdiff > 0), 0.0)
        mdm = ldiff.where((ldiff > hdiff) & (ldiff > 0), 0.0)
        tr = pd.concat([
            data['High'] - data['Low'],
            (data['High'] - data['Close'].shift()).abs(),
            (data['Low'] - data['Close'].shift()).abs()
        ], axis=1).max(axis=1)
        atr14 = tr.ewm(span=14, adjust=False).mean()
        pdi = 100 * pdm.ewm(span=14, adjust=False).mean() / atr14
        mdi = 100 * mdm.ewm(span=14, adjust=False).mean() / atr14
        dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
        data['adx'] = dx.ewm(span=14, adjust=False).mean()

    # Bollinger
    sma20 = data['Close'].rolling(20).mean()
    std20 = data['Close'].rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    data['bb_pct'] = (data['Close'] - lower) / (upper - lower)
    data['bb_width'] = (upper - lower) / sma20

    # Z-scores
    for w in [20, 50]:
        sma = data['Close'].rolling(w).mean()
        std = data['Close'].rolling(w).std()
        data[f'zscore_{w}'] = (data['Close'] - sma) / std

    # Volatility
    log_ret = np.log(data['Close'] / data['Close'].shift(1))
    for w in VOL_WINDOWS:
        data[f'rvol_{w}'] = log_ret.rolling(w).std() * np.sqrt(252)

    # ATR %
    tr = pd.concat([
        data['High'] - data['Low'],
        (data['High'] - data['Close'].shift()).abs(),
        (data['Low'] - data['Close'].shift()).abs()
    ], axis=1).max(axis=1)
    for w in [10, 20]:
        data[f'atr_pct_{w}'] = tr.ewm(span=w, adjust=False).mean() / data['Close']

    # Volume
    for w in [20, 60]:
        vm = data['Volume'].rolling(w).mean()
        vs = data['Volume'].rolling(w).std()
        data[f'vol_zscore_{w}'] = (data['Volume'] - vm) / vs

    # Stochastic
    lo14 = data['Low'].rolling(14).min()
    hi14 = data['High'].rolling(14).max()
    data['stoch_k'] = 100 * (data['Close'] - lo14) / (hi14 - lo14)

    # Volume ratio
    data['vol_ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()

    # Regime features
    sv = log_ret.rolling(10).std() * np.sqrt(252)
    lv = log_ret.rolling(60).std() * np.sqrt(252)
    data['vol_regime'] = sv / lv

    # Hurst
    def hurst_rs(s):
        if len(s) < 20:
            return np.nan
        m = s.mean()
        c = (s - m).cumsum()
        R = c.max() - c.min()
        S = s.std()
        return np.log(R / S) / np.log(len(s)) if S > 0 else np.nan
    data['hurst'] = log_ret.rolling(60).apply(hurst_rs, raw=True)

    # Distance from highs/lows
    for w in [20, 60]:
        data[f'dist_high_{w}'] = (data['Close'] - data['High'].rolling(w).max()) / data['Close']
        data[f'dist_low_{w}'] = (data['Close'] - data['Low'].rolling(w).min()) / data['Close']

    # CMF
    clv = ((data['Close'] - data['Low']) - (data['High'] - data['Close']))
    hl = (data['High'] - data['Low']).replace(0, np.nan)
    mfv = (clv / hl) * data['Volume']
    data['cmf'] = mfv.rolling(20).sum() / data['Volume'].rolling(20).sum()

    # Clean
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()

    # Feature columns (exclude OHLCV, intermediates, strategy indicators)
    exclude = {'Open', 'High', 'Low', 'Close', 'Volume',
               'fast_ma', 'slow_ma', 'ma_200', 'rsi', 'macd', 'macd_signal',
               'plus_di', 'minus_di', 'bb_upper', 'bb_lower',
               'stoch_d', 'vol_sma', 'atr', 'label', 'holding_period', 'target'}
    exclude |= {f'_sma_{w}' for w in SMA_WINDOWS}

    feature_cols = sorted([c for c in data.columns if c not in exclude])

    return data, feature_cols
