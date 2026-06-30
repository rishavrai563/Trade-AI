"""
QuantAI Strategy Engine — Rule-Based Signals (Momentum + Mean Reversion)

Key insight: ML can't predict stock returns (0.04-0.08 max correlation).
What DOES work: well-known quant rules applied in the right regime.

This module generates the PRIMARY trading signal from rules.
ML is used only as a secondary filter (see model.py).
"""

import numpy as np
import pandas as pd
from config import (
    MOMENTUM_FAST_MA, MOMENTUM_SLOW_MA,
    MOMENTUM_RSI_OVERSOLD, MOMENTUM_RSI_OVERBOUGHT, MOMENTUM_ADX_MIN,
    MEANREV_BB_LONG, MEANREV_BB_SHORT,
    MEANREV_RSI_CONFIRM_LOW, MEANREV_RSI_CONFIRM_HIGH,
    TRENDING_ADX_THRESHOLD, RANGING_ADX_THRESHOLD,
)


def compute_strategy_indicators(df):
    """Compute all indicators needed by strategy rules."""
    data = df.copy()

    # Moving averages for momentum
    data['fast_ma'] = data['Close'].rolling(MOMENTUM_FAST_MA).mean()
    data['slow_ma'] = data['Close'].rolling(MOMENTUM_SLOW_MA).mean()
    data['ma_200'] = data['Close'].rolling(200).mean()

    # RSI-14
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=13, min_periods=14).mean()
    avg_loss = loss.ewm(com=13, min_periods=14).mean()
    rs = avg_gain / avg_loss
    data['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['macd'] = ema12 - ema26
    data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
    data['macd_hist'] = data['macd'] - data['macd_signal']

    # ADX (trend strength)
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
    data['plus_di'] = pdi
    data['minus_di'] = mdi

    # Bollinger Bands
    sma20 = data['Close'].rolling(20).mean()
    std20 = data['Close'].rolling(20).std()
    data['bb_upper'] = sma20 + 2 * std20
    data['bb_lower'] = sma20 - 2 * std20
    data['bb_pct'] = (data['Close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])

    # Stochastic
    lo14 = data['Low'].rolling(14).min()
    hi14 = data['High'].rolling(14).max()
    data['stoch_k'] = 100 * (data['Close'] - lo14) / (hi14 - lo14)
    data['stoch_d'] = data['stoch_k'].rolling(3).mean()

    # Volume confirmation
    data['vol_sma'] = data['Volume'].rolling(20).mean()
    data['vol_ratio'] = data['Volume'] / data['vol_sma']

    # ATR for stop-loss
    data['atr'] = atr14

    return data


def detect_regime(row):
    """
    Detect market regime from ADX.
    Returns: 'trending', 'ranging', or 'transition'
    """
    adx = row['adx']
    if adx >= TRENDING_ADX_THRESHOLD:
        return 'trending'
    elif adx <= RANGING_ADX_THRESHOLD:
        return 'ranging'
    else:
        return 'transition'


def momentum_signal(row):
    """
    Momentum / trend-following signal.
    Used in TRENDING regime.

    LONG when:
      - Fast MA > Slow MA (uptrend)
      - MACD histogram positive (momentum confirms)
      - +DI > -DI (directional movement confirms)
      - Volume above average (conviction)

    SHORT when:
      - Fast MA < Slow MA (downtrend)
      - MACD histogram negative
      - -DI > +DI
      - Volume above average
    """
    signals = []
    direction = 0
    confirmations = 0

    # Primary: MA crossover
    if row['fast_ma'] > row['slow_ma']:
        direction = 1
    elif row['fast_ma'] < row['slow_ma']:
        direction = -1
    else:
        return 0, 0, []

    # Confirmations
    if direction == 1:
        if row['macd_hist'] > 0:
            confirmations += 1
            signals.append('MACD+')
        if row['plus_di'] > row['minus_di']:
            confirmations += 1
            signals.append('+DI>-DI')
        if row['rsi'] > 50 and row['rsi'] < MOMENTUM_RSI_OVERBOUGHT:
            confirmations += 1
            signals.append(f'RSI={row["rsi"]:.0f}')
        if row['vol_ratio'] > 1.0:
            confirmations += 1
            signals.append('Vol↑')
        if row['Close'] > row['ma_200']:
            confirmations += 1
            signals.append('>MA200')
    else:
        if row['macd_hist'] < 0:
            confirmations += 1
            signals.append('MACD-')
        if row['minus_di'] > row['plus_di']:
            confirmations += 1
            signals.append('-DI>+DI')
        if row['rsi'] < 50 and row['rsi'] > MOMENTUM_RSI_OVERSOLD:
            confirmations += 1
            signals.append(f'RSI={row["rsi"]:.0f}')
        if row['vol_ratio'] > 1.0:
            confirmations += 1
            signals.append('Vol↑')
        if row['Close'] < row['ma_200']:
            confirmations += 1
            signals.append('<MA200')

    return direction, confirmations, signals


def mean_reversion_signal(row):
    """
    Mean reversion signal.
    Used in RANGING regime.

    LONG when:
      - BB %B < threshold (price near lower band)
      - RSI < oversold (confirms oversold)
      - Stochastic < 20 (confirms oversold)

    SHORT when:
      - BB %B > threshold (price near upper band)
      - RSI > overbought
      - Stochastic > 80
    """
    signals = []
    direction = 0
    confirmations = 0

    # Primary: Bollinger Band extreme
    if row['bb_pct'] < MEANREV_BB_LONG:
        direction = 1
        signals.append(f'BB%B={row["bb_pct"]:.2f}')
        confirmations += 1

        if row['rsi'] < MEANREV_RSI_CONFIRM_LOW:
            confirmations += 1
            signals.append(f'RSI={row["rsi"]:.0f}')
        if row['stoch_k'] < 20:
            confirmations += 1
            signals.append(f'StochK={row["stoch_k"]:.0f}')
        if row['vol_ratio'] > 0.8:  # At least average volume
            confirmations += 1
            signals.append('Vol OK')

    elif row['bb_pct'] > MEANREV_BB_SHORT:
        direction = -1
        signals.append(f'BB%B={row["bb_pct"]:.2f}')
        confirmations += 1

        if row['rsi'] > MEANREV_RSI_CONFIRM_HIGH:
            confirmations += 1
            signals.append(f'RSI={row["rsi"]:.0f}')
        if row['stoch_k'] > 80:
            confirmations += 1
            signals.append(f'StochK={row["stoch_k"]:.0f}')
        if row['vol_ratio'] > 0.8:
            confirmations += 1
            signals.append('Vol OK')

    return direction, confirmations, signals


def generate_rule_signal(data):
    """
    Generate trading signal from rule-based strategy.

    Returns dict with direction, confirmations, regime, and reasoning.
    """
    strat_data = compute_strategy_indicators(data)
    latest = strat_data.iloc[-1]

    # Detect regime
    regime = detect_regime(latest)

    # Apply appropriate strategy
    if regime == 'trending':
        direction, confirmations, reasons = momentum_signal(latest)
        strategy = 'MOMENTUM'
    elif regime == 'ranging':
        direction, confirmations, reasons = mean_reversion_signal(latest)
        strategy = 'MEAN_REVERSION'
    else:
        # Transition: check both, take stronger signal
        m_dir, m_conf, m_reasons = momentum_signal(latest)
        r_dir, r_conf, r_reasons = mean_reversion_signal(latest)
        if m_conf >= r_conf:
            direction, confirmations, reasons = m_dir, m_conf, m_reasons
            strategy = 'MOMENTUM'
        else:
            direction, confirmations, reasons = r_dir, r_conf, r_reasons
            strategy = 'MEAN_REVERSION'

    # Confidence level
    if confirmations >= 3:
        confidence = 'STRONG'
    elif confirmations >= 2:
        confidence = 'MODERATE'
    elif confirmations >= 1:
        confidence = 'WEAK'
    else:
        confidence = 'NO SIGNAL'
        direction = 0

    if direction == 1:
        dir_str = 'LONG'
    elif direction == -1:
        dir_str = 'SHORT'
    else:
        dir_str = 'FLAT'

    return {
        'direction': dir_str,
        'direction_int': direction,
        'confirmations': confirmations,
        'confidence': confidence,
        'regime': regime,
        'strategy': strategy,
        'reasons': reasons,
        'indicators': {
            'fast_ma': float(latest['fast_ma']),
            'slow_ma': float(latest['slow_ma']),
            'rsi': float(latest['rsi']),
            'adx': float(latest['adx']),
            'bb_pct': float(latest['bb_pct']),
            'macd_hist': float(latest['macd_hist']),
            'stoch_k': float(latest['stoch_k']),
            'vol_ratio': float(latest['vol_ratio']),
        },
    }


def generate_rule_signals_series(data):
    """
    Generate signals for every row (for backtesting).
    Returns arrays of directions and confirmations.
    """
    strat_data = compute_strategy_indicators(data)
    strat_data = strat_data.dropna()

    n = len(strat_data)
    directions = np.zeros(n)
    confirmations = np.zeros(n)

    for i in range(n):
        row = strat_data.iloc[i]
        regime = detect_regime(row)

        if regime == 'trending':
            d, c, _ = momentum_signal(row)
        elif regime == 'ranging':
            d, c, _ = mean_reversion_signal(row)
        else:
            m_d, m_c, _ = momentum_signal(row)
            r_d, r_c, _ = mean_reversion_signal(row)
            if m_c >= r_c:
                d, c = m_d, m_c
            else:
                d, c = r_d, r_c

        directions[i] = d
        confirmations[i] = c

    return strat_data, directions, confirmations
