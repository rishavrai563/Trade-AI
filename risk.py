"""
QuantAI Risk Management — Position Sizing, Stop-Loss, Take-Profit
"""

import numpy as np
import pandas as pd
from config import (
    ATR_PERIOD, STOP_LOSS_ATR_MULT, TAKE_PROFIT_ATR_MULT,
    VOL_TARGET, MAX_POSITION_PCT, KELLY_FRACTION,
)


def compute_atr(data, period=ATR_PERIOD):
    """Compute Average True Range."""
    tr1 = data['High'] - data['Low']
    tr2 = (data['High'] - data['Close'].shift()).abs()
    tr3 = (data['Low'] - data['Close'].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def compute_position_size(signal_strength, recent_volatility,
                          vol_target=VOL_TARGET, max_pct=MAX_POSITION_PCT,
                          kelly_frac=KELLY_FRACTION):
    """
    Position size based on:
      1. Signal strength (stronger signal → larger position)
      2. Inverse volatility (higher vol → smaller position)
      3. Fractional Kelly criterion

    Returns position as fraction of capital [0, max_pct].
    """
    if recent_volatility <= 0 or signal_strength <= 0:
        return 0.0

    # Volatility scaling: target_vol / realized_vol
    vol_scalar = min(vol_target / recent_volatility, 2.0)

    # Signal-weighted Kelly fraction
    raw_size = kelly_frac * signal_strength * vol_scalar

    return min(raw_size, max_pct)


def compute_risk_levels(price, atr, direction,
                        sl_mult=STOP_LOSS_ATR_MULT,
                        tp_mult=TAKE_PROFIT_ATR_MULT):
    """
    Compute stop-loss and take-profit levels based on ATR.

    Returns dict with stop_loss, take_profit, risk_reward ratio.
    """
    if direction == "LONG":
        stop_loss = price - sl_mult * atr
        take_profit = price + tp_mult * atr
    elif direction == "SHORT":
        stop_loss = price + sl_mult * atr
        take_profit = price - tp_mult * atr
    else:
        return {
            'stop_loss': None,
            'take_profit': None,
            'risk_pct': 0.0,
            'reward_pct': 0.0,
            'risk_reward': 0.0,
        }

    risk_pct = abs(price - stop_loss) / price
    reward_pct = abs(take_profit - price) / price
    risk_reward = reward_pct / risk_pct if risk_pct > 0 else 0.0

    return {
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'risk_pct': risk_pct,
        'reward_pct': reward_pct,
        'risk_reward': risk_reward,
    }


def compute_risk_analysis(data, signal_info):
    """
    Full risk analysis for current signal.

    Args:
        data: DataFrame with OHLCV data
        signal_info: dict from model.generate_signal()

    Returns dict with position sizing, stop/TP levels, and risk metrics.
    """
    current_price = float(data['Close'].iloc[-1])
    atr = float(compute_atr(data).iloc[-1])
    direction = signal_info['direction']
    abs_signal = signal_info.get('abs_signal', abs(signal_info.get('prob_long', 0.5) - 0.5) * 2)

    # Recent realized vol (annualized)
    log_ret = np.log(data['Close'] / data['Close'].shift(1)).dropna()
    recent_vol = float(log_ret.tail(20).std() * np.sqrt(252))

    # Position sizing
    position_pct = compute_position_size(abs_signal, recent_vol)

    # Stop-loss / take-profit
    levels = compute_risk_levels(current_price, atr, direction)

    # Regime assessment
    short_vol = float(log_ret.tail(10).std() * np.sqrt(252))
    long_vol = float(log_ret.tail(60).std() * np.sqrt(252)) if len(log_ret) >= 60 else recent_vol
    vol_regime = "EXPANDING" if short_vol > long_vol * 1.2 else \
                 "CONTRACTING" if short_vol < long_vol * 0.8 else "NORMAL"

    return {
        'current_price': current_price,
        'atr': atr,
        'atr_pct': atr / current_price,
        'recent_vol': recent_vol,
        'position_pct': position_pct,
        'vol_regime': vol_regime,
        **levels,
    }
