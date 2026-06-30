"""
QuantAI Backtesting Engine — Classification-Based Walk-Forward Analysis
"""

import numpy as np
import pandas as pd
from config import (
    RISK_FREE_RATE, LONG_THRESHOLD, SHORT_THRESHOLD,
    META_CONFIDENCE_THRESHOLD,
)


def run_backtest(probabilities, meta_confidence, actuals, dates):
    """
    Simulate trading based on walk-forward out-of-sample predictions.

    Strategy:
      P(long) > LONG_THRESHOLD  AND meta_conf > threshold → LONG
      P(long) < SHORT_THRESHOLD AND meta_conf > threshold → SHORT
      Otherwise → FLAT

    Uses the actual binary outcomes (+1/-1) from triple barrier labels.
    """
    n = len(probabilities)
    positions = np.zeros(n)
    trade_returns = np.zeros(n)

    for i in range(n):
        meta_ok = meta_confidence[i] >= META_CONFIDENCE_THRESHOLD

        if probabilities[i] > LONG_THRESHOLD and meta_ok:
            positions[i] = 1.0
            # LONG: we profit if actual=1 (price hit upper barrier)
            trade_returns[i] = 1.0 if actuals[i] == 1 else -1.0
        elif probabilities[i] < SHORT_THRESHOLD and meta_ok:
            positions[i] = -1.0
            # SHORT: we profit if actual=0 (price hit lower barrier)
            trade_returns[i] = 1.0 if actuals[i] == 0 else -1.0
        else:
            positions[i] = 0.0
            trade_returns[i] = 0.0

    # Build trade log
    trades = []
    for i in range(n):
        if positions[i] != 0:
            trades.append({
                'date': dates[i],
                'direction': 'LONG' if positions[i] > 0 else 'SHORT',
                'prob': float(probabilities[i]),
                'meta_conf': float(meta_confidence[i]),
                'correct': trade_returns[i] > 0,
                'return': float(trade_returns[i]),
            })

    # Equity curve (cumulative wins minus losses)
    equity = np.cumsum(trade_returns)

    metrics = compute_metrics(trade_returns, trades, dates)

    return {
        'trades': trades,
        'trade_returns': trade_returns,
        'positions': positions,
        'equity_curve': equity,
        'dates': dates,
        'metrics': metrics,
    }


def compute_metrics(returns, trades, dates):
    """Compute performance metrics."""
    active_returns = returns[returns != 0]
    n_trades = len(trades)

    # Time span
    n_days = (dates[-1] - dates[0]).days if len(dates) > 1 else 1
    n_years = max(n_days / 365.25, 0.01)

    if n_trades > 0:
        winners = [t for t in trades if t['correct']]
        losers = [t for t in trades if not t['correct']]
        win_rate = len(winners) / n_trades

        # Long/Short breakdown
        long_trades = [t for t in trades if t['direction'] == 'LONG']
        short_trades = [t for t in trades if t['direction'] == 'SHORT']
        long_wr = np.mean([t['correct'] for t in long_trades]) if long_trades else 0.0
        short_wr = np.mean([t['correct'] for t in short_trades]) if short_trades else 0.0

        gross_profit = len(winners)
        gross_loss = len(losers)
        profit_factor = gross_profit / max(gross_loss, 1)

        # Avg probability when correct vs incorrect
        avg_prob_correct = np.mean([t['prob'] for t in winners]) if winners else 0.0
        avg_prob_incorrect = np.mean([t['prob'] for t in losers]) if losers else 0.0
    else:
        win_rate = profit_factor = 0.0
        long_wr = short_wr = 0.0
        avg_prob_correct = avg_prob_incorrect = 0.0

    # Max drawdown (in trade units)
    cum = np.cumsum(returns)
    running_max = np.maximum.accumulate(cum)
    drawdowns = cum - running_max
    max_dd = float(drawdowns.min()) if len(drawdowns) > 0 else 0.0

    # Win streak / lose streak
    if n_trades > 0:
        results = [t['correct'] for t in trades]
        max_win_streak = max_streak(results, True)
        max_lose_streak = max_streak(results, False)
    else:
        max_win_streak = max_lose_streak = 0

    exposure = np.mean(returns != 0)

    return {
        'n_trades': n_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown_trades': max_dd,
        'long_trades': len([t for t in trades if t['direction'] == 'LONG']),
        'short_trades': len([t for t in trades if t['direction'] == 'SHORT']),
        'long_win_rate': long_wr,
        'short_win_rate': short_wr,
        'max_win_streak': max_win_streak,
        'max_lose_streak': max_lose_streak,
        'exposure': exposure,
        'avg_prob_correct': avg_prob_correct,
        'avg_prob_incorrect': avg_prob_incorrect,
        'n_years': n_years,
    }


def max_streak(results, value):
    """Find longest consecutive streak of a value."""
    streak = 0
    best = 0
    for r in results:
        if r == value:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return best


def print_backtest_report(result):
    """Print formatted backtest report."""
    m = result['metrics']
    trades = result['trades']

    print("\n" + "═" * 60)
    print("  BACKTEST RESULTS (Walk-Forward Out-of-Sample)")
    print("═" * 60)

    print(f"\n  ── Trade Performance ────────────────────────────")
    print(f"  Total Trades:       {m['n_trades']}")
    print(f"  Win Rate:           {m['win_rate']:.1%}")
    print(f"  Profit Factor:      {m['profit_factor']:.2f}")
    print(f"  Long Trades:        {m['long_trades']}  (win rate: {m['long_win_rate']:.1%})")
    print(f"  Short Trades:       {m['short_trades']}  (win rate: {m['short_win_rate']:.1%})")
    print(f"  Max Win Streak:     {m['max_win_streak']}")
    print(f"  Max Lose Streak:    {m['max_lose_streak']}")
    print(f"  Max Drawdown:       {m['max_drawdown_trades']:.0f} trades")
    print(f"  Market Exposure:    {m['exposure']:.1%}")

    print(f"\n  ── Probability Calibration ──────────────────────")
    print(f"  Avg P(correct):     {m['avg_prob_correct']:.3f}")
    print(f"  Avg P(incorrect):   {m['avg_prob_incorrect']:.3f}")

    # Equity curve
    eq = result['equity_curve']
    if len(eq) > 0:
        _print_sparkline(eq)

    # Recent trades
    if trades:
        n_show = min(15, len(trades))
        print(f"\n  ── Last {n_show} Trades ────────────────────────────")
        print(f"  {'Date':>12}  {'Dir':>5}  {'P(L)':>6}  {'Meta':>6}  {'Result':>6}")
        print(f"  {'─'*12}  {'─'*5}  {'─'*6}  {'─'*6}  {'─'*6}")
        for t in trades[-n_show:]:
            d = t['date'].strftime('%Y-%m-%d') if hasattr(t['date'], 'strftime') else str(t['date'])[:10]
            result_str = "✓ WIN" if t['correct'] else "✗ LOSS"
            print(f"  {d:>12}  {t['direction']:>5}  {t['prob']:>6.3f}  {t['meta_conf']:>6.3f}  {result_str}")

    print("═" * 60)


def _print_sparkline(equity, width=50):
    """Print ASCII equity curve."""
    if len(equity) < 2:
        return

    indices = np.linspace(0, len(equity) - 1, width).astype(int)
    sampled = equity[indices]
    lo, hi = sampled.min(), sampled.max()
    if hi == lo:
        return

    height = 6
    print(f"\n  ── Equity Curve (cumulative W/L) ────────────────")
    print(f"  {hi:>+6.0f} ┐")
    for row in range(height - 1, -1, -1):
        threshold = lo + (hi - lo) * row / height
        line = "".join("█" if val >= threshold else " " for val in sampled)
        if row == height // 2:
            level = lo + (hi - lo) * row / height
            print(f"  {level:>+6.0f} │{line}│")
        else:
            print(f"        │{line}│")
    print(f"  {lo:>+6.0f} ┘")
