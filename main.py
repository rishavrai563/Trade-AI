"""
QuantAI — Quantitative Trading Signal System (v2)

Classification ensemble with:
  - Triple Barrier labeling (only label significant moves)
  - Feature selection (77 → 25 to prevent overfitting)
  - Purged walk-forward CV (no leakage from overlapping returns)
  - Meta-labeling (filter false signals)
  - LONG / SHORT / FLAT with probability + meta-confidence

Usage:
    pip install -r requirements.txt
    python main.py
"""

import warnings
warnings.filterwarnings('ignore')

import datetime
import numpy as np
import yfinance as yf

from config import START_DATE, PREDICTION_HORIZON
from features import build_features
from model import (
    walk_forward_cv, train_final_model, generate_signal,
    get_feature_importance,
)
from risk import compute_risk_analysis
from backtest import run_backtest, print_backtest_report


def get_ticker():
    """Get and validate stock ticker."""
    while True:
        ticker = input("\nEnter stock ticker (e.g., AAPL, MSFT, RELIANCE.NS): ").strip().upper()
        try:
            test = yf.Ticker(ticker).history(period="5d")
            if not test.empty:
                print(f"  ✓ Valid: {ticker}")
                return ticker
        except Exception:
            pass
        print("  ✗ Invalid ticker. Try again.")


def get_mode():
    """Get execution mode."""
    while True:
        mode = input(
            "\nSelect mode:\n"
            "  [1] signal   — Generate live trading signal\n"
            "  [2] backtest — Run walk-forward backtest\n"
            "  [3] both     — Backtest + live signal\n"
            "Choice [1/2/3]: "
        ).strip()
        modes = {'1': 'signal', '2': 'backtest', '3': 'both',
                 'signal': 'signal', 'backtest': 'backtest', 'both': 'both'}
        if mode.lower() in modes:
            return modes[mode.lower()]
        print("  ✗ Invalid. Enter 1, 2, or 3.")


def fetch_data(ticker):
    """Download historical stock data."""
    end = datetime.date.today().strftime('%Y-%m-%d')
    print(f"\n📥 Fetching {ticker} ({START_DATE} → {end})...")

    t = yf.Ticker(ticker)
    data = t.history(start=START_DATE, end=end, auto_adjust=False)
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

    idx = data.index
    if hasattr(idx, 'tz') and idx.tz is not None:  # type: ignore[union-attr]
        data.index = idx.tz_localize(None)  # type: ignore[union-attr]

    print(f"  ✓ {len(data)} trading days downloaded")
    return data


def print_signal_report(signal_info, risk_info, importance, ticker, horizon,
                        selected_features):
    """Print the live trading signal report."""
    s = signal_info
    r = risk_info

    dir_icon = {"LONG": "🟢", "SHORT": "🔴", "FLAT": "⚪"}
    conf_icon = {"STRONG": "✅", "MODERATE": "⚠️", "WEAK": "❓", "NO SIGNAL": "❌"}

    print("\n" + "═" * 60)
    print("  QuantAI — Quantitative Trading Signal")
    print("═" * 60)

    print(f"\n  Ticker:         {ticker}")
    print(f"  Horizon:        {horizon} trading days")
    print(f"  P(LONG):        {s['prob_long']:.3f}")
    print(f"  Meta-Conf:      {s['meta_confidence']:.3f}")
    print(f"  Direction:      {dir_icon.get(s['direction'], '')} {s['direction']}")
    print(f"  Confidence:     {conf_icon.get(s['confidence'], '')} {s['confidence']}")

    print(f"\n  ── Model Scores ─────────────────────────────────")
    for name, score in s['model_scores'].items():
        bar = "█" * int(score * 40)
        print(f"  {name:>12}: {score:.3f}  {bar}")
    print(f"  {'ensemble':>12}: {s['prob_long']:.3f}")
    print(f"  {'meta-label':>12}: {s['meta_confidence']:.3f}")

    if s['direction'] != "FLAT":
        print(f"\n  ── Position Sizing ──────────────────────────────")
        print(f"  Entry Price:    ${r['current_price']:.2f}")
        if r['stop_loss'] is not None:
            print(f"  Stop-Loss:      ${r['stop_loss']:.2f}  ({r['risk_pct']:+.1%})")
            print(f"  Take-Profit:    ${r['take_profit']:.2f}  ({r['reward_pct']:+.1%})")
            print(f"  Risk/Reward:    1 : {r['risk_reward']:.1f}")
        print(f"  Position Size:  {r['position_pct']:.0%} of capital (vol-adjusted)")
        print(f"  ATR:            ${r['atr']:.2f}  ({r['atr_pct']:.1%})")
        print(f"  Recent Vol:     {r['recent_vol']:.1%} (annualized)")
        print(f"  Vol Regime:     {r['vol_regime']}")

    if importance:
        print(f"\n  ── Top Alpha Factors ({len(selected_features)} selected) ──────────")
        for i, (feat, imp) in enumerate(importance, 1):
            bar = "█" * int(imp * 200)
            print(f"  {i:>2}. {feat:<25} {imp:.3f}  {bar}")

    print("═" * 60)


def main():
    """Main execution pipeline."""
    print("\n" + "═" * 60)
    print("  QuantAI v2 — Quantitative Trading Signal System")
    print("  Triple Barrier + Meta-Label + Purged Walk-Forward")
    print("═" * 60)

    # 1. Get ticker and mode
    ticker = get_ticker()
    mode = get_mode()
    print(f"\n  Mode: {mode.upper()}")

    # 2. Fetch data
    raw_data = fetch_data(ticker)

    # 3. Feature engineering + triple barrier labeling
    print("\n🔧 Building alpha factors + triple barrier labels...")
    labeled_data, feature_cols, full_data = build_features(raw_data)

    # 4. Walk-forward CV (always run)
    print("\n📊 Purged Walk-Forward Cross-Validation...")
    cv_result = walk_forward_cv(labeled_data, feature_cols)

    # 5. Backtest
    if mode in ('backtest', 'both'):
        print("\n📈 Running backtest...")
        bt_result = run_backtest(
            cv_result['probabilities'],
            cv_result['meta_confidence'],
            cv_result['actuals'],
            cv_result['dates'],
        )
        print_backtest_report(bt_result)

    # 6. Live signal
    if mode in ('signal', 'both'):
        print("\n🎯 Generating live signal...")

        selected_idx = cv_result['selected_indices']
        selected_feats = cv_result['selected_features']

        models, meta_model, scaler = train_final_model(
            labeled_data, feature_cols, selected_idx
        )

        # Use the most recent row from full_data (which includes all days)
        # Build features for the latest point
        signal_info = generate_signal(
            models, meta_model, scaler,
            labeled_data, feature_cols, selected_idx
        )

        risk_info = compute_risk_analysis(raw_data, signal_info)
        importance = get_feature_importance(models, selected_feats, top_n=10)

        print_signal_report(
            signal_info, risk_info, importance, ticker,
            PREDICTION_HORIZON, selected_feats
        )

    # 7. Summary
    cv = cv_result['overall']
    print(f"\n  ── Model Quality (Purged Walk-Forward) ──────────")
    print(f"  Raw Directional Accuracy:    {cv['accuracy']:.1%}")
    print(f"  Balanced Accuracy:           {cv['balanced_accuracy']:.1%}")
    print(f"  Meta-Filtered Accuracy:      {cv['meta_accuracy']:.1%}")
    print(f"  Meta Trade Frequency:        {cv['meta_trade_pct']:.1%}")
    print(f"  CV Folds:                    {len(cv_result['fold_metrics'])}")
    print()


if __name__ == "__main__":
    main()
