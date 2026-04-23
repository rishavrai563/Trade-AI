"""
TradeAI - Stock Movement Prediction using Deep Learning

Features:
- Bidirectional LSTM with Attention mechanism
- 30 technical indicators (RSI, MACD, Bollinger, etc.)
- GPU acceleration with mixed precision training
- Parallel data loading for fast training
- Confidence-based predictions

Usage:
    pip install -r requirements.txt
    python main.py
"""

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import yfinance as yf
import datetime

from config import (
    START_DATE, SEQUENCE_LENGTH, TRAIN_RATIO, VAL_RATIO,
    FEATURES, EPOCHS, PREDICTION_DAYS, CONFIDENCE_THRESHOLD
)
from features import prepare_data
from model import setup_compute_backend, build_model, train, evaluate, predict, scale_data, tune_threshold


TEST_HOLDOUT_DAYS = 2


def get_ticker():
    """Get and validate stock ticker from user."""
    while True:
        ticker = input("\nEnter stock ticker (e.g., AAPL, MSFT, RELIANCE.NS): ").strip().upper()
        try:
            test = yf.Ticker(ticker).history(period="5d")
            if not test.empty:
                print(f"✓ Valid: {ticker}")
                return ticker
        except:
            pass
        print("Invalid ticker. Try again.")


def get_mode():
    """Get execution mode from user."""
    while True:
        mode = input("\nSelect mode [client/testing]: ").strip().lower()
        if mode in {"client", "testing"}:
            return mode
        print("Invalid mode. Enter 'client' or 'testing'.")


def fetch_data(ticker):
    """Download historical stock data."""
    end = datetime.date.today().strftime('%Y-%m-%d')
    print(f"\nFetching {ticker} data ({START_DATE} to {end})...")

    # Use Ticker.history() for more reliable API access
    t = yf.Ticker(ticker)
    data = t.history(start=START_DATE, end=end, auto_adjust=False)
    
    # Select required columns
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    # Remove timezone from index if present
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    print(f"✓ Downloaded {len(data)} trading days")
    return data


def create_sequences(data):
    """Create sequences for LSTM input."""
    X, y = [], []
    features = data[FEATURES].values
    targets = data['Target'].values

    for i in range(len(data) - SEQUENCE_LENGTH):
        X.append(features[i:i + SEQUENCE_LENGTH])
        y.append(targets[i + SEQUENCE_LENGTH])

    X, y = np.array(X), np.array(y)
    print(f"✓ Created {len(X)} sequences of shape {X.shape}")
    return X, y


def split_data(X, y):
    """Temporal train/val/test split."""
    n = len(X)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    if min(len(X_train), len(X_val), len(X_test)) == 0:
        raise ValueError("Not enough sequences for train/val/test split. Increase history or reduce SEQUENCE_LENGTH.")

    print(f"✓ Split: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")
    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    """Main execution pipeline."""
    print("\n" + "=" * 60)
    print("  TradeAI - Stock Movement Prediction")
    print("  BiLSTM + Attention | CUDA/CPU Auto")
    print("=" * 60)

    print(f"\nConfig: {SEQUENCE_LENGTH}d lookback, {PREDICTION_DAYS}d prediction, {len(FEATURES)} features")

    # 1. Setup compute backend (CUDA GPU if available)
    runtime = setup_compute_backend()
    backend = "GPU" if runtime['use_gpu'] else "CPU"
    print(f"Active backend: {backend} | Policy: {runtime['precision_policy']}")

    # 2. Get ticker
    ticker = get_ticker()

    # 3. Select mode
    mode = get_mode()
    print(f"Mode: {mode.upper()}")

    # 4. Fetch data
    raw_data = fetch_data(ticker)

    prediction_days = PREDICTION_DAYS
    verify_info = None

    if mode == "testing":
        if len(raw_data) <= TEST_HOLDOUT_DAYS:
            raise ValueError("Not enough recent rows for testing mode")

        # Simulate an older prediction by hiding the last N trading days from the model.
        cutoff_pos = len(raw_data) - TEST_HOLDOUT_DAYS - 1
        sim_data = raw_data.iloc[:cutoff_pos + 1].copy()
        future_slice = raw_data.iloc[cutoff_pos + 1:cutoff_pos + 1 + TEST_HOLDOUT_DAYS].copy()

        if len(future_slice) < TEST_HOLDOUT_DAYS:
            raise ValueError("Not enough future rows to verify testing-mode prediction")

        prediction_days = TEST_HOLDOUT_DAYS
        verify_info = {
            'as_of_date': sim_data.index[-1],
            'base_close': float(sim_data['Close'].iloc[-1]),
            'actual_close': float(future_slice['Close'].iloc[-1]),
            'check_date': future_slice.index[-1],
        }
        data = sim_data
        print(f"Testing mode: using data only up to {verify_info['as_of_date'].date()}")
    else:
        data = raw_data

    # 5. Feature engineering
    print("\n🔧 Computing technical indicators...")
    data = prepare_data(data, prediction_days=prediction_days)

    # 6. Create sequences
    print("\n📊 Creating sequences...")
    X, y = create_sequences(data)

    # 7. Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # 8. Scale (fit on train only)
    print("\n⚖️ Scaling...")
    X_train, X_val, X_test, scaler = scale_data(X_train, X_val, X_test)

    # 9. Build model
    print("\n🏗️ Building model...")
    model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]))

    # 10. Train
    history = train(model, X_train, y_train, X_val, y_val, EPOCHS)

    # 11. Tune decision threshold on validation set
    threshold = tune_threshold(model, X_val, y_val)

    # 12. Evaluate on held-out test set
    results = evaluate(model, X_test, y_test, threshold=threshold)

    # 13. Predict
    prediction = predict(model, X_test, threshold=threshold, prediction_days=prediction_days)

    if mode == "testing" and verify_info is not None:
        actual_return = (verify_info['actual_close'] - verify_info['base_close']) / verify_info['base_close']
        actual_direction = "UP" if actual_return > 0.005 else "DOWN"
        matched = prediction['direction'] == actual_direction

        print("\n🧪 TESTING MODE CHECK")
        print(f"  As-of date (model could see up to): {verify_info['as_of_date'].date()}")
        print(f"  Checked on date: {verify_info['check_date'].date()}")
        print(f"  Predicted: {prediction['direction']}")
        print(f"  Actual: {actual_direction} ({actual_return:.2%})")
        if matched:
            print("  ✅ Prediction matched")
        else:
            print("  ❌ Prediction did not match")

    # Summary
    print("\n" + "=" * 60)
    print("  COMPLETE")
    print("=" * 60)
    print(f"  Ticker: {ticker}")
    print(f"  Accuracy: {results['accuracy']:.2%}")
    print(f"  Prediction: {prediction['direction']} ({prediction['confidence']:.2%})")
    print(f"  {prediction['recommendation']}")

    if prediction['high_confidence']:
        print("\n  ✅ HIGH CONFIDENCE SIGNAL")
    else:
        print("\n  ⚠️ Wait for stronger signal")

    print()


if __name__ == "__main__":
    main()
