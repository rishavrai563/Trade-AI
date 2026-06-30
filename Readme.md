
# QuantAI (Trade-AI)

A quantitative trading signal system that generates **LONG / SHORT / FLAT** signals with risk-managed position sizing.

## Architecture

Replaces the old LSTM binary classifier with a professional quant approach:

| Component | Technology |
|---|---|
| **Alpha Engine** | 60+ multi-timeframe technical factors (momentum, trend, mean reversion, volatility, volume, microstructure, regime) |
| **Model** | Ensemble: XGBoost (40%) + LightGBM (40%) + Ridge (20%) |
| **Validation** | Walk-forward expanding window cross-validation |
| **Risk** | ATR-based stop-loss/take-profit, volatility-scaled position sizing (fractional Kelly) |
| **Backtesting** | Full PnL curve, Sharpe, Sortino, max drawdown, Calmar, profit factor |

## Project Structure
```
config.py           # Signal thresholds, risk params, model hyperparameters
features.py         # 60+ alpha factors across multiple timeframes
model.py            # XGBoost + LightGBM + Ridge ensemble, walk-forward CV
risk.py             # Position sizing, stop-loss, take-profit, vol regime
backtest.py         # Walk-forward backtesting with performance metrics
main.py             # CLI: signal / backtest / both modes
requirements.txt    # Dependencies (no TensorFlow needed)
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/rishavrai563/Trade-AI.git
   cd Trade-AI
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
## Usage
```bash
python main.py
```

### Modes
- **Signal** — Generate a live LONG/SHORT/FLAT trading signal with position sizing
- **Backtest** — Run walk-forward backtest with Sharpe, win rate, PnL curve
- **Both** — Full analysis: backtest + live signal

## Output Example
```
══════════════════════════════════════════════════════
  QuantAI — Quantitative Trading Signal
══════════════════════════════════════════════════════

  Ticker:       AAPL
  Horizon:      5 trading days
  Signal:       +0.73  🟢 LONG
  Confidence:   ✅ STRONG

  ── Position Sizing ──────────────────────────────
  Entry Price:    $189.42
  Stop-Loss:      $184.17  (-2.8%)
  Take-Profit:    $197.15  (+4.1%)
  Risk/Reward:    1 : 1.5
  Position Size:  67% of capital (vol-adjusted)

  ── Top Alpha Factors ────────────────────────────
   1. rsi_14                    0.142  ████████████████████████████
   2. sma_cross_10_50           0.098  ███████████████████
   3. vol_regime                0.087  █████████████████
```

## Key Design Decisions

1. **Gradient boosting over LSTM** — Tree-based models dominate tabular financial data
2. **Continuous signal** — Preserves magnitude, not just direction
3. **Walk-forward CV** — Tests on every time period, mimics real production use
4. **Risk-adjusted target** — Forward returns normalized by volatility
5. **Ensemble diversity** — XGBoost (level-wise) + LightGBM (leaf-wise) + Ridge (linear baseline)
6. **No TensorFlow** — 50MB install vs 2GB, trains in seconds vs minutes

## Configuration
Edit `config.py` to adjust signal thresholds, risk parameters, or model hyperparameters.

## License
MIT License
