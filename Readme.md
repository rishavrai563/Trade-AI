
# TradeAI

TradeAI is a financial analysis tool designed to provide advanced features for financial modeling, prediction, and data analysis. This project is implemented in Python and is structured for easy extension and integration.

## Features
- Modular codebase for financial modeling
- Configurable settings via `config.py`
- Core logic in `main.py`, with features in `features.py` and models in `model.py`
- Requirements managed in `requirements.txt`

## Project Structure
```
config.py         # Configuration settings
features.py       # Feature engineering and utilities
main.py           # Main entry point for running the application
model.py          # Model definitions and training logic
requirements.txt  # Python dependencies
```

## Installation
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd TradeAI
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the main application:
```bash
python main.py
```



## Data Source & Workflow
### Data Acquisition
- Historical stock data is automatically downloaded using [Yahoo Finance](https://finance.yahoo.com/) via the `yfinance` Python package. The user specifies a stock ticker (e.g., AAPL, MSFT, RELIANCE.NS) at runtime.

### Data Processing
1. **Feature Engineering:**
   - Over 30 technical indicators are computed (RSI, MACD, Bollinger Bands, Stochastic, ATR, OBV, etc.) using the raw OHLCV (Open, High, Low, Close, Volume) data.
   - Features are added in `features.py`.
2. **Target Creation:**
   - The target is a binary label: 1 if the price increases by more than 0.5% after a configurable number of days, else 0.
3. **Data Cleaning:**
   - Infinite and missing values are handled, and only valid rows are kept.
4. **Sequence Creation:**
   - Data is split into rolling windows (sequences) for LSTM input.
5. **Scaling:**
   - Features are scaled using `RobustScaler` (fit on training data only to prevent leakage).

### Model Training
- Data is split into training, validation, and test sets (temporal split).
- The Bidirectional LSTM with Attention model is trained on the training set, validated, and then evaluated on the test set.
- Class imbalance is handled with class weights if needed.

### Results & Prediction
- The model outputs predictions for the next movement (UP/DOWN) with a confidence score.
- Evaluation metrics include accuracy, balanced accuracy, MCC, and confusion matrix.
- In "testing" mode, the model's prediction is compared to actual future data for verification.

---

## Model Architecture
The core model used in TradeAI is a **Bidirectional LSTM (Long Short-Term Memory) with an Attention mechanism**, implemented using TensorFlow and Keras. The model is designed for sequence prediction tasks and includes:
- Multiple stacked Bidirectional LSTM layers
- Attention mechanism to focus on important time steps
- Batch normalization and dropout for regularization
- Dense layers for final prediction
- Configurable for both GPU (CUDA) and CPU environments, with optional mixed precision and XLA JIT support

See `model.py` for full implementation details.

## Configuration
Edit `config.py` to adjust settings for your environment or use case.

## Contributing
Contributions are welcome! Please open issues or submit pull requests for improvements.

## License
This project is licensed under the MIT License.
