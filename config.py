"""
TradeAI Configuration - Stock Prediction Model Settings
"""

# =============================================================================
# DATA SETTINGS
# =============================================================================
START_DATE = "2015-01-01"
SEQUENCE_LENGTH = 60          # Days of history per sample
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================
LSTM_UNITS_1 = 128            # First BiLSTM layer
LSTM_UNITS_2 = 64             # Second BiLSTM layer
LSTM_UNITS_3 = 32             # Final LSTM layer
DROPOUT_RATE = 0.2
RECURRENT_DROPOUT = 0.1

# =============================================================================
# TRAINING SETTINGS
# =============================================================================
EPOCHS = 150
BATCH_SIZE = 32
LEARNING_RATE = 0.0003
EARLY_STOP_PATIENCE = 15
LR_REDUCE_PATIENCE = 7
LR_REDUCE_FACTOR = 0.5
MIN_LR = 1e-6

# =============================================================================
# PREDICTION SETTINGS
# =============================================================================
PREDICTION_DAYS = 5           # Predict 5 days ahead
CONFIDENCE_THRESHOLD = 0.65   # Minimum confidence for "high confidence" signal

# =============================================================================
# COMPUTE BACKEND
# =============================================================================
USE_GPU = True                # Enable CUDA GPU when available
USE_MIXED_PRECISION = True    # Use mixed precision on supported GPUs
USE_XLA = True                # XLA JIT compilation
NUM_WORKERS = 4               # Parallel data loading workers
PREFETCH_BUFFER = 2           # Batches to prefetch

# =============================================================================
# FEATURES - Technical Indicators
# =============================================================================
FEATURES = [
    # Price data
    'Open', 'High', 'Low', 'Close', 'Volume',
    # Moving averages
    'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
    'EMA_12', 'EMA_26',
    # Trend indicators
    'MACD', 'MACD_Signal', 'MACD_Hist',
    # Momentum indicators
    'RSI', 'Stochastic_K', 'Stochastic_D',
    'Williams_R', 'ROC',
    # Volatility indicators
    'ATR', 'Bollinger_Upper', 'Bollinger_Lower', 'Bollinger_Width',
    # Volume indicators
    'OBV', 'Volume_SMA', 'Volume_Ratio',
    # Price patterns
    'Price_Change', 'High_Low_Ratio', 'Close_Open_Ratio'
]
