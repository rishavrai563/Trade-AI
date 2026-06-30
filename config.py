"""
QuantAI Configuration — Rule-Based Quant Strategy + ML Filter
"""

# =============================================================================
# DATA SETTINGS
# =============================================================================
START_DATE = "2010-01-01"
TRAIN_MIN_DAYS = 504
WALK_FORWARD_STEP = 63
CV_PURGE_GAP = 10

# =============================================================================
# STRATEGY RULES (the actual alpha source)
# =============================================================================

# Momentum Strategy: Buy when short MA > long MA AND momentum confirms
MOMENTUM_FAST_MA = 10
MOMENTUM_SLOW_MA = 50
MOMENTUM_RSI_OVERSOLD = 30     # RSI below this → potential long (mean reversion)
MOMENTUM_RSI_OVERBOUGHT = 70   # RSI above this → potential short
MOMENTUM_ADX_MIN = 20          # ADX must be above this (trending market)

# Mean Reversion Strategy: Trade bounces from Bollinger Bands
MEANREV_BB_LONG = 0.15         # BB %B below this → oversold → long
MEANREV_BB_SHORT = 0.85        # BB %B above this → overbought → short
MEANREV_RSI_CONFIRM_LOW = 35   # RSI must confirm oversold
MEANREV_RSI_CONFIRM_HIGH = 65  # RSI must confirm overbought

# Regime Detection: Which strategy to use
TRENDING_ADX_THRESHOLD = 25    # ADX > this → trending regime → use momentum
RANGING_ADX_THRESHOLD = 20     # ADX < this → ranging regime → use mean reversion

# =============================================================================
# ML FILTER (only purpose: filter false signals from rule-based strategy)
# =============================================================================
ML_FILTER_THRESHOLD = 0.45     # ML probability must be > this to confirm signal
                                # Lower threshold = more permissive (let rules dominate)

# =============================================================================
# SIGNAL THRESHOLDS
# =============================================================================
CONFIDENCE_LEVELS = {
    'STRONG': 3,    # 3+ confirming indicators
    'MODERATE': 2,  # 2 confirming indicators
    'WEAK': 1,      # 1 confirming indicator
}

# =============================================================================
# RISK MANAGEMENT
# =============================================================================
RISK_FREE_RATE = 0.05
MAX_POSITION_PCT = 1.0
ATR_PERIOD = 14
STOP_LOSS_ATR_MULT = 2.0
TAKE_PROFIT_ATR_MULT = 3.0
VOL_TARGET = 0.15
KELLY_FRACTION = 0.25

# =============================================================================
# ML MODEL SETTINGS (lightweight — just for filtering)
# =============================================================================
XGB_PARAMS = {
    'n_estimators': 200,
    'max_depth': 3,
    'learning_rate': 0.05,
    'subsample': 0.7,
    'colsample_bytree': 0.6,
    'min_child_weight': 15,
    'reg_alpha': 2.0,
    'reg_lambda': 10.0,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss',
}

LGBM_PARAMS = {
    'n_estimators': 200,
    'max_depth': 3,
    'learning_rate': 0.05,
    'subsample': 0.7,
    'colsample_bytree': 0.6,
    'min_child_samples': 40,
    'reg_alpha': 2.0,
    'reg_lambda': 10.0,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
}

# Feature selection
MAX_FEATURES = 15               # Even fewer — ML is just a filter
CORRELATION_THRESHOLD = 0.90

# Triple Barrier (for ML training labels)
BARRIER_HORIZON = 10
BARRIER_TP_MULT = 1.5
BARRIER_SL_MULT = 1.5

# Sample weighting
SAMPLE_WEIGHT_DECAY = 0.9995

# =============================================================================
# FEATURE TIMEFRAMES
# =============================================================================
RETURN_WINDOWS = [1, 2, 3, 5, 10, 20, 60]
SMA_WINDOWS = [5, 10, 20, 50, 100, 200]
EMA_WINDOWS = [9, 12, 21, 26, 50]
RSI_PERIODS = [7, 14, 21]
VOL_WINDOWS = [5, 10, 20, 60]
