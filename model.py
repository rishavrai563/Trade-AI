"""
QuantAI ML Filter — Lightweight classifier to filter rule-based signals.

The ML model does NOT generate the trading signal.
It only answers: "Is this rule-based signal likely to be correct?"

This is the meta-labeling approach from López de Prado.
"""

import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import xgboost as xgb
import lightgbm as lgb

from config import (
    XGB_PARAMS, LGBM_PARAMS, ML_FILTER_THRESHOLD,
    TRAIN_MIN_DAYS, WALK_FORWARD_STEP, CV_PURGE_GAP,
    MAX_FEATURES, CORRELATION_THRESHOLD,
    SAMPLE_WEIGHT_DECAY,
    BARRIER_HORIZON, BARRIER_TP_MULT, BARRIER_SL_MULT,
)


# ─────────────────────────────────────────────────────────────────────────────
# Feature Selection
# ─────────────────────────────────────────────────────────────────────────────

def select_features(X_df, y, feature_cols, max_features=MAX_FEATURES,
                    corr_threshold=CORRELATION_THRESHOLD):
    """Reduce features via correlation filter + importance ranking."""
    corr_matrix = X_df[feature_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper.columns:
        correlated = upper.index[upper[col] > corr_threshold].tolist()
        if correlated:
            to_drop.add(col)

    surviving = [c for c in feature_cols if c not in to_drop]
    print(f"  Corr filter: {len(feature_cols)} → {len(surviving)}")

    if len(surviving) > max_features:
        clf = xgb.XGBClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            random_state=42, n_jobs=-1, eval_metric='logloss',
        )
        y_bin = (y > 0).astype(int) if not np.all(np.isin(y, [0, 1])) else y
        clf.fit(X_df[surviving].values, y_bin)
        imp = clf.feature_importances_
        ranked = sorted(zip(surviving, imp), key=lambda x: x[1], reverse=True)
        surviving = [n for n, _ in ranked[:max_features]]
        print(f"  Importance: → {len(surviving)} features")

    return surviving


# ─────────────────────────────────────────────────────────────────────────────
# Triple Barrier Labels (for ML training)
# ─────────────────────────────────────────────────────────────────────────────

def triple_barrier_labels(df, horizon=BARRIER_HORIZON,
                          tp_mult=BARRIER_TP_MULT, sl_mult=BARRIER_SL_MULT):
    """Label each row: +1 (TP hit), -1 (SL hit), 0 (neither)."""
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    log_ret = np.log(df['Close'] / df['Close'].shift(1))
    daily_vol = log_ret.rolling(20).std().values
    n = len(close)
    labels = np.full(n, np.nan)

    for t in range(n):
        if np.isnan(daily_vol[t]) or daily_vol[t] <= 0 or t >= n - 1:
            continue
        upper = close[t] * (1 + tp_mult * daily_vol[t])
        lower = close[t] * (1 - sl_mult * daily_vol[t])
        end = min(t + horizon, n - 1)
        label = 0
        for d in range(t + 1, end + 1):
            if high[d] >= upper:
                label = 1
                break
            if low[d] <= lower:
                label = -1
                break
        labels[t] = label

    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Sample Weights
# ─────────────────────────────────────────────────────────────────────────────

def compute_sample_weights(n, decay=SAMPLE_WEIGHT_DECAY):
    weights = np.array([decay ** (n - 1 - i) for i in range(n)])
    return weights / weights.mean()


# ─────────────────────────────────────────────────────────────────────────────
# ML Filter Training
# ─────────────────────────────────────────────────────────────────────────────

def train_ml_filter(X_train, y_train, sample_weights=None):
    """Train XGBoost + LightGBM filter (predicts P(rule signal is correct))."""
    models = {}

    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale = neg / max(pos, 1)

    xgb_p = {**XGB_PARAMS, 'scale_pos_weight': scale}
    xgb_m = xgb.XGBClassifier(**xgb_p)
    xgb_m.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)
    models['xgboost'] = xgb_m

    lgb_m = lgb.LGBMClassifier(**LGBM_PARAMS, is_unbalance=True)
    lgb_m.fit(X_train, y_train, sample_weight=sample_weights)
    models['lightgbm'] = lgb_m

    return models


def predict_ml_filter(models, X):
    """Average probability from both models."""
    p1 = models['xgboost'].predict_proba(X)[:, 1]
    p2 = models['lightgbm'].predict_proba(X)[:, 1]
    return (p1 + p2) / 2


# ─────────────────────────────────────────────────────────────────────────────
# Walk-Forward CV with Rule-Based Signals + ML Filter
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_cv(strat_data, rule_directions, rule_confirmations,
                    feature_cols, raw_df):
    """
    Walk-forward evaluation of the combined strategy:
    1. Rule-based strategy generates raw signals
    2. ML filter predicts whether each rule signal will be profitable
    3. Only trade when rules + ML agree

    Returns performance metrics and trained models.
    """
    # Compute ML labels (triple barrier) for the same rows
    labels = triple_barrier_labels(strat_data)

    # Build ML features from the strategy data
    from features import build_ml_features
    ml_data, ml_feature_cols = build_ml_features(strat_data)

    # Align all arrays
    valid_mask = ~np.isnan(labels)
    valid_idx = np.where(valid_mask)[0]

    # We need rows that have: valid labels AND valid ML features
    ml_valid = ml_data.index.isin(strat_data.index[valid_idx])
    ml_data_valid = ml_data[ml_valid].copy()

    # Get the corresponding rule signals
    aligned_directions = []
    aligned_labels = []
    aligned_indices = []
    for i, idx in enumerate(ml_data_valid.index):
        pos = strat_data.index.get_loc(idx)
        if pos < len(labels) and not np.isnan(labels[pos]):
            aligned_directions.append(rule_directions[pos])
            aligned_labels.append(labels[pos])
            aligned_indices.append(i)

    aligned_directions = np.array(aligned_directions)
    aligned_labels = np.array(aligned_labels)

    # Binary target for ML: did the label agree with the direction?
    # i.e., if rule said LONG (1) and barrier label was +1 → correct (1)
    #        if rule said SHORT (-1) and barrier label was -1 → correct (1)
    #        otherwise → incorrect (0)
    ml_correct = np.zeros(len(aligned_directions), dtype=int)
    for i in range(len(aligned_directions)):
        if aligned_directions[i] == aligned_labels[i]:
            ml_correct[i] = 1

    # Filter to only rows where rules generated a signal (direction != 0)
    has_signal = aligned_directions != 0
    if has_signal.sum() < 100:
        print("  ⚠ Not enough rule-generated signals for ML training")
        return _empty_cv_result()

    X_ml = ml_data_valid.iloc[aligned_indices][ml_feature_cols].values[has_signal]
    y_ml = ml_correct[has_signal]
    dates_ml = ml_data_valid.index[np.array(aligned_indices)[has_signal]]
    directions_ml = aligned_directions[has_signal]
    labels_ml = aligned_labels[has_signal]

    n = len(X_ml)
    min_train = min(TRAIN_MIN_DAYS, n // 3)
    step = WALK_FORWARD_STEP
    gap = CV_PURGE_GAP

    print(f"\n  Walk-Forward CV: {n} signal samples, min_train={min_train}, "
          f"step={step}, gap={gap}")

    # Feature selection on first chunk
    train_df = pd.DataFrame(X_ml[:min_train], columns=ml_feature_cols)
    selected = select_features(train_df, y_ml[:min_train], ml_feature_cols)
    sel_idx = [ml_feature_cols.index(f) for f in selected]

    all_rule_correct = []   # Was the rule signal correct?
    all_ml_confirms = []    # Did ML confirm the signal?
    all_combined = []       # Was the combined strategy correct?
    all_dates = []
    all_directions = []
    fold_metrics = []

    last_models = None
    last_scaler = None
    fold = 0
    train_end = min_train
    start_time = time.time()

    while train_end + gap + step <= n:
        test_start = train_end + gap
        test_end = min(test_start + step, n)

        X_tr = X_ml[:train_end, :][:, sel_idx]
        y_tr = y_ml[:train_end]
        X_te = X_ml[test_start:test_end, :][:, sel_idx]
        y_te = y_ml[test_start:test_end]

        # Scale
        scaler = RobustScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # Train
        weights = compute_sample_weights(len(X_tr_s))
        models = train_ml_filter(X_tr_s, y_tr, sample_weights=weights)

        # Predict
        ml_proba = predict_ml_filter(models, X_te_s)
        ml_confirms = ml_proba >= ML_FILTER_THRESHOLD

        # Metrics for this fold
        rule_acc = y_te.mean()  # % of rule signals that were correct
        ml_filtered_correct = y_te[ml_confirms]
        ml_filtered_acc = ml_filtered_correct.mean() if len(ml_filtered_correct) > 0 else 0
        ml_trade_pct = ml_confirms.sum() / len(ml_confirms) if len(ml_confirms) > 0 else 0

        fold_metrics.append({
            'fold': fold,
            'rule_accuracy': rule_acc,
            'ml_filtered_accuracy': ml_filtered_acc,
            'ml_trade_pct': ml_trade_pct,
            'n_test': len(y_te),
        })

        all_rule_correct.extend(y_te)
        all_ml_confirms.extend(ml_confirms)
        all_combined.extend(y_te[ml_confirms] if ml_confirms.sum() > 0 else [])
        all_dates.extend(dates_ml[test_start:test_end])
        all_directions.extend(directions_ml[test_start:test_end])

        last_models = models
        last_scaler = scaler
        fold += 1
        train_end = test_start + step

    elapsed = time.time() - start_time

    all_rule_correct = np.array(all_rule_correct)
    all_ml_confirms = np.array(all_ml_confirms)

    rule_only_acc = all_rule_correct.mean()
    if all_ml_confirms.sum() > 0:
        combined_acc = all_rule_correct[all_ml_confirms].mean()
        trade_pct = all_ml_confirms.sum() / len(all_ml_confirms)
    else:
        combined_acc = 0.0
        trade_pct = 0.0

    print(f"\n  ✓ {fold} folds in {elapsed:.1f}s")
    print(f"  Rule-Only Accuracy:     {rule_only_acc:.1%} (on signal days)")
    print(f"  Rule + ML Accuracy:     {combined_acc:.1%}")
    print(f"  ML Trade Filter:        {trade_pct:.1%} of signals passed")
    print(f"  Improvement:            {combined_acc - rule_only_acc:+.1%}")

    return {
        'rule_accuracy': rule_only_acc,
        'combined_accuracy': combined_acc,
        'trade_pct': trade_pct,
        'rule_correct': all_rule_correct,
        'ml_confirms': all_ml_confirms,
        'dates': all_dates,
        'directions': np.array(all_directions) if all_directions else np.array([]),
        'fold_metrics': fold_metrics,
        'last_models': last_models,
        'last_scaler': last_scaler,
        'selected_features': selected,
        'selected_indices': sel_idx,
        'ml_feature_cols': ml_feature_cols,
    }


def _empty_cv_result():
    return {
        'rule_accuracy': 0.0,
        'combined_accuracy': 0.0,
        'trade_pct': 0.0,
        'rule_correct': np.array([]),
        'ml_confirms': np.array([]),
        'dates': [],
        'directions': np.array([]),
        'fold_metrics': [],
        'last_models': None,
        'last_scaler': None,
        'selected_features': [],
        'selected_indices': [],
        'ml_feature_cols': [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Final Model + Live Filter
# ─────────────────────────────────────────────────────────────────────────────

def train_final_filter(strat_data, rule_directions, ml_feature_cols, sel_idx):
    """Train ML filter on all available data for live use."""
    from features import build_ml_features

    labels = triple_barrier_labels(strat_data)
    ml_data, _ = build_ml_features(strat_data)

    valid_mask = ~np.isnan(labels)
    valid_idx = np.where(valid_mask)[0]
    ml_valid = ml_data.index.isin(strat_data.index[valid_idx])
    ml_data_valid = ml_data[ml_valid]

    aligned_dirs = []
    aligned_labels_arr = []
    aligned_i = []
    for i, idx in enumerate(ml_data_valid.index):
        pos = strat_data.index.get_loc(idx)
        if pos < len(labels) and not np.isnan(labels[pos]):
            aligned_dirs.append(rule_directions[pos])
            aligned_labels_arr.append(labels[pos])
            aligned_i.append(i)

    aligned_dirs = np.array(aligned_dirs)
    aligned_labels_arr = np.array(aligned_labels_arr)

    ml_correct = (aligned_dirs == aligned_labels_arr).astype(int)
    has_signal = aligned_dirs != 0

    X = ml_data_valid.iloc[np.array(aligned_i)[has_signal]][ml_feature_cols].values[:, sel_idx]
    y = ml_correct[has_signal]

    scaler = RobustScaler()
    X_s = scaler.fit_transform(X)
    weights = compute_sample_weights(len(X_s))

    models = train_ml_filter(X_s, y, sample_weights=weights)
    print(f"  ✓ Final ML filter trained on {len(X)} samples")
    return models, scaler


def apply_ml_filter(models, scaler, latest_features, sel_idx):
    """Get ML filter probability for a single data point."""
    X = latest_features[sel_idx].reshape(1, -1)
    X_s = scaler.transform(X)
    proba = predict_ml_filter(models, X_s)
    return float(proba[0])


def get_feature_importance(models, selected_features, top_n=10):
    """Feature importances from ML filter models."""
    imp = np.zeros(len(selected_features))
    if 'xgboost' in models:
        xi = models['xgboost'].feature_importances_
        imp += 0.5 * (xi / max(xi.sum(), 1e-9))
    if 'lightgbm' in models:
        li = models['lightgbm'].feature_importances_.astype(float)
        imp += 0.5 * (li / max(li.sum(), 1e-9))
    idx = np.argsort(imp)[::-1][:top_n]
    return [(selected_features[i], imp[i]) for i in idx]
