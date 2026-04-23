"""
TradeAI Model - Bidirectional LSTM with Attention and CUDA/CPU runtime setup
"""

import os
import time
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    matthews_corrcoef,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import RobustScaler

from config import (
    LSTM_UNITS_1, LSTM_UNITS_2, LSTM_UNITS_3,
    DROPOUT_RATE, RECURRENT_DROPOUT, LEARNING_RATE,
    EARLY_STOP_PATIENCE, LR_REDUCE_PATIENCE, LR_REDUCE_FACTOR, MIN_LR,
    CONFIDENCE_THRESHOLD, PREDICTION_DAYS, BATCH_SIZE,
    USE_GPU, USE_MIXED_PRECISION, USE_XLA, PREFETCH_BUFFER
)


def setup_compute_backend():
    """Configure runtime to use CUDA GPU when available, else CPU."""
    import tensorflow as tf
    from tensorflow.keras import mixed_precision

    print("\n" + "=" * 50)
    print("  COMPUTE BACKEND SETUP")
    print("=" * 50)

    gpus = tf.config.list_physical_devices('GPU')
    use_gpu = bool(gpus) and USE_GPU

    if use_gpu:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ CUDA GPU detected: {len(gpus)} device(s)")
            for idx, gpu in enumerate(gpus):
                print(f"  - GPU {idx}: {gpu.name}")
        except Exception as e:
            print(f"⚠ Could not enable memory growth: {e}")

        if USE_MIXED_PRECISION:
            mixed_precision.set_global_policy('mixed_float16')
            print("✓ Mixed precision enabled (mixed_float16)")
        else:
            mixed_precision.set_global_policy('float32')
            print("✓ Mixed precision disabled (float32)")
    else:
        # CPU fallback with conservative threading to avoid oversubscription.
        cores = os.cpu_count() or 4
        inter_threads = max(1, min(4, cores // 2))
        tf.config.threading.set_intra_op_parallelism_threads(cores)
        tf.config.threading.set_inter_op_parallelism_threads(inter_threads)
        mixed_precision.set_global_policy('float32')
        if USE_GPU and not gpus:
            print("⚠ USE_GPU=True but no CUDA GPU visible to TensorFlow. Falling back to CPU.")
        print(f"✓ CPU fallback mode")
        print(f"✓ CPU cores detected: {cores}")
        print(f"✓ Intra-op threads: {cores}")
        print(f"✓ Inter-op threads: {inter_threads}")

    # XLA JIT compilation for CPU optimization
    if USE_XLA:
        tf.config.optimizer.set_jit(True)
        print("✓ XLA JIT compilation enabled")

    print(f"✓ Global precision policy: {mixed_precision.global_policy().name}")

    print("=" * 50 + "\n")
    return {
        'use_gpu': use_gpu,
        'num_gpus': len(gpus),
        'precision_policy': mixed_precision.global_policy().name
    }


def create_dataset(X, y, batch_size, shuffle=True, cache=True):
    """Create optimized tf.data.Dataset for parallel loading."""
    import tensorflow as tf

    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(X), 10000))

    dataset = dataset.batch(batch_size)

    if cache:
        dataset = dataset.cache()

    dataset = dataset.prefetch(buffer_size=PREFETCH_BUFFER)

    return dataset


def build_model(input_shape):
    """Build Bidirectional LSTM with Attention mechanism."""
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, LSTM, Bidirectional, Dense, Dropout,
        BatchNormalization, Multiply, Permute, Flatten, RepeatVector
    )
    from tensorflow.keras.optimizers import Adam

    has_gpu = len(tf.config.list_physical_devices('GPU')) > 0
    recurrent_dropout = 0.0 if has_gpu else RECURRENT_DROPOUT

    if has_gpu and RECURRENT_DROPOUT > 0:
        print("✓ GPU detected: setting recurrent_dropout=0.0 for fast cuDNN LSTM kernels")

    inputs = Input(shape=input_shape)

    # BiLSTM Layer 1
    x = Bidirectional(LSTM(
        LSTM_UNITS_1, return_sequences=True,
        dropout=DROPOUT_RATE, recurrent_dropout=recurrent_dropout
    ))(inputs)
    x = BatchNormalization()(x)

    # BiLSTM Layer 2
    x = Bidirectional(LSTM(
        LSTM_UNITS_2, return_sequences=True,
        dropout=DROPOUT_RATE, recurrent_dropout=recurrent_dropout
    ))(x)
    x = BatchNormalization()(x)

    # Attention mechanism
    attn = Dense(1, activation='tanh')(x)
    attn = Flatten()(attn)
    attn = Dense(input_shape[0], activation='softmax')(attn)
    attn = RepeatVector(LSTM_UNITS_2 * 2)(attn)
    attn = Permute([2, 1])(attn)
    x = Multiply()([x, attn])

    # Final LSTM
    x = LSTM(
        LSTM_UNITS_3, return_sequences=False,
        dropout=DROPOUT_RATE, recurrent_dropout=recurrent_dropout
    )(x)
    x = BatchNormalization()(x)
    x = Dropout(DROPOUT_RATE)(x)

    # Dense layers
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(DROPOUT_RATE / 2)(x)

    # Output layer
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    print("✓ Model built: BiLSTM + Attention")
    model.summary()
    return model


def train(model, X_train, y_train, X_val, y_val, epochs):
    """Train model with parallel data pipeline and callbacks."""
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN

    print(f"\n🚀 Training (max {epochs} epochs)...")

    # Class weights for imbalanced data
    class_weights = None
    positive_ratio = float(np.mean(y_train == 1))
    imbalance = abs(positive_ratio - 0.5)
    if imbalance >= 0.10:
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weights = dict(zip(classes.astype(int), weights))
        print(f"  Class weights: {class_weights}")
    else:
        print("  Class weights: skipped (classes are near-balanced)")

    # Optimized data pipelines
    train_ds = create_dataset(X_train, y_train, BATCH_SIZE, shuffle=True, cache=True)
    val_ds = create_dataset(X_val, y_val, BATCH_SIZE, shuffle=False, cache=False)

    callbacks = [
        EarlyStopping(monitor='val_auc', mode='max', patience=EARLY_STOP_PATIENCE,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_auc', mode='max', factor=LR_REDUCE_FACTOR,
                          patience=LR_REDUCE_PATIENCE, min_lr=MIN_LR, verbose=1),
        TerminateOnNaN()
    ]

    start = time.time()

    # Note: workers/multiprocessing not needed with tf.data.Dataset
    # Parallelism is handled by tf.data pipeline and CPU threading config
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    elapsed = time.time() - start
    trained_epochs = len(history.history['accuracy'])

    print(f"\n📊 Training complete:")
    print(f"  Epochs: {trained_epochs}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/trained_epochs:.2f}s/epoch)")
    print(f"  Best val_accuracy: {max(history.history['val_accuracy']):.2%}")
    if 'val_auc' in history.history:
        print(f"  Best val_auc: {max(history.history['val_auc']):.4f}")

    return history


def tune_threshold(model, X_val, y_val):
    """Find decision threshold that maximizes validation balanced accuracy."""
    probs = model.predict(X_val, batch_size=64, verbose=0).flatten()
    thresholds = np.arange(0.35, 0.66, 0.01)

    best_threshold = 0.5
    best_score = -1.0
    for t in thresholds:
        preds = (probs >= t).astype(int)
        score = balanced_accuracy_score(y_val, preds)
        if score > best_score:
            best_score = score
            best_threshold = float(t)

    print(f"\n🎚️ Threshold tuning (validation):")
    print(f"  Best threshold: {best_threshold:.2f}")
    print(f"  Best balanced accuracy: {best_score:.2%}")
    return best_threshold


def evaluate(model, X_test, y_test, threshold=0.5):
    """Evaluate with batch prediction for speed."""
    start = time.time()
    probs = model.predict(X_test, batch_size=64, verbose=0).flatten()
    preds = (probs >= threshold).astype(int)
    elapsed = time.time() - start

    print("\n📈 Evaluation:")
    print(f"Decision threshold: {threshold:.2f}")
    print(classification_report(y_test, preds, target_names=['DOWN', 'UP']))

    cm = confusion_matrix(y_test, preds)
    print(f"Confusion Matrix:")
    print(f"            Pred DOWN  Pred UP")
    print(f"  Act DOWN     {cm[0][0]:5d}    {cm[0][1]:5d}")
    print(f"  Act UP       {cm[1][0]:5d}    {cm[1][1]:5d}")

    acc = (preds == y_test).mean()
    bal_acc = balanced_accuracy_score(y_test, preds)
    mcc = matthews_corrcoef(y_test, preds)
    print(f"\n  Accuracy: {acc:.2%}")
    print(f"  Balanced Accuracy: {bal_acc:.2%}")
    print(f"  MCC: {mcc:.4f}")
    print(f"  Speed: {len(X_test)/elapsed:.0f} samples/sec")

    # High confidence analysis
    high_conf = (probs >= CONFIDENCE_THRESHOLD) | (probs <= 1 - CONFIDENCE_THRESHOLD)
    if high_conf.sum() > 0:
        hc_acc = (preds[high_conf] == y_test[high_conf]).mean()
        print(f"  High-conf (>{CONFIDENCE_THRESHOLD:.0%}): {high_conf.sum()} samples, {hc_acc:.2%} accuracy")

    return {
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'mcc': mcc,
        'predictions': preds,
        'probabilities': probs,
    }


def predict(model, X_test, threshold=0.5, prediction_days=PREDICTION_DAYS):
    """Predict next movement with confidence assessment."""
    latest = X_test[-1].reshape(1, X_test.shape[1], X_test.shape[2])
    prob = float(model.predict(latest, verbose=0)[0][0])

    direction = "UP" if prob >= threshold else "DOWN"
    confidence = prob if prob >= 0.5 else 1 - prob

    print(f"\n🎯 PREDICTION ({prediction_days}-day):")
    print(f"   Direction: {direction}")
    print(f"   Confidence: {confidence:.2%}")

    if confidence >= CONFIDENCE_THRESHOLD:
        print(f"   ✅ HIGH CONFIDENCE")
        rec = f"Strong signal: {direction}"
    elif confidence >= 0.55:
        print(f"   ⚠️ MODERATE CONFIDENCE")
        rec = f"Weak signal: {direction}"
    else:
        print(f"   ❌ LOW CONFIDENCE")
        rec = "No clear signal"

    return {
        'direction': direction,
        'confidence': confidence,
        'probability': prob,
        'recommendation': rec,
        'high_confidence': confidence >= CONFIDENCE_THRESHOLD
    }


def scale_data(X_train, X_val, X_test):
    """Scale sequences (fit on train only to prevent leakage)."""
    n_train, timesteps, n_features = X_train.shape
    n_val = X_val.shape[0]
    n_test = X_test.shape[0]

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, n_features))
    X_val_scaled = scaler.transform(X_val.reshape(-1, n_features))
    X_test_scaled = scaler.transform(X_test.reshape(-1, n_features))

    X_train_scaled = X_train_scaled.reshape(n_train, timesteps, n_features)
    X_val_scaled = X_val_scaled.reshape(n_val, timesteps, n_features)
    X_test_scaled = X_test_scaled.reshape(n_test, timesteps, n_features)

    print("✓ Data scaled (no leakage)")
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler
