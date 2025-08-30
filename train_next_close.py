#!/usr/bin/env python3
"""
Train an LSTM to predict the next OHLCV bar's close using data from the FastAPI.

Usage examples:
  python3 train_next_close.py --venue okx --symbol ETH-USDT --period 10
  python3 train_next_close.py --venue bnc --symbol ETHUSDT --period 10 --epochs 60 --window 64
  python3 train_next_close.py --venue okx --symbol BTC-USDT --period 100 --start 2025-08-25T00:00:00Z --end 2025-08-27T00:00:00Z

Notes:
- Data is fetched from /tick-chart of your ohlcv_api (defaults to http://localhost:8200).
- We split chronologically: 80% train / 20% test.
- We compare the model to a naïve baseline: y_pred_naive = previous close.

Requires:
  pip install tensorflow pandas numpy scikit-learn requests matplotlib
"""

import argparse
import math
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # quieter TF logs

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ----------------------------- Data IO ----------------------------- #

def fetch_ohlcv(api_base, venue, symbol, period, start=None, end=None) -> pd.DataFrame:
    params = {"venue": venue, "symbol": symbol, "period": period}
    if start: params["start"] = start
    if end:   params["end"] = end

    url = f"{api_base.rstrip('/')}/tick-chart"
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    if not data:
        raise ValueError("API returned no rows; ensure the collector is running and the query has data.")

    df = pd.DataFrame(data)

    # ---- Robust time parsing for mixed ISO-8601 inputs ----
    # Some rows: '2025-08-28T02:31:43Z'
    # Others:    '2025-08-28T02:31:43.123Z' or with '+00:00'
    time_vals = pd.Series(df["time"], dtype="string")

    try:
        # Fast path in pandas >=2.0: consistently handles Z, .fffZ, and offsets
        df["time"] = pd.to_datetime(time_vals, utc=True, format="ISO8601")
    except Exception:
        # Fallback: per-row inference
        df["time"] = pd.to_datetime(time_vals, utc=True, format="mixed")

    df = df.sort_values("time").set_index("time")

    # coerce to numerics
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()
    return df

def make_supervised(
    df: pd.DataFrame,
    window: int = 64,
    feature_cols=("open", "high", "low", "close", "volume"),
    target_col="close"
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    """
    Build X, y for sequence-to-one forecasting.
    X[i] = window of features ending at t-1
    y[i] = close at t (the NEXT bar's close)

    Returns: X, y, feature_scaler, close_scaler
    """
    if len(df) <= window:
        raise ValueError(f"Not enough rows ({len(df)}) for window={window}")

    feats = df[list(feature_cols)].copy()

    # Scale features and target using TRAIN-ONLY stats later; for now just create scalers
    feature_scaler = MinMaxScaler()
    close_scaler = MinMaxScaler()

    # We'll fit scalers after we split train/test, to avoid leakage.
    # For shape building we can use raw values; scaling will be applied inside split_and_scale().

    # Build sequences (unscaled for now)
    X_list, y_list = [], []
    values = feats.values
    target = df[target_col].values.reshape(-1, 1)
    for t in range(window, len(df)):
        X_list.append(values[t - window : t, :])   # window rows, F features
        y_list.append(target[t, 0])                # next close at t

    X = np.array(X_list)  # [N, window, F]
    y = np.array(y_list).reshape(-1, 1)  # [N, 1]
    return X, y, feature_scaler, close_scaler


def split_and_scale(
    X: np.ndarray,
    y: np.ndarray,
    df_index: pd.DatetimeIndex,
    feature_scaler: MinMaxScaler,
    close_scaler: MinMaxScaler,
    train_ratio: float = 0.8
):
    """
    Chronological split and scaling: fit scalers on train only; transform train/test.
    X shape: [N, window, F]; y shape: [N,1]
    """
    N = X.shape[0]
    split = int(N * train_ratio)

    # Fit scalers on TRAIN
    # Flatten sequences to 2D for scaler fit: [N_train * window, F]
    X_train_flat = X[:split].reshape(-1, X.shape[-1])
    feature_scaler.fit(X_train_flat)

    y_train = y[:split]
    close_scaler.fit(y_train)  # scale target independently

    # Transform X by scaling each step in each sequence
    X_scaled = X.copy()
    X_scaled = feature_scaler.transform(X_scaled.reshape(-1, X.shape[-1])).reshape(X.shape)

    y_scaled = close_scaler.transform(y)

    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y_scaled[:split], y_scaled[split:]
    idx_train, idx_test = df_index[-N:][:split], df_index[-N:][split:]

    return X_train, X_test, y_train, y_test, idx_train, idx_test


# ----------------------------- Modeling ----------------------------- #

def build_model(window: int, n_features: int) -> keras.Model:
    """
    A compact LSTM for 1-step ahead forecasting.
    Small to reduce overfit risk on noisy tick-derived bars.
    """
    inputs = keras.Input(shape=(window, n_features))
    x = layers.LSTM(64, return_sequences=False)(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1, activation="linear")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mae")
    return model


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, prev_close: np.ndarray) -> float:
    """
    Directional accuracy versus previous close: sign(y_t - y_{t-1}) vs sign(pred - y_{t-1})
    Inputs are unscaled real prices aligned on the same indices.
    """
    up_true = np.sign(y_true.flatten() - prev_close.flatten())
    up_pred = np.sign(y_pred.flatten() - prev_close.flatten())
    return float((up_true == up_pred).mean())


def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, y_naive: np.ndarray, prev_close: np.ndarray
):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-8, None))) * 100.0)

    mae_naive = mean_absolute_error(y_true, y_naive)
    rmse_naive = math.sqrt(mean_squared_error(y_true, y_naive))
    mape_naive = float(np.mean(np.abs((y_true - y_naive) / np.clip(y_true, 1e-8, None))) * 100.0)

    dir_acc = directional_accuracy(y_true, y_pred, prev_close)
    dir_acc_naive = directional_accuracy(y_true, y_naive, prev_close)

    return {
        "MAE": mae, "RMSE": rmse, "MAPE%": mape, "DirAcc": dir_acc,
        "MAE_naive": mae_naive, "RMSE_naive": rmse_naive, "MAPE%_naive": mape_naive, "DirAcc_naive": dir_acc_naive
    }


# ----------------------------- Main ----------------------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--api", default="http://macbook-server:8200", help="Base URL of ohlcv_api")
    p.add_argument("--venue", default="okx", choices=["okx", "bnc"], help="Data source")
    p.add_argument("--symbol", default="ETH-USDT", help="Instrument (OKX: BTC-XXX, Binance: BTCXXX). Server also accepts either style.")
    p.add_argument("--period", type=int, default=10, help="N ticks per bar")
    p.add_argument("--start", default=None, help="ISO datetime inclusive (e.g., 2025-08-26T00:00:00Z)")
    p.add_argument("--end", default=None, help="ISO datetime exclusive (e.g., 2025-08-27T00:00:00Z)")
    p.add_argument("--window", type=int, default=64, help="Sequence length in bars")
    p.add_argument("--epochs", type=int, default=50, help="Training epochs")
    p.add_argument("--batch", type=int, default=64, help="Batch size")
    p.add_argument("--valsplit", type=float, default=0.1, help="Validation split from TRAIN")
    p.add_argument("--plot", action="store_true", help="Save pred_vs_actual.png")
    args = p.parse_args()

    print(f"Fetching {args.venue.upper()} {args.symbol} period={args.period} … from {args.api}")
    df = fetch_ohlcv(args.api, args.venue, args.symbol, args.period, args.start, args.end)
    if len(df) < args.window + 50:
        raise SystemExit(f"Not enough bars ({len(df)}) for a meaningful train/test with window={args.window}. Try longer date range.")

    # Build supervised dataset
    X, y, feature_scaler, close_scaler = make_supervised(df, window=args.window)

    # Chronological 80/20 split and scale (fit on train only)
    Xtr, Xte, ytr, yte, idx_tr, idx_te = split_and_scale(X, y, df.index, feature_scaler, close_scaler, 0.8)

    # Model
    model = build_model(window=args.window, n_features=X.shape[-1])
    cb = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    ]
    hist = model.fit(
        Xtr, ytr,
        epochs=args.epochs,
        batch_size=args.batch,
        validation_split=args.valsplit,
        verbose=1,
        callbacks=cb
    )

    # Predict (scale back to real prices)
    y_pred_scaled = model.predict(Xte)
    y_pred = close_scaler.inverse_transform(y_pred_scaled).flatten()
    y_true = close_scaler.inverse_transform(yte).flatten()

    arr_close = df["close"].to_numpy()
    pos = df.index.get_indexer(idx_te)          # positions of test targets in the original df
    prev_pos = np.clip(pos - 1, 0, len(df)-1)   # previous bar positions (safe at boundaries)
    prev_close_test = arr_close[prev_pos]
    y_naive = prev_close_test.copy()

    # Evaluate
    metrics = evaluate_predictions(y_true, y_pred, y_naive, prev_close_test)

    print("\n=== Test Metrics ===")
    for k, v in metrics.items():
        print(f"{k:>12}: {v:,.6f}")

    # Quick verdict: did we beat the naïve baseline on MAE and Direction?
    better_mae = metrics["MAE"] < metrics["MAE_naive"]
    better_dir = metrics["DirAcc"] > metrics["DirAcc_naive"]
    print("\nVerdict:")
    print(f"  MAE better than naive? {'YES' if better_mae else 'NO'}")
    print(f"  Directional accuracy better than naive? {'YES' if better_dir else 'NO'}")

    if args.plot:
        plt.figure(figsize=(12, 5))
        plt.plot(idx_te, y_true, label="Actual")
        plt.plot(idx_te, y_pred, label="Predicted")
        plt.title(f"{args.venue.upper()} {args.symbol} • period={args.period} • next close forecast")
        plt.xlabel("Time")
        plt.ylabel("Close")
        plt.legend()
        plt.tight_layout()
        plt.savefig("pred_vs_actual.png")
        print("\nSaved plot: pred_vs_actual.png")


if __name__ == "__main__":
    main()
