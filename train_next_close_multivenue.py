#!/usr/bin/env python3
"""
Multi-venue next-close predictor for OKX using OKX + Binance OHLCV features.

Examples:
  python3 train_next_close_multivenue.py --symbol ETH-USDT --period 10 --target delta --epochs 60 --window 64 --plot
  python3 train_next_close_multivenue.py --symbol BTC-USDT --period 100 --start 2025-08-26T00:00:00Z --end 2025-08-27T00:00:00Z

Notes:
- Fetches from your FastAPI /tick-chart twice: OKX (venue=okx) and Binance (venue=bnc).
- Aligns bars using merge_asof (nearest within tolerance).
- Predicts OKX next close (either 'price' or 'delta'). If 'delta', we report price metrics by adding back prev close.
- Baselines: naive next close = previous close (persistence).
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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ----------------------------- Data IO ----------------------------- #

def fetch_venue(api_base: str, venue: str, symbol: str, period: int,
                start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch OHLCV bars for a single venue from /tick-chart.
    Ensures UTC time index, numeric columns, sorted ascending.
    """
    params = {"venue": venue, "symbol": symbol, "period": period}
    if start: params["start"] = start
    if end:   params["end"] = end
    url = f"{api_base.rstrip('/')}/tick-chart"
    r = requests.get(url, params=params, timeout=90)
    r.raise_for_status()
    data = r.json()
    if not data:
        raise ValueError(f"{venue} returned no rows.")

    df = pd.DataFrame(data)
    # robust time parsing for Z / .fffZ / +00:00 mixed
    t = pd.Series(df["time"], dtype="string")
    try:
        df["time"] = pd.to_datetime(t, utc=True, format="ISO8601")
    except Exception:
        df["time"] = pd.to_datetime(t, utc=True, format="mixed")
    df = df.sort_values("time").set_index("time")

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna()


def align_okx_binance(df_okx: pd.DataFrame, df_bnc: pd.DataFrame,
                      tolerance: str = "2s") -> pd.DataFrame:
    """
    Align OKX (left) with Binance (right) by nearest timestamp within tolerance.
    Returns a single DataFrame indexed by 'time' with suffixed columns for each venue.
    Drops rows without a match.
    """
    left  = df_okx.reset_index().rename(columns={"time": "time"})
    right = df_bnc.reset_index().rename(columns={"time": "time"})

    left  = left.sort_values("time")
    right = right.sort_values("time")

    merged = pd.merge_asof(
        left, right, on="time", direction="nearest",
        tolerance=pd.Timedelta(tolerance), suffixes=("_okx", "_bnc")
    )
    merged = merged.dropna()
    merged = merged.set_index("time")
    return merged


def build_feature_frame(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Build multi-venue features from aligned frame.
    Features include OHLCV for both venues + simple engineered features (returns, spread).
    """
    df = merged.copy()

    # One-step returns (pct change) for closes
    df["ret_okx"] = df["close_okx"].pct_change().fillna(0.0)
    df["ret_bnc"] = df["close_bnc"].pct_change().fillna(0.0)

    # Spread features
    df["spread"] = df["close_okx"] - df["close_bnc"]
    df["rel_spread"] = df["spread"] / df["close_okx"]

    # Choose core features (you can add more later)
    features = [
        "open_okx","high_okx","low_okx","close_okx","volume_okx",
        "open_bnc","high_bnc","low_bnc","close_bnc","volume_bnc",
        "ret_okx","ret_bnc","spread","rel_spread"
    ]
    return df[features + ["close_okx"]]  # keep close_okx for target ref


def make_supervised_from_df(feats_df: pd.DataFrame, target_series: pd.Series, window: int):
    """
    X[i] = window of features ending at t-1
    y[i] = target at t
    """
    if len(feats_df) <= window:
        raise ValueError(f"Not enough rows ({len(feats_df)}) for window={window}")

    F = feats_df.values
    y = target_series.values.reshape(-1, 1)

    X_list, y_list = [], []
    for t in range(window, len(feats_df)):
        X_list.append(F[t - window:t, :])
        y_list.append(y[t, 0])

    X = np.array(X_list)
    y = np.array(y_list).reshape(-1, 1)
    return X, y


def split_and_scale(
    X: np.ndarray, y: np.ndarray, index: pd.DatetimeIndex, train_ratio: float = 0.8
):
    """
    Chronological split; fit MinMax on train only.
    """
    from sklearn.preprocessing import MinMaxScaler
    N = X.shape[0]
    split = int(N * train_ratio)

    feat_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_train_flat = X[:split].reshape(-1, X.shape[-1])
    feat_scaler.fit(X_train_flat)
    y_scaler.fit(y[:split])

    X_scaled = feat_scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    y_scaled = y_scaler.transform(y)

    Xtr, Xte = X_scaled[:split], X_scaled[split:]
    ytr, yte = y_scaled[:split], y_scaled[split:]
    idx_tr, idx_te = index[-N:][:split], index[-N:][split:]
    return Xtr, Xte, ytr, yte, idx_tr, idx_te, feat_scaler, y_scaler


# ----------------------------- Modeling ----------------------------- #

def build_model(window: int, n_features: int) -> keras.Model:
    inputs = keras.Input(shape=(window, n_features))
    x = layers.LSTM(96, return_sequences=False)(inputs)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(48, activation="relu")(x)
    outputs = layers.Dense(1, activation="linear")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mae")
    return model


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, prev_close: np.ndarray) -> float:
    up_true = np.sign(y_true.flatten() - prev_close.flatten())
    up_pred = np.sign(y_pred.flatten() - prev_close.flatten())
    return float((up_true == up_pred).mean())


def evaluate_predictions(
    y_true_price: np.ndarray, y_pred_price: np.ndarray, y_naive_price: np.ndarray, prev_close: np.ndarray
):
    mae = mean_absolute_error(y_true_price, y_pred_price)
    rmse = math.sqrt(mean_squared_error(y_true_price, y_pred_price))
    mape = float(np.mean(np.abs((y_true_price - y_pred_price) / np.clip(y_true_price, 1e-8, None))) * 100.0)

    mae_naive = mean_absolute_error(y_true_price, y_naive_price)
    rmse_naive = math.sqrt(mean_squared_error(y_true_price, y_naive_price))
    mape_naive = float(np.mean(np.abs((y_true_price - y_naive_price) / np.clip(y_true_price, 1e-8, None))) * 100.0)

    dir_acc = directional_accuracy(y_true_price, y_pred_price, prev_close)
    dir_acc_naive = directional_accuracy(y_true_price, y_naive_price, prev_close)

    return {
        "MAE": mae, "RMSE": rmse, "MAPE%": mape, "DirAcc": dir_acc,
        "MAE_naive": mae_naive, "RMSE_naive": rmse_naive, "MAPE%_naive": mape_naive, "DirAcc_naive": dir_acc_naive
    }


# ----------------------------- Main ----------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://macbook-server:8200", help="Base URL of ohlcv_api")
    ap.add_argument("--symbol", default="ETH-USDT", help="OKX-style or Binance-style; server fixups accepted")
    ap.add_argument("--period", type=int, default=10, help="N ticks per bar")
    ap.add_argument("--start", default=None, help="ISO datetime inclusive")
    ap.add_argument("--end", default=None, help="ISO datetime exclusive")
    ap.add_argument("--window", type=int, default=64, help="Sequence length")
    ap.add_argument("--epochs", type=int, default=50, help="Training epochs")
    ap.add_argument("--batch", type=int, default=64, help="Batch size")
    ap.add_argument("--valsplit", type=float, default=0.1, help="Validation split (from TRAIN)")
    ap.add_argument("--target", choices=["price", "delta"], default="delta", help="Predict next close (price) or next delta (close_t - close_{t-1})")
    ap.add_argument("--tolerance", default="2s", help="Time alignment tolerance for merge_asof")
    ap.add_argument("--plot", action="store_true", help="Save pred_vs_actual_multivenue.png")
    args = ap.parse_args()

    print(f"Fetching OKX & Binance {args.symbol} period={args.period} … from {args.api}")
    df_okx = fetch_venue(args.api, "okx", args.symbol, args.period, args.start, args.end)
    df_bnc = fetch_venue(args.api, "bnc", args.symbol, args.period, args.start, args.end)

    merged = align_okx_binance(df_okx, df_bnc, tolerance=args.tolerance)
    if len(merged) < args.window + 50:
        raise SystemExit(f"After alignment there are only {len(merged)} rows. Increase date range or relax tolerance.")

    feats = build_feature_frame(merged)

    # Build target on OKX close
    close_okx = feats["close_okx"]
    if args.target == "price":
        # target is next close price
        target = close_okx.shift(-1)
    else:
        # target is next delta: close_t - close_{t-1}
        target = close_okx.shift(-1) - close_okx

    # Drop last NaN (due to shift)
    feats = feats.iloc[:-1, :]
    target = target.iloc[:-1]
    idx = feats.index

    # Remove the extra 'close_okx' we kept for convenience if predicting 'price' to avoid leakage
    feature_cols = [c for c in feats.columns if c != "close_okx"]
    feats_only = feats[feature_cols]

    # Supervised dataset
    X, y = make_supervised_from_df(feats_only, target, window=args.window)

    # Chronological split and scale (fit on train only)
    Xtr, Xte, ytr, yte, idx_tr, idx_te, feat_scaler, y_scaler = split_and_scale(X, y, idx, 0.8)

    # Model
    model = build_model(window=args.window, n_features=X.shape[-1])
    cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
    model.fit(Xtr, ytr, epochs=args.epochs, batch_size=args.batch, validation_split=args.valsplit, verbose=1, callbacks=cb)

    # Predict (inverse scale)
    y_pred_scaled = model.predict(Xte)
    y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()
    y_true = y_scaler.inverse_transform(yte).flatten()

    # Reconstruct PRICE metrics and baseline
    # prev_close for each test target = bar immediately before target time (OKX)
    arr_close_okx = df_okx["close"].to_numpy()
    # positions of test targets (in terms of original OKX index)
    pos = df_okx.index.get_indexer(idx_te, method="nearest")
    prev_pos = np.clip(pos - 1, 0, len(arr_close_okx) - 1)
    prev_close_test = arr_close_okx[prev_pos]

    if args.target == "price":
        y_pred_price = y_pred
        y_true_price = y_true
    else:
        y_pred_price = prev_close_test + y_pred
        y_true_price = prev_close_test + y_true

    # Naive baseline: predict next close = prev_close
    y_naive_price = prev_close_test.copy()

    metrics = evaluate_predictions(y_true_price, y_pred_price, y_naive_price, prev_close_test)

    print("\n=== Test Metrics (OKX target) ===")
    for k, v in metrics.items():
        print(f"{k:>12}: {v:,.6f}")

    better_mae = metrics["MAE"] < metrics["MAE_naive"]
    better_dir = metrics["DirAcc"] > metrics["DirAcc_naive"]
    print("\nVerdict:")
    print(f"  MAE better than naive? {'YES' if better_mae else 'NO'}")
    print(f"  Directional accuracy better than naive? {'YES' if better_dir else 'NO'}")

    if args.plot:
        plt.figure(figsize=(12, 5))
        plt.plot(idx_te, y_true_price, label="Actual")
        plt.plot(idx_te, y_pred_price, label="Predicted")
        plt.title(f"OKX {args.symbol} • period={args.period} • next close forecast (multi-venue, target={args.target})")
        plt.xlabel("Time"); plt.ylabel("Close"); plt.legend(); plt.tight_layout()
        plt.savefig("pred_vs_actual_multivenue.png")
        print("\nSaved plot: pred_vs_actual_multivenue.png")


if __name__ == "__main__":
    main()
