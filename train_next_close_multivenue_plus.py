#!/usr/bin/env python3
"""
Multi-venue next-close predictor (OKX target) with engineered features.

Adds:
- log returns, rolling vol (std), ATR, RSI(14), EMAs, intrabar range, close-open return
- cross-venue spread, z-score(spread), rolling corr of returns
- lagged Binance returns/volume (no leakage)
- robust Huber loss, ReduceLROnPlateau, EarlyStopping

Usage:
  python3 train_next_close_multivenue_plus.py --symbol ETH-USDT --period 10 --target delta --epochs 80 --window 64 --plot
"""

import argparse, math, os
from typing import Optional, Tuple
import numpy as np, pandas as pd, requests, matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ------------------ helpers: indicators (causal) ------------------ #

def logret(x: pd.Series) -> pd.Series:
    return np.log(x).diff().fillna(0.0)

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rolling_vol(series, win):
    # std() can start as NaN; backfill then final 0.0 for any all-NaN windows
    return (series.rolling(win, min_periods=max(2, win//2)).std()
            .bfill().fillna(0.0))

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def atr(high, low, close, win=14):
    tr = true_range(high, low, close)
    return (tr.ewm(alpha=1/win, adjust=False).mean()
            .bfill().fillna(0.0))

def zscore(series: pd.Series, win: int) -> pd.Series:
    m = series.rolling(win, min_periods=max(2, win//2)).mean()
    s = series.rolling(win, min_periods=max(2, win//2)).std()
    return ((series - m) / (s.replace(0, np.nan))).fillna(0.0)

def rolling_corr(x: pd.Series, y: pd.Series, win: int) -> pd.Series:
    return x.rolling(win, min_periods=max(2, win//2)).corr(y).fillna(0.0)

# ------------------ data fetch & align ------------------ #

def fetch(api_base: str, venue: str, symbol: str, period: int,
          start: Optional[str], end: Optional[str]) -> pd.DataFrame:
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
    t = pd.Series(df["time"], dtype="string")
    try:   df["time"] = pd.to_datetime(t, utc=True, format="ISO8601")
    except: df["time"] = pd.to_datetime(t, utc=True, format="mixed")
    df = df.sort_values("time").set_index("time")
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna()

def align(df_okx: pd.DataFrame, df_bnc: pd.DataFrame, tol="2s") -> pd.DataFrame:
    L = df_okx.reset_index().rename(columns={"time":"time"}).sort_values("time")
    R = df_bnc.reset_index().rename(columns={"time":"time"}).sort_values("time")
    merged = pd.merge_asof(L, R, on="time", direction="nearest",
                           tolerance=pd.Timedelta(tol), suffixes=("_okx","_bnc"))
    return merged.dropna().set_index("time")

def align_okx_binance(df_okx, df_bnc, tolerance="2s"):
    left  = df_okx.reset_index().rename(columns={"time": "time"}).sort_values("time")
    right = df_bnc.reset_index().rename(columns={"time": "time"}).sort_values("time")

    # Use backward so we only take Binance info from <= OKX time (no lookahead)
    m = pd.merge_asof(
        left, right, on="time",
        direction="backward",
        tolerance=pd.Timedelta(tolerance),
        suffixes=("_okx", "_bnc")
    )
    # Now keep OKX rows; Binance columns may be NaN if too stale
    # (Optional) drop overly stale rows, or forward-fill limited columns if desired
    m = m.dropna(subset=["close_bnc"])  # or keep and later .fillna(method='ffill') for selected features
    return m.set_index("time")


# ------------------ feature builder ------------------ #

def build_features(m: pd.DataFrame) -> pd.DataFrame:
    df = m.copy()

    # base returns
    df["lr_okx"] = logret(df["close_okx"])
    df["lr_bnc"] = logret(df["close_bnc"])

    # volatility & ranges
    df["vol20_okx"] = rolling_vol(df["lr_okx"], 20)
    df["vol20_bnc"] = rolling_vol(df["lr_bnc"], 20)
    df["atr14_okx"] = atr(df["high_okx"], df["low_okx"], df["close_okx"], 14)
    df["range_rel_okx"] = (df["high_okx"] - df["low_okx"]) / df["close_okx"]
    df["co_ret_okx"] = (df["close_okx"] - df["open_okx"]) / df["open_okx"]
    df["co_ret_bnc"] = (df["close_bnc"] - df["open_bnc"]) / df["open_bnc"]

    # momentum / mean-reversion
    df["rsi14_okx"] = rsi(df["close_okx"], 14)
    df["ema12_okx"] = ema(df["close_okx"], 12)
    df["ema26_okx"] = ema(df["close_okx"], 26)
    df["ema12_bnc"] = ema(df["close_bnc"], 12)
    df["ema26_bnc"] = ema(df["close_bnc"], 26)

    # spreads & cross-venue structure
    df["spread"] = df["close_okx"] - df["close_bnc"]
    df["spread_z50"] = zscore(df["spread"], 50)
    df["corr20"] = rolling_corr(df["lr_okx"], df["lr_bnc"], 20)

    # lagged Binance signals (no future info)
    df["lr_bnc_lag1"] = df["lr_bnc"].shift(1)
    df["vol20_bnc_lag1"] = df["vol20_bnc"].shift(1)
    df["co_ret_bnc_lag1"] = df["co_ret_bnc"].shift(1)

    # volume normalization
    df["vol_z_okx"] = zscore(df["volume_okx"], 50)
    df["vol_z_bnc"] = zscore(df["volume_bnc"], 50)

    # choose features (exclude raw closes to avoid target leakage on 'price' target)
    feat_cols = [
        # OKX OHLCV
        "open_okx","high_okx","low_okx","volume_okx",
        # Binance OHLCV
        "open_bnc","high_bnc","low_bnc","volume_bnc",
        # engineered
        "lr_okx","lr_bnc","vol20_okx","vol20_bnc","atr14_okx","range_rel_okx",
        "co_ret_okx","co_ret_bnc","rsi14_okx","ema12_okx","ema26_okx","ema12_bnc","ema26_bnc",
        "spread","spread_z50","corr20","lr_bnc_lag1","vol20_bnc_lag1","co_ret_bnc_lag1",
        "vol_z_okx","vol_z_bnc"
    ]
    # keep close_okx for convenience (target/prev use)
    return df[feat_cols + ["close_okx"]].dropna()

# ------------------ dataset build ------------------ #

def supervised(feats: pd.DataFrame, target: pd.Series, window: int):
    F = feats.values
    y = target.values.reshape(-1,1)
    Xs, ys = [], []
    for t in range(window, len(feats)):
        Xs.append(F[t-window:t, :])
        ys.append(y[t,0])
    X = np.array(Xs); y = np.array(ys).reshape(-1,1)
    return X, y

def split_scale(X, y, index, ratio=0.8):
    N = X.shape[0]; split = int(N*ratio)
    feat_scaler, y_scaler = MinMaxScaler(), MinMaxScaler()
    Xtr_flat = X[:split].reshape(-1, X.shape[-1])
    feat_scaler.fit(Xtr_flat)
    y_scaler.fit(y[:split])
    Xs = feat_scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    ys = y_scaler.transform(y)
    Xtr, Xte = Xs[:split], Xs[split:]; ytr, yte = ys[:split], ys[split:]
    idx_tr, idx_te = index[-N:][:split], index[-N:][split:]
    return Xtr, Xte, ytr, yte, idx_tr, idx_te, feat_scaler, y_scaler

# ------------------ model ------------------ #

def build_model(window: int, n_features: int) -> keras.Model:
    inp = keras.Input(shape=(window, n_features))
    x = layers.LSTM(128, return_sequences=True, kernel_regularizer=keras.regularizers.l2(1e-5))(inp)
    x = layers.Dropout(0.25)(x)
    x = layers.LSTM(64, return_sequences=False, kernel_regularizer=keras.regularizers.l2(1e-5))(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="linear")(x)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss=keras.losses.Huber(delta=1.0))
    return m

def dir_acc(y_true_price, y_pred_price, prev_close):
    up_t = np.sign(y_true_price.flatten() - prev_close.flatten())
    up_p = np.sign(y_pred_price.flatten() - prev_close.flatten())
    return float((up_t == up_p).mean())

def evaluate(y_true_price, y_pred_price, y_naive_price, prev_close):
    mae = mean_absolute_error(y_true_price, y_pred_price)
    rmse = math.sqrt(mean_squared_error(y_true_price, y_pred_price))
    mape = float(np.mean(np.abs((y_true_price - y_pred_price) / np.clip(y_true_price, 1e-8, None))) * 100.0)
    mae_n = mean_absolute_error(y_true_price, y_naive_price)
    rmse_n = math.sqrt(mean_squared_error(y_true_price, y_naive_price))
    mape_n = float(np.mean(np.abs((y_true_price - y_naive_price) / np.clip(y_true_price, 1e-8, None))) * 100.0)
    da = dir_acc(y_true_price, y_pred_price, prev_close)
    da_n = dir_acc(y_true_price, y_naive_price, prev_close)
    return {"MAE": mae, "RMSE": rmse, "MAPE%": mape, "DirAcc": da,
            "MAE_naive": mae_n, "RMSE_naive": rmse_n, "MAPE%_naive": mape_n, "DirAcc_naive": da_n}

# ------------------ main ------------------ #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://macbook-server:8200")
    ap.add_argument("--symbol", default="ETH-USDT")
    ap.add_argument("--period", type=int, default=10)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--valsplit", type=float, default=0.1)
    ap.add_argument("--tolerance", default="2s")
    ap.add_argument("--target", choices=["price","delta"], default="delta")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    print(f"Fetching {args.symbol} period={args.period} from {args.api}")
    okx = fetch(args.api, "okx", args.symbol, args.period, args.start, args.end)
    bnc = fetch(args.api, "bnc", args.symbol, args.period, args.start, args.end)

    # m = align(okx, bnc, tol=args.tolerance)
    m = align_okx_binance(okx, bnc, tol=args.tolerance)
    if len(m) < args.window + 50:
        raise SystemExit(f"Aligned rows {len(m)} too small for window {args.window}. Expand date range or relax tolerance.")

    feats = build_features(m)
    idx = feats.index
    close_okx = feats["close_okx"]

    # target definition (OKX)
    if args.target == "price":
        target = close_okx.shift(-1)
    else:
        target = close_okx.shift(-1) - close_okx

    # Use all feature columns except raw close (avoid trivial leakage on 'price')
    feature_cols = [c for c in feats.columns if c != "close_okx"]
    feats = feats[feature_cols].iloc[:-1, :]
    target = target.iloc[:-1]
    idx = idx[:-1]

    # supervised
    X, y = supervised(feats, target, window=args.window)

    # split+scale
    Xtr, Xte, ytr, yte, idx_tr, idx_te, fsc, ysc = split_scale(X, y, idx, 0.8)

    # model
    model = build_model(args.window, X.shape[-1])
    cbs = [
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    ]
    model.fit(Xtr, ytr, epochs=args.epochs, batch_size=args.batch, validation_split=args.valsplit, verbose=1, callbacks=cbs)

    # predict
    y_pred_s = model.predict(Xte)
    y_pred = ysc.inverse_transform(y_pred_s).flatten()
    y_true = ysc.inverse_transform(yte).flatten()

    # reconstruct price target & baseline
    arr_close_okx = okx["close"].to_numpy()
    pos = okx.index.get_indexer(idx_te, method="nearest")
    prev_pos = np.clip(pos - 1, 0, len(arr_close_okx)-1)
    prev_close = arr_close_okx[prev_pos]

    if args.target == "price":
        y_pred_price = y_pred; y_true_price = y_true
    else:
        y_pred_price = prev_close + y_pred
        y_true_price = prev_close + y_true

    y_naive_price = prev_close.copy()

    metrics = evaluate(y_true_price, y_pred_price, y_naive_price, prev_close)

    print("\n=== Test Metrics (OKX target) ===")
    for k, v in metrics.items():
        print(f"{k:>12}: {v:,.6f}")
    print("\nVerdict:")
    print(f"  MAE better than naive? {'YES' if metrics['MAE'] < metrics['MAE_naive'] else 'NO'}")
    print(f"  Directional accuracy better than naive? {'YES' if metrics['DirAcc'] > metrics['DirAcc_naive'] else 'NO'}")

    if args.plot:
        plt.figure(figsize=(12,5))
        plt.plot(idx_te, y_true_price, label="Actual")
        plt.plot(idx_te, y_pred_price, label="Predicted")
        plt.title(f"OKX {args.symbol} • period={args.period} • next close forecast (engineered, target={args.target})")
        plt.xlabel("Time"); plt.ylabel("Close"); plt.legend(); plt.tight_layout()
        plt.savefig("pred_vs_actual_multivenue_plus.png")
        print("\nSaved plot: pred_vs_actual_multivenue_plus.png")

if __name__ == "__main__":
    main()
