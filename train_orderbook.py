#!/usr/bin/env python3
import argparse, json, math
from typing import List, Tuple
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

def fetch_ob(api, symbol, start=None, end=None, step=1) -> pd.DataFrame:
    p = {"symbol": symbol, "step": step}
    if start: p["start"] = start
    if end:   p["end"] = end
    url = f"{api.rstrip('/')}/ob-top"
    r = requests.get(url, params=p, timeout=120); r.raise_for_status()
    js = r.json()
    if not js:
        raise SystemExit("No order book rows returned.")
    df = pd.DataFrame(js)
    df["time"] = pd.to_datetime(df["time"], utc=True, format="mixed")
    df = df.sort_values("time").set_index("time")
    for c in ["bid_px","ask_px","bid_sz","ask_sz","mid","spread","imbalance","microprice"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna()

def logret(x: pd.Series) -> pd.Series:
    return np.log(x).diff()

def atr_like_mid(mid: pd.Series, win=50) -> pd.Series:
    # simple volatility proxy on mid
    lr = logret(mid).abs()
    return lr.ewm(alpha=1/win, adjust=False).mean().bfill().fillna(0.0)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    mid = df["mid"]
    spread = df["spread"]
    imb = df["imbalance"].clip(-1,1).fillna(0.0)

    out["mid"] = mid
    out["spread_bps"] = (spread / mid * 1e4).clip(lower=0).fillna(0.0)
    out["imb"] = imb
    out["micro_tilt"] = (df["microprice"] - mid) / (spread.replace(0,np.nan))  # in [-0.5,0.5]
    out["micro_tilt"] = out["micro_tilt"].replace([np.inf,-np.inf], 0.0).fillna(0.0)

    # time delta (sec) between snapshots
    out["dt_sec"] = out.index.to_series().diff().dt.total_seconds().fillna(0.0).clip(0, 5.0)

    # returns & vol of mid
    out["lr_mid"] = logret(mid).fillna(0.0)
    out["vol20_mid"] = out["lr_mid"].rolling(20, min_periods=10).std().bfill().fillna(0.0)

    # lags (causal)
    for k in [1,2,3,5]:
        out[f"imb_lag{k}"]  = out["imb"].shift(k)
        out[f"tilt_lag{k}"] = out["micro_tilt"].shift(k)
        out[f"lr_mid_lag{k}"] = out["lr_mid"].shift(k)
        out[f"sbps_lag{k}"] = out["spread_bps"].shift(k)

    # volatility normalizer on mid
    out["atr_mid"] = atr_like_mid(mid, win=50)

    return out

def make_supervised(feats: pd.DataFrame, target: pd.Series, window: int) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, List[str]]:
    # drop rows with NaNs *after* making target
    Z = pd.concat([feats, target.rename("y")], axis=1).dropna()
    y = Z["y"].values.reshape(-1,1)
    feats = Z.drop(columns=["y"])
    cols = list(feats.columns)
    F = feats.values
    Xs = []
    for t in range(window, len(Z)):
        Xs.append(F[t-window:t,:])
    X = np.array(Xs)
    y = y[window:,:]
    idx = Z.index[window:]
    return X, y, idx, cols

def build_model(window, n_features):
    inp = keras.Input(shape=(window, n_features))
    x = layers.LSTM(64, return_sequences=False, kernel_regularizer=keras.regularizers.l2(1e-5))(inp)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1)(x)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(1e-3, clipnorm=1.0), loss=keras.losses.Huber(delta=1.0))
    return m

def evaluate(y_true_price, y_pred_price):
    mae  = float(np.mean(np.abs(y_true_price - y_pred_price)))
    rmse = float(np.sqrt(np.mean((y_true_price - y_pred_price)**2)))
    mape = float(np.mean(np.abs((y_true_price - y_pred_price) / np.maximum(1e-8, np.abs(y_true_price)))))
    return {"MAE": mae, "RMSE": rmse, "MAPE%": mape*100.0}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://macbook-server:8200")
    ap.add_argument("--symbol", default="ETH-USDT")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--step", type=int, default=1, help="downsample every Nth OB snapshot")
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--target", choices=["mid", "mid_delta", "mid_delta_norm"], default="mid_delta_norm")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    print(f"Fetching OB top for {args.symbol} …")
    ob = fetch_ob(args.api, args.symbol, args.start, args.end, step=args.step)

    feats = build_features(ob)
    mid = feats["mid"]

    if args.target == "mid":
        y = mid.shift(-1)
        recon = lambda prev_mid, yhat, prev_atr: yhat
    elif args.target == "mid_delta":
        y = mid.shift(-1) - mid
        recon = lambda prev_mid, yhat, prev_atr: prev_mid + yhat
    else:  # mid_delta_norm
        y = (mid.shift(-1) - mid) / (feats["atr_mid"].shift(1) + 1e-8)
        recon = lambda prev_mid, yhat, prev_atr: prev_mid + yhat * (prev_atr + 1e-8)

    X, Y, idx, cols = make_supervised(feats.drop(columns=["mid"]), y, window=args.window)
    if len(X) < 200:
        raise SystemExit("Not enough sequences; widen time range or reduce window.")

    # temporal split 80/20
    n = len(X); n_tr = int(n*0.8)
    Xtr, Xte = X[:n_tr], X[n_tr:]
    Ytr, Yte = Y[:n_tr], Y[n_tr:]
    idx_te = idx[n_tr:]

    fs = MinMaxScaler(); ys = MinMaxScaler()
    Xtr_s = fs.fit_transform(Xtr.reshape(-1, X.shape[-1])).reshape(Xtr.shape)
    Xte_s = fs.transform(Xte.reshape(-1, X.shape[-1])).reshape(Xte.shape)
    Ytr_s = ys.fit_transform(Ytr)

    model = build_model(window=args.window, n_features=X.shape[-1])
    cb = [
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    ]
    model.fit(Xtr_s, Ytr_s, validation_split=0.15, epochs=args.epochs, batch_size=256, verbose=1, callbacks=cb)

    Yhat_s = model.predict(Xte_s, verbose=0)
    y_pred = ys.inverse_transform(Yhat_s).flatten()
    y_true = Yte.flatten()

    # reconstruct to PRICE space if needed
    prev_mid = mid.reindex(idx_te).shift(1).ffill().to_numpy()
    prev_atr = feats["atr_mid"].reindex(idx_te).shift(1).ffill().to_numpy()
    y_pred_price = recon(prev_mid, y_pred, prev_atr)
    y_true_price = recon(prev_mid, y_true, prev_atr)

    m = evaluate(y_true_price, y_pred_price)
    print("\n=== Test Metrics (mid in price space) ===")
    for k,v in m.items():
        print(f"{k:>10}: {v:,.6f}")

    if args.plot:
        plt.figure(figsize=(12,4))
        plt.plot(idx_te, y_true_price, label="Actual")
        plt.plot(idx_te, y_pred_price, label="Predicted", alpha=0.8)
        plt.title(f"OKX {args.symbol} • next mid forecast (target={args.target})")
        plt.xlabel("Time"); plt.ylabel("Mid"); plt.legend(); plt.tight_layout()
        plt.savefig("ob_pred_vs_actual.png")
        print("Saved plot: ob_pred_vs_actual.png")

if __name__ == "__main__":
    main()
