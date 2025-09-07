#!/usr/bin/env python3
import argparse, json, os
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import requests
import joblib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ---------- data fetch ----------
def fetch_ob(api, symbol, start=None, end=None, step=1) -> pd.DataFrame:
    p = {"symbol": symbol, "step": step}
    if start: p["start"] = start
    if end:   p["end"] = end
    url = f"{api.rstrip('/')}/ob-top"
    r = requests.get(url, params=p, timeout=120); r.raise_for_status()
    js = r.json()
    if not js: raise SystemExit("No order book rows returned.")
    df = pd.DataFrame(js)
    df["time"] = pd.to_datetime(df["time"], utc=True, format="mixed")
    df = df.sort_values("time").set_index("time")
    num_cols = ["bid_px","ask_px","bid_sz","ask_sz","mid","spread","imbalance","microprice"]
    for c in num_cols: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna()

def logret(x: pd.Series): return np.log(x).diff()

def atr_like_mid(mid: pd.Series, win=50):
    lr = logret(mid).abs()
    return lr.ewm(alpha=1/win, adjust=False).mean().bfill().fillna(0.0)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    mid, spread = df["mid"], df["spread"]
    out["spread_bps"] = (spread / mid * 1e4).clip(lower=0).fillna(0.0)
    out["imb"] = df["imbalance"].clip(-1,1).fillna(0.0)
    out["micro_tilt"] = (df["microprice"] - mid) / (spread.replace(0,np.nan))
    out["micro_tilt"] = out["micro_tilt"].replace([np.inf,-np.inf], 0.0).fillna(0.0)
    out["dt_sec"] = out.index.to_series().diff().dt.total_seconds().fillna(0.0).clip(0,5.0)
    out["lr_mid"] = logret(mid).fillna(0.0)
    out["vol20_mid"] = out["lr_mid"].rolling(20, min_periods=10).std().bfill().fillna(0.0)
    for k in [1,2,3,5]:
        out[f"imb_lag{k}"]  = out["imb"].shift(k)
        out[f"tilt_lag{k}"] = out["micro_tilt"].shift(k)
        out[f"lr_mid_lag{k}"] = out["lr_mid"].shift(k)
        out[f"sbps_lag{k}"] = out["spread_bps"].shift(k)
    out["atr_mid"] = atr_like_mid(mid,50)
    return out.dropna()

def make_sequences(F: pd.DataFrame, window: int) -> Tuple[np.ndarray, pd.DatetimeIndex, List[str]]:
    cols = list(F.columns)
    V = F.values
    X = np.array([ V[t-window:t, :] for t in range(window, len(F)) ])
    idx = F.index[window:]
    return X, idx, cols

def build_autoencoder(window, n_features):
    inp = keras.Input(shape=(window, n_features))
    x = layers.LSTM(64, return_sequences=False)(inp)
    x = layers.RepeatVector(window)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    out = layers.TimeDistributed(layers.Dense(n_features))(x)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(1e-3, clipnorm=1.0), loss="mse")
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://macbook-server:8200")
    ap.add_argument("--symbol", default="ETH-USDT")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--step", type=int, default=5, help="downsample every Nth OB snapshot")
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--save-dir", default="models_ob_anom")
    ap.add_argument("--model-name", default=None)
    args = ap.parse_args()

    print(f"Fetching OB top for {args.symbol} …")
    ob = fetch_ob(args.api, args.symbol, args.start, args.end, step=args.step)
    feats = build_features(ob)
    X_raw, idx, cols = make_sequences(feats, window=args.window)
    if len(X_raw) < 300: raise SystemExit("Not enough sequences; widen time range.")

    # scale features to [0,1]
    fs = MinMaxScaler()
    Xs = fs.fit_transform(X_raw.reshape(-1, X_raw.shape[-1])).reshape(X_raw.shape)

    # temporal split 80/20 (unsupervised)
    n = len(Xs); n_tr = int(n*0.8)
    Xtr, Xte = Xs[:n_tr], Xs[n_tr:]
    idx_te = idx[n_tr:]

    model = build_autoencoder(args.window, Xs.shape[-1])
    cb = [
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    ]
    model.fit(Xtr, Xtr, validation_split=0.15, epochs=args.epochs, batch_size=256, verbose=1, callbacks=cb)

    # reconstruction error per sequence (MSE over entire window)
    Xhat_tr = model.predict(Xtr, verbose=0)
    Xhat_te = model.predict(Xte, verbose=0)
    err_tr = np.mean((Xtr - Xhat_tr)**2, axis=(1,2))
    err_te = np.mean((Xte - Xhat_te)**2, axis=(1,2))

    if args.plot:
        import matplotlib.dates as mdates
        thr_p95 = float(np.quantile(err_tr, 0.95))
        plt.figure(figsize=(12,3))
        plt.plot(idx_te, err_te, label="test MSE")
        plt.axhline(thr_p95, color="r", ls="--", label="train 95%")
        plt.legend(); plt.title("Reconstruction error"); plt.tight_layout()
        plt.savefig("ob_anom_error.png"); print("Saved plot: ob_anom_error.png")

    # save package
    run_id = f"okx_ob_anom_{args.symbol}_step{args.step}_w{args.window}"
    model_name = args.model_name or run_id
    out_dir = Path(args.save_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- save thresholds learned on training errors ---
    thresh = {
        "mu": float(err_tr.mean()),
        "sigma": float(err_tr.std()),
        "quantiles": {
            "0.95": float(np.quantile(err_tr, 0.95)),
            "0.975": float(np.quantile(err_tr, 0.975)),
            "0.99": float(np.quantile(err_tr, 0.99)),
            "0.995": float(np.quantile(err_tr, 0.995)),
            "0.999": float(np.quantile(err_tr, 0.999)),
        }
    }
    with open(out_dir / "thresholds.json", "w") as f:
        json.dump(thresh, f, indent=2)

    model_path = out_dir / "model.keras"
    model.save(model_path.as_posix())
    joblib.dump(fs, out_dir / "feature_scaler.pkl")
    with open(out_dir / "features.json","w") as f: json.dump(cols, f, indent=2)
    with open(out_dir / "config.json","w") as f:
        json.dump({"symbol": args.symbol, "window": args.window, "step": args.step,
                   "start": args.start, "end": args.end}, f, indent=2)
    print(f"[model] saved to {model_path}")

if __name__ == "__main__":
    main()
