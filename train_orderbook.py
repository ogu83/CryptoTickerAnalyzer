#!/usr/bin/env python3
import argparse, json, math, os
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import joblib
import psycopg

import lightgbm as lgb
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow import keras
from tensorflow.keras import layers

# ---------------- Postgres connection info (for registry) ----------------
PG_HOST = "macbook-server"
PG_PORT = 5432
PG_USER = "postgres"
PG_PASSWORD = "Postgres2839*"
PG_DBNAME = "CryptoTickers"  # target DB name

# --- helper: choose a good gate threshold on a validation tail ---
def _pick_gate_threshold(p, realized_bps, min_edge_bps, grid=None):
    """
    p:   classifier proba (for class=1)
    realized_bps: future realized return in bps (signed)
    min_edge_bps: dynamic hurdle in bps (>= costs)
    We pick thr maximizing expected net bps (rough, but practical).
    """
    if grid is None:
        grid = np.linspace(0.50, 0.95, 10)

    best_thr, best_score = 0.5, -1e9
    for thr in grid:
        mask = (p >= thr)
        if mask.sum() == 0:
            continue
        # crude net bps = directional return minus hurdle
        net = np.sign(realized_bps[mask]) * np.maximum(np.abs(realized_bps[mask]) - min_edge_bps[mask], 0.0)
        score = np.nanmean(net)
        if score > best_score:
            best_score, best_thr = score, thr
    return best_thr, best_score

# --- helper: OOF regression preds (walk-forward) ---
def _oof_reg_preds(X, y, times, n_splits=5, seed=42):
    """
    Time-ordered OOF predictions for the regressor.
    Returns: oof_pred (np.array), final_reg (fitted on full data).
    """
    splitter = TimeSeriesSplit(n_splits=n_splits)
    oof = np.full(len(y), np.nan, dtype=float)

    reg_params = dict(
        n_estimators=600, learning_rate=0.05,
        max_depth=-1, num_leaves=64,
        subsample=0.8, colsample_bytree=0.8,
        random_state=seed, n_jobs=-1
    )

    for tr, va in splitter.split(X):
        reg = LGBMRegressor(**reg_params)
        reg.fit(
            X.iloc[tr], y.iloc[tr],
            eval_set=[(X.iloc[va], y.iloc[va])],
            eval_metric="l2",
            callbacks=[lgb.log_evaluation(100)]
        )
        oof[va] = reg.predict(X.iloc[va])

    # Fit final reg on full data (optional export)
    final_reg = LGBMRegressor(**reg_params)
    final_reg.fit(
        X, y,
        eval_set=[(X, y)],
        eval_metric="l2",
        callbacks=[lgb.log_evaluation(200)]
    )

    return oof, final_reg


# ---------------- API fetch ----------------
def fetch_ob(api, symbol, start=None, end=None, step=1) -> pd.DataFrame:
    p = {"symbol": symbol, "step": step}
    if start: p["start"] = start
    if end:   p["end"] = end
    url = f"{api.rstrip('/')}/ob-top"
    r = requests.get(url, params=p, timeout=600); r.raise_for_status()
    js = r.json()
    if not js:
        raise SystemExit("No order book rows returned.")
    df = pd.DataFrame(js)
    df["time"] = pd.to_datetime(df["time"], utc=True, format="mixed")
    df = df.sort_values("time").set_index("time")
    for c in ["bid_px","ask_px","bid_sz","ask_sz","mid","spread","imbalance","microprice"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna()

# ---------------- features ----------------

def logret(x: pd.Series) -> pd.Series:
    return np.log(x).diff()

def atr_like_mid(mid: pd.Series, win=50) -> pd.Series:
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
    out["micro_tilt"] = (df["microprice"] - mid) / (spread.replace(0,np.nan))
    out["micro_tilt"] = out["micro_tilt"].replace([np.inf,-np.inf], 0.0).fillna(0.0)

    out["dt_sec"] = out.index.to_series().diff().dt.total_seconds().fillna(0.0).clip(0, 5.0)

    out["lr_mid"] = logret(mid).fillna(0.0)
    out["vol20_mid"] = out["lr_mid"].rolling(20, min_periods=10).std().bfill().fillna(0.0)

    for k in [1,2,3,5]:
        out[f"imb_lag{k}"]  = out["imb"].shift(k)
        out[f"tilt_lag{k}"] = out["micro_tilt"].shift(k)
        out[f"lr_mid_lag{k}"] = out["lr_mid"].shift(k)
        out[f"sbps_lag{k}"] = out["spread_bps"].shift(k)

    out["atr_mid"] = atr_like_mid(mid, win=50)
    return out

# ---------------- supervised frames ----------------

def make_supervised(feats: pd.DataFrame, target: pd.Series, window: int) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, List[str]]:
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

# ---------------- model ----------------

def build_model(window, n_features):
    inp = keras.Input(shape=(window, n_features))
    x = layers.LSTM(64, return_sequences=False, kernel_regularizer=keras.regularizers.l2(1e-5))(inp)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1)(x)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(1e-3, clipnorm=1.0), loss=keras.losses.Huber(delta=1.0))
    return m

# --- NEW: training path for a profitability gate ---
def train_lgbm_gate(F, feat_cols, symbol, step, out_dir, args):
    """
    F: feature dataframe you already build (one row per bar/step)
       must include:
         - 'time' (index or column), 'mid', 'spread_bps', 'atr14'
         - target columns from your builder: y_norm (mid_delta_norm) or y in price
    feat_cols: same columns used for the LGBM regressor
    """

    # 1) Targets and dynamic hurdle
    mid = F["mid"].astype(float)
    spread_bps = F["spread_bps"].astype(float).clip(lower=0)
    # realized future return (step ahead) in bps
    ret_bps = ((mid.shift(-step) - mid) / mid * 1e4).astype(float)

    # hurdle ~ fees+slip base + k*spread
    gate_base = getattr(args, "gate_base_bps", 7.0)
    k_spread = getattr(args, "k_spread", 0.3)
    min_edge_bps = gate_base + k_spread * spread_bps

    # 2) OOF regression preds as a feature to the classifier (edge estimate)
    #    y_reg = normalized delta (what you trained before)
    if "y_norm" in F.columns:
        y_reg = F["y_norm"].astype(float)
        # convert predicted norm back to bps using previous ATR (your feature name is atr_mid)
        atr_prev = F["atr_mid"].shift(1).astype(float)
        mid_prev = mid.shift(1)
        X_reg = F[feat_cols].copy()
        oof_norm, reg_full = _oof_reg_preds(X_reg, y_reg, F.index, n_splits=getattr(args, "cv", 5), seed=getattr(args, "seed", 42))
        # make it a Series aligned to F.index, handle div-by-zero / inf safely
        pred_edge_bps = pd.Series(oof_norm, index=F.index) * (atr_prev / mid_prev) * 1e4
        pred_edge_bps = pred_edge_bps.replace([np.inf, -np.inf], np.nan).fillna(0.0)    
    else:
        # Fallback: if y_norm not present, we’ll rely only on base features (still works, but weaker)
        pred_edge_bps = pd.Series(0.0, index=F.index)

    F["pred_edge_bps"] = pred_edge_bps
    F["abs_pred_edge_bps"] = pred_edge_bps.abs()
    X_cls = F[feat_cols + ["pred_edge_bps", "abs_pred_edge_bps"]].fillna(0.0)

    # 3) Profitability label for the gate (using realized returns & predicted direction)
    ok_dir = np.sign(pred_edge_bps).replace(0, np.nan) == np.sign(ret_bps).replace(0, np.nan)
    big_enough = (ret_bps.abs() >= min_edge_bps)
    y_gate = (ok_dir & big_enough).astype(int)

    # 4) Time split: train on head, pick threshold on tail
    split = int(len(F) * 0.8)
    X_tr, X_va = X_cls.iloc[:split], X_cls.iloc[split:]
    y_tr, y_va = y_gate.iloc[:split], y_gate.iloc[split:]
    ret_va, hurdle_va = ret_bps.iloc[split:], min_edge_bps.iloc[split:]

    cls = LGBMClassifier(
        n_estimators=400, learning_rate=0.05,
        num_leaves=64, max_depth=-1,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.0, reg_lambda=0.0,
        class_weight="balanced",
        random_state=getattr(args, "seed", 42),
        n_jobs=-1
    )
    cls.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="binary_logloss",
        callbacks=[lgb.log_evaluation(100)]
    )

    # 5) Pick a probability threshold that maximizes expected bps on the validation tail
    p_va = pd.Series(cls.predict_proba(X_va)[:, 1], index=X_va.index)
    thr_star, score_star = _pick_gate_threshold(p_va.values, ret_va.values, hurdle_va.values)

    # 6) Persist: model + metadata; keep your PNG too if desired
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(cls, out_dir / "gate_lgbm.pkl")

    meta = dict(
        symbol=symbol, step=step, algo="lgbm_gate",
        feat_cols=feat_cols + ["pred_edge_bps", "abs_pred_edge_bps"],
        gate_base_bps=float(gate_base), k_spread=float(k_spread),
        thr_star=float(thr_star)
    )
    with (out_dir / "gate_meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    # quick report to console
    ap = average_precision_score(y_va, p_va)
    auc = roc_auc_score(y_va, p_va)
    print(f"[gate] AUC={auc:.3f}  AP={ap:.3f}  thr*={thr_star:.3f} (val net-bps≈{score_star:.3f})")
    print(f"[model] gate saved to {out_dir.as_posix()}")


# ---------------- metrics ----------------
def evaluate(y_true_price, y_pred_price):
    mae  = float(np.mean(np.abs(y_true_price - y_pred_price)))
    rmse = float(np.sqrt(np.mean((y_true_price - y_pred_price)**2)))
    mape = float(np.mean(np.abs((y_true_price - y_pred_price) / np.maximum(1e-8, np.abs(y_true_price)))) * 100.0)
    return {"MAE": mae, "RMSE": rmse, "MAPE%": mape}

# ---------------- main ----------------
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

    # save package (like your multi-venue trainer)
    ap.add_argument("--save-dir", default="models_ob")
    ap.add_argument("--model-name", default=None)

    # model choice
    ap.add_argument("--algo", choices=["keras", "lgbm", "lgbm_gate"], default="lgbm")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--gate-base-bps", dest="gate_base_bps", type=float, default=7.0)
    ap.add_argument("--k-spread", type=float, default=0.3)

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

    if args.algo == "lgbm_gate":
        # out dir name mirrors your convention
        out_dir = f"models_ob/okx_ob_{args.symbol}_step{args.step}_gate_lgbm"
        feat_cols = [c for c in feats.columns if c not in ["mid"]]  # exclude 'mid' if needed
        train_lgbm_gate(feats, feat_cols, args.symbol, args.step, out_dir, args)
        return

    # --- LightGBM (tabular) path -----------------------------------------------
    if args.algo == "lgbm":
        # Build the supervised table: features at t, label for t->t+1
        # Reuse your feature builder outputs (includes lags & ATR-like already).
        mid = feats["mid"]
        if args.target == "mid":
            y_series = mid.shift(-1)
            def recon(prev_mid, yhat, prev_atr): return yhat
        elif args.target == "mid_delta":
            y_series = mid.shift(-1) - mid
            def recon(prev_mid, yhat, prev_atr): return prev_mid + yhat
        else:  # mid_delta_norm
            # IMPORTANT: normalize by ATR at t-1 so it's causal
            y_series = (mid.shift(-1) - mid) / (feats["atr_mid"].shift(1) + 1e-8)
            def recon(prev_mid, yhat, prev_atr): return prev_mid + yhat * (prev_atr + 1e-8)

        # Assemble dataset (drop rows where any feature/label is NaN)
        Z = pd.concat([feats, y_series.rename("y")], axis=1).dropna()
        feature_cols = [c for c in Z.columns if c not in ["y"]]  # keep 'mid' for reconstruction later

        # Chronological split 80/20
        n = len(Z)
        n_tr = int(n * 0.8)
        train_df = Z.iloc[:n_tr]
        test_df  = Z.iloc[n_tr:]

        X_tr = train_df[feature_cols].values
        y_tr = train_df["y"].values
        X_te = test_df[feature_cols].values
        y_te = test_df["y"].values
        idx_te = test_df.index

        # Model
        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=4000,
            learning_rate=0.03,
            num_leaves=127,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1e-2,
            reg_lambda=1e-2,
            random_state=42,
        )

        # Early stopping using last part of training as eval (time-safe)
        es_val_frac = 0.1
        n_es = int(len(X_tr) * (1 - es_val_frac))
        X_fit, X_val = X_tr[:n_es], X_tr[n_es:]
        y_fit, y_val = y_tr[:n_es], y_tr[n_es:]

        model.fit(
            X_fit, y_fit,
            eval_set=[(X_val, y_val)],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(100)]
        )

        # Predict label-space, then reconstruct to PRICE space
        y_pred = model.predict(X_te, num_iteration=model.best_iteration_)
        prev_mid = test_df["mid"].shift(0).to_numpy()        # mid at time t
        prev_atr = Z["atr_mid"].shift(1).reindex(idx_te).to_numpy()  # ATR at t-1 (matches y construction)

        y_pred_price = recon(prev_mid, y_pred, prev_atr)
        y_true_price = recon(prev_mid, y_te,   prev_atr)

        mask = np.isfinite(y_pred_price) & np.isfinite(y_true_price)
        y_pred_price = y_pred_price[mask]
        y_true_price = y_true_price[mask]
        idx_plot = idx_te[mask]

        # Metrics (your helper is fine too)
        mae  = float(np.mean(np.abs(y_true_price - y_pred_price)))
        rmse = float(np.sqrt(np.mean((y_true_price - y_pred_price)**2)))
        mape = float(np.mean(np.abs((y_true_price - y_pred_price) / np.maximum(1e-8, np.abs(y_true_price)))) * 100.0)

        print("\n=== LightGBM • Test Metrics (mid in price space) ===")
        print(f"{'MAE':>10}: {mae:,.6f}")
        print(f"{'RMSE':>10}: {rmse:,.6f}")
        print(f"{'MAPE%':>10}: {mape:,.6f}")

        # Plot
        plt.figure(figsize=(12,4))
        plt.plot(idx_plot, y_true_price, label="Actual")
        plt.plot(idx_plot, y_pred_price, label="Predicted", alpha=0.85)
        plt.title(f"OKX {args.symbol} • step={args.step} • LightGBM • target={args.target}")
        plt.xlabel("Time"); plt.ylabel("Mid"); plt.legend(); plt.tight_layout()
        plt.savefig("ob_pred_vs_actual.png")
        print("Saved plot: ob_pred_vs_actual.png")

        # Save package (parallel to your Keras saver)
        run_id = f"okx_ob_{args.symbol}_step{args.step}_{args.target}_lgbm"
        model_name = args.model_name or run_id
        out_dir = Path(args.save_dir) / model_name
        out_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, out_dir / "model_lgbm.pkl")
        with open(out_dir / "features.json", "w") as f:
            json.dump(feature_cols, f, indent=2)
        with open(out_dir / "config.json", "w") as f:
            json.dump({
                "algo": "lgbm",
                "symbol": args.symbol,
                "target": args.target,
                "step": args.step,
                "start": args.start,
                "end": args.end
            }, f, indent=2)
        print(f"[model] saved to {out_dir/'model_lgbm.pkl'}")

        # Optional: register to Postgres (reuse your existing registry code)
        try:
            dsn = f"host={PG_HOST} port={PG_PORT} user={PG_USER} password={PG_PASSWORD} dbname={PG_DBNAME}"
            with psycopg.connect(dsn) as conn, conn.cursor() as cur:
                cur.execute("""
                    create table if not exists ml_model_registry (
                        id bigserial primary key,
                        created_at timestamptz default now(),
                        model_name text unique,
                        path text,
                        params jsonb,
                        metrics jsonb
                    );
                """)
                cur.execute("""
                    insert into ml_model_registry (model_name, path, params, metrics)
                    values (%s, %s, %s::jsonb, %s::jsonb)
                    on conflict (model_name) do update set
                        path = EXCLUDED.path,
                        params = EXCLUDED.params,
                        metrics = EXCLUDED.metrics
                """, (
                    model_name,
                    str(out_dir),
                    json.dumps({
                        "algo": "lgbm",
                        "symbol": args.symbol, "target": args.target,
                        "step": args.step, "start": args.start, "end": args.end
                    }, allow_nan=False),
                    json.dumps({"MAE": mae, "RMSE": rmse, "MAPE%": mape}, allow_nan=False)
                ))
                conn.commit()
            print(f"[db] model registered as '{model_name}'")
        except Exception as e:
            print(f"[db] registry skipped: {e}")

        return  # prevent falling into the LSTM branch
        # --- end LightGBM path ------------------------------------------------------
    else:
        model = build_model(window=args.window, n_features=X.shape[-1])
        cb = [
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
        ]
        model.fit(Xtr_s, Ytr_s, validation_split=0.15, epochs=args.epochs, batch_size=256, verbose=1, callbacks=cb)

        Yhat_s = model.predict(Xte_s, verbose=0)
        y_pred = ys.inverse_transform(Yhat_s).flatten()
        y_true = Yte.flatten()

        # --- Reconstruct to PRICE space (robust) ---
        prev_mid = mid.reindex(idx_te).shift(1).ffill().bfill().to_numpy()
        prev_atr = feats["atr_mid"].reindex(idx_te).shift(1).ffill().bfill().to_numpy()

        y_pred_price = recon(prev_mid, y_pred, prev_atr)
        y_true_price = recon(prev_mid, y_true, prev_atr)

        # Drop any remaining non-finite values before computing metrics
        mask = np.isfinite(y_pred_price) & np.isfinite(y_true_price)
        y_pred_price = y_pred_price[mask]
        y_true_price = y_true_price[mask]

        m = evaluate(y_true_price, y_pred_price)
        print("\n=== Test Metrics (mid in price space) ===")
        for k,v in m.items():
            print(f"{k:>10}: {v:,.6f}")

        if args.plot:
            plt.figure(figsize=(12,4))
            plt.plot(idx_te[mask], y_true_price, label="Actual")
            plt.plot(idx_te[mask], y_pred_price, label="Predicted", alpha=0.8)
            plt.title(f"OKX {args.symbol} • next mid forecast (target={args.target})")
            plt.xlabel("Time"); plt.ylabel("Mid"); plt.legend(); plt.tight_layout()
            plt.savefig("ob_pred_vs_actual.png")
            print("Saved plot: ob_pred_vs_actual.png")

        # -------- save package (same pattern as multi-venue) --------
        run_id = f"okx_ob_{args.symbol}_step{args.step}_{args.target}_w{args.window}"
        model_name = args.model_name or run_id
        out_dir = Path(args.save_dir) / model_name
        out_dir.mkdir(parents=True, exist_ok=True)

        model_path = out_dir / "model.keras"
        model.save(model_path.as_posix())
        print(f"[model] saved to {model_path}")

        joblib.dump(fs, out_dir / "feature_scaler.pkl")
        joblib.dump(ys, out_dir / "target_scaler.pkl")

        with open(out_dir / "features.json", "w") as f:
            json.dump(cols, f, indent=2)

        with open(out_dir / "config.json", "w") as f:
            json.dump({
                "symbol": args.symbol,
                "target": args.target,
                "window": args.window,
                "step": args.step,
                "start": args.start,
                "end": args.end
            }, f, indent=2)

        # optional: register in Postgres
        try:
            dsn = f"host={PG_HOST} port={PG_PORT} user={PG_USER} password={PG_PASSWORD} dbname={PG_DBNAME}"
            with psycopg.connect(dsn) as conn, conn.cursor() as cur:
                cur.execute("""
                    create table if not exists ml_model_registry (
                        id bigserial primary key,
                        created_at timestamptz default now(),
                        model_name text unique,
                        path text,
                        params jsonb,
                        metrics jsonb
                    );
                """)
                cur.execute("""
                    insert into ml_model_registry (model_name, path, params, metrics)
                    values (%s, %s, %s::jsonb, %s::jsonb)
                    on conflict (model_name) do update set
                        path = EXCLUDED.path,
                        params = EXCLUDED.params,
                        metrics = EXCLUDED.metrics
                """, (
                    model_name,
                    str(out_dir),
                    json.dumps({
                        "symbol": args.symbol, "target": args.target,
                        "window": args.window, "step": args.step,
                        "start": args.start, "end": args.end
                    }, allow_nan=False),
                    json.dumps(m, allow_nan=False)
                ))
                conn.commit()
            print(f"[db] model registered as '{model_name}'")
        except Exception as e:
            print(f"[db] registry skipped: {e}")

if __name__ == "__main__":
    main()
