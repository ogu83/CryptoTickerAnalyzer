#!/usr/bin/env python3
"""
Signal generation + backtest using a saved model package.

Usage:
  python3 signal_backtest.py --model-dir models/okx_ETH-USDT_10_delta_w64 \
      --api http://macbook-server:8200 --symbol ETH-USDT --period 10 \
      --start 2025-08-28T00:00:00Z --end 2025-08-30T00:00:00Z \
      --threshold 0.15 --long-only --init-usdt 10000 --fee-bps 2 --slip-bps 1 --plot

Notes:
- Works with models trained on 'price', 'delta', or 'delta_norm'.
- Uses your same feature pipeline and causal alignment.
- Trades at next bar OPEN, exits same bar at CLOSE (one-bar holding).
"""

import os, json, math, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
import psycopg

# ---------------- Postgres connection info ----------------
PG_HOST = "macbook-server"
PG_PORT = 5432
PG_USER = "postgres"
PG_PASSWORD = "Postgres2839*"
PG_DBNAME = "CryptoTickers"   # target DB name

# ----------------- reuse key feature functions -----------------
def logret(x: pd.Series) -> pd.Series:
    return np.log(x).diff().fillna(0.0)

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0); down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    return (100 - 100/(1 + rs)).fillna(50.0)

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr

def atr(high, low, close, win=14):
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1/win, adjust=False).mean().bfill().fillna(0.0)

def rolling_vol(series, win):
    return series.rolling(win, min_periods=max(2, win//2)).std().bfill().fillna(0.0)

def zscore(series: pd.Series, win: int) -> pd.Series:
    m = series.rolling(win, min_periods=max(2, win//2)).mean()
    s = series.rolling(win, min_periods=max(2, win//2)).std()
    return ((series - m) / (s.replace(0, np.nan))).fillna(0.0)

def rolling_corr(x: pd.Series, y: pd.Series, win: int) -> pd.Series:
    return x.rolling(win, min_periods=max(2, win//2)).corr(y).fillna(0.0)

def fetch(api_base, venue, symbol, period, start=None, end=None) -> pd.DataFrame:
    params = {"venue": venue, "symbol": symbol, "period": period}
    if start: params["start"] = start
    if end:   params["end"] = end
    url = f"{api_base.rstrip('/')}/tick-chart"
    r = requests.get(url, params=params, timeout=90); r.raise_for_status()
    data = r.json()
    if not data: raise ValueError(f"{venue} returned no rows.")
    df = pd.DataFrame(data)
    t = pd.Series(df["time"], dtype="string")
    try:   df["time"] = pd.to_datetime(t, utc=True, format="ISO8601")
    except: df["time"] = pd.to_datetime(t, utc=True, format="mixed")
    df = df.sort_values("time").set_index("time")
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna()

def align_okx_binance(df_okx, df_bnc, tolerance="2s", max_ffill="5s"):
    left  = df_okx.reset_index().rename(columns={"time": "time"}).sort_values("time")
    right = df_bnc.reset_index().rename(columns={"time": "time"}).sort_values("time")
    m = pd.merge_asof(left, right, on="time", direction="backward",
                      tolerance=pd.Timedelta(tolerance), suffixes=("_okx","_bnc"))
    m["bnc_age_sec"] = (m["time"] - m["time"].where(m["close_bnc"].notna()).ffill()).dt.total_seconds()
    max_age = pd.Timedelta(max_ffill).total_seconds()
    bnc_cols = [c for c in m.columns if c.endswith("_bnc")]
    m[bnc_cols] = m[bnc_cols].ffill()
    m = m[m["bnc_age_sec"].fillna(max_age) <= max_age]
    return m.set_index("time")

def build_features(m: pd.DataFrame) -> pd.DataFrame:
    df = m.copy()
    # base
    df["lr_okx"] = logret(df["close_okx"]); df["lr_bnc"] = logret(df["close_bnc"])
    # vol & ranges
    df["vol20_okx"] = rolling_vol(df["lr_okx"], 20); df["vol20_bnc"] = rolling_vol(df["lr_bnc"], 20)
    df["atr14_okx"] = atr(df["high_okx"], df["low_okx"], df["close_okx"], 14)
    df["range_rel_okx"] = (df["high_okx"] - df["low_okx"]) / df["close_okx"]
    df["co_ret_okx"] = (df["close_okx"] - df["open_okx"]) / df["open_okx"]
    df["co_ret_bnc"] = (df["close_bnc"] - df["open_bnc"]) / df["open_bnc"]
    # lags
    df["lr_okx_lag1"] = df["lr_okx"].shift(1); df["lr_okx_lag2"] = df["lr_okx"].shift(2)
    df["vol20_okx_lag1"] = df["vol20_okx"].shift(1)
    # momentum
    df["rsi14_okx"] = rsi(df["close_okx"], 14)
    df["ema12_okx"] = ema(df["close_okx"], 12); df["ema26_okx"] = ema(df["close_okx"], 26)
    df["ema12_bnc"] = ema(df["close_bnc"], 12); df["ema26_bnc"] = ema(df["close_bnc"], 26)
    # cross-venue
    df["spread"] = df["close_okx"] - df["close_bnc"]
    df["spread_z50"] = zscore(df["spread"], 50)
    df["corr20"] = rolling_corr(df["lr_okx"], df["lr_bnc"], 20)
    df["lr_bnc_lag1"] = df["lr_bnc"].shift(1)
    df["vol20_bnc_lag1"] = df["vol20_bnc"].shift(1)
    df["co_ret_bnc_lag1"] = df["co_ret_bnc"].shift(1)
    # volume z
    df["vol_z_okx"] = zscore(df["volume_okx"], 50)
    df["vol_z_bnc"] = zscore(df["volume_bnc"], 50)

    return df

def supervised_matrix(feats: pd.DataFrame, cols_in_order, window: int):
    F = feats[cols_in_order].values
    Xs = []
    for t in range(window, len(feats)):
        Xs.append(F[t-window:t, :])
    X = np.array(Xs)
    # target timestamps (for trade timing)
    idx = feats.index[window:]
    return X, idx

def sanitize_for_json(obj):
    import math
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj

# ----------------- trade simulator -----------------
def backtest_long_short(times, sig, okx_open_series, okx_close_series,
                        init_usdt=10000, fee_bps=2.0, slip_bps=0.0,
                        hold=1, position_frac=0.2):
    """
    Executes signals with a fixed holding period:
      - At time t: if sig[t] != 0 and we're flat, ENTER at open_t
      - EXIT after 'hold' bars at close_{t+hold-1} (or last available)
      - Ignores overlapping signals while in a position
    PnL includes fees and symmetric slippage on entry+exit.
    """
    fee = fee_bps / 10000.0
    slip = slip_bps / 10000.0

    equity = init_usdt
    in_pos = False
    pos_side = 0
    exit_index = -1

    rows = []

    # For fast access:
    open_s = okx_open_series.reindex(times)
    close_s = okx_close_series  # will index at exit time

    for i, t in enumerate(times):
        if not in_pos:
            s = int(sig[i])
            if s == 0:
                rows.append((t, equity, 0, np.nan, np.nan, 0.0))
                continue

            # ENTER
            p_open = float(open_s.iloc[i])
            notional = equity * position_frac
            qty = notional / p_open

            # effective prices with slippage
            p_open_eff = p_open * (1 + slip if s == 1 else 1 - slip)

            # book the entry cashflow immediately
            cost_entry = qty * p_open_eff
            # we don't change equity on entry; we realize PnL at exit (easier book-keeping)
            in_pos = True
            pos_side = s
            exit_index = min(i + hold - 1, len(times) - 1)
            rows.append((t, equity, s, p_open, np.nan, 0.0))
        else:
            # Are we exiting at this timestamp?
            if i == exit_index:
                t_exit = t
                p_close = float(close_s.loc[t_exit])
                qty = (equity * position_frac) / float(open_s.iloc[i - (hold - 1)])  # approximate same qty as entry
                p_close_eff = p_close * (1 - slip if pos_side == 1 else 1 + slip)

                cost_entry = qty * float(open_s.iloc[i - (hold - 1)]) * (1 + slip if pos_side == 1 else 1 - slip)
                cost_exit = qty * p_close_eff

                if pos_side == 1:  # long
                    pnl = (cost_exit - cost_entry) - fee * (cost_entry + cost_exit)
                else:              # short
                    pnl = (cost_entry - cost_exit) - fee * (cost_entry + cost_exit)

                equity += pnl
                rows.append((t_exit, equity, 0, np.nan, p_close, pnl))

                in_pos = False
                pos_side = 0
                exit_index = -1
            else:
                rows.append((t, equity, pos_side, np.nan, np.nan, 0.0))

    bt = pd.DataFrame(rows, columns=["time","equity","signal","open","close","pnl"]).set_index("time")
    # metrics
    ret = bt["equity"].pct_change().fillna(0.0)
    roll_max = bt["equity"].cummax()
    drawdown = (bt["equity"] - roll_max) / roll_max
    stats = {
        "final_equity": float(bt["equity"].iloc[-1]),
        "total_return_pct": float((bt["equity"].iloc[-1] / bt["equity"].iloc[0] - 1.0) * 100.0),
        "max_drawdown_pct": float(drawdown.min() * 100.0),
        "num_trades": int((bt["signal"].diff().abs() > 0).sum()),  # rough count of entries
        "avg_trade_pnl": float(bt.loc[bt["pnl"].abs() > 0, "pnl"].mean() if (bt["pnl"].abs() > 0).any() else 0.0),
        "sharpe_like": float((ret.mean() / (ret.std() + 1e-9)) * np.sqrt(252*24*6)),
    }
    return bt, stats

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="Folder with saved Keras model + scalers + config/features")
    ap.add_argument("--api", default="http://macbook-server:8200")
    ap.add_argument("--symbol", default=None, help="Override symbol; else use model config")
    ap.add_argument("--period", type=int, default=None, help="Override period; else use model config")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--long-only", action="store_true")
    ap.add_argument("--init-usdt", type=float, default=10000.0)
    ap.add_argument("--fee-bps", type=float, default=2.0)
    ap.add_argument("--slip-bps", type=float, default=1.0)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--position-frac", type=float, default=0.2)  # was 1.0
    ap.add_argument("--signal-mode", choices=["open_ret", "norm_atr"], default="open_ret",
                    help="Signal gating mode: predicted return vs open (recommended) or normalized delta vs ATR")
    ap.add_argument("--auto-threshold", action="store_true",
                    help="Pick threshold as a quantile of |signal| (open_ret or norm_atr)")
    ap.add_argument("--quantile", type=float, default=0.9,
                    help="Quantile used when --auto-threshold is on")
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="Manual threshold; interpretation depends on --signal-mode "
                        "(fraction for open_ret, not bps; e.g., 0.0005 = 5 bps)")
    ap.add_argument("--min-edge-bps", type=float, default=5.0,
                    help="Require predicted edge to exceed this AND costs (bps); e.g., 5 = 0.05%)")
    ap.add_argument("--hold", type=int, default=1,
                    help="Holding period in bars (>=1). 1 = enter at open_t, exit at close_t")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    # Prefer native Keras format
    if (model_dir / "model.keras").exists():
        model = keras.models.load_model((model_dir / "model.keras").as_posix())
    elif (model_dir / "model.h5").exists():
        model = keras.models.load_model((model_dir / "model.h5").as_posix())
    else:
        raise FileNotFoundError(
            f"Could not find model file in {model_dir}. "
            "Expected 'model.keras' (or 'model.h5')."
        )

    fsc = joblib.load(model_dir / "feature_scaler.pkl")
    ysc = joblib.load(model_dir / "target_scaler.pkl")
    features = json.loads((model_dir / "features.json").read_text())
    cfg = json.loads((model_dir / "config.json").read_text())

    symbol = args.symbol or cfg["symbol"]
    period = args.period or cfg["period"]
    target_mode = cfg["target"]
    tolerance = cfg.get("tolerance", "2s")
    window = cfg["window"]

    print(f"[load] {model_dir.name}  symbol={symbol} period={period} target={target_mode} window={window}")

    okx = fetch(args.api, "okx", symbol, period, args.start, args.end)
    bnc = fetch(args.api, "bnc", symbol, period, args.start, args.end)
    m = align_okx_binance(okx, bnc, tolerance)

    feats_full = build_features(m).dropna()
    if len(feats_full) <= window + 1:
        raise SystemExit("Not enough rows after feature building.")

    # Build supervised matrix (no target here; we just need sequences & timestamps)
    X, idx_pred = supervised_matrix(feats_full, features, window)

    # scale with TRAIN-TIME scalers to avoid leakage in backtest
    Xs = fsc.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    y_pred_s = model.predict(Xs, verbose=0)
    y_pred = ysc.inverse_transform(y_pred_s).flatten()

    # reconstruct PREDICTED PRICE at each timestamp
    # Align prev close & prev ATR for normalization and price reconstruction
    arr_close_okx = okx["close"].to_numpy()
    pos = okx.index.get_indexer(idx_pred, method="nearest")
    prev_pos = np.clip(pos - 1, 0, len(arr_close_okx) - 1)
    prev_close = arr_close_okx[prev_pos]

    atr_okx_series = atr(okx["high"], okx["low"], okx["close"], 14)
    prev_atr = atr_okx_series.to_numpy()[prev_pos]

    if target_mode == "price":
        pred_price = y_pred
    elif target_mode == "delta":
        pred_price = prev_close + y_pred
    else:  # delta_norm
        pred_price = prev_close + y_pred * (prev_atr + 1e-8)

    # Align arrays to prediction timestamps
    open_t = okx["open"].reindex(idx_pred).to_numpy()
    atr_okx_series = atr(okx["high"], okx["low"], okx["close"], 14)
    prev_close = okx["close"].reindex(idx_pred).shift(1).ffill().to_numpy()
    prev_atr   = atr_okx_series.reindex(idx_pred).shift(1).ffill().to_numpy()   

    # Predicted open-return (this is what we monetize)
    open_ret_pred = (pred_price - open_t) / open_t  # dimensionless (e.g., 0.0005 = +5 bps)

    absr = np.abs(open_ret_pred)
    pcts = np.percentile(absr, [50, 75, 90, 95, 97, 99])  # bps = value*1e4
    print("[debug] |open_ret_pred| percentiles (bps):", (pcts*1e4).round(3).tolist())

    # Total round-trip cost in fraction (fees+slip both sides) plus a safety buffer
    cost_frac = ((args.fee_bps + args.slip_bps) * 2.0) / 10000.0
    buffer_frac = (args.min_edge_bps / 10000.0)

    if args.signal_mode == "open_ret":
        signal_series = open_ret_pred.copy()
    else:  # norm_atr (kept for experiments)
        signal_series = (pred_price - prev_close) / (prev_atr + 1e-8)

    # Pick threshold
    if args.auto_threshold:
        thr = float(np.quantile(np.abs(signal_series), args.quantile))
    else:
        thr = float(args.threshold)

    # Final gate: must beat BOTH the quantile/manual threshold AND the cost+buffer
    gate = np.maximum(thr, cost_frac + buffer_frac)

    sig = np.where(signal_series > gate,  1,
        np.where(signal_series < -gate, -1, 0))
    if args.long_only:
        sig = np.where(sig == 1, 1, 0)

    hit_rate = float((sig != 0).mean())
    print(f"[signal] mode={args.signal_mode}  thr={thr:.6f}  cost+buffer={cost_frac+buffer_frac:.6f}  trade_rate={hit_rate:.3f}")

    # normalized prediction used for signal
    pred_norm = (pred_price - prev_close) / (prev_atr + 1e-8)

    if args.auto_threshold:
        thr = float(np.quantile(np.abs(pred_norm), args.quantile))
    else:
        thr = args.threshold

    rate = float((np.abs(pred_norm) > thr).mean())
    # print(f"[signal] threshold={thr:.6f}  hit_rate(|pred_norm|>thr)={rate:.3f}")
    print(f"[signal] mode={args.signal_mode}  thr={thr:.6f}  cost+buffer={cost_frac+buffer_frac:.6f}  trade_rate={hit_rate:.3f}")

    # simulate one-bar trades with signals decided at end of t-1
    okx_open_s = okx["open"]
    okx_close_s = okx["close"]
    bt, stats = backtest_long_short(
        times=idx_pred,
        sig=sig,
        okx_open_series=okx_open_s,
        okx_close_series=okx_close_s,
        init_usdt=args.init_usdt,
        fee_bps=args.fee_bps,
        slip_bps=args.slip_bps,
        hold=args.hold,
        position_frac=args.position_frac
    )


    print("\n=== Backtest summary ===")
    for k, v in stats.items():
        print(f"{k:>18}: {v:,.6f}" if isinstance(v, float) else f"{k:>18}: {v}")

    if args.plot:
        plt.figure(figsize=(12,5))
        plt.plot(bt.index, bt["equity"], label="Equity")
        plt.title(f"Equity Curve • {model_dir.name} • {symbol} • P={period} • thr={args.threshold}")
        plt.xlabel("Time"); plt.ylabel("USDT"); plt.legend(); plt.tight_layout()
        plt.savefig("equity_curve.png")
        print("Saved plot: equity_curve.png")

    params_json = sanitize_for_json({
        "threshold": thr,
        "long_only": args.long_only,
        "init_usdt": args.init_usdt,
        "fee_bps": args.fee_bps,
        "slip_bps": args.slip_bps,
        "position_frac": args.position_frac
    })
    metrics_json = sanitize_for_json(stats)

    # -------- save summary to Postgres ----------
    with psycopg.connect(f"host={PG_HOST} port={PG_PORT} user={PG_USER} password={PG_PASSWORD} dbname={PG_DBNAME}") as conn, conn.cursor() as cur:
        cur.execute("""
            create table if not exists ml_backtest (
                id bigserial primary key,
                created_at timestamptz default now(),
                model_name text,
                symbol text,
                period int,
                start_time timestamptz,
                end_time timestamptz,
                params jsonb,
                metrics jsonb
            );
        """)
        cur.execute("""
            insert into ml_backtest (model_name, symbol, period, start_time, end_time, params, metrics)
            values (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
        """, (
            model_dir.name,
            symbol, period,
            bt.index.min().to_pydatetime() if not bt.empty else None,
            bt.index.max().to_pydatetime() if not bt.empty else None,
            json.dumps(params_json, allow_nan=False),
            json.dumps(metrics_json, allow_nan=False)
        ))
        conn.commit()

        print("[db] backtest summary saved to ml_backtest")

        if stats["num_trades"] == 0:
            print("[warn] No trades fired. Try --threshold 0.01..0.05 or --auto-threshold --quantile 0.85")
        
if __name__ == "__main__":
    main()
