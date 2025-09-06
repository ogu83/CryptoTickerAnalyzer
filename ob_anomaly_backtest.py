#!/usr/bin/env python3
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import requests, joblib
import psycopg
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ---------- fetch & features (must match training) ----------
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
    out = out.dropna()
    out["mid"] = mid.reindex(out.index)      # keep for pricing
    out["open"] = out["mid"]                  # proxy (no separate open, so use mid as open/close)
    out["close"] = out["mid"]
    return out

def make_sequences(F: pd.DataFrame, cols: list, window: int):
    F = F[cols]
    V = F.values
    X = np.array([ V[t-window:t, :] for t in range(window, len(F)) ])
    idx = F.index[window:]
    return X, idx

# ---------- backtest ----------
def backtest(times, sig, open_series, close_series, init_usdt=10_000, fee_bps=2, slip_bps=1, hold=1, position_frac=0.2):
    fee = fee_bps/10000.0; slip = slip_bps/10000.0
    equity = init_usdt
    in_pos = False; pos_side = 0; exit_index = -1
    rows=[]
    open_s = open_series.reindex(times)
    close_s = close_series  # index aligned

    for i, t in enumerate(times):
        if not in_pos:
            s = int(sig[i])
            if s == 0:
                rows.append((t, equity, 0, np.nan, np.nan, 0.0)); continue
            p_open = float(open_s.iloc[i])
            notional = equity * position_frac
            qty = notional / p_open
            p_open_eff = p_open * (1 + slip if s==1 else 1 - slip)
            in_pos = True; pos_side = s; exit_index = min(i + hold - 1, len(times)-1)
            rows.append((t, equity, s, p_open, np.nan, 0.0))
        else:
            if i == exit_index:
                t_exit = t
                p_close = float(close_s.loc[t_exit])
                qty = (equity * position_frac) / float(open_s.iloc[i - (hold - 1)])
                p_close_eff = p_close * (1 - slip if pos_side==1 else 1 + slip)
                cost_entry = qty * float(open_s.iloc[i - (hold - 1)]) * (1 + slip if pos_side==1 else 1 - slip)
                cost_exit  = qty * p_close_eff
                if pos_side==1:
                    pnl = (cost_exit - cost_entry) - fee*(cost_entry+cost_exit)
                else:
                    pnl = (cost_entry - cost_exit) - fee*(cost_entry+cost_exit)
                equity += pnl
                rows.append((t_exit, equity, 0, np.nan, p_close, pnl))
                in_pos=False; pos_side=0; exit_index=-1
            else:
                rows.append((t, equity, pos_side, np.nan, np.nan, 0.0))
    bt = pd.DataFrame(rows, columns=["time","equity","signal","open","close","pnl"]).set_index("time")
    ret = bt["equity"].pct_change().fillna(0.0)
    roll_max = bt["equity"].cummax()
    drawdown = (bt["equity"] - roll_max)/roll_max
    stats = {
        "final_equity": float(bt["equity"].iloc[-1]),
        "total_return_pct": float((bt["equity"].iloc[-1]/bt["equity"].iloc[0]-1)*100.0),
        "max_drawdown_pct": float(drawdown.min()*100.0),
        "num_trades": int((bt["signal"].diff().abs()>0).sum()),
        "avg_trade_pnl": float(bt.loc[bt["pnl"].abs()>0,"pnl"].mean() if (bt["pnl"].abs()>0).any() else 0.0),
        "sharpe_like": float((ret.mean()/(ret.std()+1e-9))*np.sqrt(252*24*6)),
    }
    return bt, stats

def sanitize_for_json(obj):
    import math
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k,v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="models_ob_anom/... directory")
    ap.add_argument("--api", default="http://macbook-server:8200")
    ap.add_argument("--symbol", default="ETH-USDT")
    ap.add_argument("--start", default=None); ap.add_argument("--end", default=None)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--quantile", type=float, default=0.99, help="anomaly threshold on train error quantile")
    ap.add_argument("--dir-alpha", type=float, default=0.5, help="direction = sign(micro_tilt + alpha*imb)")
    ap.add_argument("--hold", type=int, default=2)
    ap.add_argument("--position-frac", type=float, default=0.1)
    ap.add_argument("--fee-bps", type=float, default=2.0)
    ap.add_argument("--slip-bps", type=float, default=1.0)
    ap.add_argument("--init-usdt", type=float, default=10_000)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    model = keras.models.load_model((model_dir / "model.keras").as_posix())
    fs: MinMaxScaler = joblib.load(model_dir / "feature_scaler.pkl")
    cols = json.loads((model_dir / "features.json").read_text())

    ob = fetch_ob(args.api, args.symbol, args.start, args.end, step=args.step)
    F = build_features(ob)  # includes mid/open/close at the end
    okx_open = F["open"]; okx_close = F["close"]
    feats = F[cols].dropna()
    # align
    F = F.reindex(feats.index)

    # make sequences
    window = json.loads((model_dir / "config.json").read_text())["window"]
    X_raw = np.array([ feats.values[t-window:t,:] for t in range(window, len(feats)) ])
    idx = feats.index[window:]

    Xs = fs.transform(X_raw.reshape(-1, X_raw.shape[-1])).reshape(X_raw.shape)
    Xhat = model.predict(Xs, verbose=0)
    err = np.mean((Xs - Xhat)**2, axis=(1,2))
    # training quantile not available here; approximate using in-sample lower chunk
    thr = float(np.quantile(err[:max(1000, len(err)//2)], args.quantile))
    print(f"[anomaly] threshold (q={args.quantile:.2f}) = {thr:.6e}")

    # direction: sign(micro_tilt + alpha*imb) at the last frame
    mt = F["micro_tilt"].values[window:]
    imb = F["imb"].values[window:]
    dir_sig = np.sign(mt + args.dir_alpha * imb)
    dir_sig = np.where(np.isnan(dir_sig), 0, dir_sig)

    sig = np.where(err > thr, dir_sig, 0)

    bt, stats = backtest(idx, sig, okx_open, okx_close,
                         init_usdt=args.init_usdt, fee_bps=args.fee_bps, slip_bps=args.slip_bps,
                         hold=args.hold, position_frac=args.position_frac)

    print("\n=== Backtest (anomaly-gated) ===")
    for k,v in stats.items():
        print(f"{k:>18}: {v:,.6f}")

    if args.plot:
        plt.figure(figsize=(12,4))
        plt.plot(bt.index, bt["equity"], label="Equity")
        plt.title(f"Equity Curve • {model_dir.name} • {args.symbol} • thr@q{args.quantile}")
        plt.xlabel("Time"); plt.ylabel("USDT"); plt.legend(); plt.tight_layout()
        plt.savefig("ob_anom_equity_curve.png"); print("Saved plot: ob_anom_equity_curve.png")

    # optional: write to DB (same table you used for ML backtests)
    try:
        dsn = "host=macbook-server port=5432 user=postgres password=Postgres2839* dbname=CryptoTickers"
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute("""
                create table if not exists ml_backtest (
                    id bigserial primary key,
                    created_at timestamptz default now(),
                    model_name text,
                    symbol text, period int,
                    start_time timestamptz, end_time timestamptz,
                    params jsonb, metrics jsonb
                );
            """)
            params = sanitize_for_json({
                "type": "ob_anomaly",
                "quantile": args.quantile,
                "dir_alpha": args.dir_alpha,
                "hold": args.hold,
                "position_frac": args.position_frac,
                "fee_bps": args.fee_bps,
                "slip_bps": args.slip_bps
            })
            metrics = sanitize_for_json(stats)
            cur.execute("""
                insert into ml_backtest (model_name, symbol, period, start_time, end_time, params, metrics)
                values (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
            """, (
                model_dir.name, args.symbol, 0,
                bt.index.min().to_pydatetime() if not bt.empty else None,
                bt.index.max().to_pydatetime() if not bt.empty else None,
                json.dumps(params, allow_nan=False),
                json.dumps(metrics, allow_nan=False),
            ))
            conn.commit()
            print("[db] backtest saved to ml_backtest")
    except Exception as e:
        print(f"[db] save skipped: {e}")

if __name__ == "__main__":
    main()
