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

def logret(x: pd.Series): 
    return np.log(x).diff()

def atr_like_mid(mid: pd.Series, win=50):
    lr = logret(mid).abs()
    return lr.ewm(alpha=1/win, adjust=False).mean().bfill().fillna(0.0)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    mid, spread = df["mid"], df["spread"]

    out["spread_bps"] = (spread / mid * 1e4).clip(lower=0).fillna(0.0)
    out["imb"] = df["imbalance"].clip(-1, 1).fillna(0.0)

    out["micro_tilt"] = (df["microprice"] - mid) / (spread.replace(0, np.nan))
    out["micro_tilt"] = out["micro_tilt"].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    out["dt_sec"] = out.index.to_series().diff().dt.total_seconds().fillna(0.0).clip(0, 5.0)

    out["lr_mid"] = np.log(mid).diff().fillna(0.0)
    out["vol20_mid"] = out["lr_mid"].rolling(20, min_periods=10).std().bfill().fillna(0.0)

    for k in [1, 2, 3, 5]:
        out[f"imb_lag{k}"]   = out["imb"].shift(k)
        out[f"tilt_lag{k}"]  = out["micro_tilt"].shift(k)
        out[f"lr_mid_lag{k}"] = out["lr_mid"].shift(k)
        out[f"sbps_lag{k}"]  = out["spread_bps"].shift(k)

    out["atr_mid"] = (np.log(mid).diff().abs().ewm(alpha=1/50, adjust=False).mean()
                      .bfill().fillna(0.0))

    out["spread"] = spread  # <<< add this line so 'spread' exists later

    out = out.dropna()
    out["mid"] = mid.reindex(out.index)
    out["open"] = out["mid"]
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
    ap.add_argument("--usequantile", action="store_true", help="Use saved train quantile threshold")
    ap.add_argument("--z-thr", type=float, default=3.0, help="Z-score threshold if not using quantile")
    ap.add_argument("--consec", type=int, default=3, help="require N consecutive anomaly bars")
    ap.add_argument("--cooldown", type=int, default=10, help="bars to wait after closing a trade")
    ap.add_argument("--spread-cap-bps", type=float, default=5.0, help="skip bars with spread above this")
    ap.add_argument("--long-only", action="store_true")
    ap.add_argument("--short-only", action="store_true")
    ap.add_argument("--tilt-ema", type=int, default=3, help="EMA window for micro_tilt direction")
    ap.add_argument("--reg-model-dir", default=None,
                help="Optional: models_ob/... directory of an OB regressor (e.g., okx_ob_ETH-USDT_step5_mid_delta_norm_w64)")
    ap.add_argument("--min-edge-bps", type=float, default=3.0,
                    help="Minimum predicted edge (bps) beyond fee+slippage+buffer")
    ap.add_argument("--edge-buffer-bps", type=float, default=1.0,
                    help="Extra buffer to clear microstructure noise")
    ap.add_argument("--max-trades-per-hour", type=int, default=None,
                help="Keep only the top-|edge| anomalies per hour after all gates")


    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    model = keras.models.load_model((model_dir / "model.keras").as_posix())
    fs: MinMaxScaler = joblib.load(model_dir / "feature_scaler.pkl")
    cols = json.loads((model_dir / "features.json").read_text())

    ob = fetch_ob(args.api, args.symbol, args.start, args.end, step=args.step)
    F = build_features(ob)  # includes mid/open/close at the end
    okx_open = F["open"]; okx_close = F["close"]
    missing = [c for c in cols if c not in F.columns]
    if missing:
        raise SystemExit(f"Model expects features {missing} that are not present. "
                        f"Available columns: {list(F.columns)}")

    feats = F[cols].dropna()    # align
    F = F.reindex(feats.index)

    # make sequences
    window = json.loads((model_dir / "config.json").read_text())["window"]
    X_raw = np.array([ feats.values[t-window:t,:] for t in range(window, len(feats)) ])
    idx = feats.index[window:]

    thr_file = model_dir / "thresholds.json"
    saved = json.loads(thr_file.read_text()) if thr_file.exists() else None

    Xs = fs.transform(X_raw.reshape(-1, X_raw.shape[-1])).reshape(X_raw.shape)
    Xhat = model.predict(Xs, verbose=0)
    err = np.mean((Xs - Xhat)**2, axis=(1,2))

    if args.usequantile and saved:
        qkey = f"{args.quantile:.3f}"
        thr = float(saved["quantiles"].get(qkey, np.quantile(err, args.quantile)))
    else:
        mu  = (saved["mu"] if saved else float(err.mean()))
        sig = (saved["sigma"] if saved else float(err.std() + 1e-12))
        thr = mu + args.z_thr * sig

    print(f"[anomaly] thr = {thr:.6e}  (source={'train-quantile' if args.usequantile and saved else 'z-score'})")
    anom = err > thr

    # direction: sign(micro_tilt + alpha*imb) at the last frame
    mt = F["micro_tilt"].values[window:]
    imb = F["imb"].values[window:]

    # EMA of micro_tilt for a more stable direction
    mt_series = pd.Series(F["micro_tilt"].values, index=F.index).ewm(span=args.tilt_ema, adjust=False).mean()
    mt_s = mt_series.values[window:]
    dir_sig = np.sign(mt_s + args.dir_alpha * imb)
    dir_sig = np.where(np.isnan(dir_sig), 0, dir_sig)

    # optional long/short only
    if args.long_only:
        dir_sig = np.where(dir_sig < 0, 0, dir_sig)
    elif args.short_only:
        dir_sig = np.where(dir_sig > 0, 0, dir_sig)

    # cost guard: skip wide-spread bars
    spread_bps = (np.array(F["spread"].values[window:], dtype=float) / np.array(F["mid"].values[window:], dtype=float) * 1e4)
    ok_spread = spread_bps <= args.spread_cap_bps

    sig = np.where(err > thr, dir_sig, 0)

    # require N consecutive anomalies
    if args.consec > 1:
        from numpy.lib.stride_tricks import sliding_window_view as swv
        w = min(args.consec, len(anom))
        if w > 1:
            sw = swv(anom.astype(int), window_shape=w)
            confirmed = (sw.sum(axis=1) == w).astype(bool)
            anom_confirmed = np.r_[np.zeros(w-1, dtype=bool), confirmed]
        else:
            anom_confirmed = anom
    else:
        anom_confirmed = anom

    raw_sig = np.where(anom_confirmed & ok_spread, dir_sig, 0)

    # cooldown: prevent immediate re-entries
    sig = raw_sig.copy()
    cool = 0
    for i in range(len(sig)):
        if cool > 0:
            sig[i] = 0
            cool -= 1
            continue
        if sig[i] != 0:
            # will hold inside backtest; add external cooldown after exit
            cool = 0  # leave at 0; cooldown is applied after realized trades in backtest

    edge_ser = np.zeros_like(sig, dtype=float)  # Ensure edge_ser is always defined

    if args.reg_model_dir:
        reg_dir = Path(args.reg_model_dir)
        # load regressor package
        reg = keras.models.load_model((reg_dir / "model.keras").as_posix())
        reg_scaler = joblib.load(reg_dir / "feature_scaler.pkl")
        reg_cols = json.loads((reg_dir / "features.json").read_text())
        reg_cfg  = json.loads((reg_dir / "config.json").read_text())
        reg_window = int(reg_cfg["window"])

        # build regressor sequences on the SAME features frame F
        # (make sure reg_cols are present; if not, fail quick)
        missing = [c for c in reg_cols if c not in F.columns]
        if missing:
            raise SystemExit(f"Regressor expects features {missing} not present in F.")

        Fr = F[reg_cols].dropna()
        Vr = Fr.values
        if len(Fr) <= reg_window:
            raise SystemExit("Not enough rows to build regressor sequences.")
        Xr = np.array([Vr[t-reg_window:t, :] for t in range(reg_window, len(Fr))])
        idx_r = Fr.index[reg_window:]

        # scale and predict normalized delta (your OB model outputs next delta_norm)
        Xr_s = reg_scaler.transform(Xr.reshape(-1, Xr.shape[-1])).reshape(Xr.shape)
        y_hat = reg.predict(Xr_s, verbose=0).squeeze()  # shape (N,)

        # convert to predicted edge in bps at the prior bar (causal)
        atr_prev = F["atr_mid"].loc[idx_r].shift(1).reindex(idx_r).values
        mid_prev = F["mid"].loc[idx_r].shift(1).reindex(idx_r).values
        edge_bps = y_hat * atr_prev * 1e4
        finite = np.isfinite(edge_bps)
        print(
            f"[diag] edge_bps abs p50={np.nanmedian(np.abs(edge_bps)):.3f}, "
            f"p90={np.nanquantile(np.abs(edge_bps[finite]), 0.90):.3f}, "
            f"p99={np.nanquantile(np.abs(edge_bps[finite]), 0.99):.3f}"
        )

        edge_ser = (pd.Series(edge_bps, index=idx_r)
                    .reindex(idx)
                    .where(lambda s: np.isfinite(s), 0.0)  # non-finite -> 0
                    .values)

        costs_bps = 2*args.fee_bps + 2*args.slip_bps + args.edge_buffer_bps

        # require direction consistency and edge > costs+buffer, but only where finite
        finite_gate = np.isfinite(edge_ser)
        dir_ok = np.zeros_like(sig, dtype=bool)
        dir_ok[finite_gate] = np.sign(edge_ser[finite_gate]) == sig[finite_gate]

        mag_ok = np.zeros_like(sig, dtype=bool)
        mag_ok[finite_gate] = np.abs(edge_ser[finite_gate]) >= (costs_bps + args.min_edge_bps)

        sig = np.where((sig != 0) & dir_ok & mag_ok, sig, 0)

        # require direction consistency and edge > costs+buffer
        costs_bps = 2*args.fee_bps + 2*args.slip_bps + args.edge_buffer_bps
        dir_ok  = np.sign(np.asarray(edge_ser)) == np.asarray(sig)
        mag_ok  = np.abs(np.asarray(edge_ser)) >= (costs_bps + args.min_edge_bps)
        sig = np.where((sig != 0) & dir_ok & mag_ok, sig, 0)

    if args.max_trades_per_hour:
        hour = pd.to_datetime(idx).floor("H")
        score = np.abs(np.asarray(edge_ser)) * (np.asarray(sig) != 0)
        keep = np.zeros_like(sig, dtype=bool)
        for h in np.unique(hour):
            m = (hour == h)
            where = np.where(m)[0]
            if where.size == 0: 
                continue
            top = np.argsort(-score[where])[:args.max_trades_per_hour]
            keep[where[top]] = True
        sig = np.where(keep, sig, 0)


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

        def _count_nonzero(x): 
            return int(np.count_nonzero(x))

        print(f"[diag] total windows: {len(idx)}")
        print(f"[diag] anomalies>thr: {_count_nonzero(anom)}")

        # After consec confirmation:
        print(f"[diag] consec>={args.consec}: {_count_nonzero(anom_confirmed)}")

        # After spread cap:
        print(f"[diag] spread<=cap: {_count_nonzero(ok_spread)}")
        mask_after_spread = anom_confirmed & ok_spread
        print(f"[diag] anom∧spread: {_count_nonzero(mask_after_spread)}")

        # After direction:
        cand_dir = np.where(mask_after_spread, np.sign(dir_sig)!=0, 0)
        print(f"[diag] directionable: {_count_nonzero(cand_dir)}")

        # After regressor edge (only if provided):
        if args.reg_model_dir:
            costs_bps = 2*args.fee_bps + 2*args.slip_bps + args.edge_buffer_bps
            print(f"[diag] costs_bps≈{costs_bps:.1f}, min_edge_bps={args.min_edge_bps:.1f}")
            print(f"[diag] |edge|>=costs+min_edge: {_count_nonzero(np.abs(edge_ser) >= (costs_bps + args.min_edge_bps))}")
            print(f"[diag] dir ok: {_count_nonzero(np.sign(edge_ser) == sig)}")


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
