#!/usr/bin/env python3
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import requests, joblib
import matplotlib.pyplot as plt
import psycopg
import requests, joblib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ---------- fetching & feature engineering (mirrors train_orderbook.py) ----------
def fetch_ob(api, symbol, start=None, end=None, step=5, timeout=4800) -> pd.DataFrame:
    p = {"symbol": symbol, "step": step}
    if start: p["start"] = start
    if end:   p["end"] = end
    url = f"{api.rstrip('/')}/ob-top"
    r = requests.get(url, params=p, timeout=timeout)
    r.raise_for_status()
    js = r.json()
    if not js:
        raise SystemExit("No order book rows returned.")
    df = pd.DataFrame(js)
    df["time"] = pd.to_datetime(df["time"], utc=True, format="mixed")
    df = df.sort_values("time").set_index("time")
    for c in ["bid_px","ask_px","bid_sz","ask_sz","mid","spread","imbalance","microprice"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna()

def fetch_ob_chunked(api, symbol, start, end, step=5, timeout=900, chunk_hours=12):
    """
    Fetch OB in smaller chunks and concat. Works with your existing /ob-top.
    """
    if not start or not end:
        # fallback to single shot when range isn't specified
        return fetch_ob(api, symbol, start, end, step=step, timeout=timeout)

    t0 = pd.Timestamp(start).tz_convert("UTC") if pd.Timestamp(start).tzinfo else pd.Timestamp(start, tz="UTC")
    t1 = pd.Timestamp(end).tz_convert("UTC") if pd.Timestamp(end).tzinfo else pd.Timestamp(end, tz="UTC")
    frames = []
    cur = t0
    delta = pd.Timedelta(hours=chunk_hours)

    while cur < t1:
        chunk_start = cur.isoformat()
        chunk_end   = min(cur + delta, t1).isoformat()
        print(f"[chunk] {chunk_start} -> {chunk_end}")
        df = fetch_ob(api, symbol, start=chunk_start, end=chunk_end, step=step, timeout=timeout)
        frames.append(df)
        cur = cur + delta

    if not frames:
        raise SystemExit("No data returned for the requested interval.")
    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    mid = df["mid"]; spread = df["spread"]

    out["spread_bps"] = (spread / mid * 1e4).clip(lower=0).fillna(0.0)
    out["imb"] = df["imbalance"].clip(-1, 1).fillna(0.0)

    # microprice tilt normalized by spread
    tilt = (df["microprice"] - mid) / spread.replace(0, np.nan)
    out["micro_tilt"] = tilt.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    out["dt_sec"] = out.index.to_series().diff().dt.total_seconds().fillna(0.0).clip(0, 5.0)
    out["lr_mid"] = np.log(mid).diff().fillna(0.0)
    out["vol20_mid"] = out["lr_mid"].rolling(20, min_periods=10).std().bfill().fillna(0.0)

    for k in [1, 2, 3, 5]:
        out[f"imb_lag{k}"]    = out["imb"].shift(k)
        out[f"tilt_lag{k}"]   = out["micro_tilt"].shift(k)
        out[f"lr_mid_lag{k}"] = out["lr_mid"].shift(k)
        out[f"sbps_lag{k}"]   = out["spread_bps"].shift(k)

    # ATR-like on mid (for de-normalizing predictions)
    out["atr_mid"] = (np.log(mid).diff().abs().ewm(alpha=1/50, adjust=False).mean()
                      .bfill().fillna(0.0))

    # keep raw spread too (for dynamic thresholds)
    out["spread"] = spread

    # add prices used by simulator
    out["mid"] = mid
    out["open"] = mid  # your “open” is mid at that timestamp (top of book)
    out["close"] = mid
    return out.dropna()

# ---------- LGBM signal: from delta_norm -> edge (bps) & direction ----------
def lgbm_predict_edges(model, F: pd.DataFrame, feat_cols: list[str]) -> pd.Series:
    # Features are row-wise (no sequence). We predict next Δnorm at t, so align to t+1 when trading.
    X = F[feat_cols].copy()
    yhat_norm = model.predict(X)  # next-step normalized delta
    # de-normalize into price delta using previous ATR(mid)
    atr_prev = F["atr_mid"].shift(1).reindex(F.index).values
    edge_bps = yhat_norm * atr_prev * 1e4  # convert to bps
    return pd.Series(edge_bps, index=F.index)

# ---------- simple position-based simulator (enter next bar, exit after H bars) ----------
def backtest(times, sig, nxt_open, nxt_open_holdH, fees_bps=2, slip_bps=1,
             init_usdt=10_000, position_frac=0.1):
    fee = fees_bps/1e4; slip = slip_bps/1e4
    equity = init_usdt
    rows = []
    in_pos = False; side = 0; entry_px = None

    for i, t in enumerate(times):
        if not in_pos:
            if sig[i] == 0:
                rows.append((t, equity, 0, np.nan, np.nan, 0.0)); continue
            # enter at next-bar open (already aligned)
            px = float(nxt_open.iloc[i])
            notional = equity * position_frac
            qty = notional / px
            px_eff_in = px * (1 + slip if sig[i] > 0 else 1 - slip)
            cost_in = qty * px_eff_in
            fee_in = fee * qty * px
            in_pos = True; side = int(sig[i]); entry_px = px
            rows.append((t, equity, side, px, np.nan, 0.0))
        else:
            # exit exactly after H bars at the prebuilt series (nxt_open shifted by H)
            px_out = float(nxt_open_holdH.iloc[i])
            qty = (equity * position_frac) / entry_px
            px_eff_out = px_out * (1 - slip if side > 0 else 1 + slip)
            fee_out = fee * qty * px_out
            # PnL on notional with round-trip fees
            pnl = qty * (px_eff_out - entry_px) * (1 if side > 0 else -1) - (fee_in + fee_out)
            equity += pnl
            rows.append((t, equity, 0, np.nan, px_out, pnl))
            in_pos = False; side = 0; entry_px = None

    bt = pd.DataFrame(rows, columns=["time","equity","signal","open","close","pnl"]).set_index("time")
    ret = bt["equity"].pct_change().fillna(0.0)
    roll_max = bt["equity"].cummax()
    dd = (bt["equity"] - roll_max) / roll_max
    stats = dict(
        final_equity=float(bt["equity"].iloc[-1]),
        total_return_pct=float((bt["equity"].iloc[-1]/bt["equity"].iloc[0]-1)*100.0),
        max_drawdown_pct=float(dd.min()*100.0),
        num_trades=int((bt["signal"].abs()>0).sum()),
        avg_trade_pnl=float(bt.loc[bt["pnl"].notna(),"pnl"].mean() if bt["pnl"].notna().any() else 0.0),
        sharpe_like=float((ret.mean()/(ret.std()+1e-9))*np.sqrt(252*24*6)),
    )
    return bt, stats

def sanitize_for_json(obj):
    import math
    if isinstance(obj, dict):  return {k: sanitize_for_json(v) for k,v in obj.items()}
    if isinstance(obj, list):  return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, float): return obj if math.isfinite(obj) else None
    return obj

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="models_ob/..._lgbm directory")
    ap.add_argument("--api", default="http://macbook-server:8200")
    ap.add_argument("--symbol", default="ETH-USDT")
    ap.add_argument("--start", default=None); ap.add_argument("--end", default=None)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=300)

    # trading knobs
    ap.add_argument("--hold", type=int, default=2)
    ap.add_argument("--position-frac", type=float, default=0.1)
    ap.add_argument("--fee-bps", type=float, default=2.0)
    ap.add_argument("--slip-bps", type=float, default=1.0)
    ap.add_argument("--long-only", action="store_true", default=False)
    ap.add_argument("--short-only", action="store_true", default=False)
    ap.add_argument("--max-trades-per-hour", type=int, default=None)

    # gating
    ap.add_argument("--edge-scale", type=float, default=1.0, help="Multiply predicted edge(bps).")
    ap.add_argument("--min-edge-bps", type=float, default=0.0, help="Extra buffer beyond costs (if not auto).")
    ap.add_argument("--edge-buffer-bps", type=float, default=1.0, help="Microstructure noise buffer.")
    ap.add_argument("--spread-cap-bps", type=float, default=2.5)
    ap.add_argument("--auto-min-edge", action="store_true",
                    help="Per-bar min edge = (2*fee+2*slip+buffer) + k_spread * spread_bps.")
    ap.add_argument("--k-spread", type=float, default=0.3)

    ap.add_argument("--gate-model-dir", type=str, default=None,
                    help="Path to lgbm gate dir (with gate_lgbm.pkl & gate_meta.json)")
    ap.add_argument("--gate-thr", type=float, default=None,
                    help="Optional manual gate threshold; if omitted, use thr_star from gate_meta.json")

    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    gate_model = None
    gate_meta = None
    if args.gate_model_dir:
        gate_model = joblib.load(Path(args.gate_model_dir) / "gate_lgbm.pkl")
        with (Path(args.gate_model_dir) / "gate_meta.json").open() as f:
            gate_meta = json.load(f)
        if args.gate_thr is None:
            args.gate_thr = float(gate_meta.get("thr_star", 0.6))
        print(f"[gate] loaded • thr={args.gate_thr:.3f}")

    # load model + metadata
    mdir = Path(args.model_dir)
    model = joblib.load(mdir / "model_lgbm.pkl")
    # prefer a dedicated features file; fall back to features.json if present
    feat_file = (mdir / "features_lgbm.json") if (mdir / "features_lgbm.json").exists() else (mdir / "features.json")
    feat_cols = json.loads(feat_file.read_text())

    print(f"[load] {mdir.name}  symbol={args.symbol} step={args.step}")
    # ob = fetch_ob(args.api, args.symbol, args.start, args.end, step=args.step, timeout=args.timeout)
    ob = fetch_ob_chunked(
        args.api, args.symbol,
        start=args.start, end=args.end,
        step=args.step, timeout=args.timeout,
        chunk_hours=12,   # tune to 6/12/24 as you like
    )
    F = build_features(ob)

    # keep only rows where all features exist
    missing = [c for c in feat_cols if c not in F.columns]
    if missing:
        raise SystemExit(f"Missing features in F: {missing}")
    Fx = F[feat_cols].dropna()
    F = F.reindex(Fx.index)

    # predict edges (bps) aligned to Fx.index
    edge_bps = lgbm_predict_edges(model, F, feat_cols) * args.edge_scale

    # trade ONLY from the next bar forward (avoid any look-ahead)
    times = Fx.index[1:]  # we enter at next bar open
    edge_vec = edge_bps.reindex(times)  # already aligned, since we drop the first bar for trading

    # ---------- Gate: rebuild features identically to training ----------
    gate_thr = None
    gate_pass_mask = None
    if args.gate_model_dir:
        gate_dir = Path(args.gate_model_dir)
        gate = joblib.load(gate_dir / "gate_lgbm.pkl")
        with open(gate_dir / "gate_meta.json", "r") as f:
            gate_meta = json.load(f)

        # threshold: CLI override > saved meta > default 0.5
        gate_thr = float(args.gate_thr) if (args.gate_thr is not None) \
            else float(gate_meta.get("thr_star", 0.5))
        print(f"[gate] loaded • thr={gate_thr:.3f}")

        # We must build pred_edge_bps the same way as in train_lgbm_gate()
        # feats must include: mid, spread_bps, atr_mid (the ATR proxy)
        mid = F["mid"].astype(float)
        atr_prev = F["atr_mid"].shift(1).astype(float)

        # Use *the same* regressor feature set as saved with the LGBM reg model
        reg_feat_cols_path = Path(args.model_dir) / "features.json"
        with open(reg_feat_cols_path, "r") as f:
            reg_feat_cols = json.load(f)

        X_reg_all = F[reg_feat_cols].fillna(0.0).to_numpy()
        # model is the LGBM regressor already loaded earlier
        reg_pred_norm = model.predict(X_reg_all)
        # convert normalized delta to bps (causal de-normalization)
        pred_edge_bps = (reg_pred_norm * (atr_prev / mid) * 1e4)
        pred_edge_bps = pd.Series(pred_edge_bps, index=F.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # augment gate features exactly like training
        F["pred_edge_bps"] = pred_edge_bps
        F["abs_pred_edge_bps"] = pred_edge_bps.abs()

        # Use the exact column order saved in gate_meta
        gate_cols = gate_meta["feat_cols"]
        missing = [c for c in gate_cols if c not in F.columns]
        if missing:
            print(f"[gate][warn] missing columns in inference: {missing}")
        X_gate = F.reindex(columns=gate_cols).fillna(0.0).to_numpy()

        gate_p = gate.predict_proba(X_gate)[:, 1]
        p_series = pd.Series(gate_p, index=F.index)
        print(
            "[gate] proba stats:",
            f"min={p_series.min():.3f}, p50={p_series.quantile(0.5):.3f}, "
            f"p90={p_series.quantile(0.9):.3f}, max={p_series.max():.3f}"
        )

        # Gate decides at time t whether to take the trade at t+1.
        # Our 'times' are those t+1 bars; so get probabilities on the same index
        gate_pass_mask = (p_series.reindex(times).values >= gate_thr)
        pass_rate = gate_pass_mask.mean() * 100.0
        print(f"[gate] applied • thr={gate_thr:.3f} • pass_rate={pass_rate:.2f}%")
    else:
        gate_pass_mask = np.ones(len(times), dtype=bool)

    # -------------------------------------------------------------
    # OPTIONAL PROFITABILITY GATE (LightGBM classifier)
    # Uses the same feature list the gate was trained with.
    if gate_model is not None and gate_meta is not None:
        # build the gate feature frame on the same index as F
        # (the trainer added pred_edge features; do the same here)
        F["pred_edge_bps"] = edge_bps
        F["abs_pred_edge_bps"] = edge_bps.abs()

        gate_feat_cols = gate_meta.get("feat_cols", [])
        missing_gate = [c for c in gate_feat_cols if c not in F.columns]
        if missing_gate:
            raise SystemExit(f"[gate] missing gate features in F: {missing_gate}")

        # Gate decides at time t whether to take the trade at t+1.
        # Our 'times' are those t+1 bars; so get probabilities at index 'times'
        Xg = F[gate_feat_cols].reindex(times).fillna(0.0)
        p_take = gate_model.predict_proba(Xg)[:, 1]
        gate_keep = p_take >= float(args.gate_thr)
        keep_rate = 100.0 * (gate_keep.sum() / max(1, len(gate_keep)))
        print(f"[gate] applied • thr={args.gate_thr:.3f} • pass_rate={keep_rate:.2f}%")

        # mask: gate failing -> no trade
        # we only mask; direction/magnitude checks will run next
        gate_mask = gate_keep.astype(bool)
    else:
        gate_mask = np.ones(len(times), dtype=bool)
    # -------------------------------------------------------------

    # costs + dynamic min edge
    base_costs = 2*args.fee_bps + 2*args.slip_bps + args.edge_buffer_bps
    spread_bps = (F["spread"].reindex(times) / F["mid"].reindex(times) * 1e4).values
    if args.auto_min_edge:
        min_edge_vec = base_costs + args.k_spread * spread_bps
        print(f"[gate] dynamic min_edge = {base_costs:.2f} + {args.k_spread}*spread_bps  "
              f"(median={np.nanmedian(min_edge_vec):.2f}bps)")
    else:
        min_edge_vec = base_costs + args.min_edge_bps
        print(f"[gate] static min_edge = {base_costs:.2f} + {args.min_edge_bps:.2f} = {min_edge_vec:.2f}bps")
        min_edge_vec = np.repeat(min_edge_vec, len(times))

    # spread cap
    ok_spread = spread_bps <= args.spread_cap_bps

    # raw side = sign(edge)
    sig = np.sign(edge_vec.values).astype(int)
    if args.long_only:  sig[sig < 0] = 0
    if args.short_only: sig[sig > 0] = 0

    # magnitude gate
    mag_ok = np.abs(edge_vec.values) >= min_edge_vec
    # sig = np.where(ok_spread & mag_ok, sig, 0)
    sig = np.where(ok_spread & mag_ok & gate_mask, sig, 0)

    # optional: throttle per hour by edge “excess”
    if args.max_trades_per_hour:
        hour = pd.to_datetime(times).floor("H")
        excess = np.maximum(np.abs(edge_vec.values) - min_edge_vec, 0.0) * (sig != 0)
        keep = np.zeros_like(sig, dtype=bool)
        for h in np.unique(hour):
            idx = np.where(hour == h)[0]
            if idx.size == 0: continue
            top = np.argsort(-excess[idx])[:args.max_trades_per_hour]
            keep[idx[top]] = True
        sig = np.where(keep, sig, 0)

    # align entry/exit prices: enter at next bar open, exit after H bars at next-open+H
    nxt_open       = F["open"].reindex(times)
    nxt_open_holdH = F["open"].shift(-args.hold).reindex(times)

    # run backtest
    bt, stats = backtest(times, sig, nxt_open, nxt_open_holdH,
                         fees_bps=args.fee_bps, slip_bps=args.slip_bps,
                         init_usdt=10_000, position_frac=args.position_frac)

    print("\n=== LGBM Backtest ===")
    for k,v in stats.items():
        print(f"{k:>18}: {v:,.6f}")

    if args.plot:
        plt.figure(figsize=(12,4))
        plt.plot(bt.index, bt["equity"], label="Equity")
        plt.title(f"Equity Curve • {mdir.name} • {args.symbol}")
        plt.xlabel("Time"); plt.ylabel("USDT"); plt.legend(); plt.tight_layout()
        plt.savefig("ob_lgbm_equity_curve.png"); print("Saved plot: ob_lgbm_equity_curve.png")

    # store a DB summary like other scripts
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
                "type": "ob_lgbm",
                "hold": args.hold,
                "position_frac": args.position_frac,
                "fee_bps": args.fee_bps,
                "slip_bps": args.slip_bps,
                "spread_cap_bps": args.spread_cap_bps,
                "auto_min_edge": args.auto_min_edge,
                "k_spread": args.k_spread,
                "edge_scale": args.edge_scale,
                "min_edge_bps": args.min_edge_bps,
            })
            cur.execute("""
                insert into ml_backtest (model_name, symbol, period, start_time, end_time, params, metrics)
                values (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
            """, (
                mdir.name, args.symbol, args.step,
                bt.index.min().to_pydatetime() if not bt.empty else None,
                bt.index.max().to_pydatetime() if not bt.empty else None,
                json.dumps(params, allow_nan=False),
                json.dumps(sanitize_for_json(stats), allow_nan=False),
            ))
            conn.commit()
            print("[db] backtest saved to ml_backtest")
    except Exception as e:
        print(f"[db] save skipped: {e}")

if __name__ == "__main__":
    main()
