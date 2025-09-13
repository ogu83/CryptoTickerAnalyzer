import os
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List
from psycopg.rows import dict_row

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

# ---------------- Postgres connection info ----------------
PG_HOST = "macbook-server"
PG_PORT = 5432
PG_USER = "postgres"
PG_PASSWORD = "Postgres2839*"
PG_DBNAME = "CryptoTickers"   # target DB name

# Schemas
PG_DBO_SCHEMA_OKX = "okx"
PG_DBO_SCHEMA_BINANCE = "bnc"

CONNINFO = (
    f"host={PG_HOST} port={PG_PORT} dbname={PG_DBNAME} "
    f"user={PG_USER} password='{PG_PASSWORD}'"
)

# ---------------- FastAPI setup ----------------
app = FastAPI(title="OHLCV API (OKX/Binance)", version="1.1")

# Allow browser clients during dev; tighten in prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["GET"], allow_headers=["*"]
)

# ---------------- DB pool (sync, thread-safe) ----------------
pool = ConnectionPool(CONNINFO, min_size=1, max_size=10, kwargs={"row_factory": dict_row})

def _to_iso_z(ts: datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

def _to_num(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, Decimal):
        return float(x)
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(x)
    except Exception:
        return None

def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    s = s.strip()
    # Accept ...Z or offset; make tz-aware
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid datetime format: {s!r}. Use ISO 8601, e.g. 2025-08-26T00:00:00Z")
    # If no tzinfo provided, assume UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt

@app.get("/")
def root():
    return {
        "service": "OHLCV API",
        "endpoints": ["/tick-chart"],
        "examples": [
            "/tick-chart?symbol=BTC-USDT&period=100",                             # OKX default
            "/tick-chart?venue=bnc&symbol=BTCUSDT&period=100",                   # Binance
            "/tick-chart?symbol=BTC-USDT&period=1000&start=2025-08-26T00:00:00Z&end=2025-08-27T00:00:00Z"
        ]
    }

@app.get("/ob-top")
def get_orderbook_top(
    symbol: str,
    start: str | None = Query(None, description="ISO datetime (inclusive)"),
    end: str   | None = Query(None, description="ISO datetime (exclusive)"),
    step: int = Query(1, ge=1, description="Return every Nth snapshot (downsample)"),
) -> JSONResponse:
    schema = PG_DBO_SCHEMA_OKX  # OB is OKX-only here

    # OKX symbol fix-up
    if "-" not in symbol:
        symbol = (symbol[:-4] + "-" + symbol[-4:]) if len(symbol) > 6 else (symbol[:-3] + "-" + symbol[-3:])

    start_dt = _parse_dt(start)
    end_dt   = _parse_dt(end)

    params: List[Any] = [symbol]
    # IMPORTANT: filter by the base column names (no alias here)
    where = ["inst_id = %s"]
    if start_dt is not None:
        where.append("ts >= %s"); params.append(start_dt)
    if end_dt is not None:
        where.append("ts < %s");  params.append(end_dt)

    sql = f"""
    SELECT * FROM (
    SELECT *, row_number() OVER (ORDER BY h.ts) AS rn
    FROM
    
    WITH h AS (
    SELECT id, inst_id, ts
    FROM {schema}.orderbook_header
    WHERE inst_id = {symbol}            -- and optional ts filters
        AND ts >= {start_dt}
        AND ts <  {end_dt}
    ORDER BY ts
    )
    SELECT
    h.ts, h.inst_id,
    bb.bid_px, bb.bid_sz, bb.bid_ct,
    ba.ask_px, ba.ask_sz, ba.ask_ct
    FROM h
    LEFT JOIN LATERAL (
    SELECT i.price  AS bid_px,
            i.qty    AS bid_sz,
            i.order_count AS bid_ct
    FROM {schema}.orderbook_item i
    WHERE i.orderbook_id = h.id AND i.side = 'B'
    ORDER BY i.price DESC
    LIMIT 1
    ) bb ON TRUE
    LEFT JOIN LATERAL (
    SELECT i.price  AS ask_px,
            i.qty    AS ask_sz,
            i.order_count AS ask_ct
    FROM {schema}.orderbook_item i
    WHERE i.orderbook_id = h.id AND i.side = 'A'
    ORDER BY i.price ASC
    LIMIT 1
    ) ba ON TRUE
    ) z
    WHERE (z.rn % {step}) = 1
    ORDER BY ts;
    """

    try:
        with pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
    except psycopg.Error as e:
        # psycopg3 may not have .pgerror; use diag/sqlstate when available
        detail = getattr(e, "pgerror", None) or getattr(getattr(e, "diag", None), "message_primary", None) or str(e)
        raise HTTPException(status_code=500, detail=f"Database error: {detail}")

    # downsample if requested
    # if step > 1:
    #     rows = rows[::step]

    def fnum(x: Any) -> float | None:
        if x is None: return None
        if isinstance(x, Decimal): return float(x)
        try: return float(x)
        except Exception: return None

    out: List[Dict[str, Any]] = []
    for r in rows:
        bid_px = fnum(r["bid_px"]); ask_px = fnum(r["ask_px"])
        bid_sz = fnum(r["bid_sz"]); ask_sz = fnum(r["ask_sz"])
        mid    = (bid_px + ask_px) / 2.0 if (bid_px is not None and ask_px is not None) else None
        spread = (ask_px - bid_px) if (bid_px is not None and ask_px is not None) else None
        imb    = ((bid_sz - ask_sz) / (bid_sz + ask_sz)) if (bid_sz and ask_sz and (bid_sz + ask_sz) != 0) else None
        micro  = ((bid_px * ask_sz + ask_px * bid_sz) / (bid_sz + ask_sz)) if (bid_px and ask_px and bid_sz and ask_sz and (bid_sz + ask_sz) != 0) else None

        out.append({
            "time": _to_iso_z(r["ts"]),
            "inst_id": r["inst_id"],
            "bid_px": bid_px, "bid_sz": bid_sz, "bid_ct": r["bid_ct"],
            "ask_px": ask_px, "ask_sz": ask_sz, "ask_ct": r["ask_ct"],
            "mid": mid, "spread": spread, "imbalance": imb, "microprice": micro,
        })

    return JSONResponse(content=out)

@app.get("/tick-chart")
def get_tick_chart(
    symbol: str = Query(..., min_length=1, description="Instrument id (OKX: BTC-USDT, Binance: BTCUSDT)"),
    period: int = Query(..., ge=1, le=1_000_000, description="Number of ticks per bar"),
    venue: str = Query("okx", regex="^(okx|bnc)$", description="Data source: okx or bnc"),
    start: str | None = Query(None, description="ISO datetime (inclusive), e.g. 2025-08-26T00:00:00Z"),
    end: str | None = Query(None, description="ISO datetime (exclusive), e.g. 2025-08-27T00:00:00Z"),
) -> JSONResponse:
    # Choose schema by venue
    schema = PG_DBO_SCHEMA_OKX if venue == "okx" else PG_DBO_SCHEMA_BINANCE

    # Parse optional filters
    start_dt = _parse_dt(start)
    end_dt = _parse_dt(end)

    # symbol fixup for Binance
    if venue == "bnc" and "-" in symbol:
        symbol = symbol.replace("-", "")

    # symbol fixup for OKX
    if venue == "okx" and "-" not in symbol:
        if len(symbol) > 6:
            symbol = symbol[:-4] + "-" + symbol[-4:]
        else:
            symbol = symbol[:-3] + "-" + symbol[-3:]  

    # Build SQL; filter on the view's "time" if provided
    params: List[Any] = [symbol, period]
    sql = f'''select "time","open","close","high","low","volume"
              from {schema}.ticker_ohlcv(%s, %s)'''
    conditions: List[str] = []
    if start_dt is not None:
        conditions.append('"time" >= %s')
        params.append(start_dt)
    if end_dt is not None:
        conditions.append('"time" < %s')
        params.append(end_dt)
    if conditions:
        sql += " where " + " and ".join(conditions)
    sql += ' order by "time"'

    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
    except psycopg.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e.pgerror or str(e)}")

    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append({
            "time":   _to_iso_z(r["time"]) if r.get("time") else None,
            "open":   _to_num(r["open"]),
            "close":  _to_num(r["close"]),
            "high":   _to_num(r["high"]),
            "low":    _to_num(r["low"]),
            "volume": _to_num(r["volume"]),
        })
    return JSONResponse(content=out)
