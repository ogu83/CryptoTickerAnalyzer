import os
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List

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
