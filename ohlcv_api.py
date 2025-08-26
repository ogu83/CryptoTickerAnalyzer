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
PG_DBO_SCHEMA = "okx"         # schema name

CONNINFO = (
    f"host={PG_HOST} port={PG_PORT} dbname={PG_DBNAME} "
    f"user={PG_USER} password='{PG_PASSWORD}'"
)

# ---------------- FastAPI setup ----------------
app = FastAPI(title="OKX OHLCV API", version="1.0")

# Allow browser clients during dev; tighten in prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["GET"], allow_headers=["*"]
)

# ---------------- DB pool (sync, thread-safe) ----------------
pool = ConnectionPool(CONNINFO, min_size=1, max_size=10, kwargs={"row_factory": dict_row})

def _to_iso_z(ts: datetime) -> str:
    # Ensure RFC3339/ISO 8601 with Z suffix
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

def _to_num(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, Decimal):
        return float(x)  # chart-friendly; switch to str(x) if you prefer exactness
    if isinstance(x, (int, float)):
        return float(x)
    # Fallback if DB driver returns strings for numerics
    try:
        return float(x)
    except Exception:
        return None

@app.get("/")
def root():
    return {
        "service": "OKX OHLCV API",
        "endpoints": ["/tick-chart"],
        "example": "/tick-chart?symbol=BTC-USDT&period=100"
    }

@app.get("/tick-chart")
def get_tick_chart(
    symbol: str = Query(..., min_length=1, description="Instrument id, e.g. BTC-USDT"),
    period: int = Query(..., ge=1, le=1_000_000, description="Number of ticks per bar")
) -> JSONResponse:
    # Call okx.ticker_ohlcv(symbol, period)
    sql = f"""select "time","open","close","high","low","volume"
              from {PG_DBO_SCHEMA}.ticker_ohlcv(%s, %s);"""
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (symbol, period))
                rows = cur.fetchall()
    except psycopg.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e.pgerror or str(e)}")

    # Normalize to plain JSON
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
