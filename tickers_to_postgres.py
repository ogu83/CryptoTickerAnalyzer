import asyncio
import contextlib
import json
import signal
import sys
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Any

# ---- Windows event loop fix for psycopg async ----
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

import websockets
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

OKX_PUBLIC_WS = "wss://ws.okx.com:8443/ws/v5/public"

# Instruments to capture
INSTRUMENTS = [
    "BTC-USDT",
    # "ETH-USDT",
    # "BTC-USDT-250328",
    # "BTC-USD-250328-60000-C",
]

# Postgres connection info
PG_HOST = "macbook-server"
PG_PORT = 5432
PG_USER = "postgres"
PG_PASSWORD = "Postgres2839*"
PG_DBNAME = "OkxWebsocketDB"
PG_DBO_SCHEMA = "okx"

# ---------- Py3.8+ compatible to_thread ----------
try:
    to_thread = asyncio.to_thread  # Py 3.9+
except AttributeError:
    async def to_thread(func, /, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

# ---------- DDL ----------
DB_CREATE_TABLE_SQL = f"""
CREATE SCHEMA IF NOT EXISTS {PG_DBO_SCHEMA};

CREATE TABLE IF NOT EXISTS {PG_DBO_SCHEMA}.ticker (
    id              BIGSERIAL PRIMARY KEY,
    inst_type       varchar(16),
    inst_id         varchar(16) NOT NULL,
    ts              TIMESTAMPTZ NOT NULL,
    last            NUMERIC(36,18),
    last_sz         NUMERIC(36,18),
    ask_px          NUMERIC(36,18),
    ask_sz          NUMERIC(36,18),
    bid_px          NUMERIC(36,18),
    bid_sz          NUMERIC(36,18),
    open24h         NUMERIC(36,18),
    high24h         NUMERIC(36,18),
    low24h          NUMERIC(36,18),
    vol_ccy_24h     NUMERIC(36,18),
    vol_24h         NUMERIC(36,18),
    sod_utc0        NUMERIC(36,18),
    sod_utc8        NUMERIC(36,18)
);

CREATE INDEX IF NOT EXISTS idx_ticker_inst_ts
    ON {PG_DBO_SCHEMA}.ticker (inst_id, ts DESC);

/* If you prefer to avoid exact duplicates per (inst_id, ts), uncomment:
-- CREATE UNIQUE INDEX IF NOT EXISTS uq_ticker_inst_ts
--     ON {PG_DBO_SCHEMA}.ticker (inst_id, ts);
*/
"""

INSERT_SQL = f"""
INSERT INTO {PG_DBO_SCHEMA}.ticker
(inst_type, inst_id, ts, last, last_sz, ask_px, ask_sz, bid_px, bid_sz,
 open24h, high24h, low24h, vol_ccy_24h, vol_24h, sod_utc0, sod_utc8)
VALUES
(%(inst_type)s, %(inst_id)s, %(ts)s, %(last)s, %(last_sz)s, %(ask_px)s, %(ask_sz)s, %(bid_px)s, %(bid_sz)s,
 %(open24h)s, %(high24h)s, %(low24h)s, %(vol_ccy_24h)s, %(vol_24h)s, %(sod_utc0)s, %(sod_utc8)s)
"""

def _to_decimal(x: Any):
    if x is None or x == "":
        return None
    try:
        return Decimal(x)
    except Exception:
        return None

def normalize_for_db(t: Dict[str, Any]) -> Dict[str, Any]:
    ts = None
    if t.get("ts"):
        ts = datetime.fromtimestamp(int(t["ts"]) / 1000.0, tz=timezone.utc)
    return {
        "inst_type": t.get("instType"),
        "inst_id":   t.get("instId"),
        "ts":        ts,
        "last":         _to_decimal(t.get("last")),
        "last_sz":      _to_decimal(t.get("lastSz")),
        "ask_px":       _to_decimal(t.get("askPx")),
        "ask_sz":       _to_decimal(t.get("askSz")),
        "bid_px":       _to_decimal(t.get("bidPx")),
        "bid_sz":       _to_decimal(t.get("bidSz")),
        "open24h":      _to_decimal(t.get("open24h")),
        "high24h":      _to_decimal(t.get("high24h")),
        "low24h":       _to_decimal(t.get("low24h")),
        "vol_ccy_24h":  _to_decimal(t.get("volCcy24h")),
        "vol_24h":      _to_decimal(t.get("vol24h")),
        "sod_utc0":     _to_decimal(t.get("sodUtc0")),
        "sod_utc8":     _to_decimal(t.get("sodUtc8"))
    }

def ensure_database():
    """Create DB if missing, then create schema/table/indexes. Synchronous on purpose."""
    dsn_admin = (
        f"host={PG_HOST} port={PG_PORT} dbname=postgres "
        f"user={PG_USER} password='{PG_PASSWORD}'"
    )
    with psycopg.connect(dsn_admin, autocommit=True) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (PG_DBNAME,))
            if cur.fetchone() is None:
                cur.execute(f'CREATE DATABASE "{PG_DBNAME}"')

    dsn_target = (
        f"host={PG_HOST} port={PG_PORT} dbname={PG_DBNAME} "
        f"user={PG_USER} password='{PG_PASSWORD}'"
    )
    with psycopg.connect(dsn_target, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(DB_CREATE_TABLE_SQL)

class PostgresWriter:
    def __init__(self, dsn: str, pool_size: int = 5, batch_max: int = 200, flush_secs: float = 1.0):
        self.dsn = dsn
        # Create the pool but DO NOT open yet (avoid deprecation + ensure right loop)
        try:
            # Newer psycopg_pool
            self.pool = AsyncConnectionPool(conninfo=dsn, min_size=1, max_size=pool_size, open=False)
        except TypeError:
            # Older psycopg_pool that didn’t accept 'conninfo' but DOES accept keyword-only args
            self.pool = AsyncConnectionPool(dsn, min_size=1, max_size=pool_size, open=False)

        self.queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
        self.batch_max = batch_max
        self.flush_secs = flush_secs
        self._closing = False
        self._opened = False

        # simple stats
        self.inserted = 0

    async def enqueue(self, row: Dict[str, Any]):
        await self.queue.put(row)

    async def run(self):
        if not self._opened:
            await self.pool.open()
            self._opened = True
            print("[db] pool opened")

        try:
            while not self._closing:
                batch: List[Dict[str, Any]] = []
                try:
                    first = await asyncio.wait_for(self.queue.get(), timeout=self.flush_secs)
                    batch.append(first)
                except asyncio.TimeoutError:
                    pass

                while len(batch) < self.batch_max and not self.queue.empty():
                    batch.append(self.queue.get_nowait())

                if not batch:
                    continue

                try:
                    async with self.pool.connection() as conn:
                        async with conn.cursor() as cur:
                            await cur.executemany(INSERT_SQL, batch)
                    self.inserted += len(batch)
                except Exception as e:
                    # If something is wrong (DNS/creds/DDL), you’ll see it clearly here
                    print(f"[db] insert failed: {e.__class__.__name__}: {e}")
        finally:
            await self.pool.close()

    async def close(self):
        self._closing = True
        await asyncio.sleep(0.2)

class OkxTickerClient:
    def __init__(self, inst_ids: List[str], writer: PostgresWriter):
        self.inst_ids = inst_ids
        self.writer = writer
        self.ws = None
        self._closing = False

    def subscribe_payload(self) -> str:
        args = [{"channel": "tickers", "instId": inst} for inst in self.inst_ids]
        return json.dumps({"op": "subscribe", "args": args})

    async def run(self):
        backoff = 1
        while not self._closing:
            try:
                async with websockets.connect(
                    OKX_PUBLIC_WS,
                    ping_interval=20,
                    ping_timeout=10,
                    max_queue=None,
                    compression="deflate",
                ) as ws:
                    self.ws = ws
                    await ws.send(self.subscribe_payload())
                    print("[okx] subscribed to tickers:", ", ".join(self.inst_ids))
                    backoff = 1
                    await self.read_forever()
            except (asyncio.CancelledError, KeyboardInterrupt):
                break
            except Exception as e:
                print(f"[reconnect] {e.__class__.__name__}: {e}. retrying in {backoff}s…")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    async def read_forever(self):
        while not self._closing:
            msg = await self.ws.recv()
            data = json.loads(msg)

            if isinstance(data, dict) and data.get("event"):
                continue

            if isinstance(data, dict) and data.get("arg", {}).get("channel") == "tickers":
                for t in data.get("data", []):
                    row = normalize_for_db(t)
                    await self.writer.enqueue(row)

    async def close(self):
        self._closing = True
        try:
            if self.ws:
                await self.ws.close()
        except Exception:
            pass

async def main():
    async def reporter(writer: PostgresWriter):
        while True:
            await asyncio.sleep(5)
            print(f"[stats] queue={writer.queue.qsize()} inserted={writer.inserted}")
            
    # Create DB / schema / table off the event loop
    await to_thread(ensure_database)

    dsn = (
        f"host={PG_HOST} port={PG_PORT} dbname={PG_DBNAME} "
        f"user={PG_USER} password='{PG_PASSWORD}'"
    )
    writer = PostgresWriter(dsn=dsn, pool_size=5, batch_max=200, flush_secs=1.0)
    writer_task = asyncio.create_task(writer.run())
    report_task = asyncio.create_task(reporter(writer))

    client = OkxTickerClient(INSTRUMENTS, writer)
    ws_task = asyncio.create_task(client.run())

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _graceful(*_):
        print("\n[okx] shutting down…")
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _graceful)
        except NotImplementedError:
            pass

    await stop.wait()
    await client.close()
    await writer.close()

    ws_task.cancel()
    writer_task.cancel()
    report_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await ws_task
        await writer_task
        await report_task

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
