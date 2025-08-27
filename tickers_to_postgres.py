import asyncio
import contextlib
import json
import signal
import sys
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Any, Tuple

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

# ---------------- WebSocket endpoints ----------------
OKX_PUBLIC_WS = "wss://ws.okx.com:8443/ws/v5/public"
# Binance USDⓈ-M Futures combined stream base (lowercase stream names)
# Ref: https://developers.binance.com/.../websocket-market-streams (Connect)
BINANCE_FUTURES_COMBINED = "wss://fstream.binance.com/stream"  # add ?streams=... below

RECORD_ORDERBOOK = False  # set to True to enable OKX order book capture

# ---------------- Instruments ----------------
# Keep your OKX-style list; we’ll derive Binance symbols automatically.
INSTRUMENTS = [
    "BTC-USDT",
    "ETH-USDT",
]

# ---------------- Postgres connection info ----------------
PG_HOST = "macbook-server"
PG_PORT = 5432
PG_USER = "postgres"
PG_PASSWORD = "Postgres2839*"
PG_DBNAME = "CryptoTickers"  # target DB name

# Schemas
PG_DBO_SCHEMA_OKX = "okx"
PG_DBO_SCHEMA_BINANCE = "bnc"

# ---------- Py3.8+ compatible to_thread ----------
try:
    to_thread = asyncio.to_thread  # Py 3.9+
except AttributeError:
    async def to_thread(func, /, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

# ---------- Helpers ----------
def _to_decimal(x: Any):
    if x is None or x == "":
        return None
    try:
        return Decimal(x)
    except Exception:
        return None

def _to_int(x: Any):
    if x is None or x == "":
        return None
    try:
        return int(x)
    except Exception:
        return None

def _ts_ms_to_dt(ms: Any):
    if ms is None or ms == "":
        return None
    return datetime.fromtimestamp(int(ms) / 1000.0, tz=timezone.utc)

def _okx_to_binance_symbol(inst: str) -> Tuple[str, str]:
    """
    Convert 'BTC-USDT' -> ('BTCUSDT', 'btcusdt')
    Returns (db_symbol_upper, stream_symbol_lower)
    """
    core = inst.replace("-", "")
    return core.upper(), core.lower()

# ---------- DDL ----------
DB_CREATE_TABLE_SQL_OKX = f"""
CREATE SCHEMA IF NOT EXISTS {PG_DBO_SCHEMA_OKX};

-- OKX ticker table (as you defined)
CREATE TABLE IF NOT EXISTS {PG_DBO_SCHEMA_OKX}.ticker (
    id              BIGSERIAL PRIMARY KEY,
    inst_type       varchar(16),
    inst_id         varchar(32) NOT NULL,
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
    ON {PG_DBO_SCHEMA_OKX}.ticker (inst_id, ts DESC);

-- Order book header (one row per WS event)
CREATE TABLE IF NOT EXISTS {PG_DBO_SCHEMA_OKX}.orderbook_header (
    id            BIGSERIAL PRIMARY KEY,
    inst_type     varchar(16),
    inst_id       varchar(32) NOT NULL,
    ts            TIMESTAMPTZ NOT NULL,
    action        varchar(12),
    seq_id        BIGINT,
    prev_seq_id   BIGINT,
    checksum      BIGINT
);

CREATE INDEX IF NOT EXISTS idx_ob_hdr_inst_ts
    ON {PG_DBO_SCHEMA_OKX}.orderbook_header (inst_id, ts DESC);

-- Order book items (many rows per header)
CREATE TABLE IF NOT EXISTS {PG_DBO_SCHEMA_OKX}.orderbook_item (
    id             BIGSERIAL PRIMARY KEY,
    orderbook_id   BIGINT NOT NULL REFERENCES {PG_DBO_SCHEMA_OKX}.orderbook_header(id) ON DELETE CASCADE,
    side           CHAR(1) NOT NULL CHECK (side IN ('B','A')),  -- B=bid, A=ask
    price          NUMERIC(36,18) NOT NULL,
    qty            NUMERIC(36,18) NOT NULL,
    order_count    INTEGER
);

CREATE INDEX IF NOT EXISTS idx_ob_item_hdr_side_price
    ON {PG_DBO_SCHEMA_OKX}.orderbook_item (orderbook_id, side, price);
"""

DB_CREATE_TABLE_SQL_BINANCE = f"""
CREATE SCHEMA IF NOT EXISTS {PG_DBO_SCHEMA_BINANCE};

-- Binance USDⓈ-M Futures ticker table (aligned to OKX columns where possible)
CREATE TABLE IF NOT EXISTS {PG_DBO_SCHEMA_BINANCE}.ticker (
    id              BIGSERIAL PRIMARY KEY,
    inst_type       varchar(16),         -- e.g. 'um-futures'
    inst_id         varchar(32) NOT NULL, -- e.g. 'BTCUSDT'
    ts              TIMESTAMPTZ NOT NULL, -- from 'E' (event time)
    last            NUMERIC(36,18),      -- 'c'
    last_sz         NUMERIC(36,18),      -- 'Q' (last quantity)
    ask_px          NUMERIC(36,18),      -- (not in @ticker payload) -> NULL
    ask_sz          NUMERIC(36,18),      -- (not in @ticker payload) -> NULL
    bid_px          NUMERIC(36,18),      -- (not in @ticker payload) -> NULL
    bid_sz          NUMERIC(36,18),      -- (not in @ticker payload) -> NULL
    open24h         NUMERIC(36,18),      -- 'o'
    high24h         NUMERIC(36,18),      -- 'h'
    low24h          NUMERIC(36,18),      -- 'l'
    vol_ccy_24h     NUMERIC(36,18),      -- 'q' quote volume
    vol_24h         NUMERIC(36,18),      -- 'v' base volume
    sod_utc0        NUMERIC(36,18),      -- NULL (not provided)
    sod_utc8        NUMERIC(36,18)       -- NULL (not provided)
);

CREATE INDEX IF NOT EXISTS idx_bnc_ticker_inst_ts
    ON {PG_DBO_SCHEMA_BINANCE}.ticker (inst_id, ts DESC);
"""

INSERT_TICKER_SQL_OKX = f"""
INSERT INTO {PG_DBO_SCHEMA_OKX}.ticker
(inst_type, inst_id, ts, last, last_sz, ask_px, ask_sz, bid_px, bid_sz,
 open24h, high24h, low24h, vol_ccy_24h, vol_24h, sod_utc0, sod_utc8)
VALUES
(%(inst_type)s, %(inst_id)s, %(ts)s, %(last)s, %(last_sz)s, %(ask_px)s, %(ask_sz)s, %(bid_px)s, %(bid_sz)s,
 %(open24h)s, %(high24h)s, %(low24h)s, %(vol_ccy_24h)s, %(vol_24h)s, %(sod_utc0)s, %(sod_utc8)s)
"""

INSERT_TICKER_SQL_BINANCE = f"""
INSERT INTO {PG_DBO_SCHEMA_BINANCE}.ticker
(inst_type, inst_id, ts, last, last_sz, ask_px, ask_sz, bid_px, bid_sz,
 open24h, high24h, low24h, vol_ccy_24h, vol_24h, sod_utc0, sod_utc8)
VALUES
(%(inst_type)s, %(inst_id)s, %(ts)s, %(last)s, %(last_sz)s, %(ask_px)s, %(ask_sz)s, %(bid_px)s, %(bid_sz)s,
 %(open24h)s, %(high24h)s, %(low24h)s, %(vol_ccy_24h)s, %(vol_24h)s, %(sod_utc0)s, %(sod_utc8)s)
"""

# ---------- Normalizers ----------
def normalize_okx_ticker_for_db(t: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "inst_type": t.get("instType"),
        "inst_id":   t.get("instId"),
        "ts":        _ts_ms_to_dt(t.get("ts")),
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
        "sod_utc8":     _to_decimal(t.get("sodUtc8")),
    }

def normalize_binance_ticker_for_db(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Binance USDⓈ-M @ticker payload (24hr rolling window) example (fields we use):
    E=event time, s=symbol, c=last, Q=last qty, o=open, h=high, l=low, v=base vol, q=quote vol.
    Ref: Individual Symbol Ticker Streams. 
    """
    return {
        "inst_type": "um-futures",
        "inst_id":   d.get("s"),              # e.g. 'BTCUSDT'
        "ts":        _ts_ms_to_dt(d.get("E")),
        "last":         _to_decimal(d.get("c")),
        "last_sz":      _to_decimal(d.get("Q")),
        "ask_px":       None,                 # not in @ticker payload
        "ask_sz":       None,
        "bid_px":       None,
        "bid_sz":       None,
        "open24h":      _to_decimal(d.get("o")),
        "high24h":      _to_decimal(d.get("h")),
        "low24h":       _to_decimal(d.get("l")),
        "vol_ccy_24h":  _to_decimal(d.get("q")),
        "vol_24h":      _to_decimal(d.get("v")),
        "sod_utc0":     None,
        "sod_utc8":     None,
    }

# ---------- Order book helpers (unchanged) ----------
def _explode_side(side: str, levels: List[List[str]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for lvl in levels:
        if not lvl:
            continue
        price = _to_decimal(lvl[0]) if len(lvl) > 0 else None
        qty   = _to_decimal(lvl[1]) if len(lvl) > 1 else None
        order_count = _to_int(lvl[3]) if len(lvl) > 3 else None
        if price is None or qty is None:
            continue
        out.append({"side": side, "price": price, "qty": qty, "order_count": order_count})
    return out

def normalize_orderbook_for_db(ob: Dict[str, Any], inst_id: str, inst_type: str | None) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    header = {
        "inst_type":   inst_type,
        "inst_id":     inst_id,
        "ts":          _ts_ms_to_dt(ob.get("ts")),
        "action":      ob.get("action"),
        "seq_id":      _to_int(ob.get("seqId")),
        "prev_seq_id": _to_int(ob.get("prevSeqId")),
        "checksum":    _to_int(ob.get("checksum")),
    }
    bids = _explode_side('B', ob.get("bids") or [])
    asks = _explode_side('A', ob.get("asks") or [])
    items = bids + asks
    return header, items

# ---------- DB bootstrap ----------
def ensure_database():
    """Create DB if missing, then create schemas/tables/indexes for OKX + Binance (sync)."""
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
            cur.execute(DB_CREATE_TABLE_SQL_OKX)
            cur.execute(DB_CREATE_TABLE_SQL_BINANCE)

# ---------- Writers ----------
class _BaseWriter:
    def __init__(self, dsn: str, insert_sql: str, pool_size: int = 5, batch_max: int = 200, flush_secs: float = 1.0):
        self.insert_sql = insert_sql
        try:
            self.pool = AsyncConnectionPool(conninfo=dsn, min_size=1, max_size=pool_size, open=False)
        except TypeError:
            self.pool = AsyncConnectionPool(dsn, min_size=1, max_size=pool_size, open=False)

        self.queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
        self.batch_max = batch_max
        self.flush_secs = flush_secs
        self._closing = False
        self._opened = False
        self.inserted = 0
        self._label = "writer"

    async def enqueue(self, row: Dict[str, Any]):
        await self.queue.put(row)

    async def run(self):
        if not self._opened:
            await self.pool.open()
            self._opened = True
            print(f"[db] pool opened ({self._label})")

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
                            await cur.executemany(self.insert_sql, batch)
                    self.inserted += len(batch)
                except Exception as e:
                    print(f"[db] insert failed ({self._label}): {e.__class__.__name__}: {e}")
        finally:
            await self.pool.close()

    async def close(self):
        self._closing = True
        await asyncio.sleep(0.2)

class OkxTickerWriter(_BaseWriter):
    def __init__(self, dsn: str, **kw):
        super().__init__(dsn, INSERT_TICKER_SQL_OKX, **kw)
        self._label = "okx-ticker"

class BinanceTickerWriter(_BaseWriter):
    def __init__(self, dsn: str, **kw):
        super().__init__(dsn, INSERT_TICKER_SQL_BINANCE, **kw)
        self._label = "bnc-ticker"

class OrderBookWriter:
    def __init__(self, dsn: str, pool_size: int = 5):
        try:
            self.pool = AsyncConnectionPool(conninfo=dsn, min_size=1, max_size=pool_size, open=False)
        except TypeError:
            self.pool = AsyncConnectionPool(dsn, min_size=1, max_size=pool_size, open=False)
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=5000)
        self._closing = False
        self._opened = False
        self.headers_inserted = 0
        self.items_inserted = 0

    async def enqueue(self, header: Dict[str, Any], items: List[Dict[str, Any]]):
        await self.queue.put((header, items))

    async def run(self):
        if not self._opened:
            await self.pool.open()
            self._opened = True
            print("[db] pool opened (orderbook)")

        try:
            while not self._closing:
                try:
                    header, items = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                try:
                    async with self.pool.connection() as conn:
                        async with conn.transaction():
                            async with conn.cursor() as cur:
                                await cur.execute(f"""
                                    INSERT INTO {PG_DBO_SCHEMA_OKX}.orderbook_header
                                    (inst_type, inst_id, ts, action, seq_id, prev_seq_id, checksum)
                                    VALUES
                                    (%(inst_type)s, %(inst_id)s, %(ts)s, %(action)s, %(seq_id)s, %(prev_seq_id)s, %(checksum)s)
                                    RETURNING id
                                """, header)
                                ob_id = (await cur.fetchone())[0]
                                if items:
                                    items = [{**it, "orderbook_id": ob_id} for it in items]
                                    await cur.executemany(f"""
                                        INSERT INTO {PG_DBO_SCHEMA_OKX}.orderbook_item
                                        (orderbook_id, side, price, qty, order_count)
                                        VALUES
                                        (%(orderbook_id)s, %(side)s, %(price)s, %(qty)s, %(order_count)s)
                                    """, items)
                                    self.items_inserted += len(items)
                                self.headers_inserted += 1
                except Exception as e:
                    print(f"[db] orderbook insert failed: {e.__class__.__name__}: {e}")
        finally:
            await self.pool.close()

    async def close(self):
        self._closing = True
        await asyncio.sleep(0.2)

# ---------- WebSocket clients ----------
class OkxTickerClient:
    def __init__(self, inst_ids: List[str], writer: OkxTickerWriter):
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
                    OKX_PUBLIC_WS, ping_interval=20, ping_timeout=10,
                    max_queue=None, compression="deflate",
                ) as ws:
                    self.ws = ws
                    await ws.send(self.subscribe_payload())
                    print("[okx] subscribed to tickers:", ", ".join(self.inst_ids))
                    backoff = 1
                    await self.read_forever()
            except (asyncio.CancelledError, KeyboardInterrupt):
                break
            except Exception as e:
                print(f"[reconnect] okx tickers {e.__class__.__name__}: {e}. retrying in {backoff}s…")
                await asyncio.sleep(backoff); backoff = min(backoff * 2, 30)

    async def read_forever(self):
        while not self._closing:
            msg = await self.ws.recv()
            data = json.loads(msg)
            if isinstance(data, dict) and data.get("event"):
                continue
            if isinstance(data, dict) and data.get("arg", {}).get("channel") == "tickers":
                for t in data.get("data", []):
                    row = normalize_okx_ticker_for_db(t)
                    await self.writer.enqueue(row)

    async def close(self):
        self._closing = True
        try:
            if self.ws:
                await self.ws.close()
        except Exception:
            pass

class BinanceTickerClient:
    """
    Subscribes to Binance USDⓈ-M Futures individual symbol ticker streams via a combined stream:
      wss://fstream.binance.com/stream?streams=btcusdt@ticker/ethusdt@ticker
    Combined frames are wrapped as {"stream":"<streamName>","data":<rawPayload>}.
    All stream names are lowercase. One connection can carry up to 1024 streams. 
    Refs: Connect + Individual Symbol Ticker Streams docs. 
    """
    def __init__(self, okx_style_inst_ids: List[str], writer: BinanceTickerWriter):
        # derive Binance symbol lists from OKX-style entries
        conv = [_okx_to_binance_symbol(x) for x in okx_style_inst_ids]
        self.binance_db_symbols = [a for (a, b) in conv]      # e.g. 'BTCUSDT'
        self.binance_stream_syms = [b for (a, b) in conv]     # e.g. 'btcusdt'
        self.writer = writer
        self.ws = None
        self._closing = False

        self.streams = "/".join([f"{s}@ticker" for s in self.binance_stream_syms])
        self.url = f"{BINANCE_FUTURES_COMBINED}?streams={self.streams}"

    async def run(self):
        backoff = 1
        while not self._closing:
            try:
                async with websockets.connect(
                    self.url, ping_interval=180, ping_timeout=60, max_queue=None
                ) as ws:
                    self.ws = ws
                    print("[binance] subscribed to @ticker:", ", ".join(self.binance_db_symbols))
                    backoff = 1
                    await self.read_forever()
            except (asyncio.CancelledError, KeyboardInterrupt):
                break
            except Exception as e:
                print(f"[reconnect] binance tickers {e.__class__.__name__}: {e}. retrying in {backoff}s…")
                await asyncio.sleep(backoff); backoff = min(backoff * 2, 30)

    async def read_forever(self):
        while not self._closing:
            msg = await self.ws.recv()
            data = json.loads(msg)

            # Combined stream wrapper
            if isinstance(data, dict) and "data" in data:
                payload = data["data"]
                # Expect e='24hrTicker' payload
                if payload and payload.get("e") == "24hrTicker":
                    row = normalize_binance_ticker_for_db(payload)
                    await self.writer.enqueue(row)

    async def close(self):
        self._closing = True
        try:
            if self.ws:
                await self.ws.close()
        except Exception:
            pass

class OkxOrderBookClient:
    ORDERBOOK_CHANNEL = "books"
    def __init__(self, inst_ids: List[str], writer: OrderBookWriter):
        self.inst_ids = inst_ids
        self.writer = writer
        self.ws = None
        self._closing = False

    def subscribe_payload(self) -> str:
        args = [{"channel": self.ORDERBOOK_CHANNEL, "instId": inst} for inst in self.inst_ids]
        return json.dumps({"op": "subscribe", "args": args})

    async def run(self):
        backoff = 1
        while not self._closing:
            try:
                async with websockets.connect(
                    OKX_PUBLIC_WS, ping_interval=20, ping_timeout=10,
                    max_queue=None, compression="deflate",
                ) as ws:
                    self.ws = ws
                    await ws.send(self.subscribe_payload())
                    print(f"[okx] subscribed to {self.ORDERBOOK_CHANNEL}:", ", ".join(self.inst_ids))
                    backoff = 1
                    await self.read_forever()
            except (asyncio.CancelledError, KeyboardInterrupt):
                break
            except Exception as e:
                print(f"[reconnect] okx orderbook {e.__class__.__name__}: {e}. retrying in {backoff}s…")
                await asyncio.sleep(backoff); backoff = min(backoff * 2, 30)

    async def read_forever(self):
        while not self._closing:
            msg = await self.ws.recv()
            data = json.loads(msg)
            if isinstance(data, dict) and data.get("event"):
                continue
            if isinstance(data, dict) and data.get("arg", {}).get("channel") == self.ORDERBOOK_CHANNEL:
                arg = data.get("arg") or {}
                inst_id = arg.get("instId")
                inst_type = arg.get("instType")
                if not inst_id:
                    continue
                for ob in data.get("data", []):
                    header, items = normalize_orderbook_for_db(ob, inst_id, inst_type)
                    await self.writer.enqueue(header, items)

    async def close(self):
        self._closing = True
        try:
            if self.ws:
                await self.ws.close()
        except Exception:
            pass

# ---------- Main ----------
async def main():
    async def reporter(label: str, writer):
        while True:
            await asyncio.sleep(5)
            if isinstance(writer, OrderBookWriter):
                print(f"[stats:{label}] headers={writer.headers_inserted} items={writer.items_inserted} queue={writer.queue.qsize()}")
            else:
                print(f"[stats:{label}] queue={writer.queue.qsize()} inserted={writer.inserted}")

    # Bootstrap DB off the loop
    await to_thread(ensure_database)

    dsn = (
        f"host={PG_HOST} port={PG_PORT} dbname={PG_DBNAME} "
        f"user={PG_USER} password='{PG_PASSWORD}'"
    )

    # Writers
    okx_ticker_writer = OkxTickerWriter(dsn=dsn, pool_size=5, batch_max=200, flush_secs=1.0)
    bnc_ticker_writer = BinanceTickerWriter(dsn=dsn, pool_size=5, batch_max=200, flush_secs=1.0)

    # Optional order book (OKX)
    ob_writer = None
    if RECORD_ORDERBOOK:
        ob_writer = OrderBookWriter(dsn=dsn, pool_size=5)

    # Start writer tasks + reporters
    tasks = []
    tasks += [asyncio.create_task(okx_ticker_writer.run()),
              asyncio.create_task(reporter("okx-ticker", okx_ticker_writer))]
    tasks += [asyncio.create_task(bnc_ticker_writer.run()),
              asyncio.create_task(reporter("bnc-ticker", bnc_ticker_writer))]
    if ob_writer:
        tasks += [asyncio.create_task(ob_writer.run()),
                  asyncio.create_task(reporter("orderbook", ob_writer))]

    # WS clients
    okx_client = OkxTickerClient(INSTRUMENTS, okx_ticker_writer)
    bnc_client = BinanceTickerClient(INSTRUMENTS, bnc_ticker_writer)  # derives symbols
    ws_okx_task = asyncio.create_task(okx_client.run())
    ws_bnc_task = asyncio.create_task(bnc_client.run())
    tasks += [ws_okx_task, ws_bnc_task]

    if ob_writer:
        ob_client = OkxOrderBookClient(INSTRUMENTS, ob_writer)
        ws_ob_task = asyncio.create_task(ob_client.run())
        tasks.append(ws_ob_task)

    # Graceful shutdown
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    def _graceful(*_):
        print("\n[shutting down] …")
        stop.set()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _graceful)
        except NotImplementedError:
            pass

    await stop.wait()

    # Close clients/writers
    await okx_client.close()
    await bnc_client.close()
    await okx_ticker_writer.close()
    await bnc_ticker_writer.close()
    if RECORD_ORDERBOOK:
        await ob_client.close()
        await ob_writer.close()

    # Cancel tasks
    for t in tasks:
        t.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        for t in tasks:
            await t

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
