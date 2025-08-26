import asyncio
import json
import signal
import sys
from datetime import datetime, timezone
from typing import Dict, List

import pandas as pd
import websockets

OKX_PUBLIC_WS = "wss://ws.okx.com:8443/ws/v5/public"

# ✅ Choose the instruments you want to track
INSTRUMENTS = [
    "BTC-USDT"
    # "ETH-USDT",
    # futures/options examples:
    # "BTC-USDT-250328",  # futures
    # "BTC-USD-250328-60000-C",  # option
]

# Columns we’ll keep (OKX may add more fields in the future)
TICKER_NUM_COLS = [
    "last", "lastSz", "askPx", "askSz", "bidPx", "bidSz",
    "open24h", "high24h", "low24h", "volCcy24h", "vol24h",
    "sodUtc0", "sodUtc8"
]

def _to_float(x):
    try:
        return float(x) if x is not None and x != "" else None
    except Exception:
        return None

def normalize_row(d: dict) -> dict:
    """Convert OKX ticker dict (strings) -> typed row for pandas."""
    row = {
        "instType": d.get("instType"),
        "instId": d.get("instId"),
        "ts": pd.to_datetime(int(d["ts"]), unit="ms", utc=True) if d.get("ts") else pd.NaT,
    }
    for k in TICKER_NUM_COLS:
        row[k] = _to_float(d.get(k))
    return row

def build_subscribe_payload(inst_ids: List[str]) -> str:
    args = [{"channel": "tickers", "instId": inst} for inst in inst_ids]
    return json.dumps({"op": "subscribe", "args": args})

class OkxTickerClient:
    def __init__(self, inst_ids: List[str]):
        self.inst_ids = inst_ids
        self.rows: Dict[str, dict] = {}
        self.ws = None
        self._closing = False

    async def run(self):
        backoff = 1
        while not self._closing:
            try:
                async with websockets.connect(
                    OKX_PUBLIC_WS,
                    ping_interval=20,    # keepalive
                    ping_timeout=10,     # fail fast if unreachable
                    max_queue=None,      # don't drop messages
                    compression="deflate"
                ) as ws:
                    self.ws = ws
                    await self.subscribe()
                    backoff = 1  # reset on success
                    await self.read_forever()
            except (asyncio.CancelledError, KeyboardInterrupt):
                break
            except Exception as e:
                print(f"[reconnect] {e.__class__.__name__}: {e}. Retrying in {backoff}s...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    async def subscribe(self):
        payload = build_subscribe_payload(self.inst_ids)
        await self.ws.send(payload)
        print(f"[okx] subscribed -> {payload}")

    async def read_forever(self):
        last_print = 0
        while not self._closing:
            msg = await self.ws.recv()
            data = json.loads(msg)

            # ignore events/ack
            if isinstance(data, dict) and data.get("event"):
                # {'event': 'subscribe', 'arg': {'channel': 'tickers', 'instId': 'BTC-USDT'}, ...}
                continue

            # Market data
            if isinstance(data, dict) and data.get("arg", {}).get("channel") == "tickers":
                for t in data.get("data", []):
                    row = normalize_row(t)
                    self.rows[row["ts"]] = row

            # Periodically print/update the pandas table
            now = asyncio.get_event_loop().time()
            if now - last_print >= 2 and self.rows:
                last_print = now
                self.print_table()

    def print_table(self):
        df = pd.DataFrame.from_dict(self.rows, orient="index").sort_index()
        # Optional: compute mid price & spread (in quote units when spot)
        df["midPx"] = (df["askPx"] + df["bidPx"]) / 2
        df["spread"] = df["askPx"] - df["bidPx"]
        # Reorder columns for readability
        cols = [
            "instId", "instType", "ts", "last", "midPx", "spread",
            "bidPx", "bidSz", "askPx", "askSz",
            "open24h", "high24h", "low24h", "vol24h", "volCcy24h",
        ]
        cols = [c for c in cols if c in df.columns]
        print("\n" + "=" * 80)
        # print(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC") + " • OKX tickers")
        print(df[cols].to_string(float_format=lambda x: f"{x:.8g}"))

    async def close(self):
        self._closing = True
        try:
            if self.ws:
                await self.ws.close()
        except Exception:
            pass

async def main():
    client = OkxTickerClient(INSTRUMENTS)

    loop = asyncio.get_running_loop()
    stop = asyncio.Event()

    def _graceful(*_):
        print("\n[okx] shutting down…")
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _graceful)
        except NotImplementedError:
            pass

    task = asyncio.create_task(client.run())
    await stop.wait()
    await client.close()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

if __name__ == "__main__":
    import contextlib
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
