# OKX & Binance Tick Collector + OHLCV API + Web UI

Stream live tickers from **OKX** and **Binance USDⓈ-M Futures** into PostgreSQL, aggregate them into N-tick OHLCV bars via a SQL function, expose them through a **FastAPI** endpoint, and visualize with a lightweight **DevExtreme (jQuery)** UI.

<img width="1819" height="1122" alt="msedge_RuNIj1FRoQ" src="https://github.com/user-attachments/assets/4a69b868-0ceb-4847-9f54-db1c7c08f24a" />

---

## Contents
- [Features](#features)
- [Requirements](#requirements)
- [Environment Setup](#environment-setup)
- [Configuration](#configuration)
- [Run the Collector](#run-the-collector)
- [Run the API](#run-the-api)
- [Run the Web UI](#run-the-web-ui)
- [API Reference](#api-reference)
- [Notes & Tips](#notes--tips)
- [Troubleshooting](#troubleshooting)

---

## Features
- Ingests **OKX** tickers and **Binance** *@ticker* streams via WebSocket.
- Persists to PostgreSQL:
  - `okx.ticker`
  - `bnc.ticker`
- (Optional) Persists **OKX Order Book** (`okx.orderbook_header` / `okx.orderbook_item`).
- SQL function “parameterized view” for **N-tick bars** (OHLCV).
- REST endpoint: `GET /tick-chart?venue=...&symbol=...&period=...`
- Web UI:
  - Venue dropdown: **OKX**, **Binance**, or **OKX vs Binance (compare)**
  - Symbol dropdown (auto-formats symbol per venue, server accepts both styles)
  - Period dropdown (1 / 10 / 100 / 1000 ticks)
  - Date range picker (UTC) for server-side filtering
  - Candlestick charts (1 or 2 stacked in compare mode)
  - Runtime chart-width control

---

## Requirements
- **Python** 3.9+ (Windows & Linux tested)
- **PostgreSQL** 12+ (network-accessible from the machine running the collector & API)
- Network egress to:
  - `wss://ws.okx.com:8443/ws/v5/public`
  - `wss://fstream.binance.com/stream`

---

## Environment Setup

### 1) Create a virtual environment

**Windows (PowerShell):**
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

### 2) Install Python dependencies
```bash
pip install "psycopg[binary,pool]" websockets fastapi uvicorn
```

---

## Configuration

Open `tickers_to_postgres.py` and set your Postgres connection info and instruments:

```python
# ---------------- Postgres connection info ----------------
PG_HOST = "macbook-server"
PG_PORT = 5432
PG_USER = "postgres"
PG_PASSWORD = "Postgres2839*"
PG_DBNAME = "CryptoTickers"  # target DB name

# Schemas are fixed in code:
# PG_DBO_SCHEMA_OKX = "okx"
# PG_DBO_SCHEMA_BINANCE = "bnc"

INSTRUMENTS = [
    "BTC-USDT",
    "ETH-USDT",
]
```

> The script auto-creates the database (if missing), schemas, and tables on startup.

**Optional (Order Book):**  
In `tickers_to_postgres.py`, set:
```python
RECORD_ORDERBOOK = True
```
to capture OKX order book (`books` channel) as well.

---

## Run the Collector

From your activated venv:
```bash
# Windows PowerShell or Linux
python3 ./tickers_to_postgres.py
# (On Windows, you can also use: python .	ickers_to_postgres.py)
```

You should see logs like:
```
[db] pool opened (okx-ticker)
[db] pool opened (bnc-ticker)
[okx] subscribed to tickers: BTC-USDT, ETH-USDT
[binance] subscribed to @ticker: BTCUSDT, ETHUSDT
[stats:okx-ticker] queue=... inserted=...
[stats:bnc-ticker] queue=... inserted=...
```

---

## Run the API

From your activated venv:
```bash
uvicorn ohlcv_api:app --host 0.0.0.0 --port 8200
```

The API connects to the same database and exposes `GET /tick-chart`.

> If you changed ports, ensure your web UI points `API_BASE` to the correct one.

---

## Run the Web UI

Serve the `index.html` & `index.js` locally (simple static server):

```bash
# from the directory containing index.html
python3 -m http.server 8201
```

Open your browser to:  
`http://localhost:8201/`

**Make sure** the `API_BASE` constant in `index.js` matches your API URL (default: `http://localhost:8200`).

---

## API Reference

### `GET /tick-chart`

Returns an array of OHLCV bars aggregated from the underlying tick table via the SQL function `ticker_ohlcv(symbol, period)` (function exists in both `okx` and `bnc` schemas).

**Query Parameters**
- `venue` — `okx` (default), `bnc`, or `compare` (UI uses `compare` to load both venues separately)
- `symbol` — instrument identifier
  - OKX style: `BTC-USDT`, `ETH-USDT`
  - Binance style: `BTCUSDT`, `ETHUSDT`
  - **Symbol fixups are automatic** (server-side):
    - For `venue=bnc`, a `-` is removed (e.g., `BTC-USDT` → `BTCUSDT`)
    - For `venue=okx`, a `-` is inserted before the last 3 or 4 chars if missing (`BTCUSDT` → `BTC-USDT`)
- `period` — integer tick group size (e.g., 1 / 10 / 100 / 1000)
- `start` (optional) — ISO datetime (inclusive), e.g. `2025-08-26T00:00:00Z`
- `end` (optional) — ISO datetime (exclusive), e.g. `2025-08-27T00:00:00Z`

**Response**
```json
[
  {
    "time": "2025-08-26T12:34:56Z",
    "open":  64000.1,
    "close": 64010.3,
    "high":  64050.0,
    "low":   63990.5,
    "volume": 12.3456
  }
]
```

**Examples**
```
/tick-chart?symbol=BTC-USDT&period=100
/tick-chart?venue=bnc&symbol=BTCUSDT&period=1000
/tick-chart?symbol=ETH-USDT&period=10&start=2025-08-26T00:00:00Z&end=2025-08-27T00:00:00Z
```

---

## Notes & Tips
- **Schemas used**
  - OKX: `okx.ticker` (and order book tables if enabled)
  - Binance: `bnc.ticker`
- The OHLCV function (`ticker_ohlcv`) exists in **both** schemas; the API selects the right schema by `venue`.
- The Web UI **Compare** mode calls the API **twice** (OKX + Binance) for the same symbol/period/date range and renders two stacked candlestick charts.
- **Windows asyncio**: the collector sets `WindowsSelectorEventLoopPolicy` automatically for psycopg async compatibility.

---

## Troubleshooting

- **No rows inserted**
  - Ensure the venv is active and packages are installed in the **same** environment you run the script from.
  - Check network access to OKX/Binance WebSocket endpoints.
  - Verify Postgres host/port/user/password are correct and reachable.

- **Event loop warnings on Windows**
  - Usually harmless on shutdown. Ensure you’re running within the venv and using Python 3.9+.

- **CORS / Browser cannot reach API**
  - The API enables permissive CORS for `GET`. Ensure the port (default **8200**) is reachable (firewall).

- **UI doesn’t load data**
  - Confirm `API_BASE` in `index.js` matches your API URL (`http://localhost:8200` by default).
  - Check the browser console network tab for error responses.

---

Happy trading data building! 🚀
