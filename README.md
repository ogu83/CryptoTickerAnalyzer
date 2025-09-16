# CryptoTickerAnalyzer

Research framework for streaming **crypto tick & order book data**, training **ML/AI models** on features derived from the book, and running **backtests** with anomaly gates and profitability filters.

---

## Features

* **Data ingestion**

  * Stream live tickers from **OKX** and **Binance USDⓈ-M Futures**.
  * Persist to PostgreSQL (`okx.ticker`, `bnc.ticker`).
  * Optional: capture **OKX order book** (`okx.orderbook_header` + `okx.orderbook_item`).
* **Database utilities**

  * SQL function `okx.get_ob_top_paged(symbol, start, end, step, page, pagesize)` for efficient *top-of-book* retrieval.
  * API endpoint `/ob-top` exposes it as JSON (with computed `mid`, `spread`, `imbalance`, `microprice`).
* **Models**

  * **Neural (Keras LSTM):** predict normalized mid-price deltas.
  * **Anomaly Detection:** train on order book reconstruction error, backtest anomaly-gated signals.
  * **Tree-based (LightGBM):**

    * **Regressor**: predicts mid-price delta norm.
    * **Classifier Gate**: learns which trades are profitable given regressor outputs + market state.
* **Backtesting**

  * `ob_anomaly_backtest.py` — anomaly-based gate with rolling quantiles, dynamic min-edge logic.
  * `ob_lgbm_backtest.py` — LightGBM regressor + optional gate, cost-aware filters.
  * Sweep scripts (`sweep_gate.sh`, `sweep_gate.ps1`) to explore gate thresholds.
* **Visualization**

  * Plots like `ob_pred_vs_actual.png`, `*_equity_curve.png` saved automatically for diagnostics.

---

## Requirements

* **Python** 3.9+ (tested on Windows, Linux)
* **PostgreSQL** 12+
* Packages:

  ```bash
  pip install -r requirements.txt
  ```

  (Key: `tensorflow`, `lightgbm`, `scikit-learn`, `pandas`, `matplotlib`, `fastapi`, `uvicorn`, `psycopg[binary,pool]`)

---

## Workflow

### 1. Collect Data

```bash
python tickers_to_postgres.py
```

Streams OKX/Binance tickers (and order book if enabled) into Postgres.

### 2. Run the API

```bash
uvicorn ohlcv_api:app --host 0.0.0.0 --port 8200
```

Endpoints:

* `GET /tick-chart` — OHLCV bars.
* `GET /ob-top` — paginated order book top.

### 3. Train Models

* **Neural LSTM**

  ```bash
  python train_orderbook.py --algo keras --symbol ETH-USDT --step 5 --target mid_delta_norm --plot
  ```
* **LightGBM Regressor**

  ```bash
  python train_orderbook.py --algo lgbm --symbol ETH-USDT --step 5 --target mid_delta_norm --plot
  ```
* **LightGBM Gate**

  ```bash
  python train_orderbook.py --algo lgbm_gate --symbol ETH-USDT --step 5 --gate-base-bps 7.0 --k-spread 0.3 --cv 5
  ```

### 4. Backtest

* **Anomaly strategy**

  ```bash
  python ob_anomaly_backtest.py --model-dir models_ob_anom/okx_ob_anom_ETH-USDT_step5_w64 \
    --reg-model-dir models_ob/okx_ob_ETH-USDT_step5_mid_delta_norm_w64 \
    --symbol ETH-USDT --step 5 --roll-quantile 0.9 --roll-window 2h --plot
  ```
* **LightGBM regressor**

  ```bash
  python ob_lgbm_backtest.py --model-dir models_ob/okx_ob_ETH-USDT_step5_mid_delta_norm_lgbm \
    --symbol ETH-USDT --step 5 --auto-min-edge --k-spread 0.3 --spread-cap-bps 1.5 --plot
  ```
* **With gate**

  ```bash
  python ob_lgbm_backtest.py --model-dir models_ob/okx_ob_ETH-USDT_step5_mid_delta_norm_lgbm \
    --gate-model-dir models_ob/okx_ob_ETH-USDT_step5_gate_lgbm \
    --symbol ETH-USDT --step 5 --gate-thr 0.7 --plot
  ```

### 5. Sweep Gate Thresholds

```bash
bash sweep_gate.sh
# or on Windows
.\sweep_gate.ps1
```

---

## Notes

* **Database size**: order book tables can grow very large (50GB+ in weeks).
  Use paged queries (`page`, `pagesize`) and downsampling (`step`) aggressively.
* **Costs awareness**: backtests include trading fees (`fee_bps`) and slippage (`slip_bps`).
* **Gate philosophy**: the regressor provides direction & edge estimate; the classifier gate prunes bad trades.
* **Diagnostics**: always inspect `equity_curve.png` and logs for hit-rate, pass-rate, and min-edge thresholds.
