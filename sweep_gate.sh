#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="models_ob/okx_ob_ETH-USDT_step5_mid_delta_norm_lgbm"
GATE_DIR="models_ob/okx_ob_ETH-USDT_step5_gate_lgbm"
API="http://macbook-server:8200"
SYMBOL="ETH-USDT"
STEP=5
START=2025-09-10

# thresholds to sweep
thresholds=(0.70 0.75 0.80)

for thr in "${thresholds[@]}"; do
  echo "=== Running sweep for gate-thr=${thr} ==="
  python ob_lgbm_backtest.py \
    --model-dir "${MODEL_DIR}" \
    --gate-model-dir "${GATE_DIR}" \
    --gate-thr "${thr}" \
    --api "${API}" --symbol "${SYMBOL}" --step ${STEP} --start ${START} \
    --auto-min-edge --k-spread 0.3 --spread-cap-bps 1.5 \
    --edge-scale 1.0 --hold 1 --position-frac 0.05 \
    --max-trades-per-hour 5
done
