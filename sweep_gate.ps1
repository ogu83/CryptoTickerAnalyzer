# sweep_gate.ps1
$thresholds = 0.70, 0.75, 0.80

foreach ($thr in $thresholds) {
    python ob_lgbm_backtest.py `
        --model-dir models_ob/okx_ob_ETH-USDT_step5_mid_delta_norm_lgbm `
        --gate-model-dir models_ob/okx_ob_ETH-USDT_step5_gate_lgbm `
        --gate-thr $thr `
        --api http://macbook-server:8200 --symbol ETH-USDT --step 5 `
        --auto-min-edge --k-spread 0.3 --spread-cap-bps 1.5 `
        --edge-scale 1.0 --hold 1 --position-frac 0.05 `
        --max-trades-per-hour 5
}
