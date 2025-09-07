$qs = 0.995,0.999
$cons = 2,3
foreach ($q in $qs) {
  foreach ($c in $cons) {
    python ob_anomaly_backtest.py `
      --model-dir models_ob_anom\okx_ob_anom_ETH-USDT_step5_w64 `
      --api http://macbook-server:8200 --symbol ETH-USDT --step 5 `
      --usequantile --quantile $q --consec $c --cooldown 20 `
      --dir-alpha 0.5 --tilt-ema 5 --hold 3 --position-frac 0.05 `
      --fee-bps 2 --slip-bps 1
  }
}