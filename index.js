// ====== Config ======
const API_BASE = "http://localhost:8000";
const SYMBOLS = ["BTC-USDT", "ETH-USDT"];
const PERIODS = [1, 10, 100, 1000];
const DEFAULT_HEIGHT = 640;

// minimum pixels per candle for auto width suggestion
const MIN_PX_PER_CANDLE = 4;
const MIN_CHART_WIDTH = 600;

function toCandles(rows) {
  return rows.map(r => ({
    date: r.time ? new Date(r.time) : null,
    o: Number(r.open ?? 0),
    h: Number(r.high ?? 0),
    l: Number(r.low ?? 0),
    c: Number(r.close ?? 0),
    v: Number(r.volume ?? 0),
  }));
}

async function loadBars(symbol, period) {
  const url = `${API_BASE}/tick-chart?symbol=${encodeURIComponent(symbol)}&period=${period}`;
  const res = await fetch(url, { headers: { "Accept": "application/json" }});
  if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);
  return res.json();
}

function setStatus(text) { $("#status-text").text(text); }

$(() => {
  // --- Selectors & Button ---
  const symbolSelect = $("#symbol").dxSelectBox({
    dataSource: SYMBOLS, label: "Symbol", labelMode: "floating",
    value: SYMBOLS[0], searchEnabled: true
  }).dxSelectBox("instance");

  const periodSelect = $("#period").dxSelectBox({
    dataSource: PERIODS, label: "Period (ticks)", labelMode: "floating",
    value: PERIODS[2]
  }).dxSelectBox("instance");

  const submitBtn = $("#submit").dxButton({
    text: "Submit", type: "default", stylingMode: "contained", width: 120,
    onClick: () => refreshChart()
  }).dxButton("instance");

  // --- Width controller (NumberBox) ---
  const widthBox = $("#widthBox").dxNumberBox({
    label: "Chart width (px)", labelMode: "floating",
    value: 1000, min: 300, max: 20000, step: 100, showSpinButtons: true,
    onEnterKey: () => applyChartWidth(widthBox.option("value")),
    onValueChanged: (e) => {
      // live-apply on change; comment this out if you prefer "Apply" button
      if (e && e.value) applyChartWidth(e.value);
    }
  }).dxNumberBox("instance");

  // --- Chart ---
  const chart = $("#chart").dxChart({
    title: { text: "Tick Candles", font: { size: 16, weight: 600, color: "#cfe3ff" } },
    dataSource: [],
    size: { height: DEFAULT_HEIGHT }, // width will be set dynamically
    commonSeriesSettings: { argumentField: "date", type: "candlestick" },
    series: [{ name: "Price", openValueField: "o", highValueField: "h", lowValueField: "l", closeValueField: "c",
               reduction: { color: "#e05260" } }],
    legend: { visible: false },
    valueAxis: {
      position: "right", grid: { opacity: 0.15 },
      label: { customizeText: (e) => Number(e.value).toLocaleString() }
    },
    argumentAxis: { workdaysOnly: false, grid: { opacity: 0.1 }, valueMarginsEnabled: true, label: { format: "M/d h:m:ss" } },
    crosshair: { enabled: true, color: "#66aaff" },
    tooltip: {
      enabled: true, location: "edge",
      customizeTooltip: (arg) => {
        const d = arg.argument instanceof Date ? arg.argument.toLocaleString() : arg.argumentText;
        return { html:
          `<div>
             <div><b>${d}</b></div>
             <div>Open: ${arg.openValue}</div>
             <div>High: ${arg.highValue}</div>
             <div>Low:  ${arg.lowValue}</div>
             <div>Close:${arg.closeValue}</div>
           </div>`
        };
      }
    },
    export: { enabled: true }
  }).dxChart("instance");

  function applyChartWidth(px) {
    const width = Number(px);
    if (Number.isFinite(width) && width > 0) {
      chart.option("size", { width, height: DEFAULT_HEIGHT });
      // container scrolls horizontally if width > viewport
    } else {
      // auto width (unset), will fit container
      chart.option("size", { width: undefined, height: DEFAULT_HEIGHT });
    }
  }

  async function refreshChart() {
    const symbol = symbolSelect.option("value");
    const period = periodSelect.option("value");

    submitBtn.option("disabled", true);
    setStatus(`Loading ${symbol} x ${period}…`);

    try {
      const rows = await loadBars(symbol, period);
      const data = toCandles(rows);

      chart.option({
        title: `Tick Candles • ${symbol} • ${period} ticks`,
        dataSource: data
      });

      // auto-suggest a width so candles have some minimum visual width
      const suggested = Math.max(MIN_CHART_WIDTH, data.length * MIN_PX_PER_CANDLE);
      widthBox.option("value", suggested);
      applyChartWidth(suggested);

      // show last 200 bars if many
      if (data.length > 200) {
        const from = data[data.length - 200].date;
        const to = data[data.length - 1].date;
        chart.getArgumentAxis().visualRange({ startValue: from, endValue: to });
      } else {
        chart.getArgumentAxis().visualRange(undefined);
      }

      setStatus(`Loaded ${data.length} bars • Updated ${new Date().toLocaleTimeString()}`);
    } catch (err) {
      DevExpress.ui.notify({ message: `Error: ${err.message}`, type: "error", displayTime: 4000 });
      setStatus("Load failed");
    } finally {
      submitBtn.option("disabled", false);
    }
  }

  // Initial load
  refreshChart();
});
