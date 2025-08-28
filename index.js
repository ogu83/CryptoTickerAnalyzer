// ====== Config ======
const API_BASE = "http://localhost:8000";
const VENUES = [
  { code: "okx", name: "OKX" },
  { code: "bnc", name: "Binance" },
];
const SYMBOLS_OKX = ["BTC-USDT", "ETH-USDT"];
const SYMBOLS_BINANCE = ["BTCUSDT", "ETHUSDT"]; // UI shows venue-appropriate symbols
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

function toIsoOrNull(date) {
  if (!date) return null;
  try { return new Date(date).toISOString(); } catch { return null; }
}

async function loadBars({ venue, symbol, period, startISO, endISO }) {
  const params = new URLSearchParams();
  params.set("venue", venue);
  params.set("symbol", symbol);
  params.set("period", String(period));
  if (startISO) params.set("start", startISO);
  if (endISO) params.set("end", endISO);

  const url = `${API_BASE}/tick-chart?${params.toString()}`;
  const res = await fetch(url, { headers: { "Accept": "application/json" }});
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`HTTP ${res.status} ${res.statusText} ${text}`);
  }
  return res.json();
}

function setStatus(text) { $("#status-text").text(text); }

$(() => {
  // --- Venue selector ---
  const venueSelect = $("#venue").dxSelectBox({
    dataSource: VENUES, displayExpr: "name", valueExpr: "code",
    label: "Venue", labelMode: "floating",
    value: VENUES[0].code, elementAttr: { "aria-label": "Venue" },
    onValueChanged: e => {
      // swap symbol list to match venue format preference
      const v = e.value;
      const list = v === "bnc" ? SYMBOLS_BINANCE : SYMBOLS_OKX;
      symbolSelect.option({ dataSource: list, value: list[0] });
    }
  }).dxSelectBox("instance");

  // --- Symbol selector (changes with venue) ---
  const symbolSelect = $("#symbol").dxSelectBox({
    dataSource: SYMBOLS_OKX, label: "Symbol", labelMode: "floating",
    value: SYMBOLS_OKX[0], searchEnabled: true, elementAttr: { "aria-label": "Symbol" }
  }).dxSelectBox("instance");

  // --- Period selector ---
  const periodSelect = $("#period").dxSelectBox({
    dataSource: PERIODS, label: "Period (ticks)", labelMode: "floating",
    value: PERIODS[2], elementAttr: { "aria-label": "Period" }
  }).dxSelectBox("instance");

  // --- Date range (with time) ---
  const now = new Date();
  const msInDay = 24 * 60 * 60 * 1000;
  const initialRange = [new Date(now.getTime() - msInDay), now]; // last 1 day by default

  const rangeBox = $("#rangeBox").dxDateRangeBox({
    type: "datetime",
    label: "Date range (UTC)",
    labelMode: "floating",
    value: initialRange,
    showClearButton: true,
    applyValueMode: "useButtons",
    multiView: true,
    // displayFormat affects only how it's shown; API receives ISO strings
    displayFormat: "yyyy-MM-dd HH:mm",
  }).dxDateRangeBox("instance");

  // --- Submit button ---
  const submitBtn = $("#submit").dxButton({
    text: "Submit", type: "default", stylingMode: "contained", width: 120,
    onClick: () => refreshChart()
  }).dxButton("instance");

  // --- Width controller (NumberBox) ---
  const widthBox = $("#widthBox").dxNumberBox({
    label: "Chart width (px)", labelMode: "floating",
    value: 1200, min: 300, max: 20000, step: 100, showSpinButtons: true,
    onEnterKey: () => applyChartWidth(widthBox.option("value")),
    onValueChanged: (e) => {
      if (e && e.value) applyChartWidth(e.value);
    }
  }).dxNumberBox("instance");

  // --- Chart ---
  const chart = $("#chart").dxChart({
    title: { text: "Tick Candles", font: { size: 16, weight: 600, color: "#cfe3ff" } },
    dataSource: [],
    size: { height: DEFAULT_HEIGHT }, // width will be set dynamically
    commonSeriesSettings: { argumentField: "date", type: "candlestick" },
    series: [{
      name: "Price",
      openValueField: "o",
      highValueField: "h",
      lowValueField: "l",
      closeValueField: "c",
      reduction: { color: "#e05260" }
    }],
    legend: { visible: false },
    valueAxis: {
      position: "right", grid: { opacity: 0.15 },
      label: { customizeText: (e) => Number(e.value).toLocaleString() }
    },
    argumentAxis: {
      workdaysOnly: false, grid: { opacity: 0.1 },
      valueMarginsEnabled: true, label: { format: "MM/dd hh:mm:ss" }
    },
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
           </div>` };
      }
    },
    export: { enabled: true }
  }).dxChart("instance");

  function applyChartWidth(px) {
    const width = Number(px);
    if (Number.isFinite(width) && width > 0) {
      chart.option("size", { width, height: DEFAULT_HEIGHT });
    } else {
      chart.option("size", { width: undefined, height: DEFAULT_HEIGHT });
    }
  }

  async function refreshChart() {
    const venue = venueSelect.option("value");           // okx | bnc
    const symbol = symbolSelect.option("value");
    const period = periodSelect.option("value");

    // Date range -> ISO (API expects ISO; end is exclusive server-side)
    const [startDate, endDate] = rangeBox.option("value") || [];
    const startISO = startDate ? toIsoOrNull(startDate) : null;
    const endISO   = endDate   ? toIsoOrNull(endDate)   : null;

    submitBtn.option("disabled", true);
    setStatus(`Loading ${venue.toUpperCase()} ${symbol} x ${period}…`);

    try {
      const rows = await loadBars({ venue, symbol, period, startISO, endISO });
      const data = toCandles(rows);

      chart.option({
        title: `Tick Candles • ${venue.toUpperCase()} • ${symbol} • ${period} ticks`,
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

      const timeStamp = new Date().toLocaleTimeString();
      setStatus(`Loaded ${data.length} bars • Updated ${timeStamp}`);
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
