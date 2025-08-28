// ====== Config ======
const API_BASE = "http://localhost:8200";
const VENUES = [
  { code: "okx", name: "OKX" },
  { code: "bnc", name: "Binance" },
  { code: "compare", name: "OKX vs Binance" },
];
const SYMBOLS_OKX = ["BTC-USDT", "ETH-USDT"];
const SYMBOLS_BINANCE = ["BTCUSDT", "ETHUSDT"]; // UI shows venue-appropriate symbols
const PERIODS = [1, 10, 100, 1000, 10000, 100000];
const DEFAULT_HEIGHT = 500;

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
      const v = e.value;
      // swap symbol list to match venue format preference
      if (v === "bnc") {
        symbolSelect.option({ dataSource: SYMBOLS_BINANCE, value: SYMBOLS_BINANCE[0] });
      } else {
        // for okx or compare, default to OKX-format symbols (server fixups accept both anyway)
        const cur = symbolSelect.option("value");
        const list = SYMBOLS_OKX;
        symbolSelect.option({ dataSource: list, value: list.includes(cur) ? cur : list[0] });
      }
      // toggle second chart visibility
      toggleCharts(v);
    }
  }).dxSelectBox("instance");

  // --- Symbol selector ---
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
  const initialRange = [new Date(now.getTime() - msInDay), now];

  const rangeBox = $("#rangeBox").dxDateRangeBox({
    type: "datetime",
    label: "Date range (UTC)",
    labelMode: "floating",
    value: initialRange,
    showClearButton: true,
    applyValueMode: "useButtons",
    multiView: true,
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

  // --- Charts ---
  const chartOkx = $("#chartOkx").dxChart({
    title: { text: "Tick Candles • OKX", font: { size: 16, weight: 600, color: "#cfe3ff" } },
    dataSource: [],
    size: { height: DEFAULT_HEIGHT },
    commonSeriesSettings: { argumentField: "date", type: "candlestick" },
    series: [{ name: "Price", openValueField: "o", highValueField: "h", lowValueField: "l", closeValueField: "c", reduction: { color: "#e05260" } }],
    legend: { visible: false },
    valueAxis: { position: "right", grid: { opacity: 0.15 }, label: { customizeText: (e) => Number(e.value).toLocaleString() } },
    argumentAxis: { workdaysOnly: false, grid: { opacity: 0.1 }, valueMarginsEnabled: true, label: { format: "MM/dd HH:mm:ss" } },
    crosshair: { enabled: true, color: "#66aaff" },
    tooltip: { enabled: true, location: "edge",
      customizeTooltip: (arg) => {
        const d = arg.argument instanceof Date ? arg.argument.toLocaleString() : arg.argumentText;
        return { html: `<div><div><b>${d}</b></div><div>Open: ${arg.openValue}</div><div>High: ${arg.highValue}</div><div>Low:  ${arg.lowValue}</div><div>Close:${arg.closeValue}</div></div>` };
      }
    },
    export: { enabled: true }
  }).dxChart("instance");

  const chartBnc = $("#chartBnc").dxChart({
    title: { text: "Tick Candles • Binance", font: { size: 16, weight: 600, color: "#cfe3ff" } },
    dataSource: [],
    size: { height: DEFAULT_HEIGHT },
    commonSeriesSettings: { argumentField: "date", type: "candlestick" },
    series: [{ name: "Price", openValueField: "o", highValueField: "h", lowValueField: "l", closeValueField: "c", reduction: { color: "#e05260" } }],
    legend: { visible: false },
    valueAxis: { position: "right", grid: { opacity: 0.15 }, label: { customizeText: (e) => Number(e.value).toLocaleString() } },
    argumentAxis: { workdaysOnly: false, grid: { opacity: 0.1 }, valueMarginsEnabled: true, label: { format: "MM/dd HH:mm:ss" } },
    crosshair: { enabled: true, color: "#66aaff" },
    tooltip: { enabled: true, location: "edge",
      customizeTooltip: (arg) => {
        const d = arg.argument instanceof Date ? arg.argument.toLocaleString() : arg.argumentText;
        return { html: `<div><div><b>${d}</b></div><div>Open: ${arg.openValue}</div><div>High: ${arg.highValue}</div><div>Low:  ${arg.lowValue}</div><div>Close:${arg.closeValue}</div></div>` };
      }
    },
    export: { enabled: true }
  }).dxChart("instance");

  function toggleCharts(venueCode) {
    const both = venueCode === "compare";
    $("#chartBnc").toggle(both);
    // Titles update handled in refreshChart
  }
  toggleCharts(venueSelect.option("value"));

  function applyChartWidth(px) {
    const width = Number(px);
    const size = Number.isFinite(width) && width > 0 ? { width, height: DEFAULT_HEIGHT }
                                                     : { width: undefined, height: DEFAULT_HEIGHT };
    chartOkx.option("size", size);
    chartBnc.option("size", size);
  }

  async function refreshChart() {
    const venue = venueSelect.option("value"); // okx | bnc | compare
    const symbol = symbolSelect.option("value");
    const period = periodSelect.option("value");

    // Date range -> ISO (API expects ISO; end is exclusive server-side)
    const [startDate, endDate] = rangeBox.option("value") || [];
    const startISO = startDate ? toIsoOrNull(startDate) : null;
    const endISO   = endDate   ? toIsoOrNull(endDate)   : null;

    submitBtn.option("disabled", true);
    setStatus(`Loading ${venue.toUpperCase()} ${symbol} x ${period}…`);

    try {
      if (venue === "compare") {
        // Fetch both venues in parallel
        const [okxRows, bncRows] = await Promise.all([
          loadBars({ venue: "okx", symbol, period, startISO, endISO }),
          loadBars({ venue: "bnc", symbol, period, startISO, endISO }),
        ]);
        const okxData = toCandles(okxRows);
        const bncData = toCandles(bncRows);

        chartOkx.option({ title: `Tick Candles • OKX • ${symbol} • ${period} ticks`, dataSource: okxData });
        chartBnc.option({ title: `Tick Candles • Binance • ${symbol} • ${period} ticks`, dataSource: bncData });

        // auto width suggestion based on the larger series
        const suggested = Math.max(MIN_CHART_WIDTH, Math.max(okxData.length, bncData.length) * MIN_PX_PER_CANDLE);
        widthBox.option("value", suggested);
        applyChartWidth(suggested);

        // visual range (last 200) for each chart independently
        if (okxData.length > 200) {
          chartOkx.getArgumentAxis().visualRange({
            startValue: okxData[okxData.length - 200].date, endValue: okxData[okxData.length - 1].date
          });
        } else {
          chartOkx.getArgumentAxis().visualRange(undefined);
        }
        if (bncData.length > 200) {
          chartBnc.getArgumentAxis().visualRange({
            startValue: bncData[bncData.length - 200].date, endValue: bncData[bncData.length - 1].date
          });
        } else {
          chartBnc.getArgumentAxis().visualRange(undefined);
        }

        const ts = new Date().toLocaleTimeString();
        setStatus(`Loaded OKX:${okxData.length} • Binance:${bncData.length} bars • Updated ${ts}`);
      } else {
        // Single venue
        const rows = await loadBars({ venue, symbol, period, startISO, endISO });
        const data = toCandles(rows);

        if (venue === "okx") {
          chartOkx.option({ title: `Tick Candles • OKX • ${symbol} • ${period} ticks`, dataSource: data });
          chartBnc.option({ dataSource: [] });
        } else {
          chartBnc.option({ title: `Tick Candles • Binance • ${symbol} • ${period} ticks`, dataSource: data });
          chartOkx.option({ dataSource: [] });
        }

        const suggested = Math.max(MIN_CHART_WIDTH, data.length * MIN_PX_PER_CANDLE);
        widthBox.option("value", suggested);
        applyChartWidth(suggested);

        const axis = venue === "okx" ? chartOkx.getArgumentAxis() : chartBnc.getArgumentAxis();
        if (data.length > 200) {
          axis.visualRange({ startValue: data[data.length - 200].date, endValue: data[data.length - 1].date });
        } else {
          axis.visualRange(undefined);
        }

        const ts = new Date().toLocaleTimeString();
        setStatus(`Loaded ${venue.toUpperCase()}:${data.length} bars • Updated ${ts}`);
      }
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
