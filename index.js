// ====== Config ======
const API_BASE = "http://localhost:8000"; // use localhost in the browser (not 0.0.0.0)

// Symbols & periods for the dropdowns
const SYMBOLS = ["BTC-USDT", "ETH-USDT"];
const PERIODS = [1, 10, 100, 1000];

// Map API rows -> DevExtreme candlestick points
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
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`HTTP ${res.status} ${res.statusText} ${text}`);
  }
  return await res.json();
}

function setStatus(text) {
  $("#status-text").text(text);
}

$(() => {
  // --- UI widgets ---
  const symbolSelect = $("#symbol").dxSelectBox({
    dataSource: SYMBOLS,
    label: "Symbol",
    labelMode: "floating",
    value: SYMBOLS[0],
    searchEnabled: true,
    elementAttr: { "aria-label": "Symbol" }
  }).dxSelectBox("instance");

  const periodSelect = $("#period").dxSelectBox({
    dataSource: PERIODS,
    label: "Period (ticks)",
    labelMode: "floating",
    value: PERIODS[2], // default 100
    elementAttr: { "aria-label": "Period" }
  }).dxSelectBox("instance");

  const submitBtn = $("#submit").dxButton({
    text: "Submit",
    type: "default",
    stylingMode: "contained",
    width: 120,
    onClick: () => refreshChart()
  }).dxButton("instance");

  // Chart init
  const chart = $("#chart").dxChart({
    title: {
      text: "Tick Candles",
      font: { size: 16, weight: 600, color: "#cfe3ff" }
    },
    palette: "Material",
    dataSource: [],
    commonSeriesSettings: {
      argumentField: "date",
      type: "candlestick"
    },
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
      position: "right",
      grid: { opacity: 0.15 },
      label: {
        customizeText(e) { return Number(e.value).toLocaleString(); }
      }
    },
    argumentAxis: {
      workdaysOnly: false,
      grid: { opacity: 0.1 },
      valueMarginsEnabled: true,
      label: { format: "hh:mm:ss" }
    },
    crosshair: { enabled: true, color: "#66aaff" },
    tooltip: {
      enabled: true,
      location: "edge",
      customizeTooltip(arg) {
        const d = arg.argument instanceof Date ? arg.argument.toLocaleString() : arg.argumentText;
        return {
          html:
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

  async function refreshChart() {
    const symbol = symbolSelect.option("value");
    const period = periodSelect.option("value");

    submitBtn.option("disabled", true);
    setStatus(`Loading ${symbol} x ${period}…`);

    try {
      const rows = await loadBars(symbol, period);
      const data = toCandles(rows);

      // Update chart
      chart.option({
        title: `Tick Candles • ${symbol} • ${period} ticks`,
        dataSource: data
      });

      // Slight zoom: show latest 200 bars if huge
      if (data.length > 200) {
        const from = data[data.length - 200].date;
        const to = data[data.length - 1].date;
        chart.getArgumentAxis().visualRange({ startValue: from, endValue: to });
      } else {
        chart.getArgumentAxis().visualRange(undefined); // reset
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
