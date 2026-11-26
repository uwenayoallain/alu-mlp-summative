const CLASSES = ["cercospora", "healthy", "miner", "phoma", "rust"];
const SHORT = { cercospora: "Cercospora", healthy: "Healthy", miner: "Leaf miner", phoma: "Brown leaf spot", rust: "Leaf rust" };
const INK = "#222", GREY = "#c8c8c8", ACCENT = "#1f5130";
const charts = {};
const $ = (id) => document.getElementById(id);

Chart.defaults.font.family = "system-ui, sans-serif";
Chart.defaults.color = "#777";
Chart.defaults.plugins.legend.display = false;
Chart.defaults.animation = false;

async function api(path, opts) {
  const r = await fetch(path, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || r.statusText);
  return body;
}
const pct = (v) => (v == null ? "-" : (v * 100).toFixed(1) + "%");
function fmtUptime(s) {
  const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60);
  return (d ? d + "d " : "") + h + "h " + m + "m " + Math.floor(s % 60) + "s";
}
function bar(id, values, opts = {}) {
  if (charts[id]) charts[id].destroy();
  charts[id] = new Chart($(id), {
    type: "bar",
    data: { labels: CLASSES.map((c) => SHORT[c]), datasets: values },
    options: { scales: { x: { stacked: opts.stacked, grid: { display: false } }, y: { stacked: opts.stacked, beginAtZero: true, ticks: { precision: 0 }, title: { display: !!opts.y, text: opts.y } } },
      plugins: { legend: { display: !!opts.legend } } },
  });
}

document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === b));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === b.dataset.tab));
  if (b.dataset.tab === "data") loadData();
  if (b.dataset.tab === "retrain") loadRetrain();
}));

// overview
let uptime = null;
async function loadHealth() {
  try {
    const h = await api("/health");
    $("status").className = "status up";
    $("status").textContent = h.retraining ? "retraining" : "online";
    uptime = { s: h.uptime_seconds, at: Date.now() };
    $("version").textContent = "v" + (h.model_version ?? "-");
    $("instance").textContent = `Instance ${h.instance}, up since ${new Date(h.started_at).toLocaleString()}`;
  } catch {
    $("status").className = "status down";
    $("status").textContent = "offline";
    uptime = null;
  }
}
setInterval(() => { if (uptime) $("uptime").textContent = fmtUptime(uptime.s + (Date.now() - uptime.at) / 1000); }, 1000);

async function loadModel() {
  const m = await api("/model"), x = m.metrics || {};
  $("metrics").innerHTML = [["Accuracy", pct(x.accuracy)], ["Precision", pct(x.precision)], ["Recall", pct(x.recall)],
    ["F1", pct(x.f1)], ["Loss", x.loss?.toFixed(3) ?? "-"]].map(([k, v]) => `<div><span>${k}</span><b>${v}</b></div>`).join("");
  $("versions").querySelector("tbody").innerHTML = (m.history || []).slice().reverse().map((h) =>
    `<tr><td>v${h.version}</td><td>${h.trained_on || "-"}</td><td>${pct(h.metrics.accuracy)}</td><td>${pct(h.metrics.f1)}</td></tr>`).join("");
}

async function loadPredStats() {
  const s = await api("/stats/predictions");
  $("pred-total").textContent = s.total;
  $("pred-latency").textContent = s.avg_latency_ms ? s.avg_latency_ms + " ms" : "-";
  bar("chart-preds", [{ data: CLASSES.map((c) => s.by_class[c]), backgroundColor: INK }]);
}

// predict
let file = null;
function pick(f) {
  if (!f) return;
  file = f;
  $("preview").src = URL.createObjectURL(f);
  $("preview").style.display = "block";
  $("drop-text").style.display = "none";
  $("predict-btn").disabled = false;
}
$("predict-file").addEventListener("change", (e) => pick(e.target.files[0]));
const drop = $("drop");
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); pick(e.dataTransfer.files[0]); });

$("predict-btn").addEventListener("click", async () => {
  const btn = $("predict-btn");
  btn.disabled = true;
  const fd = new FormData(); fd.append("file", file);
  try {
    const r = await api("/predict", { method: "POST", body: fd });
    const rows = Object.entries(r.probabilities).sort((a, b) => b[1] - a[1]).map(([c, p], i) =>
      `<div class="bar-row ${i === 0 ? "top" : ""}"><span>${SHORT[c]}</span><div class="bar"><div style="width:${p * 100}%"></div></div><span>${(p * 100).toFixed(1)}%</span></div>`).join("");
    $("predict-result").innerHTML = `<div class="verdict">${r.label}</div>
      <p class="muted">${(r.confidence * 100).toFixed(1)}% confidence, ${r.latency_ms} ms, model v${r.model_version}</p>${rows}`;
    loadPredStats();
  } catch (e) {
    $("predict-result").innerHTML = `<p class="bad">${e.message}</p>`;
  } finally {
    btn.disabled = false;
  }
});

// data insights
let dataLoaded = false;
async function loadData() {
  if (dataLoaded) return;
  const s = await api("/stats/dataset"), f = s.features;
  dataLoaded = true;
  bar("chart-counts", [
    { label: "train", data: CLASSES.map((c) => s.counts.train[c]), backgroundColor: INK },
    { label: "test", data: CLASSES.map((c) => s.counts.test[c]), backgroundColor: GREY }], { stacked: true, legend: true, y: "images" });
  bar("chart-lesion", [{ data: CLASSES.map((c) => +(f.lesion_ratio[c].mean * 100).toFixed(2)), backgroundColor: INK }], { y: "% of leaf" });
  bar("chart-green", [{ data: CLASSES.map((c) => +(f.green_ratio[c].median * 100).toFixed(1)), backgroundColor: INK }], { y: "% of leaf" });
  bar("chart-texture", [{ data: CLASSES.map((c) => +f.texture[c].mean.toFixed(2)), backgroundColor: INK }], { y: "mean gradient" });
}

// retrain
const sel = $("upload-label");
CLASSES.forEach((c) => sel.insertAdjacentHTML("beforeend", `<option value="${c}">${SHORT[c]}</option>`));
$("upload-files").addEventListener("change", (e) => {
  const n = e.target.files.length;
  $("upload-text").textContent = n ? `${n} file${n > 1 ? "s" : ""}` : "Choose files";
});
$("upload-btn").addEventListener("click", async () => {
  const files = $("upload-files").files;
  if (!files.length) return;
  const fd = new FormData();
  [...files].forEach((f) => fd.append("files", f));
  if (sel.value) fd.append("label", sel.value);
  const btn = $("upload-btn"); btn.disabled = true;
  try {
    const r = await api("/upload", { method: "POST", body: fd });
    $("upload-result").innerHTML = `<p>Saved ${r.saved_total}.` +
      (r.rejected.length ? ` <span class="bad">Rejected ${r.rejected.length}: ${r.rejected.slice(0, 3).map((x) => x.file + " (" + x.reason + ")").join(", ")}</span>` : "") +
      (r.auto_retrain_started ? " Retraining started automatically." : "") + "</p>";
    $("upload-files").value = ""; $("upload-text").textContent = "Choose files";
    loadRetrain();
  } catch (e) {
    $("upload-result").innerHTML = `<p class="bad">${e.message}</p>`;
  } finally {
    btn.disabled = false;
  }
});

let poll = null;
async function loadRetrain() {
  const [u, st] = await Promise.all([api("/uploads"), api("/retrain/status")]);
  bar("chart-pending", [
    { label: "waiting", data: CLASSES.map((c) => u.pending[c]), backgroundColor: INK },
    { label: "used", data: CLASSES.map((c) => u.used[c]), backgroundColor: GREY }], { stacked: true, legend: true });
  $("threshold-text").textContent = `${u.pending_total} waiting. Auto-retrain at ${u.retrain_threshold}.`;
  $("retrain-btn").disabled = st.running || u.pending_total === 0;
  $("retrain-btn").textContent = st.running ? "Retraining..." : "Retrain";
  $("runs").querySelector("tbody").innerHTML = st.runs.map((r) => `<tr><td>${r.id}</td><td>${r.trigger}</td><td>${r.status}</td>
    <td>${r.n_new_images || "-"}</td><td>${pct(r.accuracy_before)}</td><td>${pct(r.accuracy_after)}</td><td>${pct(r.f1_after)}</td>
    <td>${r.status === "done" ? (r.promoted ? "v" + r.model_version : "no") : "-"}</td></tr>`).join("")
    || `<tr><td colspan="8" class="muted">None yet</td></tr>`;
  $("run-log").textContent = st.runs[0]?.log || "";
  clearTimeout(poll);
  if (st.running) poll = setTimeout(() => { loadRetrain(); loadHealth(); }, 2000);
  else if (st.runs[0]?.status === "done") { loadModel(); loadHealth(); }
}
$("retrain-btn").addEventListener("click", async () => {
  try { await api("/retrain", { method: "POST" }); } catch (e) { alert(e.message); }
  loadRetrain();
});

// evaluate
$("eval-btn").addEventListener("click", async () => {
  const btn = $("eval-btn"); btn.disabled = true;
  try {
    const r = await api("/evaluate?n_images=30"), t = r.test_set, l = r.live_check;
    const cm = `<table class="cm"><tr><th></th>${CLASSES.map((c) => `<th>${SHORT[c]}</th>`).join("")}</tr>` +
      t.confusion_matrix.map((row, i) => `<tr><th>${SHORT[CLASSES[i]]}</th>${row.map((v, j) => `<td class="${i === j ? "diag" : ""}">${v}</td>`).join("")}</tr>`).join("") + "</table>";
    const per = CLASSES.map((c) => { const p = t.per_class[c]; return `<tr><td>${SHORT[c]}</td><td>${pct(p.precision)}</td><td>${pct(p.recall)}</td><td>${pct(p["f1-score"])}</td><td>${p.support}</td></tr>`; }).join("");
    $("eval-result").innerHTML = `
      <h2>Model v${r.model_version}, ${t.n_samples} test images</h2>
      <div class="stats">${[["Accuracy", pct(t.accuracy)], ["Precision", pct(t.precision)], ["Recall", pct(t.recall)], ["F1", pct(t.f1)], ["Loss", t.loss.toFixed(3)]]
        .map(([k, v]) => `<div><span>${k}</span><b>${v}</b></div>`).join("")}</div>
      <div class="cols">
        <div><h2>Per class</h2><table><tr><th></th><th>Precision</th><th>Recall</th><th>F1</th><th>n</th></tr>${per}</table></div>
        <div><h2>Confusion matrix (rows: true)</h2>${cm}</div>
      </div>
      ${l ? `<p class="muted">Live check, ${l.n_samples} images through the full pipeline: accuracy ${pct(l.accuracy)}, ${r.seconds}s.</p>` : ""}`;
  } catch (e) {
    $("eval-result").innerHTML = `<p class="bad">${e.message}</p>`;
  } finally {
    btn.disabled = false;
  }
});

loadHealth(); loadModel().catch(() => {}); loadPredStats().catch(() => {});
setInterval(loadHealth, 15000);
setInterval(() => loadPredStats().catch(() => {}), 20000);
