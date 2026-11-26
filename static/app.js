const CLASSES = ["cercospora", "healthy", "miner", "phoma", "rust"];
const SHORT = { cercospora: "Cercospora", healthy: "Healthy", miner: "Leaf miner", phoma: "Brown leaf spot", rust: "Leaf rust" };
const INK = "#222", GREY = "#c8c8c8";
const $ = (id) => document.getElementById(id);

if (window.Chart) {
  Chart.defaults.font.family = "system-ui, sans-serif";
  Chart.defaults.color = "#777";
  Chart.defaults.plugins.legend.display = false;
  Chart.defaults.animation = false;
}

async function api(path, opts) {
  const r = await fetch(path, { cache: "no-store", ...opts });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || r.statusText);
  return body;
}
const pct = (v) => (v == null ? "-" : (v * 100).toFixed(1) + "%");
const tiles = (pairs) => pairs.map(([k, v]) => `<div><span>${k}</span><b>${v}</b></div>`).join("");
function fmtUptime(s) {
  const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60);
  return (d ? d + "d " : "") + h + "h " + m + "m " + Math.floor(s % 60) + "s";
}
function bar(id, datasets, opts = {}) {
  if (!window.Chart) return;
  new Chart($(id), {
    type: "bar",
    data: { labels: CLASSES.map((c) => SHORT[c]), datasets },
    options: { scales: { x: { stacked: opts.stacked, grid: { display: false } }, y: { stacked: opts.stacked, beginAtZero: true, title: { display: !!opts.y, text: opts.y } } },
      plugins: { legend: { display: !!opts.legend } } },
  });
}

document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === b));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === b.dataset.tab));
  if (b.dataset.tab === "data") loadData();
  if (b.dataset.tab === "retrain") loadRetrain();
}));

// overview: one request; retried every 3 seconds until it succeeds, then refreshed every 10 seconds
let uptime = null;
async function loadOverview() {
  try {
    const o = await api("/summary"), x = o.metrics || {};
    uptime = { s: o.uptime_seconds, at: Date.now() };
    $("version").textContent = "v" + o.model_version;
    $("pred-total").textContent = o.predictions;
    $("metrics").innerHTML = tiles([["Accuracy", pct(x.accuracy)], ["Precision", pct(x.precision)], ["Recall", pct(x.recall)],
      ["F1", pct(x.f1)], ["Loss", x.loss?.toFixed(3) ?? "-"]]);
    $("versions").querySelector("tbody").innerHTML = o.history.slice().reverse().map((v) =>
      `<tr><td>v${v.version}</td><td>${v.trained_on || "-"}</td><td>${pct(v.metrics.accuracy)}</td><td>${pct(v.metrics.f1)}</td></tr>`).join("");
  } catch {
    if (!uptime) setTimeout(loadOverview, 3000);
  }
}
setInterval(() => { if (uptime) $("uptime").textContent = fmtUptime(uptime.s + (Date.now() - uptime.at) / 1000); }, 1000);

$("eval-btn").addEventListener("click", async () => {
  const btn = $("eval-btn"); btn.disabled = true; btn.textContent = "Evaluating...";
  try {
    const r = await api("/evaluate?n_images=10"), t = r.test_set;
    const cm = `<table class="cm"><tr><th></th>${CLASSES.map((c) => `<th>${SHORT[c]}</th>`).join("")}</tr>` +
      t.confusion_matrix.map((row, i) => `<tr><th>${SHORT[CLASSES[i]]}</th>${row.map((v, j) => `<td class="${i === j ? "diag" : ""}">${v}</td>`).join("")}</tr>`).join("") + "</table>";
    loadOverview();
    $("eval-result").innerHTML = `<p class="muted">Model v${r.model_version}, ${t.n_samples} test images</p>
      <div class="stats">${tiles([["Accuracy", pct(t.accuracy)], ["Precision", pct(t.precision)], ["Recall", pct(t.recall)], ["F1", pct(t.f1)], ["Loss", t.loss.toFixed(3)]])}</div>
      <h2>Confusion matrix (rows: true class)</h2>${cm}`;
  } catch (e) {
    $("eval-result").innerHTML = `<p class="bad">${e.message}</p>`;
  } finally {
    btn.disabled = false; btn.textContent = "Evaluate current model";
  }
});

// predict
let file = null;
function pick(f) {
  if (!f) return;
  file = f;
  $("preview").src = URL.createObjectURL(f);
  $("preview").style.display = "block";
  $("drop-text").style.display = "none";
  $("predict-btn").disabled = false;
  $("reset-btn").hidden = false;
}
$("predict-file").addEventListener("change", (e) => pick(e.target.files[0]));
$("reset-btn").addEventListener("click", () => {
  file = null;
  $("predict-file").value = "";
  $("preview").style.display = "none";
  $("drop-text").style.display = "";
  $("predict-result").innerHTML = "";
  $("predict-btn").disabled = true;
  $("reset-btn").hidden = true;
});
const drop = $("drop");
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); pick(e.dataTransfer.files[0]); });

$("predict-btn").addEventListener("click", async () => {
  const btn = $("predict-btn"); btn.disabled = true; btn.textContent = "Predicting...";
  const fd = new FormData(); fd.append("file", file);
  try {
    const r = await api("/predict", { method: "POST", body: fd });
    const rows = Object.entries(r.probabilities).sort((a, b) => b[1] - a[1]).map(([c, p], i) =>
      `<div class="bar-row ${i === 0 ? "top" : ""}"><span>${SHORT[c]}</span><div class="bar"><div style="width:${p * 100}%"></div></div><span>${(p * 100).toFixed(1)}%</span></div>`).join("");
    $("predict-result").innerHTML = `<div class="verdict">${r.label}</div><p class="muted">${(r.confidence * 100).toFixed(1)}% confidence</p>${rows}`;
    loadOverview();
  } catch (e) {
    $("predict-result").innerHTML = `<p class="bad">${e.message}</p>`;
  } finally {
    btn.disabled = false; btn.textContent = "Predict";
  }
});

// data
let dataLoaded = false;
async function loadData() {
  if (dataLoaded) return;
  const s = await api("/stats/dataset"), f = s.features;
  dataLoaded = true;
  bar("chart-counts", [
    { label: "train", data: CLASSES.map((c) => s.counts.train[c]), backgroundColor: INK },
    { label: "test", data: CLASSES.map((c) => s.counts.test[c]), backgroundColor: GREY }], { stacked: true, legend: true, y: "images" });
  bar("chart-lesion", [{ data: CLASSES.map((c) => +(f.lesion_ratio[c].mean * 100).toFixed(2)), backgroundColor: INK }], { y: "% of leaf" });
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
  const btn = $("upload-btn"); btn.disabled = true; btn.textContent = "Uploading...";
  try {
    const r = await api("/upload", { method: "POST", body: fd });
    $("upload-result").textContent = `Saved ${r.saved_total} images to the database.` +
      (r.rejected.length ? ` Rejected ${r.rejected.length}.` : "") + (r.auto_retrain_started ? " Retraining started automatically." : "");
    $("upload-files").value = ""; $("upload-text").textContent = "Choose files";
    loadRetrain();
  } catch (e) {
    $("upload-result").textContent = e.message;
  } finally {
    btn.disabled = false; btn.textContent = "Upload";
  }
});

let poll = null;
async function loadRetrain() {
  const [u, st] = await Promise.all([api("/uploads"), api("/retrain/status")]);
  $("pending-text").textContent = `${u.pending_total} new images waiting. Retraining also starts automatically at ${u.retrain_threshold}.`;
  $("retrain-btn").disabled = st.running || u.pending_total === 0;
  $("retrain-btn").textContent = st.running ? "Retraining..." : "Retrain";
  $("runs").querySelector("tbody").innerHTML = st.runs.map((r) => `<tr><td>${r.id}</td><td>${r.status}</td><td>${r.n_new_images || "-"}</td>
    <td>${pct(r.accuracy_before)}</td><td>${pct(r.accuracy_after)}</td><td>${r.promoted ? "v" + r.model_version : "-"}</td></tr>`).join("")
    || `<tr><td colspan="6" class="muted">None yet</td></tr>`;
  $("run-log").textContent = st.runs[0]?.log || "";
  clearTimeout(poll);
  if (st.running) poll = setTimeout(loadRetrain, 3000);
  else loadOverview();
}
$("retrain-btn").addEventListener("click", async () => {
  try { await api("/retrain", { method: "POST" }); } catch (e) { alert(e.message); }
  loadRetrain();
});

loadOverview();
setInterval(loadOverview, 10000);
