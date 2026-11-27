// PrecisionPilot demo — vanilla JS, no build step.
"use strict";

const $ = (sel) => document.querySelector(sel);
const state = { meta: null, gpu: null, bs: null, budget: 1.0, model: null, selection: null, sample: null };
const METHOD_LABEL = {
  greedy: "Sensitivity-guided greedy",
  optimum: "Exhaustive optimum",
  "all-fp16": "All FP16",
  "all-int8": "All INT8",
};

const fmt = {
  ms: (v) => (v >= 10 ? v.toFixed(1) : v.toFixed(2)) + " ms",
  pp: (v) => fmt.signed(-v) + " pp",
  // accuracy *change*: negative = lost accuracy; exact zero shows unsigned
  signed: (v) => (Math.abs(v) < 0.005 ? "0.00" : (v < 0 ? "−" : "+") + Math.abs(v).toFixed(2)),
  x: (v) => v.toFixed(2) + "×",
  pct: (v) => (v * 100).toFixed(1) + "%",
  mb: (v) => (v >= 100 ? v.toFixed(0) : v.toFixed(1)) + " MB",
};

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (_) { /* not json */ }
    throw new Error(msg);
  }
  return res.json();
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const c of children) if (c != null) node.append(c);
  return node;
}

function segmented(container, options, current, onPick) {
  container.replaceChildren();
  for (const opt of options) {
    const b = el("button", {
      type: "button", role: "radio", "aria-checked": String(opt.value === current),
      text: opt.label,
      onclick: () => {
        container.querySelectorAll("button").forEach((x) => x.setAttribute("aria-checked", "false"));
        b.setAttribute("aria-checked", "true");
        onPick(opt.value);
      },
    });
    container.append(b);
  }
}

// ---------------------------------------------------------------- explore
function notice(html) {
  const n = $("#notice");
  n.innerHTML = html;
  n.hidden = false;
}

function renderControls() {
  const gpus = state.meta.gpus;
  segmented($("#gpu-options"), gpus.map((g) => ({ value: g.id, label: g.label })), state.gpu, (v) => {
    state.gpu = v;
    renderBatchOptions();
    renderModelOptions();
    refresh();
  });
  renderBatchOptions();
  renderModelOptions();
}

function renderModelOptions() {
  const gpu = state.meta.gpus.find((g) => g.id === state.gpu);
  const ctl = $("#model-control");
  ctl.hidden = !gpu.has_context_model;
  state.model = gpu.has_context_model ? (state.model || "context") : "additive";
  if (!gpu.has_context_model) return;
  segmented($("#model-options"), [
    { value: "context", label: "Context-aware" },
    { value: "additive", label: "Additive table" },
  ], state.model, (v) => {
    state.model = v;
    refresh();
  });
}

function renderBatchOptions() {
  const gpu = state.meta.gpus.find((g) => g.id === state.gpu);
  if (!gpu.batch_sizes.includes(state.bs)) state.bs = gpu.batch_sizes[gpu.batch_sizes.length - 1];
  segmented($("#bs-options"), gpu.batch_sizes.map((b) => ({ value: b, label: String(b) })), state.bs, (v) => {
    state.bs = v;
    refresh();
  });
}

function sizeSub(g) {
  if (g.model_mb == null) return "rebuild the latency table to see sizes";
  const parts = [g.size_saving_pct >= 0.5 ? `${g.size_saving_pct.toFixed(0)}% smaller than FP16` : "same as FP16"];
  if (g.measured && g.measured.gpu_footprint_mb != null) parts.push(`${fmt.mb(g.measured.gpu_footprint_mb)} on the GPU`);
  return parts.join(" · ");
}

function bits(config) {
  const span = el("span", { class: "bits" });
  for (const ch of config) span.append(ch === "1" ? el("b", { text: "1" }) : "0");
  return span;
}

function renderSelection(sel) {
  const sens = Object.fromEntries((state.meta.sensitivity || []).map((s) => [s.unit, s.drop]));
  const units = $("#units");
  units.replaceChildren();
  for (const u of sel.units) {
    const isInt8 = u.precision === "int8";
    const saved = u.fp16_ms - u.int8_ms;
    units.append(el("li", {
      class: "unit", "data-p": u.precision,
      title: `${u.name}: FP16 ${fmt.ms(u.fp16_ms)} · INT8 ${fmt.ms(u.int8_ms)}`,
    },
      el("span", { class: "unit__p", text: isInt8 ? "INT8" : "FP16" }),
      el("span", { class: "unit__name", text: u.name }),
      el("span", {
        class: "unit__drop",
        text: u.unit in sens ? `Δacc ${fmt.signed(-sens[u.unit])}` : `Δt ${fmt.signed(-saved)} ms`,
      }),
    ));
  }

  const g = sel.methods.find((m) => m.method === "greedy");
  const opt = sel.methods.find((m) => m.method === "optimum");
  const latency = g.measured ? g.measured.median_ms : g.est_latency_ms;
  const latencySub = g.measured
    ? `measured median · p95 ${fmt.ms(g.measured.p95_ms)}`
    : "estimated from per-unit table";
  const testSub = g.test_acc != null ? `test accuracy ${g.test_acc.toFixed(2)}%` : `val accuracy ${g.val_acc.toFixed(2)}%`;
  const metrics = [
    ["Speedup vs FP16", fmt.x(g.speedup), `${g.config.split("1").length - 1} of 10 units in INT8`],
    ["Model size", g.model_mb != null ? fmt.mb(g.model_mb) : "n/a", sizeSub(g)],
    ["Latency", fmt.ms(latency), latencySub],
    ["Accuracy change", fmt.pp(g.drop_pp), testSub],
    ["Evaluations", `${g.n_evals}`, `of ${sel.space} · ${Math.abs(g.gap_pct) < 0.05 ? "matches the optimum" : `${g.gap_pct.toFixed(1)}% slower than optimum`}`],
  ];
  const dl = $("#metrics");
  dl.replaceChildren();
  for (const [label, value, sub] of metrics) {
    dl.append(el("div", { class: "metric" },
      el("dt", { text: label }),
      el("dd", {}, el("span", { class: "metric__value", text: value }), el("span", { class: "metric__sub", text: sub }))));
  }

  const tbody = $("#compare tbody");
  tbody.replaceChildren();
  for (const m of sel.methods) {
    const lat = m.measured ? m.measured.median_ms : m.est_latency_ms;
    tbody.append(el("tr", { class: m.method === "greedy" ? "is-ours" : "" },
      el("td", { text: METHOD_LABEL[m.method] || m.method }),
      el("td", {}, bits(m.config)),
      el("td", { class: "num", text: fmt.pp(m.drop_pp) }),
      el("td", { class: "num", text: fmt.ms(lat) }),
      el("td", { class: "num", text: m.model_mb != null ? fmt.mb(m.model_mb) : "n/a" }),
      el("td", { class: "num", text: fmt.x(m.speedup) }),
      el("td", { class: "num", text: m.method.startsWith("all-") ? "1" : String(m.n_evals) }),
      el("td", {}, el("span", {
        class: `status ${m.feasible ? "status--ok" : "status--bad"}`,
        text: m.feasible ? "Within" : "Exceeds",
      })),
    ));
  }
  const measured = sel.methods.some((m) => m.measured);
  const backend = (state.meta.gpus.find((x) => x.id === state.gpu) || {}).backend;
  const source = { trt: "TensorRT", torch: "PyTorch CUDA", mock: "mock backend, simulated — not a benchmark" }[backend] || backend;
  const model = sel.latency_model === "context" ? "context-aware estimate" : "additive lookup-table estimate";
  $("#compare-note").textContent = measured
    ? `Latency: measured full-model median (${source}) where available, otherwise the ${model}.`
    : `Latency: ${model} (${source}).`;
  void opt;
}

let refreshTimer = null;
function refresh() {
  clearTimeout(refreshTimer);
  refreshTimer = setTimeout(async () => {
    const main = $("#explore");
    main.classList.add("is-loading");
    try {
      const q = new URLSearchParams({ gpu: state.gpu, batch_size: state.bs, budget: state.budget });
      if (state.model) q.set("latency_model", state.model);
      state.selection = await api(`/api/select?${q}`);
      renderSelection(state.selection);
      if (state.sample != null) predict();
    } catch (e) {
      notice(`Could not compute a selection: ${e.message}`);
    } finally {
      main.classList.remove("is-loading");
    }
  }, 120);
}

// ---------------------------------------------------------------- predict
function currentConfig() {
  return state.selection ? state.selection.methods.find((m) => m.method === "greedy").config : "0000000000";
}

function bars(title, items, ref) {
  const col = el("div", {}, el("h3", { text: title }));
  for (const it of items) {
    const fill = el("div", { class: "bar__fill" });
    fill.style.width = fmt.pct(it.prob);
    col.append(el("div", { class: `bar${ref ? " bar--ref" : ""}` },
      el("span", { text: it.label }),
      el("span", { class: "bar__pct", text: fmt.pct(it.prob) }),
      el("div", { class: "bar__track" }, fill)));
  }
  return col;
}

function renderPrediction(out, imgSrc, caption) {
  const box = $("#result");
  box.replaceChildren(
    el("div", { class: "result__top" },
      el("img", { class: "result__img", src: imgSrc, alt: caption }),
      el("div", {},
        el("p", { class: "result__title", text: caption }),
        el("p", { class: "muted small" }, "Configuration ", bits(out.config)))),
    el("div", { class: "result__cols" },
      bars("FP32 reference", out.fp32, true),
      bars("Selected mixed precision", out.mixed, false)),
    el("p", {
      class: `agree status ${out.agree ? "status--ok" : "status--bad"}`,
      text: out.agree ? "Same top prediction as FP32" : "Top prediction differs from FP32",
    }),
  );
}

async function predict() {
  const box = $("#result");
  box.classList.add("is-loading");
  try {
    if (state.sample.kind === "sample") {
      const q = new URLSearchParams({ sample_id: state.sample.id, config: currentConfig() });
      const out = await api(`/api/predict?${q}`);
      renderPrediction(out, state.sample.png, `True label: ${out.label}`);
    } else {
      const fd = new FormData();
      fd.append("file", state.sample.file);
      const out = await api(`/api/upload?config=${currentConfig()}`, { method: "POST", body: fd });
      renderPrediction(out, out.png, "Your image (resized to 32×32)");
    }
  } catch (e) {
    box.replaceChildren(el("p", { class: "muted", text: `Prediction unavailable: ${e.message}` }));
  } finally {
    box.classList.remove("is-loading");
  }
}

async function loadSamples() {
  const gallery = $("#gallery");
  try {
    const samples = await api("/api/samples?n=12");
    for (const s of samples) {
      const btn = el("button", {
        type: "button", class: "thumb", "aria-pressed": "false", "aria-label": `Sample: ${s.label}`,
        onclick: () => {
          gallery.querySelectorAll(".thumb").forEach((t) => t.setAttribute("aria-pressed", "false"));
          btn.setAttribute("aria-pressed", "true");
          state.sample = { kind: "sample", id: s.id, png: s.png };
          predict();
        },
      }, el("img", { src: s.png, alt: s.label, loading: "lazy" }));
      gallery.append(btn);
    }
  } catch (e) {
    $("#result").replaceChildren(el("p", { class: "muted", text: `Samples unavailable: ${e.message}` }));
  }
}

$("#upload").addEventListener("change", (ev) => {
  const file = ev.target.files[0];
  if (!file) return;
  document.querySelectorAll(".thumb").forEach((t) => t.setAttribute("aria-pressed", "false"));
  state.sample = { kind: "upload", file };
  predict();
});

// ---------------------------------------------------------------- boot
const budget = $("#budget");
budget.addEventListener("input", () => {
  state.budget = Number(budget.value);
  $("#budget-out").textContent = `${state.budget.toFixed(1)} pp`;
  refresh();
});

(async function boot() {
  try {
    state.meta = await api("/api/meta");
  } catch (e) {
    notice(`Cannot reach the demo server: ${e.message}`);
    return;
  }
  $("#footer-source").textContent = `Results: ${state.meta.results_dir}`;
  if (!state.meta.ready) {
    notice("No experiment results found yet. Run <code>run-accuracy</code> and <code>run-gpu</code> " +
      "(or unzip the Kaggle results into <code>results/</code>), then reload. " +
      "Predictions below still work once a checkpoint is available.");
  } else {
    state.budget = state.meta.main_budget;
    budget.value = state.budget;
    $("#budget-out").textContent = `${state.budget.toFixed(1)} pp`;
    state.gpu = state.meta.gpus[0].id;
    renderControls();
    refresh();
  }
  loadSamples();
})();
