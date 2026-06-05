const state = {
  mode: "domain",
  profile: "freshness_accuracy",
  profiles: [],
};

const $ = (id) => document.getElementById(id);

function setVisible(id, show) {
  $(id).classList.toggle("hidden", !show);
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function profileDefaults(id) {
  const p = state.profiles.find((x) => x.id === id);
  if (!p) return { retrieval_k: 10, top_k: 3 };
  return { retrieval_k: p.retrieval_k ?? p.top_k ?? 10, top_k: p.top_k ?? 3 };
}

function applyRetrievalDefaults(id) {
  const d = profileDefaults(id);
  $("retrievalK").value = d.retrieval_k;
  $("topK").value = d.top_k;
}

async function loadConfig() {
  const [configRes, healthRes] = await Promise.all([
    fetch("/api/config"),
    fetch("/api/health"),
  ]);
  const config = await configRes.json();
  const health = await healthRes.json();
  state.profiles = config.profiles;

  const list = $("profileList");
  list.innerHTML = "";
  config.profiles.forEach((p) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "profile-option" + (p.id === state.profile ? " selected" : "");
    btn.dataset.profile = p.id;
    btn.innerHTML = `
      <span class="label">${escapeHtml(p.label)}</span>
      <span class="meta">default top-k ${p.top_k} · retrieve ${p.retrieval_k}</span>
      ${p.index_ready ? "" : '<span class="warn">Index missing — run ingest</span>'}
    `;
    btn.addEventListener("click", () => selectProfile(p.id));
    list.appendChild(btn);
  });

  const status = $("indexStatus");
  const ready = health.index_ready;
  status.innerHTML = `
    <li><span>Unified index</span><span class="${ready ? "dot-ok" : "dot-bad"}">${ready ? "ready" : "missing"}</span></li>
  `;

  applyRetrievalDefaults(state.profile);
}

function selectProfile(id) {
  state.profile = id;
  document.querySelectorAll(".profile-option").forEach((el) => {
    el.classList.toggle("selected", el.dataset.profile === id);
  });
  applyRetrievalDefaults(id);
}

function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll("#modeToggle button").forEach((b) => {
    b.classList.toggle("active", b.dataset.mode === mode);
  });
  setVisible("profilePanel", mode === "domain");
}

document.querySelectorAll("#modeToggle button").forEach((btn) => {
  btn.addEventListener("click", () => setMode(btn.dataset.mode));
});

$("queryForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = $("queryInput").value.trim();
  if (!query) return;

  $("errorBox").classList.add("hidden");
  $("answerCard").classList.add("hidden");
  $("latencySection").classList.add("hidden");
  $("sourcesSection").classList.add("hidden");

  const btn = $("submitBtn");
  btn.disabled = true;
  $("statusText").textContent = "Running retrieval + generation…";

  const body = {
    query,
    mode: state.mode,
    profile: state.profile,
    retrieval_k: parseInt($("retrievalK").value, 10) || 10,
    top_k: parseInt($("topK").value, 10) || 3,
  };

  try {
    const res = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || res.statusText);
    }
    renderResult(data);
    $("statusText").textContent = "Done";
  } catch (err) {
    $("errorBox").textContent = err.message || String(err);
    $("errorBox").classList.remove("hidden");
    $("statusText").textContent = "";
  } finally {
    btn.disabled = false;
  }
});

function renderResult(data) {
  $("answerCard").classList.remove("hidden");
  $("answerBody").textContent = data.answer || "";

  const badges = $("badges");
  badges.innerHTML = "";
  if (data.mode === "domain" && data.profile) {
    addBadge(badges, data.profile);
  } else {
    addBadge(badges, "baseline");
  }
  if (data.is_mock_answer) {
    addBadge(badges, "mock LLM", "mock");
  }
  addBadge(badges, "llm: auto");
  if (data.retrieval_k != null && data.top_k != null) {
    addBadge(badges, `retrieve ${data.retrieval_k} → top-k ${data.top_k}`);
  }
  const gen = data.latency_ms?.generation_ms;
  if (gen?.chunks_sent_to_llm != null) {
    addBadge(badges, `${gen.chunks_sent_to_llm} chunks to LLM`);
  }

  renderLatencyBreakdown(data.latency_ms || {});

  const chunks = data.retrieved || [];
  if (chunks.length) {
    $("sourcesSection").classList.remove("hidden");
    const list = $("sourcesList");
    list.innerHTML = chunks
      .map(
        (c, i) => `
      <div class="source-item">
        <header>
          <span>#${i + 1} ${escapeHtml(shortPath(c.source_uri))}</span>
          <span class="score">${(c.score ?? 0).toFixed(3)}</span>
        </header>
        <p>${escapeHtml(c.text || "")}</p>
      </div>`
      )
      .join("");
  }
}

function addBadge(container, text, extraClass = "") {
  const span = document.createElement("span");
  span.className = "badge " + extraClass;
  span.textContent = text;
  container.appendChild(span);
}

function fmtMs(n, precise = false) {
  if (n == null || Number.isNaN(n)) return "—";
  if (precise) return `${Number(n).toFixed(3)} ms`;
  return `${Number(n).toFixed(1)} ms`;
}

function renderLatencyBreakdown(lat) {
  if (!lat || lat.total == null) return;
  $("latencySection").classList.remove("hidden");
  const root = $("latencyBreakdown");
  const ret = lat.retrieval_ms || {};
  const gen = lat.generation_ms || {};
  const sections = [];

  sections.push({
    title: "End-to-end",
    rows: [{ label: "Total", value: lat.total, total: true }],
  });

  const setup = [];
  if (lat.index_load_ms > 0) setup.push({ label: "Load index from disk", value: lat.index_load_ms });
  if (lat.external_sync_ms > 0) setup.push({ label: "External doc sync (on query)", value: lat.external_sync_ms });
  if (lat.index_reload_ms > 0) setup.push({ label: "Reload index after sync", value: lat.index_reload_ms });
  if (setup.length) sections.push({ title: "Setup (before retrieval)", rows: setup });

  const retrievalRows = [
    { label: "Retrieval total", value: lat.retrieval ?? ret.total, total: true },
    { label: "Vector search (embed + candidates)", value: ret.vector_search_ms, sub: true },
    { label: "Profile post-process", value: ret.profile_postprocess_ms, sub: true },
  ];
  if (ret.privacy_context_mask_ms > 0) {
    retrievalRows.push({
      label: "Privacy: mask context chunks",
      value: ret.privacy_context_mask_ms,
      sub: true,
      precise: true,
    });
  }
  sections.push({ title: "Retrieval", rows: retrievalRows });

  const generationRows = [
    { label: "LLM request (send → response)", value: gen.llm_request_ms ?? lat.generation, total: true },
    { label: "Build prompt", value: gen.prompt_build_ms, sub: true },
  ];
  if (gen.prompt_chars > 0) {
    generationRows.push({
      label: `Prompt size (${gen.chunks_sent_to_llm ?? "?"} chunks)`,
      value: null,
      note: `${gen.prompt_chars.toLocaleString()} chars`,
      sub: true,
    });
  }
  if (gen.privacy_answer_mask_ms > 0) {
    generationRows.push({
      label: "Privacy: mask answer",
      value: gen.privacy_answer_mask_ms,
      sub: true,
      precise: true,
    });
  }
  if (gen.answer_postprocess_ms > 0) {
    generationRows.push({ label: "Answer post-process", value: gen.answer_postprocess_ms, sub: true });
  }
  sections.push({ title: "Generation", rows: generationRows });

  root.innerHTML = sections
    .map(
      (sec) => `
    <div class="latency-group">
      <h3>${escapeHtml(sec.title)}</h3>
      ${sec.rows
        .filter((r) => r.note || (r.value != null && (r.total || r.sub || r.value > 0)))
        .map(
          (r) => `
        <div class="latency-row ${r.total ? "total-row" : ""} ${r.sub ? "sub" : ""}">
          <span class="label">${escapeHtml(r.label)}</span>
          <span class="value">${r.note ? escapeHtml(r.note) : fmtMs(r.value, r.precise)}</span>
        </div>`
        )
        .join("")}
    </div>`
    )
    .join("");
}

function shortPath(uri) {
  if (!uri) return "";
  const parts = uri.split("/");
  return parts.length > 2 ? parts.slice(-2).join("/") : uri;
}

loadConfig().catch((e) => {
  $("errorBox").textContent = "Failed to load config: " + e.message;
  $("errorBox").classList.remove("hidden");
});
