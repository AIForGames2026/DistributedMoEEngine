// The operator dashboard's page (TASK-25): polls /api/state and sends control requests.
"use strict";

const $ = (id) => document.getElementById(id);
let last = null;
let polling = null;
let liveShown = false;
let modelId = null;

function el(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "class") {
      e.className = v;
    } else if (k.startsWith("on")) {
      e.addEventListener(k.slice(2), v);
    } else if (v !== false && v != null) {
      e.setAttribute(k, v === true ? "" : v);
    }
  }
  for (const kid of kids) {
    if (kid != null) {
      e.append(kid instanceof Node ? kid : String(kid));
    }
  }
  return e;
}

function gb(mb) {
  return (mb / 1024).toFixed(1) + " GB";
}

function size(bytes) {
  return (bytes / 1e9).toFixed(1) + " GB";
}

async function api(path, body) {
  const res = await fetch(path, body === undefined ? { cache: "no-store" } : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (res.status === 401) {
    showLogin();
    throw new Error("Sign in first.");
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const msg = typeof data.error === "string" ? data.error : data.error && data.error.message;
    throw new Error(msg || res.statusText);
  }
  return data;
}

function showLogin() {
  $("login").hidden = false;
  $("app").hidden = true;
  $("settingsview").hidden = true;
  $("status").hidden = true;
  $("logout").hidden = true;
  $("settingsbtn").hidden = true;
  if (polling) {
    clearInterval(polling);
    polling = null;
  }
  if (location.hash === "#link-expired") {
    $("loginerror").textContent = "That sign-in link was already used or has expired. Open the dashboard again from the tray icon, or paste the admin token.";
    $("loginerror").hidden = false;
    history.replaceState(null, "", "/");
  }
  $("token").focus();
}

function showApp() {
  $("login").hidden = true;
  $("status").hidden = false;
  $("logout").hidden = false;
  $("settingsbtn").hidden = false;
  if (!polling) {
    polling = setInterval(refresh, 1000);
  }
  showView();
  refresh();
}

// The Settings view lives at #settings, so the tray can open it directly.
function showView() {
  const settings = location.hash === "#settings";
  $("app").hidden = settings;
  $("settingsview").hidden = !settings;
  $("settingsbtn").textContent = settings ? "Back to the dashboard" : "Settings";
  if (settings) {
    loadSettings();
  }
}

window.addEventListener("hashchange", () => {
  if ($("login").hidden) {
    showView();
  }
});

$("settingsbtn").addEventListener("click", () => {
  location.hash = location.hash === "#settings" ? "" : "#settings";
});

let secrets = {};

function fillSettings(s) {
  secrets = { token: s.admin_token, secret: s.node_secret, key: s.api_key };
  const d = s.dashboard;
  $("s-models").value = d.models;
  $("s-open").checked = d.open_to_network;
  $("s-key").checked = d.api_key;
  $("s-port").value = d.port;
  $("s-node-port").value = d.node_port;
  $("s-router-port").value = d.router_port;
  $("s-expert-port").value = d.expert_port;
  $("apikeyrow").hidden = !s.api_key;
  $("settingsfile").textContent = s.file || "";
  const managed = s.file != null;
  $("unmanaged").hidden = managed;
  for (const e of document.querySelectorAll("#settingsview .form input, #savesettings")) {
    e.disabled = !managed;
  }
  $("restartbar").hidden = !(managed && s.restart_needed);
}

async function loadSettings() {
  $("settingserror").textContent = "";
  try {
    fillSettings(await api("/api/settings"));
  } catch (e) {
    $("settingserror").textContent = e.message;
  }
}

// The clipboard API needs a secure page (localhost counts); over plain HTTP elsewhere, copy through a hidden field.
async function copyText(text, what) {
  try {
    await navigator.clipboard.writeText(text);
  } catch (e) {
    const t = el("textarea", { readonly: true, style: "position:fixed;opacity:0" });
    t.value = text;
    document.body.append(t);
    t.select();
    document.execCommand("copy");
    t.remove();
  }
  $("copied").textContent = `Copied the ${what}. Treat it like a password.`;
}

$("copytoken").addEventListener("click", () => copyText(secrets.token, "admin token"));
$("copysecret").addEventListener("click", () => copyText(secrets.secret, "node secret"));
$("copykey").addEventListener("click", () => copyText(secrets.key, "API key"));

$("savesettings").addEventListener("click", async () => {
  $("settingserror").textContent = "";
  $("settingsmsg").hidden = true;
  try {
    const s = await api("/api/settings", {
      models: $("s-models").value.trim(),
      open_to_network: $("s-open").checked,
      api_key: $("s-key").checked,
      port: Number($("s-port").value),
      node_port: Number($("s-node-port").value),
      router_port: Number($("s-router-port").value),
      expert_port: Number($("s-expert-port").value),
    });
    fillSettings(s);
    $("settingsmsg").textContent = "Saved. The models folder and the API key apply now.";
    $("settingsmsg").hidden = false;
    nodesSig = "";
    $("model").dataset.names = "";
  } catch (e) {
    $("settingserror").textContent = e.message;
  }
});

$("restart").addEventListener("click", async () => {
  $("restart").disabled = true;
  try {
    await api("/api/restart", {});
    $("restartbar").textContent = "Restarting. The tray launcher opens the dashboard again in a new tab, signed in; you can close this one.";
  } catch (e) {
    $("settingserror").textContent = e.message;
    $("restart").disabled = false;
  }
});

async function refresh() {
  try {
    const s = await api("/api/state");
    render(s);
  } catch (e) {
    // A failed poll keeps the last view; the next one retries.
  }
}

function render(s) {
  last = s;
  renderStatus(s.run);
  renderNodes(s);
  renderModelForm(s);
  renderRun(s);
  renderEndpoint(s);
}

const stateLook = {
  idle: ["", "Idle"],
  starting: ["warn", "Loading"],
  running: ["ok", "Running"],
  stopping: ["warn", "Stopping"],
  failed: ["bad", "Failed"],
};

function renderStatus(run) {
  const [cls, label] = stateLook[run.state] || ["", run.state];
  $("statusdot").className = "dot " + cls;
  $("statusstate").textContent = label;
  $("statusmsg").textContent = run.model ? run.model : "";
}

function rankCell(r) {
  if (!r) {
    return el("span", { class: "muted" }, "none");
  }
  const look = r.state === "listening" ? "ok" : r.state === "exited" ? "bad" : "warn";
  const text = `rank ${r.rank} of ${r.ranks}, ${r.state}` + (r.exit != null ? ` (${r.exit})` : "");
  return el("span", { class: "badge wrap" }, el("span", { class: "dot " + look }), text);
}

let nodesSig = "";

function renderNodes(s) {
  const body = $("nodes");
  // Rebuilt only when what it shows changes, so buttons stay put under the pointer.
  const sig = JSON.stringify([s.local.name, Math.round(s.local.mem_available_mb / 100), s.local.models.length, s.run.state, s.node_port,
    s.nodes.map((n) => [n.name, n.approved, n.online, n.addr, n.report && [Math.round(n.report.mem_available_mb / 100), n.report.mem_total_mb, n.report.gpu,
      n.report.models.length, n.report.rdma.map((d) => d.dev).join(","), n.report.rank && [n.report.rank.rank, n.report.rank.state, n.report.rank.exit, !!n.report.rank.log]]])]);
  if (sig === nodesSig) {
    for (const n of s.nodes) {
      const pre = body.querySelector(`pre[data-node="${CSS.escape(n.name)}"]`);
      if (pre && n.report && n.report.rank) {
        pre.textContent = n.report.rank.log;
      }
    }
    return;
  }
  nodesSig = sig;
  const open = new Set([...body.querySelectorAll("details[open]")].map((d) => d.dataset.node));
  body.replaceChildren();
  const local = s.local;
  body.append(el("tr", {},
    el("td", {}, el("b", {}, local.name), el("div", { class: "muted" }, "this machine, rank 0")),
    el("td", {}, el("span", { class: "badge" }, el("span", { class: "dot ok" }), "dashboard")),
    el("td", {}, `${gb(local.mem_available_mb)} of ${gb(local.mem_total_mb)}`),
    el("td", {}, local.gpu || el("span", { class: "muted" }, "none")),
    el("td", {}, String(local.models.length)),
    el("td", {}, s.run.state === "running" || s.run.state === "starting" ? el("span", { class: "badge" }, "router") : el("span", { class: "muted" }, "none")),
    el("td", {}),
  ));
  let online = 0;
  for (const n of s.nodes) {
    const r = n.report;
    online += n.online ? 1 : 0;
    const status = !n.online ? ["", "offline"] : n.approved ? ["ok", "online"] : ["warn", "waiting for approval"];
    const actions = el("td", { class: "actions" });
    if (!n.approved) {
      actions.append(el("button", { class: "small primary", onclick: () => act("/api/nodes/approve", { name: n.name }) }, "Approve"));
    }
    actions.append(el("button", { class: "small", onclick: () => act("/api/nodes/forget", { name: n.name }) }, n.approved ? "Remove" : "Ignore"));
    const rdma = r && r.rdma.length ? el("div", { class: "muted mono" }, r.rdma.map((d) => d.dev).join(", ")) : null;
    const row = el("tr", {},
      el("td", {}, el("b", {}, n.name), el("div", { class: "muted mono" }, n.addr), rdma),
      el("td", {}, el("span", { class: "badge" }, el("span", { class: "dot " + status[0] }), status[1])),
      el("td", {}, r ? `${gb(r.mem_available_mb)} of ${gb(r.mem_total_mb)}` : "-"),
      el("td", {}, r ? (r.gpu || el("span", { class: "muted" }, "none")) : "-"),
      el("td", {}, r ? String(r.models.length) : "-"),
      el("td", {}, r ? rankCell(r.rank) : "-"),
      actions,
    );
    body.append(row);
    if (r && r.rank && r.rank.log) {
      const logRow = el("tr", {}, el("td", { colspan: 7 },
        el("details", { "data-node": n.name, open: open.has(n.name) }, el("summary", {}, `Rank log on ${n.name}`), el("pre", { class: "mono", "data-node": n.name }, r.rank.log))));
      body.append(logRow);
    }
  }
  if (s.nodes.length === 0) {
    body.append(el("tr", {}, el("td", { colspan: 7, class: "empty" }, "No node agents have linked yet. Open \"Add a node\" below to start one.")));
    $("addnode").open = true;
  }
  $("nodecount").textContent = `${s.nodes.length} known, ${online} online`;
  const host = location.hostname || "DASHBOARD_IP";
  $("agentcmd").textContent = `tfswarm node --dashboard ${host}:${s.node_port} --models /path/to/models --secret-file node.secret`;
}

function renderModelForm(s) {
  const sel = $("model");
  const names = s.local.models.map((m) => m.name);
  const current = sel.value || s.run.model || names[0];
  if (sel.dataset.names !== names.join("|")) {
    sel.replaceChildren(...s.local.models.map((m) => el("option", { value: m.name }, `${m.name} (${size(m.bytes)})`)));
    sel.dataset.names = names.join("|");
    if (names.length === 0) {
      sel.append(el("option", { value: "" }, s.managed ? "No models found; set the models folder in Settings" : "No models found in the --models directory"));
    }
  }
  if (names.includes(current)) {
    sel.value = current;
  }
  const model = s.local.models.find((m) => m.name === sel.value);
  const checked = new Set([...document.querySelectorAll("#picks input:checked")].map((i) => i.value));
  if ($("picks").dataset.init !== "1" && s.run.nodes) {
    s.run.nodes.forEach((n) => checked.add(n));
  }
  $("picks").dataset.init = "1";
  const usable = s.nodes.filter((n) => n.approved).map((n) => {
    const has = n.report && model && n.report.models.some((m) => m.name === model.name && m.bytes === model.bytes);
    return [n.name, !n.online ? "offline" : !has ? "no copy of this model" : ""];
  });
  const sig = JSON.stringify(usable);
  if (sig !== $("picks").dataset.sig) {
    $("picks").dataset.sig = sig;
    renderPicks(usable, checked);
  }
  const busy = s.run.state === "starting" || s.run.state === "stopping";
  $("load").disabled = busy || !model;
  $("stop").disabled = !(s.run.state === "running" || s.run.state === "starting");
  $("load").textContent = s.run.state === "running" ? "Reload" : "Load";
  if (!$("ctx").dataset.init && s.run.settings) {
    const st = s.run.settings;
    $("transport").value = st.transport;
    $("ctx").value = st.ctx;
    $("payload").value = st.payload;
    $("kv").value = st.kv_format;
    $("cache").value = st.cache;
    $("gpu").checked = st.gpu;
    $("reuse").checked = st.reuse;
  }
  $("ctx").dataset.init = "1";
}

function renderPicks(usable, checked) {
  const picks = [];
  for (const [name, why] of usable) {
    const box = el("input", { type: "checkbox", value: name, disabled: why !== "" });
    box.checked = why === "" && checked.has(name);
    picks.push(el("label", { class: "pick" + (why ? " off" : ""), title: why }, box, name, why ? el("span", { class: "muted" }, ` (${why})`) : null));
  }
  if (picks.length === 0) {
    picks.push(el("span", { class: "muted" }, "None approved yet; the model runs on this machine alone."));
  }
  $("picks").replaceChildren(...picks);
}

function renderRun(s) {
  const run = s.run;
  const secs = Math.round(run.since_s);
  const when = run.state === "idle" ? "" : ` (${secs < 120 ? secs + " s" : Math.round(secs / 60) + " min"})`;
  $("runmsg").textContent = (run.message || "Nothing loaded yet.") + when;
  $("runmsg").className = "message" + (run.state === "failed" ? " error" : "");
  $("command").textContent = run.command || "(no model loaded)";
  $("routerlog").textContent = s.router_log || "(no router log yet)";
  const live = run.state === "running";
  if (live !== liveShown) {
    liveShown = live;
    $("live").hidden = !live;
    $("liveempty").hidden = live;
    $("live").src = live ? "/live" : "about:blank";
    modelId = null;
  }
}

function renderEndpoint(s) {
  const base = `${location.origin}/v1`;
  const running = s.run.state === "running";
  $("endpointsub").textContent = s.api_key ? "API key required" : "open on this network";
  if (running && modelId === null) {
    modelId = "";
    fetch("/v1/models").then((r) => r.json()).then((d) => { modelId = d.data[0].id; }).catch(() => { modelId = null; });
  }
  const id = modelId || "MODEL";
  const auth = s.api_key ? ` \\\n  -H "Authorization: Bearer $API_KEY"` : "";
  $("endpoint").textContent = `${base}\n\ncurl ${base}/chat/completions${auth} \\\n  -H "Content-Type: application/json" \\\n  -d '{"model": "${id}", "messages": [{"role": "user", "content": "Hello"}]}'`;
  $("send").disabled = !running;
}

async function act(path, body) {
  $("loaderror").textContent = "";
  try {
    await api(path, body);
  } catch (e) {
    $("loaderror").textContent = e.message;
  }
  refresh();
}

$("loginform").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  $("loginerror").hidden = true;
  try {
    await api("/api/login", { token: $("token").value });
    $("token").value = "";
    showApp();
  } catch (e) {
    $("loginerror").textContent = e.message;
    $("loginerror").hidden = false;
  }
});

$("logout").addEventListener("click", async () => {
  await fetch("/api/logout", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
  showLogin();
});

$("load").addEventListener("click", () => {
  const nodes = [...document.querySelectorAll("#picks input:checked")].map((i) => i.value);
  act("/api/load", {
    model: $("model").value,
    nodes,
    transport: $("transport").value,
    ctx: Number($("ctx").value),
    payload: $("payload").value,
    kv_format: $("kv").value,
    cache: Number($("cache").value),
    gpu: $("gpu").checked,
    reuse: $("reuse").checked,
  });
});

$("stop").addEventListener("click", () => act("/api/stop", {}));

$("send").addEventListener("click", async () => {
  const text = $("prompt").value.trim();
  if (!text) {
    return;
  }
  $("send").disabled = true;
  $("reply").textContent = "";
  $("sendstats").textContent = "waiting for the first token...";
  const t0 = performance.now();
  try {
    const res = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: modelId || "default", messages: [{ role: "user", content: text }], stream: true, max_tokens: 400, stream_options: { include_usage: true } }),
    });
    if (!res.ok) {
      const d = await res.json().catch(() => ({}));
      throw new Error((d.error && d.error.message) || res.statusText);
    }
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    let first = null;
    let usage = null;
    for (;;) {
      const { value, done } = await reader.read();
      if (done) {
        break;
      }
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const event = buf.slice(0, i);
        buf = buf.slice(i + 2);
        const line = event.split("\n").find((l) => l.startsWith("data: "));
        if (!line || line === "data: [DONE]") {
          continue;
        }
        const d = JSON.parse(line.slice(6));
        if (d.usage) {
          usage = d.usage;
        }
        const delta = d.choices && d.choices[0] && d.choices[0].delta;
        const piece = delta && (delta.content || delta.reasoning_content);
        if (piece) {
          if (first === null) {
            first = performance.now() - t0;
          }
          $("reply").textContent += piece;
          $("sendstats").textContent = `first token ${Math.round(first)} ms`;
        }
      }
    }
    const secs = (performance.now() - t0) / 1000;
    if (usage) {
      $("sendstats").textContent = `${usage.completion_tokens} tokens in ${secs.toFixed(1)} s, first token ${Math.round(first || 0)} ms`;
    }
  } catch (e) {
    $("sendstats").textContent = "";
    $("reply").textContent = "Request failed: " + e.message;
  }
  $("send").disabled = !(last && last.run.state === "running");
});

fetch("/api/state", { cache: "no-store" }).then((res) => {
  if (res.status === 401) {
    showLogin();
  } else {
    showApp();
  }
}).catch(showLogin);
