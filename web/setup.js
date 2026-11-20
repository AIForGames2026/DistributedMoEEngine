// The first-run setup page (TASK-44): picks the role and writes the settings file.
"use strict";

const $ = (id) => document.getElementById(id);
let current = null;

async function api(path, body) {
  const res = await fetch(path, body === undefined ? { cache: "no-store" } : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.error || res.statusText);
  }
  return data;
}

function role() {
  const r = document.querySelector("input[name=role]:checked");
  return r ? r.value : "";
}

function showRole() {
  const r = role();
  $("host").hidden = r !== "dashboard";
  $("join").hidden = r !== "node";
  $("savebar").hidden = r === "";
}

function note(id, cls, text) {
  $(id).className = "note " + cls;
  $(id).textContent = text;
}

async function checkModels(input, out) {
  const dir = $(input).value.trim();
  if (!dir) {
    note(out, "warn", "Type the folder that holds the models.");
    return;
  }
  try {
    const r = await api("/api/models", { dir });
    if (!r.exists) {
      note(out, "bad", "There is no folder at that path.");
      return;
    }
    const names = r.models.map((m) => `${m.name} (${(m.bytes / 1e9).toFixed(1)} GB)`);
    const missing = r.unindexed ? ` ${r.unindexed} .gguf file${r.unindexed === 1 ? " has" : "s have"} no .tfsum index yet (made with tfload index on Linux).` : "";
    if (names.length === 0) {
      note(out, "warn", "No loadable models in this folder yet." + missing);
    } else {
      note(out, "ok", `Found ${names.length}: ${names.join(", ")}.` + missing);
    }
  } catch (e) {
    note(out, "bad", e.message);
  }
}

function joinFields() {
  return {
    dashboard: $("j-dashboard").value.trim(),
    models: $("j-models").value.trim(),
    name: $("j-name").value.trim(),
    advertise: $("j-advertise").value.trim(),
  };
}

function hostFields() {
  return {
    models: $("h-models").value.trim(),
    open_to_network: $("h-open").checked,
    api_key: $("h-key").checked,
    port: Number($("h-port").value),
    node_port: Number($("h-node-port").value),
    router_port: Number($("h-router-port").value),
    expert_port: Number($("h-expert-port").value),
  };
}

async function testLink() {
  const n = joinFields();
  note("j-test-note", "", "Linking...");
  $("j-test").disabled = true;
  try {
    const r = await api("/api/test-node", { dashboard: n.dashboard, secret: $("j-secret").value, name: n.name });
    note("j-test-note", r.ok ? "ok" : "bad", r.message);
  } catch (e) {
    note("j-test-note", "bad", e.message);
  }
  $("j-test").disabled = false;
}

async function save() {
  $("saveerror").textContent = "";
  const r = role();
  const body = { role: r, dashboard: hostFields(), node: joinFields(), node_secret: $("j-secret").value };
  $("save").disabled = true;
  try {
    await api("/api/save", body);
    $("form").hidden = true;
    $("saved").hidden = false;
    if (r === "dashboard") {
      $("saved-title").textContent = "The dashboard is starting";
      $("saved-text").textContent = "It opens in a new tab, already signed in. From then on, open it again from the tray icon.";
    } else {
      $("saved-title").textContent = "The node agent is starting";
      $("saved-text").textContent = "It runs from the tray icon and links to the dashboard. Approve this PC there, under Nodes, and it can hold experts from the next load.";
    }
  } catch (e) {
    $("saveerror").textContent = e.message;
    $("save").disabled = false;
  }
}

function fill(s) {
  const d = s.settings.dashboard;
  const n = s.settings.node;
  $("h-models").value = d.models;
  $("h-open").checked = d.open_to_network;
  $("h-key").checked = d.api_key;
  $("h-port").value = d.port;
  $("h-node-port").value = d.node_port;
  $("h-router-port").value = d.router_port;
  $("h-expert-port").value = d.expert_port;
  $("j-dashboard").value = n.dashboard;
  $("j-models").value = n.models;
  $("j-name").value = n.name;
  $("j-name").placeholder = s.host;
  $("j-advertise").value = n.advertise;
  if (s.has_node_secret) {
    $("j-secret").placeholder = "A node secret is saved; leave empty to keep it";
  }
  $("file").textContent = "Settings file: " + s.file;
  for (const r of document.querySelectorAll("input[name=role]")) {
    r.checked = r.value === s.settings.role;
  }
  showRole();
}

for (const r of document.querySelectorAll("input[name=role]")) {
  r.addEventListener("change", showRole);
}
$("h-check").addEventListener("click", () => checkModels("h-models", "h-models-note"));
$("j-check").addEventListener("click", () => checkModels("j-models", "j-models-note"));
$("h-models").addEventListener("change", () => checkModels("h-models", "h-models-note"));
$("j-models").addEventListener("change", () => checkModels("j-models", "j-models-note"));
$("j-test").addEventListener("click", testLink);
$("save").addEventListener("click", save);

api("/api/setup").then((s) => {
  current = s;
  fill(s);
}).catch((e) => {
  $("saveerror").textContent = e.message;
});
