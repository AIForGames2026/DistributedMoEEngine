// TensorFold-Swarm inspector: WebGL view of MoE tokens being routed to experts across nodes.
// Everything on screen is driven by a telemetry trace (docs/telemetry-schema.md): a synthetic
// one generated in the browser, the 10,000-packet stress trace, or a trace file the viewer loads.
// TASK-16 adds traces recorded from the real engine; the player does not change.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { MODEL, LINKS, HW, NODES, PRESETS } from './sim.js';
import { generateTrace, validateTrace, summarize, KIND, STATUS } from './trace.js';
import { STRESS } from './presets.js';
import './style.css';

const COLORS = {
  idle: new THREE.Color(0x2b3446),
  resident: new THREE.Color(0x7dd3fc),
  prefetched: new THREE.Color(0x4ade80),
  miss: new THREE.Color(0xf87171),
  rdma: new THREE.Color(0x38bdf8),
  tcp: new THREE.Color(0xfb923c),
  shm: new THREE.Color(0xc4b5fd),
};
const STATUS_COLOR = STATUS.map((s) => COLORS[s]);
const FLASH_MS = 1100;
const MAX_PACKETS = 16384;
const LAYER_SCREEN_MS = 1500; // the median layer takes this long at 1x
const MIN_FLIGHT_MS = 250;
const MAX_FLIGHT_MS = 1000;

const params = new URLSearchParams(location.search);
const state = {
  source: params.get('source') === 'stress' ? 'stress' : 'live',
  preset: 'sparks', topK: MODEL.topK, lookahead: true, recall: 0.85, speed: 1, playing: true, seed: 42,
};

if (hasWebGL2()) start();
else showFallback();

function hasWebGL2() {
  try {
    return !!document.createElement('canvas').getContext('webgl2');
  } catch {
    return false;
  }
}

function showFallback() {
  document.getElementById('viewport').innerHTML =
    '<div class="fallback"><h2>WebGL 2 is not available</h2>' +
    '<p>This inspector needs a browser with WebGL 2 (current Chrome, Firefox, Edge, or Safari). ' +
    'Nothing needs to be installed; try enabling hardware acceleration.</p></div>';
}

function start() {
  const viewport = document.getElementById('viewport');
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  viewport.appendChild(renderer.domElement);

  const labels = new CSS2DRenderer();
  labels.domElement.className = 'label-layer';
  viewport.appendChild(labels.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b1020);
  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 200);
  const controls = new OrbitControls(camera, labels.domElement);
  controls.enableDamping = true;

  scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  const sun = new THREE.DirectionalLight(0xffffff, 1.2);
  sun.position.set(5, 10, 6);
  scene.add(sun);
  const grid = new THREE.GridHelper(30, 30, 0x1c2540, 0x141b2e);
  grid.position.y = -0.2;
  scene.add(grid);

  // Packets: one instanced mesh, recycled from a pool. Hidden instances have scale 0.
  const packetMesh = new THREE.InstancedMesh(
    new THREE.SphereGeometry(0.08, 8, 6),
    new THREE.MeshBasicMaterial(),
    MAX_PACKETS,
  );
  packetMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  for (let i = 0; i < MAX_PACKETS; i++) packetMesh.setColorAt(i, COLORS.rdma);
  packetMesh.frustumCulled = false;
  scene.add(packetMesh);
  const packets = [];
  const freeSlots = [];

  let topo = null;
  let trace = null;
  let summary = null;
  let cursor = 0;
  let traceNow = 0; // ns
  let visNow = 0; // screen ms, scaled by speed
  let nsPerMs = 1;
  let layer = 0;
  const stats = { packets: 0, remote: 0, prefetched: 0, misses: 0, experts: 0, fetchBytes: 0, dropped: 0 };
  // fps is capped by the display; workMs (main-thread time per frame, worst
  // over the last second) shows the headroom left under 16.7 ms for 60 FPS.
  const perf = { fps: 0, frames: 0, since: performance.now(), maxInFlight: 0, minFps: Infinity, workMs: 0, maxWorkMs: 0, worst: 0 };
  window.__inspector = perf;

  function buildTopology(tr) {
    if (topo) {
      scene.remove(topo.group);
      topo.group.traverse((o) => {
        if (o.isCSS2DObject) o.element.remove();
        o.geometry?.dispose();
        o.material?.dispose?.();
      });
    }
    const group = new THREE.Group();
    const n = tr.nodes.length;
    const radius = n === 1 ? 0 : n === 2 ? 4.5 : 5.2;
    const nodePos = tr.nodes.map((_, i) => {
      const a = n === 2 ? Math.PI * i : (2 * Math.PI * i) / n - Math.PI / 2;
      return new THREE.Vector3(Math.cos(a) * radius, 0, Math.sin(a) * radius);
    });

    tr.nodes.forEach((node, i) => {
      const known = NODES[node.id];
      const platform = new THREE.Mesh(
        new THREE.BoxGeometry(3.4, 0.3, 3.4),
        new THREE.MeshStandardMaterial({ color: node.kind === 'desktop' ? 0x3b2a1e : 0x1e293b, roughness: 0.7 }),
      );
      platform.position.copy(nodePos[i]);
      group.add(platform);
      const el = document.createElement('div');
      el.className = 'node-label';
      const b = document.createElement('b');
      b.textContent = node.label ?? node.id;
      const sub = document.createElement('span');
      sub.textContent = known?.sub ?? node.kind;
      el.append(b, sub);
      if (known?.note) {
        const note = document.createElement('em');
        note.textContent = known.note;
        el.append(note);
      }
      const label = new CSS2DObject(el);
      label.position.copy(nodePos[i]).add(new THREE.Vector3(0, 0.2, 2.2));
      group.add(label);
    });

    // Experts: a grid of cubes on the platform of the node whose process owns them.
    const experts = tr.model.experts;
    const expertNode = tr.expert_owner.map((p) => tr.processes[p].node);
    const expertMesh = new THREE.InstancedMesh(
      new THREE.BoxGeometry(0.3, 0.3, 0.3),
      new THREE.MeshStandardMaterial({ roughness: 0.4 }),
      experts,
    );
    const expertPos = [];
    const dummy = new THREE.Object3D();
    tr.nodes.forEach((_, ni) => {
      const mine = [...Array(experts).keys()].filter((e) => expertNode[e] === ni);
      const cols = Math.max(1, Math.ceil(Math.sqrt(mine.length)));
      const spacing = Math.min(0.38, 3.0 / cols);
      mine.forEach((e, i) => {
        const x = (i % cols) - (cols - 1) / 2;
        const z = Math.floor(i / cols) - (Math.ceil(mine.length / cols) - 1) / 2;
        const p = nodePos[ni].clone().add(new THREE.Vector3(x * spacing, 0.32, z * spacing));
        expertPos[e] = p;
        dummy.position.copy(p);
        dummy.scale.setScalar(spacing / 0.38);
        dummy.updateMatrix();
        expertMesh.setMatrixAt(e, dummy.matrix);
        expertMesh.setColorAt(e, COLORS.idle);
      });
    });
    group.add(expertMesh);

    const transport = new Map();
    for (const l of tr.links) {
      const pts = [nodePos[l.a].clone().setY(0.18), nodePos[l.b].clone().setY(0.18)];
      const geo = new THREE.BufferGeometry().setFromPoints(pts);
      const color = COLORS[l.transport] ?? COLORS.rdma;
      const mat = l.transport === 'tcp'
        ? new THREE.LineDashedMaterial({ color, dashSize: 0.3, gapSize: 0.2 })
        : new THREE.LineBasicMaterial({ color });
      const line = new THREE.Line(geo, mat);
      if (l.transport === 'tcp') line.computeLineDistances();
      group.add(line);
      const el = document.createElement('div');
      el.className = `link-label ${l.transport}`;
      const info = LINKS[l.transport];
      el.textContent = info ? `${info.label} · ${info.us.toLocaleString()} µs` : l.transport;
      const label = new CSS2DObject(el);
      label.position.copy(pts[0]).lerp(pts[1], 0.5);
      group.add(label);
      transport.set(`${l.a}|${l.b}`, l.transport);
      transport.set(`${l.b}|${l.a}`, l.transport);
    }

    scene.add(group);
    return {
      view: new THREE.Vector3(0, n <= 2 ? 9 : 11, n <= 2 ? 11 : 13),
      group, nodePos, expertMesh, expertPos,
      procNode: tr.processes.map((p) => p.node),
      transport: (a, b) => transport.get(`${a}|${b}`) ?? 'rdma',
      flash: new Array(experts).fill(null),
    };
  }

  function clearPackets() {
    packets.length = 0;
    freeSlots.length = 0;
    for (let i = MAX_PACKETS - 1; i >= 0; i--) freeSlots.push(i);
    dummy.scale.setScalar(0);
    dummy.updateMatrix();
    for (let i = 0; i < MAX_PACKETS; i++) packetMesh.setMatrixAt(i, dummy.matrix);
    packetMesh.instanceMatrix.needsUpdate = true;
  }

  function loadTrace(tr, label) {
    validateTrace(tr);
    trace = tr;
    const starts = tr.events.filter((e) => e[1] === KIND.layer).map((e) => e[0]);
    summary = { ...summarize(tr), layerEvents: starts.length };
    const gaps = starts.slice(1).map((t, i) => t - starts[i]).sort((a, b) => a - b);
    const median = gaps.length ? gaps[gaps.length >> 1] : summary.durationNs || 1;
    nsPerMs = Math.max(1e-3, median / LAYER_SCREEN_MS);
    topo = buildTopology(tr);
    frame();
    setBadge(label, tr);
    rewind();
  }

  function rewind() {
    clearPackets();
    cursor = 0;
    traceNow = 0;
    visNow = 0;
    layer = 0;
    Object.assign(stats, { packets: 0, remote: 0, prefetched: 0, misses: 0, experts: 0, fetchBytes: 0, dropped: 0 });
    perf.maxInFlight = 0;
    perf.minFps = Infinity;
    perf.maxWorkMs = 0;
    topo.flash.fill(null);
  }

  // (Re)builds the trace for the current source and controls.
  function reload() {
    try {
      if (state.source === 'stress') loadTrace(generateTrace(STRESS), 'Stress test: synthetic trace');
      else if (state.source === 'live') {
        loadTrace(generateTrace({
          preset: state.preset, passes: 2, topK: state.topK, lookahead: state.lookahead,
          recall: state.recall, seed: state.seed,
        }), 'Synthetic trace (simulated routing)');
      } else if (trace) rewind();
      setError('');
    } catch (err) {
      setError(err.message);
    }
  }

  function launchPacket(from, to, color, ms) {
    const slot = freeSlots.pop();
    if (slot === undefined) {
      stats.dropped++;
      return;
    }
    packetMesh.setColorAt(slot, color);
    packets.push({ slot, from, to, start: visNow, ms });
  }

  function apply(ev) {
    switch (ev[1]) {
      case KIND.layer:
        layer = ev[2];
        break;
      case KIND.dispatch:
      case KIND.combine: {
        stats.packets++;
        const src = topo.procNode[ev[7]];
        const dst = topo.procNode[ev[8]];
        if (src === dst) break;
        stats.remote++;
        const ms = Math.min(MAX_FLIGHT_MS, Math.max(MIN_FLIGHT_MS, ev[2] / nsPerMs));
        const nodeAt = (ni) => topo.nodePos[ni].clone().setY(0.4);
        const expertAt = topo.expertPos[ev[6]];
        const color = COLORS[topo.transport(src, dst)] ?? COLORS.rdma;
        if (ev[1] === KIND.dispatch) launchPacket(nodeAt(src), expertAt, color, ms);
        else launchPacket(expertAt, nodeAt(dst), color, ms);
        break;
      }
      case KIND.expert:
        stats.experts++;
        if (ev[7] === 1) stats.prefetched++;
        if (ev[7] === 2) stats.misses++;
        topo.flash[ev[5]] = { color: STATUS_COLOR[ev[7]], until: visNow + FLASH_MS };
        break;
      case KIND.fetch:
        stats.fetchBytes += ev[6];
        break;
      default:
        break; // unknown kinds are ignored, per the schema
    }
  }

  const tmp = new THREE.Vector3();
  const dummy = new THREE.Object3D();
  function updatePackets() {
    for (let i = packets.length - 1; i >= 0; i--) {
      const p = packets[i];
      const u = (visNow - p.start) / p.ms;
      if (u >= 1) {
        dummy.scale.setScalar(0);
        dummy.position.set(0, 0, 0);
        dummy.updateMatrix();
        packetMesh.setMatrixAt(p.slot, dummy.matrix);
        freeSlots.push(p.slot);
        packets[i] = packets[packets.length - 1];
        packets.pop();
        continue;
      }
      tmp.copy(p.from).lerp(p.to, u);
      tmp.y += Math.sin(Math.PI * u) * 1.6;
      dummy.position.copy(tmp);
      dummy.scale.setScalar(1);
      dummy.updateMatrix();
      packetMesh.setMatrixAt(p.slot, dummy.matrix);
    }
    packetMesh.instanceMatrix.needsUpdate = true;
    if (packetMesh.instanceColor) packetMesh.instanceColor.needsUpdate = true;
    perf.maxInFlight = Math.max(perf.maxInFlight, packets.length);
  }

  const c = new THREE.Color();
  function updateExperts() {
    for (let e = 0; e < topo.flash.length; e++) {
      const f = topo.flash[e];
      const k = f ? Math.max(0, (f.until - visNow) / FLASH_MS) : 0;
      c.copy(COLORS.idle);
      if (k > 0) c.lerp(f.color, Math.min(1, k * 1.6));
      topo.expertMesh.setColorAt(e, c);
    }
    topo.expertMesh.instanceColor.needsUpdate = true;
  }

  const $ = (id) => document.getElementById(id);
  const hud = {
    layer: $('stat-layer'), tps: $('stat-tps'), layerUs: $('stat-layer-us'), hit: $('stat-hit'),
    remote: $('stat-remote'), packets: $('stat-packets'), fps: $('stat-fps'), progress: $('trace-progress'),
  };
  const pct = (a, b) => (b ? `${((100 * a) / b).toFixed(1)}%` : '…');
  function updateHud() {
    const layers = summary.layerEvents || 1;
    hud.layer.textContent = `${layer + 1} / ${trace.model.layers}`;
    hud.tps.textContent = summary.durationNs ? `${(summary.tokens / (summary.durationNs * 1e-9)).toFixed(0)} tok/s` : '…';
    hud.layerUs.textContent = `${(summary.durationNs / layers / 1e3).toFixed(1)} µs`;
    hud.hit.textContent = pct(stats.prefetched, stats.prefetched + stats.misses);
    hud.remote.textContent = pct(stats.remote, stats.packets);
    hud.packets.textContent = `${stats.remote.toLocaleString()} drawn · ${packets.length} in flight`;
    hud.fps.textContent = `${perf.fps.toFixed(0)} · ${perf.workMs.toFixed(1)} ms/frame`;
    hud.progress.value = summary.durationNs ? Math.min(1, traceNow / summary.durationNs) : 0;
  }

  function setBadge(label, tr) {
    const s = summarize(tr);
    $('badge').textContent = `${label} · ${tr.source} · ${s.remote.toLocaleString()} remote packets`;
  }
  function setError(msg) {
    $('trace-error').textContent = msg;
  }

  function frame() {
    camera.position.copy(topo.view).multiplyScalar(Math.max(1, 1.5 / camera.aspect));
    controls.target.set(0, 0, 0);
    controls.update();
  }

  function resize() {
    const { clientWidth: w, clientHeight: h } = viewport;
    renderer.setSize(w, h);
    labels.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    if (topo) frame();
  }

  bindControls({ reload, loadFile, regenerate: () => { if (state.source === 'live') reload(); } });
  window.addEventListener('resize', resize);
  resize();
  reload();

  function loadFile(file) {
    const reader = new FileReader();
    reader.onload = () => {
      try {
        loadTrace(JSON.parse(reader.result), `File: ${file.name}`);
        state.source = 'file';
        $('source').value = 'file';
        syncEnabled();
        setError('');
      } catch (err) {
        setError(err.message);
      }
    };
    reader.readAsText(file);
  }

  let last = performance.now();
  renderer.setAnimationLoop((now) => {
    const dt = Math.min(100, now - last);
    last = now;
    perf.frames++;
    if (now - perf.since >= 1000) {
      perf.fps = (perf.frames * 1000) / (now - perf.since);
      if (packets.length > 0) perf.minFps = Math.min(perf.minFps, perf.fps);
      perf.workMs = perf.worst;
      perf.maxWorkMs = Math.max(perf.maxWorkMs, perf.worst);
      perf.worst = 0;
      perf.frames = 0;
      perf.since = now;
    }
    if (state.playing) {
      visNow += dt * state.speed;
      traceNow += dt * state.speed * nsPerMs;
      const events = trace.events;
      while (cursor < events.length && events[cursor][0] <= traceNow) apply(events[cursor++]);
      if (cursor >= events.length && packets.length === 0) {
        // Loop. Live traces get a new seed so each loop routes differently.
        if (state.source === 'live') {
          state.seed++;
          reload();
        } else rewind();
      }
    }
    updatePackets();
    updateExperts();
    updateHud();
    controls.update();
    renderer.render(scene, camera);
    labels.render(scene, camera);
    perf.worst = Math.max(perf.worst, performance.now() - now);
  });
}

// Cluster, top-k, and lookahead controls only shape generated traces.
function syncEnabled() {
  for (const id of ['preset', 'topk', 'lookahead', 'recall']) document.getElementById(id).disabled = state.source !== 'live';
  if (state.source === 'live') document.getElementById('recall').disabled = !state.lookahead;
}

function bindControls({ reload, loadFile, regenerate }) {
  const $ = (id) => document.getElementById(id);
  const source = $('source');
  source.value = state.source;
  source.addEventListener('change', () => {
    if (source.value === 'file') {
      $('trace-file').click();
      source.value = state.source; // switches only once a file loads
      return;
    }
    state.source = source.value;
    syncEnabled();
    reload();
  });
  $('trace-file').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) loadFile(file);
    e.target.value = '';
  });

  const preset = $('preset');
  for (const [key, p] of Object.entries(PRESETS)) preset.add(new Option(p.label, key));
  preset.value = state.preset;
  preset.addEventListener('change', () => { state.preset = preset.value; regenerate(); });

  const bindRange = (id, key, fmt, onChange) => {
    const input = $(id);
    const out = $(`${id}-out`);
    const sync = () => { state[key] = Number(input.value); out.textContent = fmt(state[key]); };
    input.addEventListener('input', sync);
    if (onChange) input.addEventListener('change', onChange);
    sync();
  };
  bindRange('topk', 'topK', (v) => `${v}`, regenerate);
  bindRange('recall', 'recall', (v) => `${Math.round(v * 100)}%`, regenerate);
  bindRange('speed', 'speed', (v) => `${v}x`);

  const lookahead = $('lookahead');
  lookahead.addEventListener('change', () => {
    state.lookahead = lookahead.checked;
    syncEnabled();
    regenerate();
  });

  const play = $('play');
  play.addEventListener('click', () => {
    state.playing = !state.playing;
    play.textContent = state.playing ? 'Pause' : 'Play';
  });
  $('restart').addEventListener('click', reload);
  syncEnabled();

  $('sources').innerHTML = [
    `RDMA latency: ${LINKS.rdma.us} µs (${LINKS.rdma.source})`,
    `TCP latency: ${LINKS.tcp.us.toLocaleString()} µs (${LINKS.tcp.source})`,
    `Memory bandwidth: ${HW.memGBps.value} GB/s (${HW.memGBps.source})`,
    `NVMe bandwidth: ${HW.nvmeGBps.value} GB/s (${HW.nvmeGBps.source})`,
  ].map((s) => `<li>${s}</li>`).join('');
}
