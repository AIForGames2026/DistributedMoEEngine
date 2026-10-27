// TensorFold-Swarm inspector: WebGL view of MoE tokens being routed to experts across nodes.
// Phase 1 (TASK-04) drives it from the simulation in sim.js; TASK-16 swaps in recorded engine traces.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { MODEL, LINKS, HW, NODES, PRESETS, Router, owner, layerCost } from './sim.js';
import './style.css';

const COLORS = {
  idle: new THREE.Color(0x2b3446),
  resident: new THREE.Color(0x7dd3fc),
  prefetched: new THREE.Color(0x4ade80),
  miss: new THREE.Color(0xf87171),
  rdma: new THREE.Color(0x38bdf8),
  tcp: new THREE.Color(0xfb923c),
};
const PHASE_MS = { dispatch: 900, compute: 450, missStall: 350, combine: 900 };
const FLASH_MS = 1100;
const BATCH = 8;
const MAX_PACKETS = 2048;
const HISTORY = 64;

const state = { preset: 'sparks', topK: MODEL.topK, lookahead: true, recall: 0.85, speed: 1, playing: true };

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
  camera.position.set(0, 11, 13);
  const controls = new OrbitControls(camera, labels.domElement);
  controls.enableDamping = true;
  controls.target.set(0, 0, 0);

  scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  const sun = new THREE.DirectionalLight(0xffffff, 1.2);
  sun.position.set(5, 10, 6);
  scene.add(sun);

  const grid = new THREE.GridHelper(30, 30, 0x1c2540, 0x141b2e);
  grid.position.y = -0.2;
  scene.add(grid);

  // Packets: one instanced mesh, recycled from a pool.
  const packetMesh = new THREE.InstancedMesh(
    new THREE.SphereGeometry(0.09, 10, 8),
    new THREE.MeshBasicMaterial(),
    MAX_PACKETS,
  );
  packetMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  for (let i = 0; i < MAX_PACKETS; i++) packetMesh.setColorAt(i, COLORS.rdma);
  scene.add(packetMesh);
  const packets = [];
  const freeSlots = [...Array(MAX_PACKETS).keys()].reverse();

  let topo = null;
  let router = new Router(42);
  let simNow = 0;
  let layer = 0;
  let phase = 'dispatch';
  let phaseEnd = 0;
  let events = [];
  const stats = { layers: [], tokens: 0, packets: 0, remote: 0, prefetched: 0, misses: 0 };

  function buildTopology(key) {
    if (topo) {
      scene.remove(topo.group);
      topo.group.traverse((o) => {
        if (o.isCSS2DObject) o.element.remove();
        o.geometry?.dispose();
        o.material?.dispose?.();
      });
    }
    const preset = PRESETS[key];
    const group = new THREE.Group();
    const n = preset.nodes.length;
    const radius = n === 2 ? 4.5 : 5.2;
    const nodePos = {};
    preset.nodes.forEach((id, i) => {
      const a = n === 2 ? Math.PI * i : (2 * Math.PI * i) / n - Math.PI / 2;
      nodePos[id] = new THREE.Vector3(Math.cos(a) * radius, 0, Math.sin(a) * radius);
    });

    // Node platforms and labels.
    for (const id of preset.nodes) {
      const info = NODES[id];
      const platform = new THREE.Mesh(
        new THREE.BoxGeometry(3.4, 0.3, 3.4),
        new THREE.MeshStandardMaterial({ color: id === 'desktop' ? 0x3b2a1e : 0x1e293b, roughness: 0.7 }),
      );
      platform.position.copy(nodePos[id]);
      group.add(platform);
      const el = document.createElement('div');
      el.className = 'node-label';
      el.innerHTML = `<b>${info.label}</b><span>${info.sub}</span><em>${info.note}</em>`;
      const label = new CSS2DObject(el);
      label.position.copy(nodePos[id]).add(new THREE.Vector3(0, 0.2, 2.2));
      group.add(label);
    }

    // Experts: a grid of cubes on each node's platform.
    const expertMesh = new THREE.InstancedMesh(
      new THREE.BoxGeometry(0.3, 0.3, 0.3),
      new THREE.MeshStandardMaterial({ roughness: 0.4 }),
      MODEL.experts,
    );
    const expertPos = [];
    const dummy = new THREE.Object3D();
    for (const id of preset.nodes) {
      const mine = [...Array(MODEL.experts).keys()].filter((e) => owner(e, preset.nodes) === id);
      const cols = Math.ceil(Math.sqrt(mine.length));
      mine.forEach((e, i) => {
        const x = (i % cols) - (cols - 1) / 2;
        const z = Math.floor(i / cols) - (Math.ceil(mine.length / cols) - 1) / 2;
        const p = nodePos[id].clone().add(new THREE.Vector3(x * 0.38, 0.32, z * 0.38));
        expertPos[e] = p;
        dummy.position.copy(p);
        dummy.updateMatrix();
        expertMesh.setMatrixAt(e, dummy.matrix);
        expertMesh.setColorAt(e, COLORS.idle);
      });
    }
    group.add(expertMesh);

    // Links between nodes.
    const linkUsByPair = new Map();
    for (const [a, b, type] of preset.links) {
      const pts = [nodePos[a].clone().setY(0.18), nodePos[b].clone().setY(0.18)];
      const geo = new THREE.BufferGeometry().setFromPoints(pts);
      const mat = type === 'tcp'
        ? new THREE.LineDashedMaterial({ color: COLORS.tcp, dashSize: 0.3, gapSize: 0.2 })
        : new THREE.LineBasicMaterial({ color: COLORS.rdma });
      const line = new THREE.Line(geo, mat);
      if (type === 'tcp') line.computeLineDistances();
      group.add(line);
      const el = document.createElement('div');
      el.className = `link-label ${type}`;
      el.textContent = `${LINKS[type].label} · ${LINKS[type].us} µs`;
      const label = new CSS2DObject(el);
      label.position.copy(pts[0]).lerp(pts[1], 0.5);
      group.add(label);
      linkUsByPair.set(`${a}|${b}`, LINKS[type].us);
      linkUsByPair.set(`${b}|${a}`, LINKS[type].us);
    }
    const linkType = Object.fromEntries(preset.links.flatMap(([a, b, t]) => [[`${a}|${b}`, t], [`${b}|${a}`, t]]));

    scene.add(group);
    return {
      view: new THREE.Vector3(0, n === 2 ? 9 : 11, n === 2 ? 11 : 13),
      group, preset, nodePos, expertMesh, expertPos, linkType,
      flash: new Array(MODEL.experts).fill(null),
      linkUs: (a, b) => linkUsByPair.get(`${a}|${b}`) ?? 0,
    };
  }

  function launchPacket(from, to, color, duration) {
    const slot = freeSlots.pop();
    if (slot === undefined) return;
    packetMesh.setColorAt(slot, color);
    packets.push({ slot, from, to, start: simNow, duration });
  }

  function beginLayer() {
    const nodes = topo.preset.nodes;
    events = [];
    for (let t = 0; t < BATCH; t++) {
      const home = nodes[t % nodes.length];
      const actual = router.route(layer, state.topK);
      const predicted = state.lookahead ? router.predict(actual, state.recall) : new Set();
      for (const e of actual) {
        const to = owner(e, nodes);
        const status = router.resident[layer].has(e) ? 'resident' : predicted.has(e) ? 'prefetched' : 'miss';
        events.push({ token: t, from: home, to, expert: e, status });
      }
    }
    for (const ev of events) {
      stats.packets++;
      if (ev.status === 'prefetched') stats.prefetched++;
      if (ev.status === 'miss') stats.misses++;
      if (ev.from === ev.to) continue;
      stats.remote++;
      const type = topo.linkType[`${ev.from}|${ev.to}`];
      launchPacket(topo.nodePos[ev.from].clone().setY(0.4), topo.expertPos[ev.expert], COLORS[type], PHASE_MS.dispatch);
    }
    stats.layers.push(layerCost(events, nodes, topo.linkUs).us);
    if (stats.layers.length > HISTORY) stats.layers.shift();
    phase = 'dispatch';
    phaseEnd = simNow + PHASE_MS.dispatch;
  }

  function advancePhase() {
    if (phase === 'dispatch') {
      for (const ev of events) topo.flash[ev.expert] = { color: COLORS[ev.status], until: simNow + FLASH_MS };
      phase = 'compute';
      phaseEnd += PHASE_MS.compute + (events.some((e) => e.status === 'miss') ? PHASE_MS.missStall : 0);
    } else if (phase === 'compute') {
      for (const ev of events) {
        if (ev.from === ev.to) continue;
        const type = topo.linkType[`${ev.from}|${ev.to}`];
        launchPacket(topo.expertPos[ev.expert], topo.nodePos[ev.from].clone().setY(0.4), COLORS[type], PHASE_MS.combine);
      }
      phase = 'combine';
      phaseEnd += PHASE_MS.combine;
    } else {
      layer = (layer + 1) % MODEL.layers;
      if (layer === 0) stats.tokens += BATCH;
      beginLayer();
    }
  }

  const tmp = new THREE.Vector3();
  const dummy = new THREE.Object3D();
  function updatePackets() {
    for (let i = packets.length - 1; i >= 0; i--) {
      const p = packets[i];
      const u = (simNow - p.start) / p.duration;
      if (u >= 1) {
        dummy.scale.setScalar(0);
        dummy.updateMatrix();
        packetMesh.setMatrixAt(p.slot, dummy.matrix);
        freeSlots.push(p.slot);
        packets.splice(i, 1);
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
  }

  const c = new THREE.Color();
  function updateExperts() {
    for (let e = 0; e < MODEL.experts; e++) {
      const f = topo.flash[e];
      const k = f ? Math.max(0, (f.until - simNow) / FLASH_MS) : 0;
      c.copy(COLORS.idle);
      if (k > 0) c.lerp(f.color, Math.min(1, k * 1.6));
      topo.expertMesh.setColorAt(e, c);
    }
    topo.expertMesh.instanceColor.needsUpdate = true;
  }

  const hud = {
    layer: document.getElementById('stat-layer'),
    tps: document.getElementById('stat-tps'),
    layerUs: document.getElementById('stat-layer-us'),
    hit: document.getElementById('stat-hit'),
    remote: document.getElementById('stat-remote'),
    tokens: document.getElementById('stat-tokens'),
  };
  function updateHud() {
    const avg = stats.layers.reduce((a, b) => a + b, 0) / Math.max(1, stats.layers.length);
    const tps = BATCH / (avg * MODEL.layers * 1e-6);
    const needed = stats.prefetched + stats.misses;
    hud.layer.textContent = `${layer + 1} / ${MODEL.layers} · ${phase}`;
    hud.tps.textContent = Number.isFinite(tps) ? `${tps.toFixed(0)} tok/s` : '…';
    hud.layerUs.textContent = `${avg.toFixed(1)} µs`;
    hud.hit.textContent = needed ? `${((100 * stats.prefetched) / needed).toFixed(1)}%` : '…';
    hud.remote.textContent = stats.packets ? `${((100 * stats.remote) / stats.packets).toFixed(1)}%` : '…';
    hud.tokens.textContent = `${stats.tokens}`;
  }

  function reset() {
    for (const p of packets) freeSlots.push(p.slot);
    packets.length = 0;
    for (let i = 0; i < MAX_PACKETS; i++) {
      dummy.scale.setScalar(0);
      dummy.updateMatrix();
      packetMesh.setMatrixAt(i, dummy.matrix);
    }
    Object.assign(stats, { layers: [], tokens: 0, packets: 0, remote: 0, prefetched: 0, misses: 0 });
    router = new Router(42);
    layer = 0;
    simNow = 0;
    topo = buildTopology(state.preset);
    frame();
    beginLayer();
  }

  // Pull the camera back on narrow viewports so every node stays in view.
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

  bindControls(reset);
  window.addEventListener('resize', resize);
  resize();
  reset();

  let last = performance.now();
  renderer.setAnimationLoop((now) => {
    const dt = Math.min(100, now - last);
    last = now;
    if (state.playing) {
      simNow += dt * state.speed;
      while (simNow >= phaseEnd) advancePhase();
    }
    updatePackets();
    updateExperts();
    updateHud();
    controls.update();
    renderer.render(scene, camera);
    labels.render(scene, camera);
  });
}

function bindControls(reset) {
  const $ = (id) => document.getElementById(id);
  const preset = $('preset');
  for (const [key, p] of Object.entries(PRESETS)) preset.add(new Option(p.label, key));
  preset.value = state.preset;
  preset.addEventListener('change', () => { state.preset = preset.value; reset(); });

  const bindRange = (id, key, fmt) => {
    const input = $(id);
    const out = $(`${id}-out`);
    const sync = () => { state[key] = Number(input.value); out.textContent = fmt(state[key]); };
    input.addEventListener('input', sync);
    sync();
  };
  bindRange('topk', 'topK', (v) => `${v}`);
  bindRange('recall', 'recall', (v) => `${Math.round(v * 100)}%`);
  bindRange('speed', 'speed', (v) => `${v}x`);

  const lookahead = $('lookahead');
  lookahead.addEventListener('change', () => {
    state.lookahead = lookahead.checked;
    $('recall').disabled = !lookahead.checked;
  });

  const play = $('play');
  play.addEventListener('click', () => {
    state.playing = !state.playing;
    play.textContent = state.playing ? 'Pause' : 'Play';
  });
  $('restart').addEventListener('click', reset);

  $('sources').innerHTML = [
    `RDMA latency: ${LINKS.rdma.us} µs (${LINKS.rdma.source})`,
    `TCP latency: ${LINKS.tcp.us} µs (${LINKS.tcp.source})`,
    `Memory bandwidth: ${HW.memGBps.value} GB/s (${HW.memGBps.source})`,
    `NVMe bandwidth: ${HW.nvmeGBps.value} GB/s (${HW.nvmeGBps.source})`,
  ].map((s) => `<li>${s}</li>`).join('');
}
