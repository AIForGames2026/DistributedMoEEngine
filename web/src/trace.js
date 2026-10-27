// Telemetry traces (docs/telemetry-schema.md): a synthetic generator built on
// the routing model in sim.js, and a validator for traces loaded from files.
// Pure JavaScript with no DOM use, so Node scripts can import it too.
import { MODEL, HW, PRESETS, NODES, Router, owner, EXPERT_BYTES, linkTypeOf, LINKS } from './sim.js';

export const SCHEMA = 'tensorfold.trace';
export const VERSION = 0;
export const PACKET_BYTES = 7552;
export const STATUS = ['resident', 'prefetched', 'miss'];
export const KIND = { layer: 'L', dispatch: 'D', expert: 'X', combine: 'C', fetch: 'F' };

// Generates a trace of `passes` forward passes over all layers. Timing comes
// from the cost model: link latency per hop, expert compute as reading its
// weights at memory bandwidth, and NVMe fetch time for misses.
export function generateTrace({
  preset = 'sparks', passes = 1, batch = 8, topK = MODEL.topK, lookahead = true, recall = 0.85, seed = 42,
} = {}) {
  const p = PRESETS[preset];
  const nodes = p.nodes;
  const router = new Router(seed);
  const computeNs = EXPERT_BYTES / HW.memGBps.value; // bytes / (GB/s) = ns
  const fetchNs = EXPERT_BYTES / HW.nvmeGBps.value;
  const linkNs = (a, b) => (a === b ? 0 : LINKS[linkTypeOf(p, nodes[a], nodes[b])].us * 1e3);
  const proc = (e) => nodes.indexOf(owner(e, nodes));

  const events = [];
  let t = 0;
  let batchId = 0;
  for (let pass = 0; pass < passes; pass++) {
    for (let layer = 0; layer < MODEL.layers; layer++) {
      events.push([t, KIND.layer, layer, batchId]);
      const byExpert = new Map();
      for (let tok = 0; tok < batch; tok++) {
        const home = tok % nodes.length;
        const actual = router.route(layer, topK);
        const predicted = lookahead ? router.predict(actual, recall) : new Set();
        for (const e of actual) {
          if (!byExpert.has(e)) {
            const status = router.resident[layer].has(e) ? 0 : predicted.has(e) ? 1 : 2;
            byExpert.set(e, { status, tokens: [] });
          }
          byExpert.get(e).tokens.push([tok, home]);
        }
      }

      // Dispatch, then each expert runs once all its tokens and its weights
      // are there; experts on one process run one after another.
      const busy = new Array(nodes.length).fill(t);
      const order = [];
      for (const [e, info] of byExpert) {
        const dst = proc(e);
        let ready = t;
        info.tokens.forEach(([tok, src]) => {
          const dur = linkNs(src, dst);
          events.push([t, KIND.dispatch, dur, batchId, tok, layer, e, src, dst, PACKET_BYTES]);
          ready = Math.max(ready, t + dur);
        });
        if (info.status === 1) events.push([Math.max(0, t - fetchNs), KIND.fetch, fetchNs, layer, e, dst, EXPERT_BYTES, 0]);
        if (info.status === 2) {
          events.push([t, KIND.fetch, fetchNs, layer, e, dst, EXPERT_BYTES, 1]);
          ready = Math.max(ready, t + fetchNs);
        }
        order.push({ e, dst, ready, info });
      }
      order.sort((a, b) => a.ready - b.ready || a.e - b.e);
      let end = t;
      for (const { e, dst, ready, info } of order) {
        const start = Math.max(busy[dst], ready);
        busy[dst] = start + computeNs;
        events.push([start, KIND.expert, computeNs, batchId, layer, e, dst, info.status]);
        info.tokens.forEach(([tok, home]) => {
          const dur = linkNs(dst, home);
          events.push([busy[dst], KIND.combine, dur, batchId, tok, layer, e, dst, home, PACKET_BYTES]);
          end = Math.max(end, busy[dst] + dur);
        });
      }
      t = Math.ceil(end);
    }
    batchId++;
  }
  events.forEach((ev) => { ev[0] = Math.round(ev[0]); if (ev[1] !== KIND.layer) ev[2] = Math.round(ev[2]); });
  events.sort((a, b) => a[0] - b[0]);

  return {
    schema: SCHEMA,
    version: VERSION,
    source: 'synthetic',
    time_unit: 'ns',
    model: { name: MODEL.name, layers: MODEL.layers, experts: MODEL.experts, top_k: topK, d_model: MODEL.dModel, d_ff: MODEL.dFf },
    nodes: nodes.map((id) => ({ id, label: NODES[id].label, kind: id === 'desktop' ? 'desktop' : 'gb10' })),
    links: p.links.map(([a, b, transport]) => ({ a: nodes.indexOf(a), b: nodes.indexOf(b), transport })),
    processes: nodes.map((_, i) => ({ id: i, node: i, rank: i })),
    expert_owner: [...Array(MODEL.experts).keys()].map(proc),
    batch_tokens: batch,
    generator: { preset, passes, batch, top_k: topK, lookahead, recall, seed },
    events,
  };
}

// Throws an Error describing the first problem; returns the trace otherwise.
export function validateTrace(tr) {
  const fail = (msg) => { throw new Error(`invalid trace: ${msg}`); };
  if (!tr || tr.schema !== SCHEMA) fail(`schema must be "${SCHEMA}"`);
  if (tr.version !== VERSION) fail(`unsupported version ${tr.version} (this player reads ${VERSION})`);
  if (tr.time_unit !== 'ns') fail('time_unit must be "ns"');
  if (!Array.isArray(tr.nodes) || !tr.nodes.length) fail('nodes is empty');
  if (!Array.isArray(tr.processes) || !tr.processes.length) fail('processes is empty');
  if (!Array.isArray(tr.links)) fail('links missing');
  if (!tr.model || !(tr.model.experts > 0) || !(tr.model.layers > 0)) fail('model shape missing');
  if (!Array.isArray(tr.expert_owner) || tr.expert_owner.length !== tr.model.experts) fail('expert_owner length != model.experts');
  if (!Array.isArray(tr.events)) fail('events missing');
  const nProc = tr.processes.length;
  for (const pr of tr.processes) if (!(pr.node >= 0 && pr.node < tr.nodes.length)) fail(`process ${pr.id} on unknown node`);
  for (const l of tr.links) if (!(l.a in tr.nodes) || !(l.b in tr.nodes)) fail('link references unknown node');
  let prev = -Infinity;
  for (let i = 0; i < tr.events.length; i++) {
    const ev = tr.events[i];
    if (!Array.isArray(ev) || typeof ev[0] !== 'number') fail(`event ${i} is not [t, kind, ...]`);
    if (ev[0] < prev) fail(`event ${i} is out of time order`);
    prev = ev[0];
    if (ev[1] === KIND.dispatch || ev[1] === KIND.combine) {
      if (ev.length < 10 || !(ev[7] < nProc) || !(ev[8] < nProc) || !(ev[6] < tr.model.experts)) fail(`event ${i}: bad packet fields`);
    } else if (ev[1] === KIND.expert) {
      if (ev.length < 8 || !(ev[5] < tr.model.experts) || !(ev[6] < nProc) || !(ev[7] in STATUS)) fail(`event ${i}: bad expert fields`);
    }
  }
  return tr;
}

export function summarize(tr) {
  let packets = 0, remote = 0, prefetched = 0, misses = 0, experts = 0;
  for (const ev of tr.events) {
    if (ev[1] === KIND.dispatch || ev[1] === KIND.combine) {
      packets++;
      if (ev[7] !== ev[8]) remote++;
    } else if (ev[1] === KIND.expert) {
      experts++;
      if (ev[7] === 1) prefetched++;
      if (ev[7] === 2) misses++;
    }
  }
  const last = tr.events.at(-1);
  const durationNs = last ? last[0] + (last[1] === KIND.layer ? 0 : last[2]) : 0;
  const batches = new Set(tr.events.filter((e) => e[1] === KIND.layer).map((e) => e[3])).size;
  return { packets, remote, prefetched, misses, experts, durationNs, tokens: batches * tr.batch_tokens };
}
