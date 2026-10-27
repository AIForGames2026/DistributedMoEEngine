// Simulated MoE routing and cost model for the TensorFold-Swarm inspector.
// Every constant names its source so the demo never presents a guess as a measurement.

export const MODEL = {
  name: 'OLMoE-1B-7B',
  layers: 16,
  experts: 64,
  topK: 8,
  dModel: 2048,
  dFf: 1024,
};

// NVFP4 SwiGLU expert: 3 matrices of d_model x d_ff at 0.5625 bytes per weight.
export const EXPERT_BYTES = 3 * MODEL.dModel * MODEL.dFf * 0.5625;

export const LINKS = {
  rdma: {
    label: 'RoCE v2 RDMA',
    us: 3.26,
    source: 'measured: ib_write_lat loopback on gx10-03, 7,552-byte packet',
  },
  tcp: {
    label: 'TCP over Tailscale',
    us: 2000,
    source: 'assumed until TASK-02 measures it',
  },
};

export const HW = {
  memGBps: { value: 273, source: 'DGX Spark spec (LPDDR5X peak)' },
  nvmeGBps: { value: 6.5, source: 'estimate for PCIe Gen 4 x4, pending TASK-03 fio baseline' },
};

export const NODES = {
  'gx10-01': { label: 'gx10-01', sub: 'DGX Spark / GB10', note: 'busy today (GLM 5.3)' },
  'gx10-02': { label: 'gx10-02', sub: 'DGX Spark / GB10', note: 'busy today (GLM 5.3)' },
  'gx10-03': { label: 'gx10-03', sub: 'DGX Spark / GB10', note: 'available (shared)' },
  desktop: { label: 'Desktop', sub: 'RTX 5070 Ti', note: 'dev workstation' },
};

export const PRESETS = {
  sparks: {
    label: 'Target: three Sparks, 200 Gb/s full mesh',
    nodes: ['gx10-01', 'gx10-02', 'gx10-03'],
    links: [
      ['gx10-01', 'gx10-02', 'rdma'],
      ['gx10-01', 'gx10-03', 'rdma'],
      ['gx10-02', 'gx10-03', 'rdma'],
    ],
  },
  today: {
    label: 'Today: gx10-03 + desktop over Tailscale',
    nodes: ['gx10-03', 'desktop'],
    links: [['gx10-03', 'desktop', 'tcp']],
  },
};

// xorshift32: small, fast, and deterministic so runs are reproducible.
export function makeRng(seed) {
  let s = seed >>> 0 || 1;
  return () => {
    s ^= s << 13; s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5; s >>>= 0;
    return s / 4294967296;
  };
}

export class Router {
  constructor(seed = 42) {
    this.rng = makeRng(seed);
    // Per-layer Zipf-like expert popularity: real MoE routing is skewed, not uniform.
    this.logPop = [];
    this.resident = [];
    for (let l = 0; l < MODEL.layers; l++) {
      const order = [...Array(MODEL.experts).keys()];
      for (let i = order.length - 1; i > 0; i--) {
        const j = Math.floor(this.rng() * (i + 1));
        [order[i], order[j]] = [order[j], order[i]];
      }
      const logPop = new Float64Array(MODEL.experts);
      order.forEach((e, rank) => { logPop[e] = -0.8 * Math.log(rank + 1); });
      this.logPop.push(logPop);
      // Memory is tight (about 11 GB free on gx10-03), so only the more popular half stays resident.
      this.resident.push(new Set(order.slice(0, MODEL.experts / 2)));
    }
  }

  gumbel() {
    return -Math.log(-Math.log(this.rng() + 1e-12) + 1e-12);
  }

  // Top-k over popularity plus Gumbel noise: a sample of k distinct experts.
  route(layer, k) {
    const scores = this.logPop[layer].map((p) => p + this.gumbel());
    return [...scores.keys()].sort((a, b) => scores[b] - scores[a]).slice(0, k);
  }

  // A lookahead predictor with the given recall: each true expert is kept with
  // probability `recall`, otherwise replaced by a wrong guess.
  predict(actual, recall) {
    const out = new Set();
    for (const e of actual) {
      if (this.rng() < recall) out.add(e);
      else out.add(Math.floor(this.rng() * MODEL.experts));
    }
    return out;
  }
}

export function owner(expert, nodes) {
  return nodes[expert % nodes.length];
}

// Modeled (not measured) cost of one layer for one batch.
export function layerCost(events, nodes, linkUs) {
  const computeUs = EXPERT_BYTES / (HW.memGBps.value * 1e3);
  const missUs = EXPERT_BYTES / (HW.nvmeGBps.value * 1e3);
  const perNode = new Map(nodes.map((n) => [n, { active: new Set(), misses: new Set() }]));
  let comm = 0;
  for (const ev of events) {
    const slot = perNode.get(ev.to);
    slot.active.add(ev.expert);
    if (ev.status === 'miss') slot.misses.add(ev.expert);
    if (ev.from !== ev.to) comm = Math.max(comm, 2 * linkUs(ev.from, ev.to));
  }
  let worst = 0;
  for (const { active, misses } of perNode.values()) {
    worst = Math.max(worst, active.size * computeUs + misses.size * missUs);
  }
  return { us: comm + worst, commUs: comm, computeUs, missUs };
}
