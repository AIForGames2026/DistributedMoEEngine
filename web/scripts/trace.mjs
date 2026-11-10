// Checks the synthetic trace generator against the v0 schema (run by CI), or
// writes a trace to a file for use with "Load trace file" in the inspector.
//   node scripts/trace.mjs check
//   node scripts/trace.mjs write <out.json> [preset] [batch] [passes]
import { writeFileSync } from 'node:fs';
import { generateTrace, validateTrace, summarize } from '../src/trace.js';
import { STRESS, MISSION } from '../src/presets.js';
import { PRESETS } from '../src/sim.js';

// The guided mission's lessons must hold in the model it runs on.
function checkMission() {
  const speed = (preset, lookahead) => {
    const s = summarize(generateTrace({ preset, lookahead, recall: MISSION.recall, passes: MISSION.passes, seed: MISSION.seed }));
    return s.tokens / (s.durationNs * 1e-9);
  };
  const fail = (msg) => { throw new Error(`mission: ${msg}`); };
  if (PRESETS.today.label !== MISSION.relayLabel) fail('step text names a Cluster option that does not exist');
  if (speed('sparks', false) >= speed('sparks', true)) fail('turning off lookahead must slow the three-Spark cluster');
  for (const lookahead of [true, false]) {
    if (speed('today', lookahead) >= MISSION.targetTps) fail('the relay must not reach the target');
    if (speed('direct', lookahead) < MISSION.targetTps) fail('direct TCP must reach the target');
  }
  console.log(`mission check: ok (${speed('sparks', true).toFixed(0)}, ${speed('sparks', false).toFixed(0)}, ` +
    `${speed('today', true).toFixed(1)}, ${speed('direct', true).toFixed(0)} tok/s, target ${MISSION.targetTps})`);
}

function check() {
  const cases = {
    'live sparks': generateTrace({ preset: 'sparks' }),
    'live today': generateTrace({ preset: 'today' }),
    'live direct': generateTrace({ preset: 'direct' }),
    'live laptop': generateTrace({ preset: 'laptop' }),
    'no lookahead': generateTrace({ lookahead: false }),
    stress: generateTrace(STRESS),
  };
  for (const [name, tr] of Object.entries(cases)) {
    validateTrace(JSON.parse(JSON.stringify(tr)));
    const s = summarize(tr);
    console.log(`${name}: ${tr.events.length} events, ${s.packets} packets (${s.remote} remote), ` +
      `${s.experts} expert runs (${s.misses} misses), ${(s.durationNs / 1e3).toFixed(0)} us`);
  }
  const s = summarize(cases.stress);
  if (s.remote < 10000) throw new Error(`stress trace has ${s.remote} drawn packets, need at least 10,000`);
  if (summarize(cases['no lookahead']).prefetched !== 0) throw new Error('prefetches without lookahead');
  let threw = false;
  try { validateTrace({ ...cases.stress, version: 1 }); } catch { threw = true; }
  if (!threw) throw new Error('validator accepted version 1');
  checkMission();
  console.log('trace check: ok');
}

const [cmd, out, preset = 'sparks', batch = '8', passes = '1'] = process.argv.slice(2);
if (cmd === 'check') check();
else if (cmd === 'write' && out) {
  const tr = generateTrace({ preset, batch: Number(batch), passes: Number(passes) });
  writeFileSync(out, JSON.stringify(tr));
  console.log(`wrote ${out}: ${tr.events.length} events`);
} else {
  console.error('usage: node scripts/trace.mjs check | write <out.json> [preset] [batch] [passes]');
  process.exit(2);
}
