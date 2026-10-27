// Checks the synthetic trace generator against the v0 schema (run by CI), or
// writes a trace to a file for use with "Load trace file" in the inspector.
//   node scripts/trace.mjs check
//   node scripts/trace.mjs write <out.json> [preset] [batch] [passes]
import { writeFileSync } from 'node:fs';
import { generateTrace, validateTrace, summarize } from '../src/trace.js';
import { STRESS } from '../src/presets.js';

function check() {
  const cases = {
    'live sparks': generateTrace({ preset: 'sparks' }),
    'live today': generateTrace({ preset: 'today' }),
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
