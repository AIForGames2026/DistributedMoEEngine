// Replays every recording on the built or public site in Firefox over WebDriver BiDi (TASK-18), and checks that
// each plays, its per-rank counts match the engine's, and nothing reached the console or broke the CSP.
//   firefox -headless -no-remote -profile DIR --remote-debugging-port 9333 about:blank
//   node scripts/check-firefox.mjs http://localhost:4173/ [port] [screenshot-dir]
import { writeFileSync } from 'node:fs';
import { join } from 'node:path';

const base = new URL(process.argv[2] || 'http://localhost:4173/');
const port = process.argv[3] || '9333';
const shots = process.argv[4] || null;

const ws = new WebSocket(`ws://127.0.0.1:${port}/session`);
let next = 1;
const waiting = new Map();
const logs = [];
ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && waiting.has(m.id)) {
    const { ok, bad } = waiting.get(m.id);
    waiting.delete(m.id);
    if (m.type === 'error') {
      bad(new Error(`${m.error}: ${m.message}`));
    } else {
      ok(m.result);
    }
  } else if (m.method === 'log.entryAdded') {
    logs.push(`${m.params.level}: ${m.params.text}`);
  }
};
await new Promise((ok, bad) => {
  ws.onopen = ok;
  ws.onerror = () => bad(new Error(`no WebDriver BiDi on port ${port}; start Firefox with --remote-debugging-port`));
});

function send(method, params = {}) {
  const id = next++;
  ws.send(JSON.stringify({ id, method, params }));
  return new Promise((ok, bad) => waiting.set(id, { ok, bad }));
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const session = await send('session.new', { capabilities: {} });
console.log(`browser: ${session.capabilities.browserName} ${session.capabilities.browserVersion}`);
await send('session.subscribe', { events: ['log.entryAdded'] });
await send('script.addPreloadScript', {
  functionDeclaration: "() => { window.__csp = []; document.addEventListener('securitypolicyviolation', (e) => window.__csp.push(e.violatedDirective + ' ' + e.blockedURI)); }",
});
const context = (await send('browsingContext.getTree')).contexts[0].context;

async function evaluate(expression) {
  const r = await send('script.evaluate', { expression, target: { context }, awaitPromise: true, resultOwnership: 'none' });
  if (r.type === 'exception') {
    throw new Error(r.exceptionDetails.text);
  }
  return r.result.value;
}

const state = `JSON.stringify({
  conn: document.getElementById('conn').textContent,
  pick: document.getElementById('pick').options.length,
  stats: [...document.querySelectorAll('.stat')].map((s) => s.innerText.replace(/\\n/g, ' | ')),
  problems: [...document.querySelectorAll('.msg.error')].map((e) => e.textContent),
  csp: window.__csp,
})`;

await send('browsingContext.navigate', { context, url: base.href, wait: 'complete' });
const list = JSON.parse(await evaluate("fetch('recordings/index.json').then((r) => r.text())")).recordings;
let failed = 0;
for (const rec of list) {
  logs.length = 0;
  const url = new URL(`?r=${rec.id}`, base).href;
  await send('browsingContext.navigate', { context, url, wait: 'complete' });
  await sleep(1500);
  await evaluate("const s = document.getElementById('speed'); s.value = '10'; s.dispatchEvent(new Event('change')); true");
  await sleep(8000);
  const s = JSON.parse(await evaluate(state));
  const ranks = s.stats.filter((t) => /^rank \d+ expert calls/.test(t));
  const played = s.stats.find((t) => t.startsWith('recording'));
  const steps = Number((played || '').match(/\| ([\d,]+) of/)?.[1].replace(/,/g, '') || 0);
  const why = [];
  if (!s.conn.startsWith('recorded on')) {
    why.push(`header says "${s.conn}"`);
  }
  if (!steps) {
    why.push('no steps played');
  }
  if (!ranks.length || !ranks.every((t) => t.includes("matches the engine's own count"))) {
    why.push(`counts: ${ranks.join(' / ')}`);
  }
  if (!/0 dropped while recording · 0 lines rejected/.test(played || '')) {
    why.push(`recording: ${played}`);
  }
  if (s.problems.length) {
    why.push(`errors shown: ${s.problems.join('; ')}`);
  }
  if (s.csp.length) {
    why.push(`CSP violations: ${s.csp.join('; ')}`);
  }
  const loud = logs.filter((l) => !l.startsWith('debug'));
  if (loud.length) {
    why.push(`console: ${loud.join('; ')}`);
  }
  if (shots) {
    const png = await send('browsingContext.captureScreenshot', { context });
    writeFileSync(join(shots, `firefox-${rec.id}.png`), Buffer.from(png.data, 'base64'));
  }
  console.log(`${why.length ? 'FAIL' : 'ok  '} ${rec.id}: ${steps.toLocaleString('en-US')} steps played; ${ranks.map((t) => t.split(' | ').slice(0, 2).join(' ')).join(', ')}${why.length ? `\n     ${why.join('\n     ')}` : ''}`);
  if (why.length) {
    failed++;
  }
}
await send('session.end').catch(() => {});
ws.close();
if (failed) {
  console.error(`firefox check: ${failed} of ${list.length} recordings failed`);
  process.exit(1);
}
console.log(`firefox check: ok (${list.length} recordings)`);
