// Builds the public site's main page (TASK-18) into dist/ after `vite build`: the engine's own
// 2D page (live.html) in showcase mode, with the recordings listed in showcase.json.
//   node scripts/showcase.mjs [dist]
import { createHash } from 'node:crypto';
import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gunzipSync } from 'node:zlib';

const web = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const repo = resolve(web, '..');
const dist = resolve(process.argv[2] || join(web, 'dist'));
const blob = 'https://github.com/AIForGames2026/DistributedMoEEngine/blob/main/';

// The public page loads only its own files and the recordings beside it; it never connects anywhere else.
const CSP = [
  "default-src 'none'",
  "script-src 'self'",
  "style-src 'self'",
  "img-src 'self' data:",
  "connect-src 'self'",
  "base-uri 'none'",
  "form-action 'none'",
].join('; ');

function fail(msg) {
  console.error(`showcase: ${msg}`);
  process.exit(1);
}

function only(html, re, what) {
  const all = [...html.matchAll(re)];
  if (all.length !== 1) {
    fail(`live.html must have exactly one inline ${what}, found ${all.length}`);
  }
  return all[0];
}

function emit(name, ext, text) {
  const hash = createHash('sha256').update(text).digest('hex').slice(0, 10);
  const file = `${name}-${hash}.${ext}`;
  writeFileSync(join(dist, file), text);
  return file;
}

// A recording must be one the page will replay, and its label must fit it.
function checkRecording(rec, bytes) {
  const text = (bytes[0] === 0x1f && bytes[1] === 0x8b ? gunzipSync(bytes) : bytes).toString('utf8');
  const lines = text.split('\n').filter((l) => l.trim());
  const hello = JSON.parse(lines[0]);
  if (hello.t !== 'hello' || hello.schema !== 'tensorfold.live') {
    fail(`${rec.source} does not start with a TensorFold hello line`);
  }
  if (rec.ranks && rec.ranks.length !== hello.ranks.length) {
    fail(`${rec.id} names ${rec.ranks.length} ranks but the recording has ${hello.ranks.length}`);
  }
  let steps = 0;
  for (const l of lines.slice(1)) {
    if (JSON.parse(l).t === 'step') {
      steps++;
    }
  }
  if (!steps) {
    fail(`${rec.source} holds no engine steps`);
  }
  return steps;
}

const manifest = JSON.parse(readFileSync(join(web, 'showcase.json'), 'utf8'));
mkdirSync(join(dist, 'recordings'), { recursive: true });
const listed = [];
for (const rec of manifest.recordings) {
  if (!/^[a-z0-9-]+$/.test(rec.id) || !rec.title || !rec.source) {
    fail(`every recording needs an id (lowercase letters, digits, dashes), a title, and a source`);
  }
  const src = join(repo, rec.source);
  const bytes = readFileSync(src);
  const steps = checkRecording(rec, bytes);
  const ext = bytes[0] === 0x1f && bytes[1] === 0x8b ? '.jsonl.gz' : '.jsonl';
  const file = `recordings/${rec.id}${ext}`;
  copyFileSync(src, join(dist, file));
  listed.push({
    id: rec.id,
    title: rec.title,
    summary: rec.summary || '',
    note: rec.note || '',
    file,
    results: rec.results ? blob + rec.results : null,
    ranks: rec.ranks || null,
    speed: rec.speed || 1,
  });
  console.log(`showcase: ${file} (${steps.toLocaleString('en-US')} steps, ${(bytes.length / 1024).toFixed(0)} KB) from ${rec.source}`);
}
writeFileSync(join(dist, 'recordings', 'index.json'), JSON.stringify({ recordings: listed }, null, 2) + '\n');

let html = readFileSync(join(web, 'live.html'), 'utf8');
const style = only(html, /<style>([\s\S]*?)<\/style>/g, '<style>');
const script = only(html, /<script>([\s\S]*?)<\/script>/g, '<script>');
const css = emit('live', 'css', style[1].replace(/^\n/, ''));
const js = emit('live', 'js', script[1].replace(/^\n/, ''));
html = html.replace(style[0], `<link rel="stylesheet" href="${css}">`);
html = html.replace(script[0], `<script src="${js}"></script>`);
const head = [
  `<meta http-equiv="Content-Security-Policy" content="${CSP}">`,
  '<meta name="tensorfold-showcase" content="recordings/index.json">',
  '<meta name="description" content="Recorded runs of TensorFold-Swarm, a distributed Mixture-of-Experts inference engine: each token\'s experts light up on the machine that ran them.">',
  '<link rel="icon" href="data:,">',
].join('\n');
const viewport = /<meta name="viewport"[^>]*>/;
if (!viewport.test(html)) {
  fail('live.html has no viewport meta to put the showcase settings after');
}
html = html.replace(viewport, (m) => `${m}\n${head}`);
html = html.replace(/<title>[^<]*<\/title>/, '<title>TensorFold-Swarm: recorded runs</title>');
if (/<(script|style)>/.test(html)) {
  fail('an inline script or style is left in the page, which its Content-Security-Policy would block');
}
writeFileSync(join(dist, 'index.html'), html);
console.log(`showcase: index.html, ${css}, ${js}`);
