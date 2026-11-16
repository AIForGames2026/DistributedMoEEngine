// Fails the Pages deploy (TASK-18) if the built site holds an executable or installer, a page
// without a Content-Security-Policy, or a main page heavier than 5 MB to load.
//   node scripts/check-bundle.mjs [dist]
//   node scripts/check-bundle.mjs --self-test
import { mkdtempSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync, mkdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export const INITIAL_LOAD_LIMIT = 5_000_000;

const BANNED_EXT = [
  '.exe', '.dmg', '.pkg', '.mpkg', '.msi', '.msix', '.appx', '.app', '.deb', '.rpm', '.apk', '.jar', '.appimage',
  '.sh', '.bash', '.zsh', '.command', '.run', '.bin', '.bat', '.cmd', '.ps1', '.vbs', '.scr', '.com', '.dll', '.so', '.dylib',
];

// Executables and installers by their first bytes, so a renamed one is caught too.
const MAGIC = [
  { name: 'a Windows executable', bytes: [0x4d, 0x5a] },
  { name: 'an ELF executable', bytes: [0x7f, 0x45, 0x4c, 0x46] },
  { name: 'a Mach-O executable', bytes: [0xcf, 0xfa, 0xed, 0xfe] },
  { name: 'a Mach-O executable', bytes: [0xce, 0xfa, 0xed, 0xfe] },
  { name: 'a Mach-O executable', bytes: [0xfe, 0xed, 0xfa, 0xcf] },
  { name: 'a Mach-O universal binary', bytes: [0xca, 0xfe, 0xba, 0xbe] },
  { name: 'a Windows installer (OLE)', bytes: [0xd0, 0xcf, 0x11, 0xe0, 0xa1, 0xb1, 0x1a, 0xe1] },
  { name: 'a macOS installer package (xar)', bytes: [0x78, 0x61, 0x72, 0x21] },
  { name: 'a script with a #! line', bytes: [0x23, 0x21] },
];

function walk(dir, out = []) {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) {
      if (name.toLowerCase().endsWith('.app')) {
        out.push(p);
      }
      walk(p, out);
    } else {
      out.push(p);
    }
  }
  return out;
}

function startsWith(buf, bytes) {
  return buf.length >= bytes.length && bytes.every((b, i) => buf[i] === b);
}

// The main page and what it fetches before anything plays: its scripts and styles, the list of recordings, and the first recording.
export function initialLoad(dist) {
  const parts = [];
  const add = (rel) => {
    const clean = rel.split(/[?#]/)[0].replace(/^\.\//, '');
    parts.push({ file: clean, bytes: statSync(join(dist, clean)).size });
  };
  add('index.html');
  const html = readFileSync(join(dist, 'index.html'), 'utf8');
  for (const m of html.matchAll(/<script[^>]*\ssrc="([^"]+)"/g)) {
    add(m[1]);
  }
  for (const m of html.matchAll(/<link[^>]*rel="(?:stylesheet|modulepreload)"[^>]*href="([^"]+)"/g)) {
    add(m[1]);
  }
  const manifest = html.match(/<meta name="tensorfold-showcase" content="([^"]+)"/);
  if (manifest) {
    add(manifest[1]);
    const list = JSON.parse(readFileSync(join(dist, manifest[1]), 'utf8')).recordings;
    add(list[0].file);
  }
  return { parts, total: parts.reduce((s, p) => s + p.bytes, 0) };
}

export function checkBundle(dist) {
  const problems = [];
  const files = walk(dist);
  for (const p of files) {
    const rel = relative(dist, p).replaceAll('\\', '/');
    const lower = rel.toLowerCase();
    const ext = BANNED_EXT.find((e) => lower.endsWith(e));
    if (ext) {
      problems.push(`${rel}: ${ext} files may not be published`);
      continue;
    }
    if (statSync(p).isDirectory()) {
      continue;
    }
    const buf = readFileSync(p);
    const head = buf.subarray(0, 16);
    const magic = MAGIC.find((m) => startsWith(head, m.bytes));
    if (magic) {
      problems.push(`${rel}: looks like ${magic.name}`);
    }
    if (buf.length >= 512 && buf.subarray(buf.length - 512, buf.length - 508).toString('latin1') === 'koly') {
      problems.push(`${rel}: looks like a macOS disk image`);
    }
    if (lower.endsWith('.html') && !/<meta http-equiv="Content-Security-Policy" content="[^"]+"/i.test(buf.toString('utf8'))) {
      problems.push(`${rel}: has no Content-Security-Policy`);
    }
  }
  let load = null;
  try {
    load = initialLoad(dist);
    if (load.total >= INITIAL_LOAD_LIMIT) {
      problems.push(`the main page loads ${(load.total / 1e6).toFixed(2)} MB before playing, over the 5 MB limit`);
    }
  } catch (e) {
    problems.push(`could not measure the main page's initial load: ${e.message}`);
  }
  return { problems, files: files.length, load };
}

function report(dist) {
  const { problems, files, load } = checkBundle(dist);
  if (load) {
    for (const p of load.parts) {
      console.log(`  ${p.file}: ${(p.bytes / 1024).toFixed(1)} KB`);
    }
    console.log(`initial load: ${(load.total / 1e6).toFixed(3)} MB (limit 5 MB); ${files} files in the bundle`);
  }
  for (const p of problems) {
    console.error(`bundle check: ${p}`);
  }
  if (problems.length) {
    process.exit(1);
  }
  console.log('bundle check: ok');
}

// Builds small sites in a temporary folder and checks that each planted problem is caught.
function selfTest() {
  const csp = '<meta http-equiv="Content-Security-Policy" content="default-src \'self\'">';
  const page = `<!doctype html><html><head>${csp}<meta name="tensorfold-showcase" content="recordings/index.json"></head><body><script src="a.js"></script></body></html>`;
  const make = (extra) => {
    const d = mkdtempSync(join(tmpdir(), 'tf-bundle-'));
    mkdirSync(join(d, 'recordings'));
    writeFileSync(join(d, 'index.html'), page);
    writeFileSync(join(d, 'a.js'), 'console.log(1);');
    writeFileSync(join(d, 'recordings', 'index.json'), JSON.stringify({ recordings: [{ file: 'recordings/r.jsonl.gz' }] }));
    writeFileSync(join(d, 'recordings', 'r.jsonl.gz'), Buffer.from([0x1f, 0x8b, 8, 0]));
    extra(d);
    return d;
  };
  const cases = [
    ['a clean site', () => {}, null],
    ['an .exe', (d) => writeFileSync(join(d, 'setup.exe'), 'x'), 'may not be published'],
    ['a .dmg', (d) => writeFileSync(join(d, 'TensorFold.dmg'), 'x'), 'may not be published'],
    ['a .pkg', (d) => writeFileSync(join(d, 'a.pkg'), 'x'), 'may not be published'],
    ['an .msi', (d) => writeFileSync(join(d, 'a.MSI'), 'x'), 'may not be published'],
    ['a shell installer', (d) => writeFileSync(join(d, 'install.sh'), 'echo'), 'may not be published'],
    ['a renamed Windows executable', (d) => writeFileSync(join(d, 'data.json'), Buffer.from('MZ\x90\x00')), 'Windows executable'],
    ['a renamed ELF', (d) => writeFileSync(join(d, 'x.wasm'), Buffer.from([0x7f, 0x45, 0x4c, 0x46, 2])), 'ELF'],
    ['a script renamed to text', (d) => writeFileSync(join(d, 'notes.txt'), '#!/bin/sh\ncurl x | sh\n'), '#! line'],
    ['a page without a CSP', (d) => writeFileSync(join(d, 'other.html'), '<!doctype html><p>hi'), 'Content-Security-Policy'],
    ['an initial load over 5 MB', (d) => writeFileSync(join(d, 'a.js'), Buffer.alloc(INITIAL_LOAD_LIMIT)), 'over the 5 MB limit'],
    ['a large file that is not loaded at first', (d) => writeFileSync(join(d, 'recordings', 'big.jsonl.gz'), Buffer.alloc(INITIAL_LOAD_LIMIT + 1)), null],
  ];
  let bad = 0;
  for (const [name, plant, want] of cases) {
    const d = make(plant);
    const { problems } = checkBundle(d);
    rmSync(d, { recursive: true, force: true });
    const ok = want === null ? problems.length === 0 : problems.some((p) => p.includes(want));
    console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${name}${problems.length ? `: ${problems.join('; ')}` : ''}`);
    if (!ok) {
      bad++;
    }
  }
  if (bad) {
    console.error(`bundle check self-test: ${bad} case(s) failed`);
    process.exit(1);
  }
  console.log(`bundle check self-test: ok (${cases.length} cases)`);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv[2] === '--self-test') {
    selfTest();
  } else {
    report(resolve(process.argv[2] || join(dirname(fileURLToPath(import.meta.url)), '..', 'dist')));
  }
}
