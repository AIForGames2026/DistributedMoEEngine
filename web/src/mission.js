// The guided mission: the inspector's first-run path for someone who has never
// heard of a Mixture-of-Experts model. Four simulated experiments end in a
// challenge and a report card, with no developer help. Only a line of text the
// tester copies by hand leaves the page; nothing is sent anywhere.
import { MISSION } from './presets.js';

const BENCH = 'https://github.com/AIForGames2026/DistributedMoEEngine/blob/main/bench';
const tps = (r) => `${r.tps.toFixed(r.tps < 100 ? 1 : 0)} tok/s`;
const narrow = () => window.matchMedia('(max-width: 760px)').matches; // matches style.css

const STEPS = [
  {
    title: 'Experiment 1 of 4: watch it work',
    body: `<p>This is the setup the project is aiming for: three machines joined by a very fast cable. Press <b>Run</b> and watch a batch of 8 words pass through all 16 layers.</p>
      <ul class="key">
        <li><i style="background:#7dd3fc"></i>Blue cube: the expert was already in memory</li>
        <li><i style="background:#4ade80"></i>Green cube: it was loaded early, because the system predicted it</li>
        <li><i style="background:#f87171"></i>Red cube: nobody saw it coming, so the expert had to wait for the disk</li>
      </ul>`,
    enter: { preset: 'sparks', lookahead: true },
    need: () => null,
    result: (r) => `<p>Finished at <b>${tps(r)}</b> (tokens are roughly words) with <b>${r.misses}</b> red cubes. Remember those numbers.</p>`,
  },
  {
    title: 'Experiment 2 of 4: remove the prediction',
    body: '<p><b>Lookahead prefetch</b> guesses which experts come next and loads them early. Turn it <b>off</b> in the left panel, then press <b>Run</b> and watch the colors.</p>',
    spot: ['lookahead'],
    need: (c) => (c.lookahead ? 'First uncheck Lookahead prefetch in the left panel.' : null),
    result: (r, runs) => `<p>Now <b>${tps(r)}</b> with <b>${r.misses}</b> red cubes. Before: ${tps(runs[0])} with ${runs[0].misses}.</p>`,
    question: {
      ask: 'Why did it get slower?',
      options: [
        ['The cable between the machines got slower', false],
        ['More experts had to wait for the disk', true],
        ['The model now has more experts to run', false],
      ],
      explain: 'Right: the machines and the model did not change. Without a prediction, an expert that is not in memory holds up its layer while it is read from disk.',
      retry: 'Not quite. Only one setting changed. Compare the colors of the cubes in the two runs.',
    },
  },
  {
    title: 'Experiment 3 of 4: the real situation',
    body: `<p>Only one of the three machines is free right now, so the model has to be split between it and a separate desktop. Their traffic goes through a relay on the internet.</p>
      <p>Turn Lookahead prefetch back <b>on</b>, set <b>Cluster</b> to "${MISSION.relayLabel}", then press <b>Run</b>.</p>`,
    spot: ['preset', 'lookahead'],
    need: (c) => {
      if (c.preset !== 'today') return `Set Cluster to "${MISSION.relayLabel}".`;
      if (!c.lookahead) return 'Turn Lookahead prefetch back on.';
      return null;
    },
    result: (r, runs) => `<p>Only <b>${tps(r)}</b>, against ${tps(runs[0])} in the first run. Prediction is on, and there are few red cubes.</p>`,
    question: {
      ask: 'What is slowing it down now?',
      options: [
        ['The connection between the two machines', true],
        ['Reading experts from disk', false],
        ['The experts doing their math', false],
      ],
      explain: 'Right: each trip over the relay takes about 16.5 ms. A disk read takes about 0.5 ms, and an expert runs in about 0.013 ms.',
      retry: 'Not quite. The labels on the line between the machines and the time per layer are good clues.',
    },
  },
  {
    title: 'Final challenge: fix the swarm',
    body: `<p>Get these two machines to <b>${MISSION.targetTps} tok/s</b> or more. Use the left panel and run as many times as you like. The other two machines are still busy, so you cannot use them.</p>`,
    spot: ['preset', 'lookahead'],
    challenge: true,
    need: (c) => (c.preset === 'sparks' ? 'Those three machines are busy today. Pick a two-machine Cluster.' : null),
    result: (r, runs, tries) => {
      if (r.tps >= MISSION.targetTps) return `<p><b>${tps(r)}: target reached.</b></p>`;
      const hint = tries >= 2
        ? ' Hint: the Cluster menu has another way to connect the same two machines.'
        : ' Prediction cannot help much when something else is this slow.';
      return `<p><b>${tps(r)}</b>, and the target is ${MISSION.targetTps}.${hint}</p>`;
    },
  },
];

export function createMission(api) {
  const card = document.createElement('div');
  card.className = 'mission';
  card.setAttribute('role', 'region');
  card.setAttribute('aria-label', 'Guided mission');
  card.setAttribute('aria-live', 'polite');
  card.hidden = true;
  document.getElementById('viewport').append(card);

  let step = 0;
  let phase = 'intro'; // intro | ready | running | asking | answered | done
  let note = '';
  let runs = [];
  let wrong = 0;
  let tries = 0;
  let startedAt = 0;
  let last = null;

  const clearSpots = () => document.querySelectorAll('.spot').forEach((e) => e.classList.remove('spot'));
  function spotlight(ids = []) {
    clearSpots();
    for (const id of ids) (document.getElementById(id).closest('label.check') ?? document.getElementById(id)).classList.add('spot');
    if (ids.length && narrow()) document.getElementById(ids[0]).scrollIntoView({ block: 'center' });
  }

  function close() {
    card.hidden = true;
    clearSpots();
  }

  function exit() {
    close();
    api.exit();
  }

  function restart() {
    runs = [];
    wrong = 0;
    tries = 0;
    showIntro();
  }

  function footer() {
    return '<button type="button" class="link" data-act="exit">Exit and explore freely</button>';
  }

  function render(html) {
    card.hidden = false;
    card.innerHTML = html;
    card.scrollTop = 0;
    card.querySelector('[data-act="exit"]')?.addEventListener('click', exit);
    card.querySelector('[data-act="restart"]')?.addEventListener('click', restart);
    card.querySelector('.primary')?.focus({ preventScroll: true });
  }

  function showIntro() {
    clearSpots();
    phase = 'intro';
    api.enter();
    render(`<h2>Find out what slows down a giant AI model, then fix it</h2>
      <p>The largest AI models do not fit on one machine, so this project splits one across several.</p>
      <p>For every word, each of the model's 16 layers picks 8 of its 64 specialist parts, called <b>experts</b>. Those experts live on different machines, so the word's data has to fly to them and back. You will see that happen.</p>
      <p>This is a <b>simulation</b> driven by speeds measured on the real hardware. It takes about 3 minutes, and there is nothing to install.</p>
      <div class="row"><button type="button" class="primary" data-act="start">Start</button></div>
      ${footer()}`);
    card.querySelector('[data-act="start"]').addEventListener('click', () => {
      startedAt = performance.now();
      enterStep(0);
    });
  }

  function enterStep(i) {
    step = i;
    phase = 'ready';
    note = '';
    last = null;
    const s = STEPS[i];
    if (s.enter) api.setConfig(s.enter);
    spotlight(s.spot);
    renderStep();
  }

  function renderStep() {
    const s = STEPS[step];
    const blocked = phase === 'ready' ? s.need(api.config()) : null;
    let bottom = '';
    if (phase === 'ready') {
      bottom = `<p class="need">${blocked ?? '&nbsp;'}</p>
        <div class="row"><button type="button" class="primary" data-act="run" ${blocked ? 'disabled' : ''}>Run</button></div>`;
    } else if (phase === 'running') {
      bottom = '<p class="need">Running. Watch the cubes and the packets flying between machines.</p>';
    } else if (phase === 'asking') {
      const q = s.question;
      bottom = `<p class="ask">${q.ask}</p><div class="answers">${q.options.map(([text], n) => `<button type="button" data-a="${n}">${text}</button>`).join('')}</div><p class="need">${note || '&nbsp;'}</p>`;
    } else {
      const label = step === STEPS.length - 1 ? 'See your results' : 'Next';
      bottom = `${note ? `<p class="good">${note}</p>` : ''}<div class="row"><button type="button" class="primary" data-act="next">${label}</button></div>`;
    }
    const resultHtml = last && phase !== 'running' ? s.result(last, runs, tries) : '';
    render(`<h2>${s.title}</h2>${phase === 'ready' ? s.body : ''}${resultHtml}${bottom}${footer()}`);
    card.querySelector('[data-act="run"]')?.addEventListener('click', tryRun);
    card.querySelector('[data-act="next"]')?.addEventListener('click', () => {
      if (step === STEPS.length - 1) showReport();
      else enterStep(step + 1);
    });
    card.querySelectorAll('[data-a]').forEach((b) => b.addEventListener('click', () => answer(Number(b.dataset.a))));
  }

  function answer(n) {
    const q = STEPS[step].question;
    if (q.options[n][1]) {
      phase = 'answered';
      note = q.explain;
    } else {
      wrong++;
      note = q.retry;
    }
    renderStep();
  }

  function tryRun() {
    if (phase !== 'ready' || STEPS[step].need(api.config())) return;
    phase = 'running';
    last = null;
    clearSpots();
    renderStep();
    if (narrow()) window.scrollTo(0, 0);
    api.run();
  }

  // Called by the player when a mission run reaches the end of its trace.
  function onRunEnd(r) {
    if (card.hidden || phase !== 'running') return;
    const s = STEPS[step];
    runs.push(r);
    last = r;
    if (s.challenge) {
      tries++;
      if (r.tps >= MISSION.targetTps) {
        phase = 'answered';
      } else {
        phase = 'ready';
        spotlight(s.spot);
      }
    } else {
      phase = s.question ? 'asking' : 'answered';
    }
    note = '';
    renderStep();
  }

  // Called when a control changes. A run in progress starts over from Ready.
  function onConfigChanged() {
    if (card.hidden) return;
    if (phase === 'running') {
      phase = 'ready';
      spotlight(STEPS[step].spot);
    }
    if (phase === 'ready') renderStep();
  }

  function showReport() {
    const seconds = Math.round((performance.now() - startedAt) / 1000);
    const final = runs[runs.length - 1];
    const line = `TFS-alpha time=${seconds}s wrong=${wrong} challenge_tries=${tries} runs=${runs.length} final=${Math.round(final.tps)}tok/s`;
    phase = 'done';
    clearSpots();
    render(`<h2>Swarm fixed: ${tps(final)}</h2>
      <p>You split a model across two machines, found the bottleneck, and fixed it. It took you ${Math.floor(seconds / 60)} min ${seconds % 60} s, with ${wrong} wrong ${wrong === 1 ? 'answer' : 'answers'} and ${tries} ${tries === 1 ? 'try' : 'tries'} at the challenge.</p>
      <table class="runs"><thead><tr><th>Run</th><th>Speed</th><th>Red cubes</th></tr></thead><tbody>
      ${runs.map((r) => `<tr><td>${r.label}${r.lookahead ? '' : ', no prediction'}</td><td>${tps(r)}</td><td>${r.misses}</td></tr>`).join('')}
      </tbody></table>
      <h3>What the real engine did</h3>
      <p>The fix you found is real: this week the actual engine moved from the relay to a direct connection. The same model, split across the same two machines on their CPUs, took a 17-token prompt from <a href="${BENCH}/olmoe/2026-11-05.md" target="_blank" rel="noopener">24.3 s</a> to <a href="${BENCH}/transport/2026-11-05-direct-path.md" target="_blank" rel="noopener">3.9 s</a>. The simulation assumes both machines compute at full speed, so its numbers are higher than that CPU run.</p>
      <p class="ask">Paste this line into your feedback form:</p>
      <input class="code" type="text" readonly value="${line}" aria-label="Result line to copy" />
      <div class="row"><button type="button" class="primary" data-act="copy">Copy</button><button type="button" data-act="restart">Play again</button></div>
      ${footer()}`);
    const code = card.querySelector('.code');
    const copy = card.querySelector('[data-act="copy"]');
    copy.addEventListener('click', async () => {
      code.select();
      try {
        await navigator.clipboard.writeText(line);
        copy.textContent = 'Copied';
      } catch {
        copy.textContent = 'Press Ctrl+C to copy';
      }
    });
    window.__mission = { line, seconds, wrong, tries, runs: runs.length };
  }

  return { open: restart, tryRun, onRunEnd, onConfigChanged, isOpen: () => !card.hidden };
}
