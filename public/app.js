import { cloneConfig } from '/src/config.js';
import { Match } from '/src/engine.js';
import { RULESETS } from '/src/rules/index.js';
import { JudgeClient } from '/src/clip.js';
import { Camera } from '/src/camera.js';
import { StrikeCoordinator, EventDeduper } from '/src/multicam.js';
import { POSE_CONNECTIONS } from '/src/util.js';
import { swatch } from '/src/appearance.js';
import { buildReportHtml } from '/src/report.js';
import { decideThrow } from '/src/throwgrade.js';

const $ = (s) => document.querySelector(s);
const cfg = cloneConfig();
const canvas = $('#view');
const ctx = canvas.getContext('2d');

let sport = 'boxing';
let match = null;
const judgeClient = new JudgeClient(); // CLIP backend
let lastVerdict = null;
let markers = [];
let lastBoardDraw = 0;
let flashSide = null;
let bannerTimer = null;

/* ----------------------------------------------------------------- UI build */

function buildSportTabs() {
  const nav = $('#sportTabs');
  nav.innerHTML = '';
  for (const r of Object.values(RULESETS)) {
    const b = document.createElement('button');
    b.textContent = `${r.icon} ${r.name}`;
    b.dataset.sport = r.id;
    b.onclick = () => setSport(r.id);
    nav.appendChild(b);
  }
}

function setSport(id, variant) {
  sport = id;
  const rules = RULESETS[id];
  match = new Match(rules, variant);
  match.on(onMatchEvent);
  for (const b of document.querySelectorAll('#sportTabs button')) b.classList.toggle('on', b.dataset.sport === id);
  const sel = $('#variant');
  sel.innerHTML = '';
  for (const [k, v] of Object.entries(rules.variants)) sel.add(new Option(v.label, k));
  sel.value = match.variantId;
  markers = [];
  setRecoverDefault();
  buildManual();
  buildRules();
  renderLog();
  renderBoard();
  $('#status').textContent = 'Press Start (or run a source)';
}

function buildManual() {
  const r = RULESETS[sport];
  const el = $('#tab-manual');
  const row = (label, desc, auto, handlerA, handlerB) => {
    const d = document.createElement('div');
    d.className = 'ovr';
    d.innerHTML = `<div>${label}${auto ? '<span class="au" title="Also detected automatically by the AI">AI</span>' : ''}<div class="d">${desc || ''}</div></div>`;
    const a = Object.assign(document.createElement('button'), { textContent: 'RED', className: 'ra', onclick: handlerA });
    const b = Object.assign(document.createElement('button'), { textContent: 'BLUE', className: 'rb', onclick: handlerB });
    d.append(a, b);
    return d;
  };
  const act = (ev) => () => {
    if (match.status === 'ready') match.start();
    match.handle(ev);
  };
  el.innerHTML = '<div id="undoBar" class="undobar"></div><h4>Scores / actions</h4>';
  for (const s of r.manualScores) el.appendChild(row(s.label, '', false, act({ type: 'score', by: 'a', kind: s.kind }), act({ type: 'score', by: 'b', kind: s.kind })));
  el.insertAdjacentHTML('beforeend', '<h4>Fouls (button = side that committed it)</h4>');
  for (const [id, f] of Object.entries(r.fouls)) {
    el.appendChild(row(f.label + (f.direct ? ' ⚠ direct' : ''), f.desc, f.auto, act({ type: 'foul', by: 'a', foul: id }), act({ type: 'foul', by: 'b', foul: id })));
  }
}

/** round-by-round 10-point-must card (boxing, kickboxing): points, with the strikes / knockdowns / deductions behind them */
function scorecardHtml() {
  const cards = match.roundCards;
  if (!cards?.length) return '';
  const cell = (c) => `<b>${c.pts}</b><small>${c.strikes} strike${c.strikes === 1 ? '' : 's'}${c.kd ? ` · KD ${c.kd}` : ''}${c.ded ? ` · −${c.ded}` : ''}</small>`;
  const tot = (side) => match.sides[side].score;
  return `<table class="scorecard"><thead><tr><th>Rd</th><th class="a">RED</th><th class="b">BLUE</th></tr></thead><tbody>${cards.map((r) => `<tr><td>${r.round}</td><td>${cell(r.a)}</td><td>${cell(r.b)}</td></tr>`).join('')}</tbody><tfoot><tr><td>Total</td><td class="a"><b>${tot('a')}</b></td><td class="b"><b>${tot('b')}</b></td></tr></tfoot></table>`;
}

/** "↩ Undo last penalty": lets the referee withdraw a wrong automatic call (e.g. a passivity shido, a foul from a mis-read clip) */
function undoHtml() {
  const p = match.lastPenalty?.();
  if (!p || match.status === 'ready') return '';
  const name = match.sides[p.side].name;
  return `<div class="undoline"><button data-undo title="Withdraw the most recent penalty: restores the score, the penalty count and, if it ended the bout, the bout">↩ Undo last penalty: ${escapeHtml(p.label.split(' (')[0])} → ${escapeHtml(p.action)} (${name})</button></div>`;
}

function renderUndoBar() {
  const bar = $('#undoBar');
  if (bar) bar.innerHTML = undoHtml() || '<span class="dim">No penalty to undo.</span>';
}

function buildRules() {
  const r = RULESETS[sport];
  const el = $('#tab-rules');
  let h = `<div class="rules"><h4>${r.icon} ${r.name} ruleset loaded</h4>`;
  for (const sec of r.summary) h += `<h5>${sec.title}</h5><ul>${sec.items.map((i) => `<li>${i}</li>`).join('')}</ul>`;
  h += `<div class="verify"><b>Verify before real use:</b><ul>${(r.verify || []).map((v) => `<li>${v}</li>`).join('')}</ul></div>`;
  h += `<h5>Sources</h5><ul>${r.sources.map((s) => `<li><a href="${s.url}" target="_blank" rel="noopener">${s.title}</a></li>`).join('')}</ul></div>`;
  el.innerHTML = h;
}

const TUNE = [
  ['Punch: min speed (m/s)', 'punch', 'minSpeed', 0.8, 5, 0.1],
  ['Punch: min extension (m)', 'punch', 'minExtension', 0.05, 0.4, 0.01],
  ['Kick: min speed (m/s)', 'kick', 'minSpeed', 1, 6, 0.1],
  ['Kick: min foot lift (torso)', 'kick', 'minLift', 0.2, 1.2, 0.05],
  ['Reap: min leg speed', 'reap', 'minSpeed', 1, 8, 0.1],
  ['Fall: min torso tilt (°)', 'fall', 'minPhi', 30, 80, 1],
];

function buildTune() {
  const el = $('#tab-tune');
  el.innerHTML = '<div class="tune"></div><div class="dbg" id="debugOut">waiting for frames…</div>';
  const box = el.firstChild;
  for (const [label, grp, key, lo, hi, step] of TUNE) {
    const l = document.createElement('label');
    l.innerHTML = `<span>${label}</span><input type="range" min="${lo}" max="${hi}" step="${step}" value="${cfg[grp][key]}"><span>${cfg[grp][key]}</span>`;
    const inp = l.querySelector('input'), val = l.lastElementChild;
    inp.oninput = () => {
      cfg[grp][key] = Number(inp.value);
      val.textContent = inp.value;
    };
    box.appendChild(l);
  }
}

/* ------------------------------------------------------------- match output */

function onMatchEvent(ev) {
  if (ev.kind === 'call') {
    renderLog();
    const e = ev.entry;
    if (ev.big) showBanner(ev.big, e, ev);
    if (ev.speak) speak(ev.speak);
    if (e.kind === 'score') flashSide = e.side;
    renderBoard();
  } else if (ev.kind === 'state') renderBoard();
  else if (ev.kind === 'tick') {
    const n = performance.now();
    if (n - lastBoardDraw > 200) renderBoard();
  }
}

function showBanner(text, entry, ev) {
  const b = $('#banner');
  const cls = entry.kind === 'result' ? 'result' : entry.kind === 'foul' ? 'foul' : entry.side || 'n';
  b.className = `banner show ${cls}`;
  const sub = entry.kind === 'result' || entry.kind === 'foul' || entry.kind === 'score' ? `<small>${escapeHtml(entry.text)}</small>` : '';
  b.innerHTML = `${escapeHtml(text)}${sub}`;
  clearTimeout(bannerTimer);
  bannerTimer = setTimeout(() => b.classList.remove('show'), entry.kind === 'result' ? 6000 : 2200);
}

function speak(text) {
  if (!$('#optVoice').checked || !('speechSynthesis' in window)) return;
  try {
    const u = new SpeechSynthesisUtterance(text.replace(/[–—]/g, ' '));
    u.rate = 1.05;
    if (!/^\d+$/.test(text)) speechSynthesis.cancel();
    speechSynthesis.speak(u);
  } catch { /* ignore */ }
}

const escapeHtml = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

function renderLog() {
  const ol = $('#log');
  ol.innerHTML = match.log
    .slice(0, 80)
    .map((l) => {
      const dot = l.side ? `<span class="dot ${l.side}"></span>` : '';
      return `<li class="${l.kind}"><span class="t">${l.round > 1 || RULESETS[sport].variants[match.variantId].rounds > 1 ? 'R' + l.round + ' ' : ''}${l.clock}</span><span class="x">${l.auto ? '<span class="ai">AI</span>' : l.kind === 'info' ? '' : '<span class="ref">REF</span>'}${dot}${escapeHtml(l.text)}</span></li>`;
    })
    .join('');
}

function statusText() {
  switch (match.status) {
    case 'ready': return 'READY';
    case 'running': return match.golden ? 'GOLDEN SCORE' : 'LIVE';
    case 'paused': return 'BREAK';
    case 'rest': return 'REST';
    case 'counting': return `COUNT ${Math.floor(match.count?.n ?? 0)}`;
    case 'ended': return 'FINAL';
  }
}

function sideHtml(id) {
  const s = match.sides[id];
  const chips = [];
  let big = '', sub = '';
  if (sport === 'boxing' || sport === 'kickboxing') {
    big = s.punchesTotal;
    sub = `clean ${sport === 'boxing' ? 'punches' : 'strikes'} · ${s.punches} this round · card ${s.score} pts`;
    if (s.kdTotal) chips.push(`<span class="chip bad">KD ×${s.kdTotal}</span>`);
    if (s.deductTotal) chips.push(`<span class="chip warn">−${s.deductTotal} pt</span>`);
  } else if (sport === 'judo') {
    big = s.wazaari;
    sub = `waza-ari · yuko ${s.yuko}`;
    chips.push(`<span class="chip ${s.shido ? 'warn' : ''}">shido ${'●'.repeat(s.shido)}${'○'.repeat(Math.max(0, 3 - s.shido))}</span>`);
  }
  for (const p of s.penalties) chips.push(`<span class="chip ${p.action === 'dq' || p.action === 'hm' ? 'bad' : 'warn'}" title="${escapeHtml(p.label)}">${escapeHtml(p.label.split(' (')[0])} → ${p.action}</span>`);
  const win = match.status === 'ended' && match.winner === id;
  return `<div class="side ${id} ${win ? 'win' : ''} ${flashSide === id ? 'flash' : ''}" id="side-${id}"><div class="nm">${s.name}${win ? ' 🏆' : ''}</div><div class="big">${big}</div><div class="sub">${sub}</div><div class="chips">${chips.join('')}</div></div>`;
}

function renderBoard() {
  lastBoardDraw = performance.now();
  const v = match.cfg;
  const roundTxt = v.rounds > 1 ? `Round ${match.round}/${v.rounds}` : match.golden ? 'Golden score' : v.label.split(' (')[0];
  const res = match.status === 'ended' ? `<div class="result show">${match.winner ? `${match.sides[match.winner].name} wins` : 'No winner'} by ${escapeHtml(match.method)}</div>` : '<div class="result"></div>';
  $('#board').innerHTML = `
    <div class="sb-top"><div class="sb-clock">${match.golden ? '+' : ''}${match.clockText()}</div>
      <div class="sb-meta"><b>${statusText()}</b><br>${roundTxt}</div></div>
    <div class="sides">${sideHtml('a')}${sideHtml('b')}</div>${res}${scorecardHtml()}${undoHtml()}`;
  flashSide = null;
  renderUndoBar();
}

// one handler for both undo buttons (the scoreboard one and the override-panel one); they are re-created on every redraw
document.addEventListener('click', (ev) => {
  if (!ev.target.closest?.('[data-undo]')) return;
  if (match.undoLastPenalty()) renderLog();
});

/* ------------------------------------------------------------------ cameras */

const MAX_CAMS = 3;
const CAM_STORE = 'cameraSetup';
let cams = []; // live Camera objects (one per camera row), or a single video-file camera
let primary = 0; // which camera the big view shows
let mode = null; // 'camera' | 'video' | null
const coordinator = new StrikeCoordinator(); // merges the same strike seen by several cameras
const deduper = new EventDeduper(); // counts a throw / fall once even if several cameras report it
let devices = []; // [{deviceId, label}] of the connected cameras
let setup = (() => {
  try {
    const s = JSON.parse(localStorage.getItem(CAM_STORE));
    return Array.isArray(s) && s.length ? s.slice(0, MAX_CAMS) : null;
  } catch { return null; }
})() || [{ id: '', flip: false }]; // one row per camera angle: {id: deviceId, flip: RED is on the right in this view}

const saveSetup = () => { try { localStorage.setItem(CAM_STORE, JSON.stringify(setup)); } catch { /* ignore */ } };
const clockNow = () => (mode === 'video' && cams[0] ? cams[0].lastMedia : performance.now()); // the clock the frames are stamped with

let recoverTouched = false; // the user ticked / unticked 'Find 2nd fighter' themselves: stop applying the per-source default
// Judo needs both skeletons (throws, falls, the passivity timer) and its slower leg actions tolerate a lower frame rate; boxing and
// kickboxing punches need 10+ fps, so a live camera there stays fast. Uploaded video always searches (it slows its playback to keep up).
function setRecoverDefault(srcMode = mode) {
  if (recoverTouched) return;
  const on = srcMode === 'video' || sport === 'judo';
  $('#optRecover').checked = on;
  cams.forEach((c) => c.pose && (c.pose.recover = on));
}

async function newPose() {
  const { createPose } = await import('/src/pose.js'); // each camera needs its own tracker instance
  return createPose((t) => ($('#status').textContent = t), { recover: $('#optRecover').checked });
}

function stopAll() {
  for (const c of cams) c.stop();
  cams = [];
  primary = 0;
  mode = null;
  coordinator.reset();
  deduper.reset();
  renderThumbs();
}

function beginMatch() {
  for (const c of cams) {
    c.pipeline.reset();
    c.clipBuf.reset();
  }
  coordinator.reset();
  deduper.reset();
  lastVerdict = null;
  markers = [];
  match.reset();
  match.start();
}

let camNote = ''; // last problem found while detecting cameras (shown under the rows)
let permState = 'unknown';

async function refreshCameras() {
  try {
    devices = (await navigator.mediaDevices.enumerateDevices())
      .filter((d) => d.kind === 'videoinput')
      // browsers hide camera names until this page has been allowed to use a camera: then label is ''
      .map((d, i) => ({ deviceId: d.deviceId, rawLabel: d.label, label: d.label || `Camera ${i + 1} (name hidden until allowed)` }));
  } catch (e) {
    console.warn('enumerateDevices failed', e);
    devices = [];
  }
  try { permState = (await navigator.permissions.query({ name: 'camera' })).state; } catch { permState = 'unknown'; }
  renderCamRows();
  renderCamInfo();
}

function browserName() {
  const u = navigator.userAgent;
  if (/Claude\//.test(u)) return 'Claude app preview';
  if (navigator.brave) return 'Brave';
  if (/Edg\//.test(u)) return 'Edge';
  if (/Firefox\//.test(u)) return 'Firefox';
  if (/Chrome\//.test(u)) return 'Chrome';
  return 'browser';
}

function renderCamInfo() {
  const hidden = devices.length > 0 && devices.every((d) => !d.rawLabel);
  const embedded = /Claude\//.test(navigator.userAgent);
  const bits = [`in ${browserName()}`, `${devices.length} camera${devices.length === 1 ? '' : 's'} reported`, `permission: ${permState}`];
  let hint = '';
  if (embedded) hint = `You are viewing this inside the Claude app's built-in preview, which cannot use cameras. Open ${location.origin} in Brave, Chrome or Edge.`;
  else if (permState === 'denied') hint = 'Camera access is BLOCKED for this site. Click the lock / camera icon at the left of the address bar, set Camera to Allow, then reload the page.';
  else if (!devices.length) hint = 'No cameras reported. Plug one in; if it is plugged in, allow access (press Detect cameras) or check Windows Settings → Privacy & security → Camera.';
  else if (hidden) hint = 'Names are hidden until you allow camera access: press Detect cameras.';
  $('#camInfo').innerHTML = `<span>${bits.join(' · ')}</span>${hint ? `<br><b>${escapeHtml(hint)}</b>` : ''}${camNote ? `<br><b class="warn">${escapeHtml(camNote)}</b>` : ''}<br><a href="/camera-test.html" target="_blank" rel="noopener">Camera not showing up? Run the camera test page</a>`;
}

/** Ask the browser for camera access once (then release the camera at once) so every camera's real name becomes visible. */
async function detectCameras() {
  camNote = '';
  try {
    const s = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    s.getTracks().forEach((t) => t.stop());
  } catch (e) {
    camNote = e.name === 'NotReadableError'
      ? 'A camera is in use by another tab or app (e.g. an older browser tab of this app, Teams, Zoom). Close it, then press Detect cameras again.'
      : e.name === 'NotAllowedError'
        ? 'Camera access was not allowed. Click the camera/lock icon in the address bar and allow it, then press Detect cameras again.'
        : `Could not open a camera: ${e.name || e.message}`;
  }
  await refreshCameras();
}

function renderCamRows() {
  const box = $('#camRows');
  box.innerHTML = '';
  setup.forEach((row, i) => {
    const d = document.createElement('div');
    d.className = 'camrow';
    const opts = `<option value="">${i === 0 ? 'System default camera' : 'Choose a camera…'}</option>` + devices.map((x) => `<option value="${escapeHtml(x.deviceId)}">${escapeHtml(x.label)}</option>`).join('');
    d.innerHTML = `<b class="camname">CAM ${i + 1}</b>
      <select class="camsel" data-i="${i}" title="Which camera is this angle">${opts}</select>
      <label class="camflip" title="Tick if RED is on the right (or far) side in this camera's view"><input type="checkbox" class="camflipbox" data-i="${i}" ${row.flip ? 'checked' : ''}> RED on the right (only used when clothes look alike)</label>
      ${i > 0 ? `<button class="camdel" data-i="${i}" title="Remove this camera angle">✕</button>` : ''}`;
    box.appendChild(d);
    d.querySelector('select').value = devices.some((x) => x.deviceId === row.id) ? row.id : '';
  });
  $('#btnAddCam').disabled = setup.length >= MAX_CAMS;
}

function renderThumbs() {
  const box = $('#camThumbs');
  box.innerHTML = '';
  box.hidden = cams.length < 2;
  cams.forEach((c, i) => {
    const d = document.createElement('div');
    d.className = 'thumb' + (i === primary ? ' on' : '');
    d.dataset.i = i;
    d.innerHTML = `<canvas width="320" height="${Math.round(320 / c.aspect)}"></canvas><span>${c.id.toUpperCase()} · ${escapeHtml(c.label)} <em class="idbadge"></em></span>`;
    c.badge = d.querySelector('.idbadge');
    box.appendChild(d);
    c.thumb = d.querySelector('canvas');
  });
}

function setMainAspect() {
  const a = cams[primary]?.aspect || 16 / 9;
  canvas.width = 960;
  canvas.height = Math.round(960 / a);
}

async function startCameras() {
  stopAll();
  const made = [];
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw Object.assign(new Error('Camera API unavailable. Open http://localhost:5173 in Chrome, Edge, Brave or Firefox (not an embedded viewer).'), { name: 'NoAPI' });
    await refreshCameras();
    // camera ids change when the browser resets permissions or restarts: a saved id that no longer exists would fail with OverconstrainedError
    const known = new Set(devices.map((d) => d.deviceId).filter(Boolean));
    if (known.size) setup.forEach((r) => { if (r.id && !known.has(r.id)) r.id = ''; });
    saveSetup();
    renderCamRows();
    const chosen = setup.map((r) => r.id).filter(Boolean);
    if (new Set(chosen).size !== chosen.length) throw Object.assign(new Error('Two camera rows use the same camera. Pick a different camera for each angle.'), { name: 'Duplicate' });
    const blank = setup.findIndex((r, i) => i > 0 && !r.id);
    if (blank > 0) throw Object.assign(new Error(`Choose a camera for CAM ${blank + 1} (or remove the row).`), { name: 'Duplicate' });
    for (let i = 0; i < setup.length; i++) {
      $('#status').textContent = `Opening camera ${i + 1} of ${setup.length}…`;
      const cam = new Camera(i, cfg);
      cam.pipeline.tracker.flip = !!setup[i].flip;
      cam.pipeline.tracker.appearance = $('#optColour').checked;
      if (i > 0) cam.pipeline.tracker.globalRef = made[0].pipeline.tracker.ref; // later cameras take RED / BLUE from camera 1's clothing colours
      made.push(cam); // so a failure below releases it
      try { await cam.open(setup[i].id); } catch (e) { e.camIndex = i; throw e; }
      if (made.some((o) => o !== cam && o.deviceId && o.deviceId === cam.deviceId)) throw Object.assign(new Error(`CAM ${i + 1} opened the same camera as another row. Pick a different camera.`), { name: 'Duplicate' });
      setRecoverDefault('camera'); // live camera: on for judo, off for boxing / kickboxing (the search costs frame rate, and punches need >= 10 fps)
      cam.pose = await newPose();
    }
    cams = made;
    primary = 0;
    mode = 'camera';
    coordinator.waitMs = cams.length > 1 ? 700 : 0; // with several cameras, give the others time to report the same strike
    cams.forEach((c, i) => (setup[i].id = c.deviceId)); // remember the cameras that were actually used
    saveSetup();
    await refreshCameras(); // names are visible now that permission is granted
    setMainAspect();
    renderThumbs();
    $('#status').textContent = `Live: ${cams.map((c) => c.label).join(' + ')}`;
    beginMatch();
  } catch (e) {
    made.forEach((c) => c.stop());
    cams = [];
    mode = null;
    console.error(e);
    $('#status').textContent = 'Camera error: ' + (e.name || e.message);
    showCameraHelp(e);
  }
}

async function showCameraHelp(e) {
  let perm = 'unknown';
  try { perm = (await navigator.permissions.query({ name: 'camera' })).state; } catch { /* unsupported */ }
  const fixes = {
    NotAllowedError: perm === 'denied'
      ? 'Camera is BLOCKED for this site. Click the padlock / camera icon left of the address bar → set Camera to Allow → reload. (Chrome: chrome://settings/content/camera, remove localhost from "Not allowed".) Edge: edge://settings/content/camera.'
      : 'Permission was dismissed or denied. Reload the page and click Start again, then choose Allow.',
    NotFoundError: 'No camera detected. Plug one in, or check Windows Settings → Privacy & security → Camera → "Let desktop apps access your camera" is ON, and no other app has exclusive use.',
    OverconstrainedError: 'The camera chosen for that row is no longer available (browsers change camera ids after a restart or permission reset). Pick the camera again in its dropdown, or leave CAM 1 on "System default camera", then press Start.',
    NotReadableError: 'The camera is in use by another app (Teams/Zoom/Camera app) or by another row. Close it and retry.',
    NoAPI: e.message,
    Duplicate: e.message,
  };
  const which = e.camIndex !== undefined ? `CAM ${e.camIndex + 1}: ` : '';
  const msg = which + (fixes[e.name] || String(e.message || e));
  const b = $('#banner');
  b.className = 'banner show foul';
  b.innerHTML = `Camera unavailable<small style="font-size:14px;letter-spacing:0;text-transform:none;line-height:1.4">${escapeHtml(msg)}<br>Permission state: ${perm}. Meanwhile you can analyse a recorded clip with "Video file…".</small>`;
  clearTimeout(bannerTimer);
  bannerTimer = setTimeout(() => b.classList.remove('show'), 20000);
}

async function startVideo(file) {
  stopAll();
  const cam = new Camera(0, cfg);
  try {
    await cam.openFile(file, Number($('#rate').value));
    setRecoverDefault('video'); // uploaded video: find both fighters; playback slows down by itself to keep up (see adaptPlayback)
    cam.pose = await newPose();
    cams = [cam];
    primary = 0;
    mode = 'video';
    coordinator.waitMs = 0;
    setMainAspect();
    renderThumbs();
    $('#status').textContent = `Analysing ${file.name}`;
    beginMatch();
  } catch (e) {
    cam.stop();
    console.error(e);
    $('#status').textContent = 'Video error: ' + (e.message || e);
  }
}

/* ------------------------------------------------------------------- frames */

const COL = { a: '#ff4757', b: '#3b82f6' };

function onDetections(cam, res) {
  const backend = judgeClient.online;
  const multi = cams.length > 1;
  const now = clockNow();
  for (const e of res.events) {
    if (e.type === 'punch_thrown' || e.type === 'kick_thrown') {
      if (!backend) continue; // offline: the geometry events below are used instead
      const hint = res.events.find((x) => /_(landed|missed)$/.test(x.type) && x.by === e.by && x.hand === e.hand && (x.strike || 'punch') === (e.strike || 'punch'));
      const { cluster, created } = coordinator.add(cam.id, e, now, hint || null);
      if (created) {
        cluster.marker = { cam: cam.id, t0: performance.now(), x: e.at.x / cam.aspect, y: e.at.y, text: 'judging…', color: '#9aa4b2', hold: true };
        markers.push(cluster.marker);
      }
      continue;
    }
    const landed = e.type === 'strike_landed' || e.type === 'kick_landed';
    if (landed || e.type === 'strike_missed' || e.type === 'kick_missed') {
      if (backend) continue; // the CLIP backend decides contact; the geometry result is only a fallback / hint
      if (multi && landed && !deduper.accept(e, now)) continue;
      markers.push({ cam: cam.id, t0: performance.now(), x: e.at.x / cam.aspect, y: e.at.y, text: landed ? `${e.strike === 'punch' || !e.strike ? '' : e.strike.toUpperCase() + ' '}${e.target.toUpperCase()}` : 'miss', color: landed ? COL[e.by] : '#9aa4b2' });
      if (!landed) continue;
      match.handle({ ...e, auto: true, clock: now });
      continue;
    }
    if (multi && !deduper.accept(e, now)) continue; // another camera already reported this fall / throw
    match.handle({ ...e, auto: true, clock: now });
    if (e.type === 'down') classifyFall(cam, e, now); // judo: which throw put him there? (video model on the backend)
  }
}

/**
 * Judo: a fighter went down next to the opponent and no leg-reap rule explained it. Ask the backend's throw-type model
 * (trained on the judo_throws clips: ippon seoi nage / o goshi / osoto gari / uchi mata) and score a throw only if it is confident.
 * The model has no "not a throw" class, so decideThrow() first rules out falls that are clearly not throws.
 */
async function classifyFall(cam, e, now) {
  if (sport !== 'judo' || e.thrown || !judgeClient.online || !judgeClient.info?.throw_model) return;
  const near = (cam.pairHist || []).filter((p) => now - p.t <= 2000).map((p) => p.d);
  const minPair = near.length ? Math.min(...near) : null; // null: both fighters were never tracked together (common when bodies overlap)
  if (decideThrow({ down: e, minPair, result: { technique: '?', confidence: 1 }, gradeCfg: cfg.reap }).kind === 'ignore') return; // cheap checks before uploading frames
  try {
    const frames = await cam.clipBuf.capture(now, 2400, 150);
    if (frames.length < 3) return;
    const result = await judgeClient.throwType({ frames });
    const d = decideThrow({ down: e, minPair, result, gradeCfg: cfg.reap });
    if (d.kind === 'throw') {
      match.say(d.note, { kind: 'info', side: d.event.by, auto: true });
      match.handle({ ...d.event, clock: now });
    } else if (d.kind === 'unsure') match.say(d.note, { kind: 'info', auto: true });
  } catch (err) {
    console.warn('throw-type model call failed', err);
  }
}

/** one strike (merged across cameras) -> every camera's clip at the same moment -> CLIP backend -> verdict -> rules */
async function dispatchStrike(cl) {
  const hint = cl.hint;
  const evidence = {
    thrower: cl.by, hand: cl.hand, strike: cl.strike, peak_t_ms: cl.peakT, speed_mps: cl.speed, extension_m: cl.extension,
    geo_hint: hint ? { landed: /_landed$/.test(hint.type), target: hint.target ?? null, virtual: !!hint.virtual } : null,
    legal_zones: match.cfg.legalZones || ['head', 'torso'], // target zones the ruleset in force allows
  };
  const mk = cl.marker || { cam: cl.atCam, t0: performance.now(), x: 0.5, y: 0.5, text: '', color: '#9aa4b2', hold: true };
  try {
    const views = [];
    for (const cam of cams) {
      // every camera is captured at the same peak time, not only the one(s) that triggered: a camera that did not fire
      // may still have the best view of whether the strike touched
      const frames = await cam.clipBuf.capture(cl.peakT);
      if (frames.length >= 3) views.push({ frames, evidence, return_crops: true, camera: cam.id });
    }
    if (!views.length) throw new Error('no frames captured around the strike');
    const v = views.length > 1 ? await judgeClient.judgeMulti({ views, return_crops: true }) : await judgeClient.judge(views[0]);
    // the frames the judge looked at, kept in memory (this session only) for the slow-motion replay
    const replay = { peakT: cl.peakT, cams: views.map((vw) => ({ camera: vw.camera, frames: vw.frames.map((f) => ({ t: f.t_ms, src: `data:image/jpeg;base64,${f.jpeg_b64}` })) })) };
    onVerdict(cl, evidence, v, mk, replay);
  } catch (err) {
    console.error(err);
    mk.hold = false;
    mk.t0 = performance.now();
    mk.text = 'backend error';
    mk.color = '#fb923c';
    if (hint && /_landed$/.test(hint.type)) match.handle({ ...hint, auto: true }); // keep the bout going on geometry
  }
}

function onVerdict(e, evidence, v, mk, replay) {
  const cls = v.contact_location ? v.genuine_punch.detail.classification : '';
  mk.hold = false;
  mk.t0 = performance.now();
  const notStrike = v.strike_check && !v.strike_check.value; // a step / raised leg, not a kick
  mk.text = notStrike ? 'NOT A KICK' : v.foul ? 'FOUL' : v.scored ? 'SCORES' : v.review ? 'REVIEW' : v.contact.value ? `NO SCORE: ${String(cls).toUpperCase()}` : 'MISS';
  mk.color = notStrike ? '#9aa4b2' : v.foul ? '#fb923c' : v.scored ? '#34d399' : v.review ? '#fbbf24' : '#9aa4b2';
  lastVerdict = { v, e, evidence };
  addReview(e, v, replay);
  match.handle({ type: 'strike_verdict', by: e.by, against: v.intended_target.value ?? (e.by === 'a' ? 'b' : 'a'), verdict: v, auto: true, clock: e.peakT });
}

const pct = (x) => `${Math.round(x * 100)}%`;
const yn = (b) => (b === null || b === undefined ? 'n/a' : b ? 'Yes' : 'No');
const tick = (b, c) => (b === null || b === undefined ? '' : b ? `<span class="ok">✓ ${pct(c)}</span>` : `<span class="no">✗ ${pct(c)}</span>`);

/* Punch reviews are a permanent history (newest first), saved in this browser so they survive a reload. */
const reviews = (() => {
  try { return JSON.parse(localStorage.getItem('punchReviews') || '[]'); } catch { return []; }
})();
let reviewSeq = reviews.reduce((m, r) => Math.max(m, r.id), 0);

function saveReviews() {
  // screenshots are ~30 KB per review: if the browser quota is hit, drop screenshots of the oldest ones first
  for (const keep of [60, 20, 8, 0]) {
    try {
      localStorage.setItem('punchReviews', JSON.stringify(reviews.slice(0, 60).map((r, i) => (i < keep ? r : { ...r, v: { ...r.v, crops: {} } }))));
      return;
    } catch { /* quota exceeded: retry with fewer screenshots */ }
  }
}

const replays = new Map(); // review id -> {peakT, cams:[{camera, frames:[{t, src}]}]}: memory only, newest 15
const replayTimers = new Map();

function addReview(e, v, replay) {
  const rev = { at: Date.now(), id: ++reviewSeq, by: e.by, hand: e.hand, strike: v.strike || 'punch', sport, clock: match.clockText(), round: match.round, v, open: true, status: null };
  reviews.unshift(rev);
  if (replay) {
    replays.set(rev.id, replay);
    while (replays.size > 15) replays.delete(replays.keys().next().value);
  }
  if (reviews.length > 60) reviews.length = 60;
  saveReviews();
  renderVerdict();
}

function reviewHtml(r) {
  const v = r.v;
  const dist = (c) => Object.entries(c.detail.distribution || {}).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([k, x]) => `${k.replace(/_/g, ' ')} ${pct(x)}`).join(' · ');
  const rows = [
    ...(v.strike_check ? [['0 Real ' + v.strike + '? (not a step)', yn(v.strike_check.value), tick(v.strike_check.value, v.strike_check.confidence)]] : []),
    ['1 Who threw it?', `${v.thrower.detail.name} (${r.hand} ${r.strike === 'punch' || !r.strike ? 'hand' : r.strike === 'kick' ? 'leg' : 'knee'})`, `<span class="ok">${pct(v.thrower.confidence)}</span>`],
    ['2 Intended target', v.intended_target.detail.name ?? 'none in frame', ''],
    ['3 Contact made?', yn(v.contact.value), tick(v.contact.value, v.contact.confidence)],
    ['4 Where?', String(v.contact_location.value).replace(/_/g, ' '), `<span class="dim">${pct(v.contact_location.confidence)}</span>`],
    ['5 Legal scoring area?', yn(v.legal_area.value), tick(v.legal_area.value, v.legal_area.confidence)],
    [(v.strike || 'punch') === 'punch' ? '6 Legal glove part?' : '6 Legal striking surface?', `${yn(v.legal_glove_part.value)}${v.legal_glove_part.detail.part ? ' · ' + v.legal_glove_part.detail.part.replace(/_/g, ' ') : ''}`, tick(v.legal_glove_part.value, v.legal_glove_part.confidence)],
    ['7 Genuine punch?', `${yn(v.genuine_punch.value)} · ${v.genuine_punch.detail.classification ?? ''}`, tick(v.genuine_punch.value, v.genuine_punch.confidence)],
    ...(v.strike_check?.detail?.trained_kick ? [['Trained kick detector', `${v.strike_check.detail.trained_kick.positive_frames} of ${v.strike_check.detail.trained_kick.frames_considered} frames show a kick`, `<span class="dim">${pct(v.strike_check.detail.trained_kick.mean)}</span>`]] : []),
    ...(v.punch_type ? [['Punch type (experimental)', v.punch_type.label, `<span class="dim" title="${escapeHtml(v.punch_type.note || '')}">${pct(v.punch_type.confidence)}</span>`]] : []),
  ];
  const cls = v.foul ? 'foul' : v.scored ? 'scored' : v.review ? 'review' : 'none';
  const word = (v.strike || 'punch') === 'knee' ? 'KNEE' : (v.strike || 'punch').toUpperCase();
  const txt = v.strike_check && !v.strike_check.value ? `NOT A REAL ${word}` : v.foul ? `FOUL: ${v.foul.replace(/_/g, ' ')}` : v.scored ? `SCORING ${word}` : v.review ? 'NEEDS REFEREE REVIEW' : 'NO SCORE';
  const crops = v.crops || {};
  // single camera: frame / impact / weapon / reaction.  several cameras: keys are "cam2_impact" etc., grouped by camera
  const RANK = { frame: 0, impact: 1, weapon: 2, glove: 2, reaction: 3 }; // 'glove' = reviews saved before kicks existed
  const split = (k) => { const m = k.match(/^(.+)_(frame|impact|weapon|glove|reaction)$/); return m ? [m[1], m[2]] : ['', k]; };
  const order = Object.keys(crops).sort((a, b) => { const [ca, ta] = split(a), [cb, tb] = split(b); return ca === cb ? (RANK[ta] ?? 9) - (RANK[tb] ?? 9) : ca < cb ? -1 : 1; });
  const shots = order.map((k) => { const [cm, t] = split(k); const cap = cm ? `${cm} ${t}` : t; return `<figure><img alt="${cap}" data-zoom src="data:image/jpeg;base64,${crops[k]}"><figcaption>${cap}</figcaption></figure>`; }).join('');
  const camTable = (v.views || []).length
    ? `<h4>Cameras (${v.views.length}): each view is weighted by how well it sees the strike</h4><table class="vt">${v.views.map((x) => `<tr><td class="q">${escapeHtml(x.camera)}</td><td>${x.available ? 'sees both fighters' : 'cannot see the opponent'}${x.available && x.visibility != null ? ` · visibility ${pct(x.visibility)} · side-on ${pct(x.foreshortening ?? 1)}` : ''}</td><td class="c">weight ${pct(x.quality)}${x.p_contact != null ? ` · contact ${pct(x.p_contact)} · ${escapeHtml(String(x.location).replace(/_/g, ' '))}` : ''}</td></tr>`).join('')}</table>${v.contact?.detail?.veto_by_good_view ? '<div class="rstat">A clear miss from a well-placed camera overruled an apparent hit from another angle.</div>' : ''}`
    : '';
  const side = r.by === 'a' ? 'RED' : 'BLUE';
  const status = r.status ? `<div class="rstat">Referee: ${r.status}</div>` : '';
  const btns = v.review && !r.status ? `<div class="vbtn"><button class="primary" data-act="award" data-id="${r.id}">Award ${r.strike === 'knee' ? 'knee strike' : r.strike || 'punch'}</button><button data-act="dismiss" data-id="${r.id}">Dismiss</button></div>` : '';
  return `<div class="rv ${cls}" data-id="${r.id}">
    <div class="rvh"><span class="dot ${r.by}"></span><b>#${r.id}</b> ${side} ${r.hand} ${r.strike || 'punch'} · <b class="t-${cls}">${txt}</b> · ${pct(v.confidence)}<span class="rvt">${r.round > 1 ? 'R' + r.round + ' ' : ''}${r.clock} · ${Math.round(v.latency_ms)} ms</span></div>
    <div class="thumbs">${shots}</div>
    ${replays.has(r.id) ? `<button class="rbtn" data-act="replay" data-id="${r.id}">▶ Replay in slow motion</button>` : ''}<div class="replaybox" data-id="${r.id}"></div>
    <details data-id="${r.id}" ${r.open ? 'open' : ''}><summary>Seven questions &amp; evidence</summary>
      ${camTable}
      <table class="vt">${rows.map(([q, a, c]) => `<tr><td class="q">${q}</td><td>${escapeHtml(a)}</td><td class="c">${c}</td></tr>`).join('')}</table>
      <div class="vline ${cls}">${txt}<span>${escapeHtml(v.summary)}</span></div>
      <div class="dbg">location: ${escapeHtml(dist(v.contact_location))}\ngenuine: ${escapeHtml(dist(v.genuine_punch))}\nclip: ${escapeHtml(JSON.stringify({ contact: v.contact.detail.clip, genuine: v.genuine_punch.detail.clip }))}\ngeometry: ${escapeHtml(JSON.stringify({ contact: v.contact.detail.geometry, genuine: v.genuine_punch.detail.geometry }))}\nmodel: ${escapeHtml(v.model)}</div>
    </details>${btns}${status}</div>`;
}

function stopReplay(id) {
  clearInterval(replayTimers.get(id));
  replayTimers.delete(id);
  const box = document.querySelector(`.replaybox[data-id="${id}"]`);
  if (box) box.innerHTML = '';
}

/** slow-motion replay of the frames the judge saw (all cameras side by side), impact frame marked */
function toggleReplay(id) {
  if (replayTimers.has(id)) return stopReplay(id);
  const rp = replays.get(id), box = document.querySelector(`.replaybox[data-id="${id}"]`);
  if (!rp || !box) return;
  box.innerHTML = rp.cams.map((c) => `<figure><img alt="${escapeHtml(c.camera)} replay"><figcaption></figcaption></figure>`).join('');
  const imgs = [...box.querySelectorAll('img')], caps = [...box.querySelectorAll('figcaption')];
  const peak = rp.cams.map((c) => c.frames.reduce((bi, f, i) => (Math.abs(f.t - rp.peakT) < Math.abs(c.frames[bi].t - rp.peakT) ? i : bi), 0));
  const n = Math.max(...rp.cams.map((c) => c.frames.length));
  let k = 0, loops = 0;
  const step = () => {
    rp.cams.forEach((c, ci) => {
      const idx = Math.min(k, c.frames.length - 1);
      imgs[ci].src = c.frames[idx].src;
      imgs[ci].classList.toggle('peak', idx === peak[ci]);
      caps[ci].textContent = `${rp.cams.length > 1 ? c.camera + ' · ' : ''}frame ${idx + 1}/${c.frames.length}${idx === peak[ci] ? ' · IMPACT' : ''} (slow motion)`;
    });
    if (++k >= n) { k = 0; if (++loops >= 5) stopReplay(id); }
  };
  step();
  replayTimers.set(id, setInterval(step, 150));
}

function renderVerdict() {
  const el = $('#verdict');
  const head = `<div class="rvtop"><h4>Punch reviews (${reviews.length}) · CLIP backend</h4>${reviews.length ? '<button data-act="clear">Clear all</button>' : ''}</div>`;
  if (!reviews.length) {
    el.innerHTML = head + '<div class="empty">Each candidate punch is sent to the backend as a short clip. Its answers to the seven scoring questions and its screenshots are kept here.</div>';
    return;
  }
  el.innerHTML = head + `<div class="rvlist">${reviews.map(reviewHtml).join('')}</div>`;
}

$('#verdict').addEventListener('click', (ev) => {
  const t = ev.target;
  if (t.matches('img[data-zoom]')) { t.classList.toggle('zoomed'); return; }
  const act = t.dataset?.act;
  if (!act) return;
  if (act === 'replay') { toggleReplay(Number(t.dataset.id)); return; }
  if (act === 'clear') {
    if (confirm('Delete all saved punch reviews and their screenshots?')) { reviews.length = 0; saveReviews(); renderVerdict(); }
    return;
  }
  const r = reviews.find((x) => x.id === Number(t.dataset.id));
  if (!r) return;
  if (act === 'award') {
    const kind = r.strike === 'knee' ? 'knee' : r.strike === 'kick' ? (r.v.contact_location.value === 'legs' ? 'leg_kick' : 'kick') : 'punch';
    match.handle({ type: 'score', kind, by: r.by });
    r.status = 'awarded';
  }
  if (act === 'dismiss') r.status = 'dismissed';
  saveReviews();
  renderVerdict();
});
// remember which entries the referee expanded (toggle does not bubble, so listen in the capture phase)
$('#verdict').addEventListener('toggle', (ev) => {
  const r = reviews.find((x) => x.id === Number(ev.target.dataset?.id));
  if (r) { r.open = ev.target.open; saveReviews(); }
}, true);


function updateBackendPill() {
  const el = $('#backend');
  const busy = judgeClient.inflight ? ` · judging ${judgeClient.inflight}` : '';
  if (judgeClient.online) {
    el.textContent = `CLIP ✓ ${judgeClient.info.model.replace('openai/clip-', '')} · ${judgeClient.info.device}${busy}`;
    el.className = 'pill okp';
  } else {
    el.textContent = 'CLIP backend offline: geometry only';
    el.className = 'pill badp';
  }
}

// Frame rate each sport needs (tested on the scripted fights): punches are found 3 of 3 at 12 fps but about 1 of 3 at 8 fps; the Judo
// throw is found 7 of 8 times at 6 fps (and falls 8 of 8 down to 4 fps).
const minFps = () => (sport === 'judo' ? 6 : 12);

/** an uploaded video plays in real time, so a slow computer would skip frames; slow the playback down so about 15 video frames a second get analysed */
function adaptPlayback() {
  const userRate = Number($('#rate').value) || 1;
  for (const c of cams) {
    if (c.kind !== 'video' || !c.stepMs) continue;
    const want = Math.max(0.1, Math.min(userRate, 1000 / 15 / (c.stepMs + 20)));
    if (Math.abs(c.video.playbackRate - want) > 0.04) c.video.playbackRate = +want.toFixed(2);
  }
}

function stepCams() {
  for (const cam of cams) {
    const t0 = performance.now();
    const res = cam.step();
    if (!res) continue;
    cam.stepMs = 0.9 * (cam.stepMs ?? 70) + 0.1 * (performance.now() - t0); // smoothed cost of one processed frame
    if (cam.looped) beginMatch(); // the video clip restarted
    const pd = res.debug?.throw?.pairDist; // how close the two fighters are (when both are tracked): evidence for 'this fall was a throw'
    if (pd != null) {
      (cam.pairHist ||= []).push({ t: res.t, d: pd });
      while (cam.pairHist.length && res.t - cam.pairHist[0].t > 3000) cam.pairHist.shift();
    }
    onDetections(cam, res);
  }
  for (const cl of coordinator.due(clockNow())) dispatchStrike(cl); // fire and forget: the verdict arrives later
}

/* ------------------------------------------------------------------- render */

function drawPerson(c, P, id, W, H, aspect) {
  const s = W / 960;
  const col = id === 'a' ? '#ff4757' : '#3b82f6';
  const X = (i) => (P.lm[i].x / aspect) * W;
  const Y = (i) => P.lm[i].y * H;
  c.lineWidth = 4 * s;
  c.lineCap = 'round';
  c.strokeStyle = col;
  c.shadowColor = '#000';
  c.shadowBlur = 4;
  for (const [a, b] of POSE_CONNECTIONS) {
    if (P.lm[a].v < 0.3 || P.lm[b].v < 0.3) continue;
    c.beginPath();
    c.moveTo(X(a), Y(a));
    c.lineTo(X(b), Y(b));
    c.stroke();
  }
  c.shadowBlur = 0;
  c.fillStyle = '#fff';
  for (const i of [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]) {
    if (P.lm[i].v < 0.3) continue;
    c.beginPath();
    c.arc(X(i), Y(i), (i === 15 || i === 16 ? 6 : 4) * s, 0, 7);
    c.fill();
  }
  // head ring + label
  const hr = 0.5 * P.torso * H;
  c.strokeStyle = col;
  c.lineWidth = 2 * s;
  c.beginPath();
  c.arc(X(0), Y(0), hr, 0, 7);
  c.stroke();
  c.font = `700 ${Math.max(10, 15 * s)}px system-ui, sans-serif`;
  c.fillStyle = col;
  c.textAlign = 'center';
  c.fillText(id === 'a' ? 'RED' : 'BLUE', X(0), Y(0) - hr - 8 * s);
}

function drawCamView(c, cam, W, H, now) {
  const s = W / 960;
  if (cam.video.readyState >= 2) c.drawImage(cam.video, 0, 0, W, H);
  else {
    c.fillStyle = '#0b0f16';
    c.fillRect(0, 0, W, H);
  }
  if ($('#optSkel').checked && cam.lastRes) {
    for (const id of ['a', 'b']) if (cam.lastRes.people[id]) drawPerson(c, cam.lastRes.people[id], id, W, H, cam.aspect);
  }
  for (const m of markers) {
    if (m.cam !== cam.id) continue;
    const u = m.hold ? 0.3 + 0.15 * Math.sin(now / 120) : (now - m.t0) / 1400;
    c.globalAlpha = m.hold ? 1 : 1 - u;
    c.strokeStyle = m.color;
    c.lineWidth = 4 * s;
    c.beginPath();
    c.arc(m.x * W, m.y * H, (12 + u * 34) * s, 0, 7);
    c.stroke();
    c.fillStyle = m.color;
    c.font = `800 ${Math.max(10, 16 * s)}px system-ui`;
    c.textAlign = 'center';
    c.fillText(m.text, m.x * W, m.y * H - (22 + u * 12) * s);
    c.globalAlpha = 1;
  }
}

function render() {
  const W = canvas.width, H = canvas.height;
  const now = performance.now();
  markers = markers.filter((m) => m.hold || now - m.t0 < 1400);
  const cam = cams[primary];
  if (cam) {
    const want = Math.round(960 / cam.aspect);
    if (canvas.height !== want) setMainAspect();
    drawCamView(ctx, cam, canvas.width, canvas.height, now);
  } else {
    ctx.fillStyle = '#0b0f16';
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = 'rgba(255,255,255,.75)';
    ctx.font = '600 22px system-ui';
    ctx.textAlign = 'center';
    ctx.fillText('Add your camera(s) below and press Start, or load a video file', W / 2, H / 2);
  }
  if (cams.length > 1) for (const c of cams) if (c.thumb) drawCamView(c.thumb.getContext('2d'), c, c.thumb.width, c.thumb.height, now);
}

/** how this camera tells RED from BLUE, with the clothing colour it has learned for each (so you can check it by eye) */
function identityHtml(cam) {
  const t = cam.pipeline.tracker;
  const ra = t.ref.a || t.globalRef?.a, rb = t.ref.b || t.globalRef?.b;
  const dot = (s, name) => (s ? `<i class="sw" title="${name}'s clothing colour as learned by ${cam.id}" style="background:${swatch(s)}"></i>` : '');
  let how;
  if (!t.appearance) how = 'by position';
  else if (!(ra && rb)) how = 'learning colours…';
  else how = t.info.src === 'colour' ? 'by colour ✓' : 'by position (clothes look alike)';
  return `RED${dot(ra, 'RED')} BLUE${dot(rb, 'BLUE')} ${how}`;
}

let last = performance.now(), fpsN = 0, fpsT = 0;
function loop(ts) {
  const dt = Math.min(500, ts - last);
  last = ts;
  stepCams();
  match.tick(dt);
  render();
  fpsN++;
  fpsT += dt;
  if (fpsT > 1000) {
    $('#fps').textContent = `${fpsN} fps${cams[0]?.pose ? ' · ' + cams[0].pose.delegate : ''}${cams.length > 1 ? ` · ${cams.length} cameras` : ''}`;
    $('#idinfo').innerHTML = cams[primary] ? identityHtml(cams[primary]) : '';
    for (const c of cams) if (c.badge) c.badge.innerHTML = identityHtml(c);
    const fpsNow = fpsN;
    const slow = cams.length && cams.some((c) => c.kind === 'camera') && fpsNow < minFps();
    $('#fps').classList.toggle('slowfps', !!slow);
    if (slow) $('#fps').textContent += ` ⚠ too slow for ${sport === 'judo' ? 'throws' : 'punches'} (< ${minFps()} fps): untick 'Find 2nd fighter' / 'Match fighters by clothing colour', or close other tabs`;
    const vid = cams.find((c) => c.kind === 'video');
    if (vid) $('#fps').textContent += ` · video at ${vid.video.playbackRate.toFixed(2)}×`;
    adaptPlayback();
    fpsN = 0;
    fpsT = 0;
    const d = $('#debugOut');
    const lr = cams[primary]?.lastRes;
    if (d && lr) d.textContent = JSON.stringify(lr.debug, (k, v) => (typeof v === 'number' ? +v.toFixed(2) : v), 1).slice(0, 1400);
  }
  requestAnimationFrame(loop);
}

/* ------------------------------------------------------------------- wiring */

buildSportTabs();
buildTune();
setSport('boxing');

$('#variant').onchange = (e) => {
  const v = e.target.value;
  match = new Match(RULESETS[sport], v);
  match.on(onMatchEvent);
  renderLog();
  renderBoard();
};

// camera rows: choose cameras, say which side RED is on in each view, add / remove angles
$('#btnCam').onclick = startCameras;
$('#btnAddCam').onclick = () => {
  if (setup.length >= MAX_CAMS) return;
  setup.push({ id: '', flip: false });
  saveSetup();
  renderCamRows();
};
$('#camRows').addEventListener('change', (ev) => {
  const t = ev.target, i = Number(t.dataset.i);
  if (!setup[i]) return;
  if (t.classList.contains('camsel')) setup[i].id = t.value;
  if (t.classList.contains('camflipbox')) {
    setup[i].flip = t.checked;
    if (cams[i]) cams[i].setFlip(t.checked); // takes effect live
  }
  saveSetup();
});
$('#camRows').addEventListener('click', (ev) => {
  const t = ev.target.closest('.camdel');
  if (!t) return;
  setup.splice(Number(t.dataset.i), 1);
  saveSetup();
  renderCamRows();
});
$('#camThumbs').addEventListener('click', (ev) => {
  const t = ev.target.closest('.thumb');
  if (!t) return;
  primary = Number(t.dataset.i); // click a small view to make it the big one
  setMainAspect();
  renderThumbs();
});
navigator.mediaDevices?.addEventListener?.('devicechange', refreshCameras); // plugging a webcam in/out updates the lists
$('#btnDetect').onclick = detectCameras;
refreshCameras().then(() => {
  // already allowed for this site: reveal the camera names without needing a click (the camera is released at once)
  if (permState === 'granted' && devices.length && devices.every((d) => !d.rawLabel)) detectCameras();
});
try { navigator.permissions.query({ name: 'camera' }).then((p) => (p.onchange = refreshCameras)); } catch { /* not supported */ }
$('#fileVideo').onchange = (e) => e.target.files[0] && startVideo(e.target.files[0]);
$('#rate').onchange = () => adaptPlayback(); // the chosen speed is the ceiling; adaptPlayback lowers it when the computer is too slow
judgeClient.onChange = updateBackendPill;
$('#backend').onclick = () => {
  const u = prompt('CLIP backend URL', judgeClient.url);
  if (u) judgeClient.setUrl(u);
};
judgeClient.start();
renderVerdict();
$('#btnStart').onclick = () => (match.status === 'ended' ? (match.reset(), match.start()) : match.start());
$('#btnPause').onclick = () => (match.status === 'paused' ? match.start() : match.pause());
$('#btnReset').onclick = () => {
  for (const c of cams) c.pipeline.reset();
  coordinator.reset();
  match.reset();
  renderLog();
};
/** open the printable match report in a new tab (falls back to a download if pop-ups are blocked) */
function openReport() {
  const html = buildReportHtml({ match, reviews: reviews.filter((r) => (r.at ?? 0) >= match.startedAt) });
  const url = URL.createObjectURL(new Blob([html], { type: 'text/html' }));
  if (!window.open(url, '_blank')) {
    const a = document.createElement('a');
    a.href = url;
    a.download = `ai-referee-report-${match.rules.id}-${new Date().toISOString().slice(0, 10)}.html`;
    a.click();
  }
}
$('#btnReport').onclick = openReport;
$('#optSolo').onchange = (e) => (cfg.punch.soloVirtualOpponent = e.target.checked);
$('#optRecover').onchange = (e) => {
  recoverTouched = true;
  cams.forEach((c) => c.pose && (c.pose.recover = e.target.checked));
};
$('#optColour').onchange =(e) => cams.forEach((c) => (c.pipeline.tracker.appearance = e.target.checked));
for (const b of document.querySelectorAll('#tabs button')) {
  b.onclick = () => {
    for (const x of document.querySelectorAll('#tabs button')) x.classList.toggle('on', x === b);
    for (const p of document.querySelectorAll('.tabpane')) p.hidden = p.id !== 'tab-' + b.dataset.tab;
  };
}

// debug hook for tests: build cameras without hardware, push frames, trigger a strike
window.__ref = {
  buildReport: () => buildReportHtml({ match, reviews: reviews.filter((r) => (r.at ?? 0) >= match.startedAt) }), replays, toggleReplay,
  get match() { return match; }, cfg, setSport, judgeClient, coordinator, deduper, Camera, dispatchStrike, onVerdict, onDetections, stepCams, adaptPlayback,
  get cams() { return cams; }, get lastVerdict() { return lastVerdict; }, get markers() { return markers; },
};
requestAnimationFrame(loop);
