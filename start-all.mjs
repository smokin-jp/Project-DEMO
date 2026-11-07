// One command to run everything:   npm run all     (or double-click start.bat on Windows)
//   - starts the CLIP backend  (python venv in ./backend/.venv)  on API_PORT (default 8000)
//   - starts the web app                                          on WEB_PORT (default 5173)
//   - waits until the model is really loaded, then opens the app in your default browser
// Anything that is already running on those ports is reused, not started twice.
// Options: NO_OPEN=1 (don't open a browser), WEB_PORT=..., API_PORT=...
import { spawn, execFile } from 'node:child_process';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.dirname(fileURLToPath(import.meta.url));
const WEB = Number(process.env.WEB_PORT || 5173);
const API = Number(process.env.API_PORT || 8000);
const children = [];

const log = (tag, line) => console.log(`[${tag}] ${line}`);
const pipe = (tag, stream) => {
  let buf = '';
  stream.on('data', (d) => {
    buf += d;
    const lines = buf.split(/\r?\n/);
    buf = lines.pop();
    for (const l of lines) if (l.trim() && !/symlink|HF_TOKEN|Loading weights|warnings\.warn/.test(l)) log(tag, l);
  });
};

function get(url, timeout = 2000) {
  return new Promise((resolve) => {
    const req = http.get(url, { timeout }, (res) => {
      let body = '';
      res.on('data', (c) => (body += c));
      res.on('end', () => resolve({ status: res.statusCode, body }));
    });
    req.on('error', () => resolve(null));
    req.on('timeout', () => { req.destroy(); resolve(null); });
  });
}
const apiReady = async () => {
  const r = await get(`http://127.0.0.1:${API}/api/health`);
  try { return !!(r && JSON.parse(r.body).loaded); } catch { return false; }
};
const webReady = async () => !!(await get(`http://127.0.0.1:${WEB}/`));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function shutdown(code = 0) {
  for (const c of children) { try { c.kill(); } catch { /* already gone */ } }
  process.exit(code);
}
process.on('SIGINT', () => shutdown(0));
process.on('SIGTERM', () => shutdown(0));

function start(tag, cmd, args, cwd) {
  const c = spawn(cmd, args, { cwd, stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true });
  children.push(c);
  pipe(tag, c.stdout);
  pipe(tag, c.stderr);
  c.on('exit', (code) => { if (code) { log(tag, `exited with code ${code}`); shutdown(1); } });
  c.on('error', (e) => { log(tag, `could not start: ${e.message}`); shutdown(1); });
}

// ---------- backend ----------
const py = [path.join(root, 'backend', '.venv', 'Scripts', 'python.exe'), path.join(root, 'backend', '.venv', 'bin', 'python')].find((p) => fs.existsSync(p));
if (await apiReady()) log('api', `already running on port ${API}: reusing it`);
else {
  if (!py) {
    console.error('The Python backend is not set up yet. Once only:\n  python -m venv backend/.venv\n  backend\\.venv\\Scripts\\pip install torch --index-url https://download.pytorch.org/whl/cpu\n  backend\\.venv\\Scripts\\pip install -r backend/requirements.txt');
    process.exit(1);
  }
  log('api', `starting the CLIP backend on port ${API} (loading the model takes ~15 s) ...`);
  start('api', py, ['-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', String(API), '--log-level', 'warning'], path.join(root, 'backend'));
}

// ---------- web app ----------
if (await webReady()) log('web', `already running on port ${WEB}: reusing it`);
else {
  log('web', `starting the web app on port ${WEB} ...`);
  start('web', process.execPath, [path.join(root, 'server.mjs'), String(WEB)], root);
}

// ---------- wait, then open ----------
const t0 = Date.now();
while (!(await webReady() && await apiReady())) {
  if (Date.now() - t0 > 180000) { console.error('Timed out waiting for the servers (3 min). See the [api] / [web] lines above.'); shutdown(1); }
  await sleep(1000);
}
const url = `http://localhost:${WEB}/${API === 8000 ? '' : `?api=${encodeURIComponent(`http://127.0.0.1:${API}`)}`}`;
console.log(`\n  AI Referee is ready (${Math.round((Date.now() - t0) / 1000)} s)\n  app:      ${url}\n  backend:  http://127.0.0.1:${API}/api/health\n  camera test page: http://localhost:${WEB}/camera-test.html\n\n  Press Ctrl+C to stop both servers.\n`);
if (!process.env.NO_OPEN) {
  if (process.platform === 'win32') execFile('cmd', ['/c', 'start', '', url], { windowsHide: true });
  else execFile(process.platform === 'darwin' ? 'open' : 'xdg-open', [url]);
}
