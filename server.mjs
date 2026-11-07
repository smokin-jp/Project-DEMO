// Zero-dependency static server. Camera access needs http://localhost (a secure context).
//   node server.mjs [port]
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.dirname(fileURLToPath(import.meta.url));
const port = Number(process.argv[2] || process.env.PORT || 5173);

const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8', '.json': 'application/json', '.wasm': 'application/wasm',
  '.task': 'application/octet-stream', '.png': 'image/png', '.svg': 'image/svg+xml', '.mp4': 'video/mp4',
};

http
  .createServer((req, res) => {
    let p;
    try { p = decodeURIComponent(new URL(req.url, 'http://x').pathname); } catch { res.writeHead(400).end('bad request'); return; }
    if (p === '/') p = '/index.html';
    // /src/* comes from ./src, everything else from ./public; nothing outside those two folders is ever served
    const dir = p.startsWith('/src/') ? path.join(root, 'src') : path.join(root, 'public');
    const file = path.normalize(path.join(root, p.startsWith('/src/') ? p : path.join('public', p)));
    if (p.includes('\0') || !file.startsWith(dir + path.sep)) { res.writeHead(403).end('forbidden'); return; }
    fs.stat(file, (err, st) => {
      if (err || !st.isFile()) { res.writeHead(404).end('not found'); return; }
      res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream', 'Cache-Control': 'no-cache' });
      fs.createReadStream(file).on('error', () => res.destroy()).pipe(res);
    });
  })
  .listen(port, () => console.log(`AI Referee running at http://localhost:${port}`));
