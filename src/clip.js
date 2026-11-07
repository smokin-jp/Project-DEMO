// Browser side of the CLIP backend: a rolling frame buffer, clip capture around each candidate strike,
// and the HTTP client. The pose tracker is the cheap trigger; the backend makes the call.

const KEYPOINTS = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
const PRE_MS = 500, POST_MS = 300, MAX_SEND = 16, WIDTH = 640;

/** compact skeletons for the backend: {a: {idx: [x,y,v]}, b: ...}; x,y as image fractions */
export function compactPoses(people, aspect) {
  const out = {};
  for (const id of ['a', 'b']) {
    const P = people[id];
    if (!P) { out[id] = null; continue; }
    const o = {};
    for (const i of KEYPOINTS) o[i] = [+(P.lm[i].x / aspect).toFixed(4), +P.lm[i].y.toFixed(4), +P.lm[i].v.toFixed(2)];
    out[id] = o;
  }
  return out;
}

export class ClipBuffer {
  constructor() {
    this.frames = []; // {t, bmp: Promise<ImageBitmap>, poses}
    this.maxMs = PRE_MS + POST_MS + 2600; // a kick can be held back ~1.3 s while the pipeline rules out a throw
    this.canvas = document.createElement('canvas');
  }

  reset() {
    for (const f of this.frames) f.bmp.then((b) => b.close?.()).catch(() => {});
    this.frames = [];
  }

  /** call once per processed frame. source: video element or canvas */
  push(t, source, poses) {
    const w = source.videoWidth || source.width, h = source.videoHeight || source.height;
    const bmp = createImageBitmap(source, { resizeWidth: WIDTH, resizeHeight: Math.round((WIDTH * h) / w), resizeQuality: 'low' });
    bmp.catch(() => {}); // a failed grab surfaces when the clip is encoded; do not also log an unhandled rejection
    this.frames.push({ t, bmp, poses });
    while (this.frames.length && t - this.frames[0].t > this.maxMs) {
      this.frames.shift().bmp.then((b) => b.close?.()).catch(() => {});
    }
  }

  /** JPEG-encode the frames around a strike (every camera is captured at the same peak time, so the clips line up) */
  async capture(peak, pre = PRE_MS, post = POST_MS) {
    const frames = this.select(peak, pre, post);
    return frames.length ? this.encode(frames) : [];
  }

  select(peak, pre = PRE_MS, post = POST_MS) {
    let win = this.frames.filter((f) => f.t >= peak - pre && f.t <= peak + post);
    if (win.length > MAX_SEND) {
      const near = new Set(win.filter((f) => Math.abs(f.t - peak) <= 100));
      const rest = win.filter((f) => !near.has(f));
      const keep = MAX_SEND - near.size;
      const stride = rest.length / Math.max(1, keep);
      const picked = new Set(near);
      for (let i = 0; i < keep; i++) picked.add(rest[Math.floor(i * stride)]);
      win = win.filter((f) => picked.has(f));
    }
    return win;
  }

  // OffscreenCanvas.convertToBlob does not wait for a render frame (canvas.toBlob does, and stalls in background tabs)
  async encode(frames) {
    return Promise.all(
      frames.map(async (f) => {
        const bmp = await f.bmp;
        const oc = new OffscreenCanvas(bmp.width, bmp.height);
        oc.getContext('2d').drawImage(bmp, 0, 0);
        const blob = await oc.convertToBlob({ type: 'image/jpeg', quality: 0.85 });
        const bytes = new Uint8Array(await blob.arrayBuffer());
        let bin = '';
        for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
        return { t_ms: f.t, jpeg_b64: btoa(bin), poses: f.poses };
      })
    );
  }
}

export class JudgeClient {
  constructor() {
    const q = typeof location !== 'undefined' ? new URLSearchParams(location.search).get('api') : null; // http://localhost:5173/?api=http://127.0.0.1:8090
    this.url = q || (typeof localStorage !== 'undefined' && localStorage.getItem('backendUrl')) || 'http://127.0.0.1:8000'; // not "localhost": on Windows that costs ~2 s per request (IPv6 attempt first)
    this.online = false;
    this.info = null;
    this.inflight = 0;
    this.onChange = () => {};
  }

  setUrl(u) {
    this.url = u.replace(/\/$/, '');
    try { localStorage.setItem('backendUrl', this.url); } catch { /* ignore */ }
    this.poll();
  }

  async poll() {
    try {
      const r = await fetch(this.url + '/api/health', { cache: 'no-store' });
      const j = await r.json();
      this.online = !!(j.ok && j.loaded);
      this.info = j;
    } catch {
      this.online = false;
      this.info = null;
    }
    this.onChange();
  }

  start() {
    this.poll();
    setInterval(() => this.poll(), 4000);
  }

  /** frames around a throw-like fall -> which of the 4 trained judo throws (needs backend/models/throw_clip.pt) */
  throwType(body) {
    return this.judge(body, '/api/throw_type');
  }

  /** several cameras' clips of the same strike -> one fused verdict */
  judgeMulti(body) {
    return this.judge(body, '/api/judge_multi');
  }

  async judge(body, path = '/api/judge') {
    this.inflight++;
    this.onChange();
    try {
      const r = await fetch(this.url + path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      if (!r.ok) throw new Error(`backend ${r.status}: ${(await r.text()).slice(0, 200)}`);
      return await r.json();
    } finally {
      this.inflight--;
      this.onChange();
    }
  }
}
