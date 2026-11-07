// MediaPipe Pose Landmarker wrapper (runs fully in the browser, assets are vendored locally).
import { FilesetResolver, PoseLandmarker } from '/vendor/vision_bundle.mjs';
import { bodyOf, whyNot, personBox, Confirmer } from './second.js';

// ---- second-fighter recovery -------------------------------------------------------------------------------------------
// MediaPipe often reports ONE pose when two fighters are clearly in view: once it is tracking one body it rarely
// re-detects the other, and a person who is partly hidden or tangled with the first is dropped. When exactly one pose comes
// back, a second look is taken: the first fighter's box is blanked out and the picture searched again (whole frame, then
// the left and right parts of the plain frame, which gives a small or half-hidden person a larger share of the model's input).
// On 1,098 frames of real two-person judo footage this raised "both fighters found" from 7 % to about 33 %.
// A recovered skeleton must pass sanity checks (similar body size, close to the first fighter, inside the frame, well-supported
// joints, knees visible) and has to be seen twice in a row at the same place, so a banner, logo or spectator is not
// mistaken for a fighter.

const WORK_W = 640;     // the search runs on a picture this wide (the model shrinks everything further anyway)
const MASK_PAD = 0.18;  // blank out the first fighter's box plus this much margin
const NEAR_MS = 700;    // after a find, the next searches start in a box around where the second fighter was (cheap) for this long
const RETRY_MS = 200;   // after a fruitless search, wait this long before looking again (a lone fighter stays cheap)

export async function createPose(onStatus = () => {}, { recover = true } = {}) {
  onStatus('Loading pose model…');
  const fileset = await FilesetResolver.forVisionTasks('/vendor/wasm');
  const make = (delegate, mode = 'VIDEO', numPoses = 2, conf = 0.45) =>
    PoseLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: '/models/pose_landmarker_full.task', delegate },
      runningMode: mode,
      numPoses,
      minPoseDetectionConfidence: conf,
      minPosePresenceConfidence: conf,
      minTrackingConfidence: conf,
    });
  let lm, delegate = 'GPU';
  try {
    lm = await make('GPU');
  } catch (e) {
    console.warn('GPU delegate failed, using CPU', e);
    delegate = 'CPU';
    lm = await make('CPU');
  }
  let lastTs = 0;
  // warm-up: the first inference compiles GPU shaders (seconds); do it now, not on the first live frame
  onStatus('Warming up pose model…');
  const c = document.createElement('canvas');
  c.width = 256;
  c.height = 144;
  lm.detectForVideo(c, (lastTs = performance.now()));
  // a second, stateless instance for the second-fighter search (single pose, lower confidence bar)
  let finder = null;
  try {
    finder = await make(delegate, 'IMAGE', 2, 0.25); // two poses back: the first fighter again must not hide the second
    finder.detect(c);
  } catch (e) {
    console.warn('second-fighter search unavailable', e);
  }
  onStatus(`Pose model ready (${delegate})`);

  const work = document.createElement('canvas');
  const part = document.createElement('canvas');
  let nextTry = 0;
  let lastSecond = null; // {box, t}: where the second fighter was last found (the next search starts there)
  const confirm = new Confirmer();
  const stats = { frames: 0, native2: 0, one: 0, zero: 0, searched: 0, recovered: 0, noBox: 0, candidates: 0, rejected: {} }; // rejected: why candidates were turned down

  /** the poses found in a region of the picture on `work` (default: all of it), mapped back to full-frame coordinates */
  function search(x0, x1, aspect, y0 = 0, y1 = 1) {
    const W = work.width, H = work.height;
    let src = work;
    if (x0 > 0 || x1 < 1 || y0 > 0 || y1 < 1) {
      part.width = Math.max(8, Math.round((x1 - x0) * W));
      part.height = Math.max(8, Math.round((y1 - y0) * H));
      part.getContext('2d').drawImage(work, x0 * W, y0 * H, part.width, part.height, 0, 0, part.width, part.height);
      src = part;
    }
    const r = finder.detect(src);
    return (r.landmarks || []).map((l, i) => ({
      lm: l.map((p) => ({ ...p, x: x0 + p.x * (x1 - x0), y: y0 + p.y * (y1 - y0) })),
      world: r.worldLandmarks?.[i],
    })).map((p) => ({ ...p, body: bodyOf(p.lm, aspect) }));
  }

  function findSecond(video, first) {
    const aspect = (video.videoWidth || video.width) / (video.videoHeight || video.height) || 16 / 9;
    const wh = Math.max(8, Math.round(WORK_W / aspect));
    if (work.width !== WORK_W || work.height !== wh) { work.width = WORK_W; work.height = wh; } // resizing a canvas clears it: only when needed
    const g = work.getContext('2d');
    g.drawImage(video, 0, 0, work.width, work.height);
    const fb = bodyOf(first.lm, aspect);
    const box = personBox(first.lm, MASK_PAD);
    if (!box) { stats.noBox++; return null; }
    const { x0, x1, y0, y1 } = box;
    const remember = (hit) => {
      lastSecond = { box: personBox(hit.lm, 0), t: performance.now() };
      return hit;
    };
    // fast path: the second fighter was seen a moment ago, so look only in a box around that place (one small search instead of up to three)
    if (lastSecond?.box && performance.now() - lastSecond.t < NEAR_MS) {
      const b = lastSecond.box, w = b.x1 - b.x0, h = b.y1 - b.y0;
      const hit = tryNear(Math.max(0, b.x0 - 0.7 * w), Math.min(1, b.x1 + 0.7 * w), Math.max(0, b.y0 - 0.5 * h), Math.min(1, b.y1 + 0.5 * h), aspect, fb);
      if (hit) return remember(hit);
    }
    g.fillStyle = 'rgb(127,127,127)';
    g.fillRect(x0 * work.width, y0 * work.height, (x1 - x0) * work.width, (y1 - y0) * work.height);
    // 1) the frame with the first fighter blanked out, as a whole; 2) the plain left and right parts of the frame (a small or
    // half-hidden fighter gets a bigger share of the model's input). Plain halves find people the blanked frame cannot.
    const tryCands = (cands) => {
      for (const cand of cands) {
        stats.candidates++;
        const why = whyNot(cand.body, fb);
        if (!why) return { lm: cand.lm, world: cand.world, body: cand.body };
        stats.rejected[why] = (stats.rejected[why] || 0) + 1;
      }
      return null;
    };
    const hit = tryCands(search(0, 1, aspect));
    if (hit) return remember(hit);
    g.drawImage(video, 0, 0, work.width, work.height); // un-blank the picture
    const hit2 = tryCands(search(0, 0.62, aspect)) || tryCands(search(0.38, 1, aspect));
    if (hit2) return remember(hit2);
    lastSecond = null;
    return null;
  }

  /** no skeleton at all from the video model: look for up to two people in the whole frame (lower confidence bar) */
  function findAny(video) {
    const aspect = (video.videoWidth || video.width) / (video.videoHeight || video.height) || 16 / 9;
    const wh = Math.max(8, Math.round(WORK_W / aspect));
    if (work.width !== WORK_W || work.height !== wh) { work.width = WORK_W; work.height = wh; }
    work.getContext('2d').drawImage(video, 0, 0, work.width, work.height);
    const ok = [];
    for (const cand of search(0, 1, aspect)) {
      stats.candidates++;
      const why = whyNot(cand.body, null);
      if (why) stats.rejected[why] = (stats.rejected[why] || 0) + 1;
      else ok.push(cand);
    }
    if (ok.length === 2 && whyNot(ok[1].body, ok[0].body)) ok.pop(); // the pair must look like two different fighters, else keep one
    return ok.map((c) => ({ lm: c.lm, world: c.world, body: c.body }));
  }

  /** search a box of the (un-blanked) picture; the first fighter inside it is rejected by the sanity checks ("same body") */
  function tryNear(x0, x1, y0, y1, aspect, fb) {
    for (const cand of search(x0, x1, aspect, y0, y1)) {
      stats.candidates++;
      const why = whyNot(cand.body, fb);
      if (!why) return { lm: cand.lm, world: cand.world, body: cand.body };
      stats.rejected[why] = (stats.rejected[why] || 0) + 1;
    }
    return null;
  }

  const api = {
    delegate,
    recover, // can be switched off live (public/app.js checkbox)
    stats,
    /** returns the poses of the current video frame in the pipeline's raw format (without .t / .aspect) */
    detect(video) {
      const ts = Math.max(lastTs + 1, performance.now()); // MediaPipe needs strictly increasing timestamps
      lastTs = ts;
      const r = lm.detectForVideo(video, ts);
      const poses = (r.landmarks || []).map((l, i) => ({ lm: l, world: r.worldLandmarks?.[i] }));
      stats.frames++;
      if (poses.length >= 2) { stats.native2++; confirm.reset(); return poses; }
      if (poses.length === 1) stats.one++;
      if (!api.recover || !finder || poses.length > 1) return poses;
      const now = performance.now();
      if (now < nextTry) return poses;
      if (poses.length === 0) {
        stats.zero++;
        stats.searched++;
        let any = [];
        try { any = findAny(video); } catch (e) { console.warn('skeleton search failed', e); }
        if (!any.length) { nextTry = now + RETRY_MS; confirm.reset(); return poses; }
        if (!confirm.see(now, any[0].body)) return poses; // seen twice in the same place before it is believed
        stats.recovered += any.length;
        return any.map((c) => ({ lm: c.lm, world: c.world }));
      }
      stats.searched++;
      let found = null;
      try { found = findSecond(video, poses[0]); } catch (e) { console.warn('second-fighter search failed', e); }
      if (!found) { nextTry = now + RETRY_MS; confirm.reset(); return poses; }
      if (!confirm.see(now, found.body)) return poses; // seen twice, in the same place, before it is believed
      stats.recovered++;
      poses.push({ lm: found.lm, world: found.world });
      return poses;
    },
    /** free the WebAssembly / GPU memory of this instance (call when the camera is stopped) */
    close() {
      for (const x of [lm, finder]) { try { x?.close(); } catch { /* already closed */ } }
    },
  };
  return api;
}
