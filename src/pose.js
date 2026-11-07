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
    finder = await make(delegate, 'IMAGE', 1, 0.25);
    finder.detect(c);
  } catch (e) {
    console.warn('second-fighter search unavailable', e);
  }
  onStatus(`Pose model ready (${delegate})`);

  const work = document.createElement('canvas');
  const part = document.createElement('canvas');
  let nextTry = 0;
  const confirm = new Confirmer();
  const stats = { frames: 0, native2: 0, one: 0, searched: 0, recovered: 0, noBox: 0, candidates: 0, rejected: {} }; // rejected: why candidates were turned down

  /** the poses of the picture on `work` (full frame) or its left/right part, mapped back to full-frame coordinates */
  function search(x0, x1, aspect) {
    const W = work.width, H = work.height;
    let src = work;
    if (x0 > 0 || x1 < 1) {
      part.width = Math.round((x1 - x0) * W);
      part.height = H;
      part.getContext('2d').drawImage(work, x0 * W, 0, part.width, H, 0, 0, part.width, H);
      src = part;
    }
    const r = finder.detect(src);
    return (r.landmarks || []).map((l, i) => ({
      lm: l.map((p) => ({ ...p, x: x0 + p.x * (x1 - x0) })),
      world: r.worldLandmarks?.[i],
    })).map((p) => ({ ...p, body: bodyOf(p.lm, aspect) }));
  }

  function findSecond(video, first) {
    const aspect = (video.videoWidth || video.width) / (video.videoHeight || video.height) || 16 / 9;
    work.width = WORK_W;
    work.height = Math.max(8, Math.round(WORK_W / aspect));
    const g = work.getContext('2d');
    g.drawImage(video, 0, 0, work.width, work.height);
    const fb = bodyOf(first.lm, aspect);
    const box = personBox(first.lm, MASK_PAD);
    if (!box) { stats.noBox++; return null; }
    const { x0, x1, y0, y1 } = box;
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
    if (hit) return hit;
    g.drawImage(video, 0, 0, work.width, work.height); // un-blank the picture
    return tryCands(search(0, 0.62, aspect)) || tryCands(search(0.38, 1, aspect));
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
      if (!api.recover || !finder || poses.length !== 1) return poses;
      const now = performance.now();
      if (now < nextTry) return poses;
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
