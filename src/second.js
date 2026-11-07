// Sanity checks for a skeleton found by the second-fighter search (see pose.js). Pure functions, no browser needed.
// Landmarks are MediaPipe's {x, y, visibility} in image fractions; x is multiplied by the aspect ratio so distances are
// in image-height units, as everywhere else.

const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
const vis = (p) => p.visibility ?? 1;

export function bodyOf(lm, aspect) {
  const P = lm.map((p) => ({ x: p.x * aspect, y: p.y, v: vis(p) }));
  const sh = mid(P[11], P[12]), hip = mid(P[23], P[24]);
  const seen = P.filter((p) => p.v > 0.3);
  const xs = seen.map((p) => p.x), ys = seen.map((p) => p.y);
  const box = seen.length ? { x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys), y1: Math.max(...ys) } : null; // where the body is on screen
  return { P, sh, hip, box, torso: Math.hypot(sh.x - hip.x, sh.y - hip.y) };
}

const touching = (a, b, margin) => !!a && !!b && a.x0 - margin < b.x1 && b.x0 - margin < a.x1 && a.y0 - margin < b.y1 && b.y0 - margin < a.y1;

/** why `cand` is NOT a believable second fighter next to `first` (both from bodyOf, same aspect), or null if it is.
 *  Without `first` only the checks a body must pass on its own are made. */
export function whyNot(cand, first) {
  const { P, torso, hip } = cand;
  if (torso < 0.04) return 'too small';
  const core = [11, 12, 23, 24];
  if (core.reduce((s, i) => s + P[i].v, 0) / 4 < 0.6) return 'weak joints'; // shoulders and hips must be well supported
  if (!first) return [25, 26].filter((i) => P[i].v > 0.4).length < 1 ? 'no knee' : core.some((i) => P[i].y < 0 || P[i].y > 1) ? 'off frame' : null; // nobody to compare with
  const ratio = torso / first.torso;
  if (ratio < 0.5 || ratio > 2) return 'size'; // roughly the same size as the first fighter
  const d = Math.hypot(hip.x - first.hip.x, hip.y - first.hip.y);
  if (d <= 0.5 * Math.max(torso, first.torso)) return 'same body'; // the same body again
  if (d > 5 * Math.max(torso, first.torso)) return 'too far'; // a fight is two bodies close together (not a banner in the corner)
  // Both fighters stand on the same mat: a hip line far ABOVE the first fighter's belongs to a spectator or a banner. The exception
  // is a body that overlaps the first one (a fighter lifted in a throw really is higher), so the check only applies when apart.
  const up = (first.hip.y - hip.y) / Math.max(torso, first.torso);
  if (up < -3) return 'above floor';
  if (up > 0.8 && !touching(cand.box, first.box, 0.25 * Math.max(torso, first.torso))) return 'above floor';
  if (core.some((i) => P[i].y < 0 || P[i].y > 1)) return 'off frame';
  if ([25, 26].filter((i) => P[i].v > 0.4).length < 1) return 'no knee'; // a real body shows at least one knee
  return null;
}


export const plausibleSecond = (cand, first) => whyNot(cand, first) === null;

/** bounding box (image fractions) of the landmarks that are reasonably visible, expanded by `pad` of its size */
export function personBox(lm, pad = 0.18) {
  const ok = lm.filter((p) => vis(p) > 0.2);
  if (ok.length < 4) return null;
  const xs = ok.map((p) => p.x), ys = ok.map((p) => p.y);
  const w = Math.max(...xs) - Math.min(...xs), h = Math.max(...ys) - Math.min(...ys);
  return {
    x0: Math.max(0, Math.min(...xs) - pad * w), x1: Math.min(1, Math.max(...xs) + pad * w),
    y0: Math.max(0, Math.min(...ys) - pad * h), y1: Math.min(1, Math.max(...ys) + pad * h),
  };
}

/** a recovered skeleton is only believed once it shows up twice in a row in the same place */
export class Confirmer {
  constructor(windowMs = 800, distTorso = 1.5) {
    this.windowMs = windowMs;
    this.dist = distTorso;
    this.last = null;
  }

  reset() {
    this.last = null;
  }

  /** @returns true when this sighting repeats the previous one (so it can be used) */
  see(now, body) {
    const prev = this.last;
    this.last = { t: now, hip: body.hip };
    return !!prev && now - prev.t < this.windowMs && Math.hypot(body.hip.x - prev.hip.x, body.hip.y - prev.hip.y) < this.dist * body.torso;
  }
}
