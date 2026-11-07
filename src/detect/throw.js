import { dist2, sign } from '../util.js';
import { gradeThrow } from '../throwgrade.js';

// Osoto-gari (major outer reap) detector = leg reap by tori + uke falling backwards over it.
//  1. "reap": tori's ankle whips towards uke fast (relative to tori's hip) and passes close to uke's legs
//  2. "down": uke's FallDetector fires within linkMs, falling AWAY from tori, backwards-facing
// Grade follows IJF landing criteria as far as 2D pose allows:
//   ippon      - lands largely on the back (torso ~horizontal) with speed/force
//   waza-ari   - substantial but not all of ippon
//   yuko       - side / seated landing (2025 rules reintroduced yuko)

export class ThrowDetector {
  constructor(cfg) {
    this.cfg = cfg;
    this.reaps = [];
    this.cool = { a: 0, b: 0 };
    this.prevRel = {};
    this.debug = {};
  }

  reset() {
    this.reaps = [];
    this.cool = { a: 0, b: 0 };
    this.prevRel = {};
  }

  // detect reaps for this frame
  scan(fr) {
    const c = this.cfg.reap;
    const A = fr.people.a, B = fr.people.b;
    this.debug = {};
    if (!A || !B) {
      this.prevRel = {};
      return;
    }
    const pair = dist2(A.c, B.c) / ((A.torso + B.torso) / 2);
    this.debug.pairDist = +pair.toFixed(2);
    for (const [X, Y] of [[A, B], [B, A]]) {
      const u = sign(Y.c.x - X.c.x);
      for (const ai of [27, 28]) {
        const key = X.id + ai;
        const an = X.lm[ai];
        const rel = { x: (an.x - X.hip.x) / X.torso, y: (an.y - X.hip.y) / X.torso, t: X.t };
        const pr = this.prevRel[key];
        this.prevRel[key] = rel;
        if (!pr || an.v < 0.3) continue;
        const dt = (rel.t - pr.t) / 1000;
        if (dt < 0.001 || dt > 0.2) continue;
        const vTow = ((rel.x - pr.x) / dt) * u;
        this.debug[key] = +vTow.toFixed(1);
        if (pair > c.maxPairDist || vTow < c.minSpeed || X.t < this.cool[X.id]) continue;
        // ankle must sweep past the opponent's legs
        const near = Math.min(
          ...[25, 26, 27, 28].map((i) => dist2(an, Y.lm[i]))
        ) / X.torso;
        if (near > c.reachDist) continue;
        this.cool[X.id] = X.t + 1500;
        this.reaps.push({ by: X.id, against: Y.id, t: X.t, u, speed: vTow });
      }
    }
    // expire
    const now = fr.t;
    this.reaps = this.reaps.filter((r) => now - r.t < this.cfg.reap.linkMs + 500);
  }

  // match 'down' events against pending reaps; returns throw events and flags the downs
  resolve(fr, downs) {
    const c = this.cfg.reap;
    const out = [];
    for (const d of downs) {
      const r = this.reaps.find(
        (x) => x.against === d.by && d.durationMs !== undefined && fr.t - x.t <= c.linkMs && fr.t >= x.t - 100
      );
      if (!r) continue;
      if (d.lean !== r.u) continue; // must fall away from tori
      const grade = gradeThrow(d, c);
      d.thrown = true;
      this.reaps = this.reaps.filter((x) => x !== r);
      out.push({
        type: 'throw', by: r.by, against: d.by, technique: 'Osoto-gari', grade,
        phi: d.phi, speed: d.speed, durationMs: d.durationMs, reapSpeed: +r.speed.toFixed(1),
      });
    }
    return out;
  }
}
