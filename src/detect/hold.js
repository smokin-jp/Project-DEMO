import { dist2 } from '../util.js';

// Sustained clinch ("holding") detector: hips stay within ~1 torso length for several seconds.
// Blame goes to the fighter whose wrists are closer to the opponent's body.

export class HoldDetector {
  constructor(cfg) {
    this.cfg = cfg;
    this.since = null;
    this.cool = 0;
  }

  reset() {
    this.since = null;
    this.cool = 0;
  }

  update(fr) {
    const c = this.cfg.hold;
    const A = fr.people.a, B = fr.people.b;
    if (!A || !B) { this.since = null; return []; }
    const d = dist2(A.c, B.c) / ((A.torso + B.torso) / 2);
    if (d > c.maxDist) { this.since = null; return []; }
    if (this.since == null) this.since = fr.t;
    if (fr.t - this.since >= c.ms && fr.t > this.cool) {
      this.cool = fr.t + c.cooldownMs;
      const wr = (P, O) => (dist2(P.lm[15], O.c) + dist2(P.lm[16], O.c)) / 2;
      const by = wr(A, B) <= wr(B, A) ? 'a' : 'b';
      return [{ type: 'holding', by, ms: Math.round(fr.t - this.since) }];
    }
    return [];
  }
}
