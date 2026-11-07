import { dist2 } from '../util.js';
import { tilt } from './fall.js';

// Pose-geometry fouls that need no image model: the skeleton alone shows them (roughly).
//   'headbutt' : a fighter's head is thrust at the opponent's head and they meet (head closing fast, mostly the attacker's
//                own movement, and a real neck / upper-body thrust relative to his own shoulders)
//   'leg_grab' : a wrist stays on the opponent's leg below the belt for >= legGrabMs  (judo: shido)
//   'leg_hold' : the same grab kept up for >= legHoldMs  (kickboxing: a kicked leg held on to; catching it briefly is legal)
//   'low_head' : head kept below the opponent's belt line while engaged (boxing: ducking below the waist)
// The rules decide which of these mean anything in the sport being refereed. All distances are in torso lengths.
//
// Honest limits: a 2D skeleton cannot tell a deliberate headbutt from heads clashing in a clinch, or a grip on the thigh from
// a hand resting near it, and MediaPipe often loses one fighter in a tangle (no call is made without both skeletons). These
// calls are therefore flagged as AI calls the referee should check with Undo, and they work best from a side-on camera.

const near = (P, i) => (P.lm[i].v ?? 1) >= 0.35;

export class FoulDetector {
  constructor(cfg) {
    this.cfg = cfg;
    this.reset();
  }

  reset() {
    this.hist = [];
    this.cool = { headbutt: { a: 0, b: 0 }, leg_grab: { a: 0, b: 0 }, leg_hold: { a: 0, b: 0 }, low_head: { a: 0, b: 0 } };
    this.grab = { a: null, b: null }; // since (ms) when this fighter's wrist first touched the opponent's leg
    this.low = { a: null, b: null };
    this.debug = {};
  }

  /** wrist-to-leg distance (opponent torso lengths) for fighter X's wrists against Y's legs, below Y's belt only */
  legReach(X, Y) {
    let best = 9;
    for (const wi of [15, 16]) {
      if (!near(X, wi)) continue;
      const w = X.lm[wi];
      if (w.y < Y.hip.y + this.cfg.foul.belowBelt * Y.torso) continue; // wrist is not below the belt line
      for (const [h, k, a] of [[23, 25, 27], [24, 26, 28]]) {
        for (const [p, q] of [[h, k], [k, a]]) {
          if (!near(Y, k) || (q === a && !near(Y, a))) continue;
          best = Math.min(best, segDist(w, Y.lm[p], Y.lm[q]) / Y.torso);
        }
      }
    }
    return best;
  }

  update(fr) {
    const c = this.cfg.foul;
    const out = [];
    const A = fr.people.a, B = fr.people.b;
    const t = fr.t;
    if (!A || !B) {
      this.hist = [];
      this.grab.a = this.grab.b = this.low.a = this.low.b = null;
      return out;
    }
    const downA = tilt(A) >= c.maxTilt, downB = tilt(B) >= c.maxTilt; // someone on the mat: ne-waza / knockdown, not these fouls
    const snap = { t };
    for (const [id, P] of [['a', A], ['b', B]]) snap[id] = { nose: { ...P.lm[0] }, sh: { ...P.sh }, rel: { x: P.lm[0].x - P.sh.x, y: P.lm[0].y - P.sh.y }, torso: P.torso };
    this.hist.push(snap);
    while (this.hist.length && t - this.hist[0].t > 600) this.hist.shift();
    // the snapshot closest to windowMs ago (frame rates vary); measured movement is scaled to a per-window rate
    let ref = null;
    for (const h of this.hist) {
      const age = t - h.t;
      if (age >= c.windowMs * 0.6 && age <= c.windowMs * 1.8 && (!ref || Math.abs(age - c.windowMs) < Math.abs(t - ref.t - c.windowMs))) ref = h;
    }
    const rate = ref ? c.windowMs / (t - ref.t) : 1;

    const avgTorso = (A.torso + B.torso) / 2;
    const gap = dist2(A.lm[0], B.lm[0]) / avgTorso;
    this.debug = { headGap: +gap.toFixed(2) };

    // ---------- headbutt
    if (ref && near(A, 0) && near(B, 0) && !downA && !downB && gap <= c.headContact) {
      const gap0 = dist2(ref.a.nose, ref.b.nose) / avgTorso;
      const closing = (gap0 - gap) * rate; // torso lengths closed per window
      if (closing >= c.minClose) {
        for (const [X, Y, id, oid] of [[A, B, 'a', 'b'], [B, A, 'b', 'a']]) {
          const r = ref[id];
          const ux = ref[oid].nose.x - r.nose.x, uy = ref[oid].nose.y - r.nose.y;
          const un = Math.hypot(ux, uy) || 1;
          const u = { x: ux / un, y: uy / un };
          const me = (((X.lm[0].x - r.nose.x) * u.x + (X.lm[0].y - r.nose.y) * u.y) / avgTorso) * rate; // my head towards his head
          const thrust = ((((X.lm[0].x - X.sh.x) - r.rel.x) * u.x + ((X.lm[0].y - X.sh.y) - r.rel.y) * u.y) / avgTorso) * rate; // head relative to own shoulders
          const share = me / closing;
          // a body lunge or lean carries the head along; a headbutt moves the head far more than the shoulders
          const shoulders = (((X.sh.x - r.sh.x) * u.x + (X.sh.y - r.sh.y) * u.y) / avgTorso) * rate;
          const headVsBody = me / Math.max(shoulders, 0.05);
          this.debug[id] = { me: +me.toFixed(2), thrust: +thrust.toFixed(2), share: +share.toFixed(2), headVsBody: +headVsBody.toFixed(2) };
          if (thrust >= c.minThrust && share >= c.minShare && headVsBody >= c.minHeadVsBody && t > this.cool.headbutt[id]) {
            this.cool.headbutt[id] = t + c.cooldownMs;
            out.push({ type: 'headbutt', by: id, against: oid, thrust: +thrust.toFixed(2), closing: +closing.toFixed(2) });
          }
        }
      }
    }

    // ---------- leg grab / hold, ducking low
    for (const [X, Y, id, oid, down] of [[A, B, 'a', 'b', downB], [B, A, 'b', 'a', downA]]) {
      const reach = this.legReach(X, Y);
      if (reach <= c.legTouch && !down && dist2(X.c, Y.c) / avgTorso < c.engageDist) {
        if (this.grab[id] == null) this.grab[id] = { since: t, grab: false, hold: false };
        const g = this.grab[id], held = t - g.since;
        if (held >= c.legGrabMs && !g.grab && t > this.cool.leg_grab[id]) { // one call per continuous grip
          g.grab = true;
          this.cool.leg_grab[id] = t + c.cooldownMs;
          out.push({ type: 'leg_grab', by: id, against: oid, ms: Math.round(held) });
        }
        if (held >= c.legHoldMs && !g.hold && t > this.cool.leg_hold[id]) {
          g.hold = true;
          this.cool.leg_hold[id] = t + c.cooldownMs;
          out.push({ type: 'leg_hold', by: id, against: oid, ms: Math.round(held) });
        }
      } else this.grab[id] = null;

      const lowNow = near(X, 0) && X.lm[0].y > Y.hip.y + c.duckBelow * Y.torso && tilt(X) < c.duckMaxTilt && !down && dist2(X.c, Y.c) / avgTorso < c.duckDist;
      if (lowNow) {
        if (this.low[id] == null) this.low[id] = t;
        if (t - this.low[id] >= c.duckMs && t > this.cool.low_head[id]) {
          this.cool.low_head[id] = t + c.cooldownMs;
          out.push({ type: 'low_head', by: id, against: oid, ms: Math.round(t - this.low[id]) });
        }
      } else this.low[id] = null;
    }
    return out;
  }
}

function segDist(p, a, b) {
  const abx = b.x - a.x, aby = b.y - a.y;
  const n = abx * abx + aby * aby;
  const u = n < 1e-12 ? 0 : Math.max(0, Math.min(1, ((p.x - a.x) * abx + (p.y - a.y) * aby) / n));
  return Math.hypot(p.x - (a.x + u * abx), p.y - (a.y + u * aby));
}
