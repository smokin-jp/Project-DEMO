import { dist2, dist3, angle3, other } from '../util.js';

// Straight-punch detector (jab / cross) using MediaPipe *world* landmarks (metres) so it works
// side-on and head-on. A punch = fast arm-reach growth, near-full extension, enough peak speed.
// "Landing" is then decided in image space by checking the wrist against the opponent's zones.

const ARMS = { L: { s: 11, e: 13, w: 15 }, R: { s: 12, e: 14, w: 16 } };

function zones(O) {
  const t = O.torso;
  const xs = [O.lm[11].x, O.lm[12].x, O.lm[23].x, O.lm[24].x];
  const m = 0.3 * t; // profile views have near-zero shoulder width, so pad for body thickness
  return {
    head: { x: O.lm[0].x, y: O.lm[0].y, r: 0.5 * t },
    x0: Math.min(...xs) - m,
    x1: Math.max(...xs) + m,
    shY: O.sh.y,
    hipY: O.hip.y,
    knY: (O.lm[25].y + O.lm[26].y) / 2,
    t,
  };
}

// -> 'head' | 'torso' | 'low' | null
export function classifyHit(w, O, margin) {
  const z = zones(O);
  const g = margin * z.t;
  if (dist2(w, z.head) <= z.head.r + g) return 'head';
  if (w.x >= z.x0 - g && w.x <= z.x1 + g) {
    if (w.y >= z.shY - 0.2 * z.t && w.y <= z.hipY + 0.05 * z.t) return 'torso';
    if (w.y > z.hipY + 0.05 * z.t && w.y <= z.knY) return 'low';
  }
  return null;
}

function virtualTarget(P, w) {
  if (w.y < P.sh.y + 0.05 * P.torso) return 'head';
  if (w.y < P.hip.y) return 'torso';
  return 'low';
}

export class PunchDetector {
  constructor(cfg) {
    this.cfg = cfg;
    this.st = {};
    this.debug = {};
  }

  reset() {
    this.st = {};
  }

  arm(id, side) {
    return (this.st[id + side] ??= { state: 'idle', rh: [], prevRel: null, prevT: null, prevReach: null, cool: 0 });
  }

  update(fr) {
    const c = this.cfg.punch;
    const out = [];
    for (const id of ['a', 'b']) {
      const P = fr.people[id];
      if (!P || !P.world) continue;
      for (const side of ['L', 'R']) {
        const A = ARMS[side];
        const st = this.arm(id, side);
        if (P.lm[A.w].v < 0.35 || P.lm[A.s].v < 0.35) {
          st.state = 'idle';
          st.prevT = null;
          continue;
        }
        const S = P.world[A.s], E = P.world[A.e], W = P.world[A.w];
        const reach = dist3(S, W);
        const armLen = dist3(S, E) + dist3(E, W);
        const elbow = angle3(S, E, W);
        const rel = { x: W.x - S.x, y: W.y - S.y, z: W.z - S.z };
        const dt = st.prevT == null ? 0 : (P.t - st.prevT) / 1000;
        const ok = dt > 0.001 && dt < 0.2;
        const rate = ok ? (reach - st.prevReach) / dt : 0;
        const speed = ok && st.prevRel ? Math.hypot(rel.x - st.prevRel.x, rel.y - st.prevRel.y, rel.z - st.prevRel.z) / dt : 0;
        st.prevT = P.t;
        st.prevReach = reach;
        st.prevRel = rel;
        st.rh.push({ t: P.t, reach });
        while (st.rh.length && P.t - st.rh[0].t > 300) st.rh.shift();
        this.debug[id + side] = { reach, elbow, speed, rate, state: st.state };

        if (st.state === 'idle') {
          if (rate > c.startRate && P.t > st.cool) {
            st.state = 'ext';
            st.t0 = P.t;
            st.startReach = Math.min(...st.rh.map((r) => r.reach));
            st.peakReach = reach;
            st.peakSpeed = speed;
            st.maxElbow = elbow;
            st.peakLm = P.lm[A.w];
            st.peakArm = armLen;
            st.peakT = P.t;
          }
        } else {
          if (reach > st.peakReach) {
            st.peakReach = reach;
            st.peakLm = P.lm[A.w];
            st.peakArm = armLen;
            st.peakT = P.t;
          }
          st.peakSpeed = Math.max(st.peakSpeed, speed);
          st.maxElbow = Math.max(st.maxElbow, elbow);
          if (reach < st.peakReach - 0.015 || P.t - st.t0 > 450) {
            const ext = st.peakReach - st.startReach;
            const valid = ext >= c.minExtension && st.peakSpeed >= c.minSpeed && st.peakReach / st.peakArm >= c.minReachRatio;
            st.state = 'idle';
            st.cool = P.t + c.cooldownMs;
            if (valid) this.resolve(out, fr, P, id, side, st);
          }
        }
      }
    }
    return out;
  }

  resolve(out, fr, P, id, side, st) {
    const c = this.cfg.punch;
    const w = { x: st.peakLm.x, y: st.peakLm.y };
    const hand = side === 'L' ? 'left' : 'right';
    const base = { by: id, hand, speed: +st.peakSpeed.toFixed(2), extension: +(st.peakReach - st.startReach).toFixed(2), at: w, peakT: st.peakT };
    out.push({ type: 'punch_thrown', ...base });

    const opp = fr.people[other(id)];
    if (opp) {
      const shoulder = P.lm[side === 'L' ? 11 : 12];
      // must be heading towards the opponent
      const towards = dist2(w, opp.c) < dist2(shoulder, opp.c);
      const target = towards ? classifyHit(w, opp, c.landMargin) : null;
      if (target) out.push({ type: 'strike_landed', ...base, against: other(id), target, virtual: false });
      else out.push({ type: 'strike_missed', ...base });
    } else if (c.soloVirtualOpponent) {
      out.push({ type: 'strike_landed', ...base, against: other(id), target: virtualTarget(P, w), virtual: true });
    } else {
      out.push({ type: 'strike_missed', ...base });
    }
  }
}
