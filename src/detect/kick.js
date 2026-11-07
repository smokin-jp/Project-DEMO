import { dist2, dist3, angle3, other } from '../util.js';
import { classifyHit } from './punch.js';
import { tilt } from './fall.js';

// Kick / knee-strike CANDIDATE detector. Uses world landmarks (metres) for speed and leg shape and image landmarks for
// how far the foot or knee has been lifted above its standing height (torso lengths). It is deliberately sensitive:
// the CLIP backend makes the real call ("is this really a kick, or a step?") and judges contact, target and legality.
//
//   kick  = foot swings fast relative to the hip, rises well above standing height, leg fairly straight at the peak
//   knee  = knee drives up fast with the leg still folded
// Walking and stepping stay under the lift / speed thresholds. A swing made while the fighter's torso is tipping over
// (the flailing legs of someone being thrown or knocked down) is not a kick.

const LEGS = { L: { h: 23, k: 25, a: 27 }, R: { h: 24, k: 26, a: 28 } };

const sub = (a, b) => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z });
const len = (v) => Math.hypot(v.x, v.y, v.z);

export class KickDetector {
  constructor(cfg) {
    this.cfg = cfg;
    this.st = {};
    this.debug = {};
  }

  reset() {
    this.st = {};
  }

  leg(id, side) {
    return (this.st[id + side] ??= { state: 'idle', prev: null, prevT: null, cool: 0 });
  }

  /** returns bundles: [{ by, events: [kick_thrown, kick_landed|kick_missed] }] */
  update(fr) {
    const c = this.cfg.kick;
    const out = [];
    for (const id of ['a', 'b']) {
      const P = fr.people[id];
      if (!P || !P.world) continue;
      for (const side of ['L', 'R']) {
        const J = LEGS[side];
        const st = this.leg(id, side);
        if (P.lm[J.a].v < 0.35 || P.lm[J.h].v < 0.35) {
          st.state = 'idle';
          st.prevT = null;
          continue;
        }
        const H = P.world[J.h], K = P.world[J.k], A = P.world[J.a];
        const relA = sub(A, H), relK = sub(K, H);
        const legLen = dist3(H, K) + dist3(K, A);
        const dt = st.prevT == null ? 0 : (P.t - st.prevT) / 1000;
        const ok = dt > 0.001 && dt < 0.2;
        const vA = ok ? len(sub(relA, st.prev.A)) / dt : 0;
        const vK = ok ? len(sub(relK, st.prev.K)) / dt : 0;
        st.prev = { A: relA, K: relK };
        st.prevT = P.t;

        const t = P.torso, hipY = P.lm[J.h].y;
        const liftA = (hipY + 1.76 * t - P.lm[J.a].y) / t; // 0 = foot on the floor
        const liftK = (hipY + 0.9 * t - P.lm[J.k].y) / t;
        const kneeAngle = angle3(H, K, A);
        const reach = len(relA) / Math.max(legLen, 1e-6);
        this.debug[id + side] = { vA, vK, liftA, liftK, kneeAngle, reach, state: st.state };

        if (st.state === 'idle') {
          if (P.t > st.cool && ((vA > c.startSpeed && liftA > 0.3) || (vK > c.startSpeed && liftK > 0.3))) {
            st.state = 'swing';
            st.t0 = P.t;
            st.startA = relA;
            st.peakV = Math.max(vA, vK);
            st.maxPhi = tilt(P);
            st.kick = { disp: 0, t: P.t, lm: P.lm[J.a], angle: kneeAngle, liftMax: liftA };
            st.knee = { lift: liftK, t: P.t, lm: P.lm[J.k], angle: kneeAngle, reach };
          }
          continue;
        }

        st.peakV = Math.max(st.peakV, vA, vK);
        st.maxPhi = Math.max(st.maxPhi, tilt(P));
        const disp = len(sub(relA, st.startA));
        st.kick.liftMax = Math.max(st.kick.liftMax, liftA);
        if (disp > st.kick.disp) st.kick = { ...st.kick, disp, t: P.t, lm: P.lm[J.a], angle: kneeAngle };
        if (liftK > st.knee.lift) st.knee = { lift: liftK, t: P.t, lm: P.lm[J.k], angle: kneeAngle, reach };

        const settled = vA < 0.35 * st.peakV && vK < 0.35 * st.peakV && P.t - st.t0 > 120;
        if (!settled && P.t - st.t0 <= 900) continue;

        st.state = 'idle';
        st.cool = P.t + c.cooldownMs;
        const speedOk = st.peakV >= c.minSpeed;
        const kickOk = speedOk && st.kick.disp >= c.minTravel && st.kick.liftMax >= c.minLift && st.kick.angle >= c.kickKneeAngle;
        const kneeOk = speedOk && st.knee.lift >= c.minKneeLift && st.knee.angle < 115 && st.knee.reach < 0.8;
        const falling = st.maxPhi > 45;
        const strike = falling ? null : kickOk ? 'kick' : kneeOk ? 'knee' : null;
        if (!strike) continue;
        const peak = strike === 'kick' ? st.kick : st.knee;
        out.push(this.resolve(fr, P, id, side, J, strike, peak, st));
      }
    }
    return out;
  }

  resolve(fr, P, id, side, J, strike, peak, st) {
    const c = this.cfg;
    const w = { x: peak.lm.x, y: peak.lm.y };
    const leg = side === 'L' ? 'left' : 'right';
    const base = { strike, by: id, leg, hand: leg, speed: +st.peakV.toFixed(2), extension: +(strike === 'kick' ? st.kick.disp : st.knee.lift).toFixed(2), at: w, peakT: peak.t };
    const events = [{ type: 'kick_thrown', ...base }];
    const opp = fr.people[other(id)];
    if (opp) {
      const hip = P.lm[J.h];
      const towards = dist2(w, opp.c) < dist2(hip, opp.c);
      const t = towards ? classifyHit(w, opp, c.punch.landMargin) : null;
      const target = t === 'low' ? 'leg' : t;
      if (target) events.push({ type: 'kick_landed', ...base, against: other(id), target, virtual: false });
      else events.push({ type: 'kick_missed', ...base });
    } else if (c.punch.soloVirtualOpponent) {
      const target = w.y < P.sh.y ? 'head' : w.y < P.hip.y ? 'torso' : 'leg';
      events.push({ type: 'kick_landed', ...base, against: other(id), target, virtual: true });
    } else events.push({ type: 'kick_missed', ...base });
    return { by: id, events };
  }
}
