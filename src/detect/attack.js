import { dist2 } from '../util.js';
import { tilt } from './fall.js';

// Judo non-combativity support. Two signals, no judgement of its own:
//   'attack'   : a fighter steps / drives in towards the opponent, or swings a leg at them (an attempt to engage)
//   'tracking' : heartbeat (every ~0.5 s) while BOTH fighters are in view and on their feet. The rules use it to know the
//                standing contest is really being observed (no tracking = ground work or nobody in view: the 30 s timer pauses)
// Motion is measured along the image x-axis between the two fighters, so it works best from a camera that sees the pair
// side-on; from a camera looking straight down the line between them, steps in are hard to see.

export class AttackDetector {
  constructor(cfg) {
    this.cfg = cfg;
    this.reset();
  }

  reset() {
    this.prev = null;
    this.cool = { a: 0, b: 0 };
    this.lastTrack = -1e9;
  }

  update(fr) {
    const c = this.cfg.attack;
    const out = [];
    const A = fr.people.a, B = fr.people.b;
    if (!A || !B) {
      this.prev = null;
      return out;
    }
    const t = fr.t;
    const upright = tilt(A) < 35 && tilt(B) < 35;
    if (upright && t - this.lastTrack >= 500) {
      this.lastTrack = t;
      out.push({ type: 'tracking' });
    }
    const snap = { t };
    for (const [X, Y, id] of [[A, B, 'a'], [B, A, 'b']]) {
      snap[id] = { hipX: X.c.x, an: [27, 28].map((i) => (X.lm[i].x - X.hip.x) / X.torso) };
    }
    const pv = this.prev;
    this.prev = snap;
    if (!pv || !upright) return out;
    const dt = (t - pv.t) / 1000;
    if (dt < 0.001 || dt > 0.2) return out;
    for (const [X, Y, id] of [[A, B, 'a'], [B, A, 'b']]) {
      const dir = Y.c.x > X.c.x ? 1 : -1; // which way the opponent is
      const step = ((X.c.x - pv[id].hipX) * dir) / X.torso / dt; // torso lengths / s the hips move towards the opponent
      let leg = 0;
      [27, 28].forEach((ai, k) => {
        const rel = (X.lm[ai].x - X.hip.x) / X.torso;
        leg = Math.max(leg, ((rel - pv[id].an[k]) * dir) / dt); // ankle swinging towards the opponent relative to the hip
      });
      if ((step >= c.stepSpeed || leg >= c.legSpeed) && t > this.cool[id] && dist2(X.c, Y.c) / X.torso < c.engageDist) {
        this.cool[id] = t + c.cooldownMs;
        out.push({ type: 'attack', by: id, kind: leg >= c.legSpeed ? 'leg' : 'step', speed: +Math.max(step, leg).toFixed(2) });
      }
    }
    return out;
  }
}
