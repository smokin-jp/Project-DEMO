import { sign } from '../util.js';

// Fall detector: emits 'down' when a fighter's torso goes from upright to near-horizontal with
// the hips dropping fast (knockdown / throw landing), and 'up' when they get back on their feet.
// Torso tilt phi = angle of hip->shoulder vector from vertical (0 upright, 90 horizontal).

export function tilt(P) {
  const dx = P.sh.x - P.hip.x, dy = P.sh.y - P.hip.y;
  return (Math.atan2(Math.abs(dx), -dy) * 180) / Math.PI;
}

export class FallDetector {
  constructor(cfg) {
    this.cfg = cfg;
    this.s = { a: this.fresh(), b: this.fresh() };
    this.debug = {};
  }

  fresh() {
    return { state: 'up', base: null, settleAt: 0, maxPhi: 0, maxDrop: 0, startT: 0, pre: null };
  }

  reset() {
    this.s = { a: this.fresh(), b: this.fresh() };
  }

  update(fr) {
    const c = this.cfg.fall;
    const out = [];
    for (const id of ['a', 'b']) {
      const P = fr.people[id];
      const st = this.s[id];
      if (!P) continue;
      const phi = tilt(P);
      if (st.state === 'up' && phi < 28) {
        st.base = st.base == null ? P.hip.y : st.base * 0.97 + P.hip.y * 0.03;
      }
      // running facing estimate, only trusted when nearly vertical (a backward lean cancels the nose offset)
      if (st.state === 'up' && phi < 12) {
        const dx = P.lm[0].x - P.sh.x;
        if (Math.abs(dx) > 0.04 * P.torso) st.faceAcc = 0.9 * (st.faceAcc || 0) + 0.1 * sign(dx);
      }
      const hipDrop = st.base == null ? 0 : (P.hip.y - st.base) / P.torso;
      this.debug[id] = { phi: +phi.toFixed(0), hipDrop: +hipDrop.toFixed(2), state: st.state };

      if (st.state === 'up') {
        const win = fr.hist[id].filter((p) => P.t - p.t <= c.windowMs);
        const minSh = Math.min(...win.map((p) => p.sh.y));
        const shDrop = (P.sh.y - minSh) / P.torso;
        if (phi >= c.minPhi && hipDrop >= c.minHipDrop && shDrop >= c.minShoulderDrop) {
          st.state = 'settling';
          st.settleAt = P.t + c.settleMs;
          st.maxPhi = phi;
          st.maxDrop = hipDrop;
          // fall start = last time the torso was still upright-ish in the window
          const upright = win.filter((p) => p.sh.y - p.hip.y < 0 && Math.abs(p.sh.x - p.hip.x) < 0.5 * p.torso);
          st.startT = upright.length ? upright[upright.length - 1].t : win[0].t;
          // peak shoulder drop speed (torso lengths / s)
          let peak = 0;
          for (let i = 1; i < win.length; i++) {
            const dtt = (win[i].t - win[i - 1].t) / 1000;
            if (dtt > 0.001) peak = Math.max(peak, (win[i].sh.y - win[i - 1].sh.y) / P.torso / dtt);
          }
          st.speed = peak;
        }
      } else if (st.state === 'settling') {
        st.maxPhi = Math.max(st.maxPhi, phi);
        st.maxDrop = Math.max(st.maxDrop, hipDrop);
        // keep tracking fall speed while settling
        const h = fr.hist[id];
        if (h.length > 1) {
          const dtt = (h[h.length - 1].t - h[h.length - 2].t) / 1000;
          if (dtt > 0.001) st.speed = Math.max(st.speed, (h[h.length - 1].sh.y - h[h.length - 2].sh.y) / P.torso / dtt);
        }
        if (P.t >= st.settleAt) {
          st.state = 'down';
          const lean = sign(P.sh.x - P.hip.x);
          const facingOk = Math.abs(st.faceAcc || 0) > 0.5;
          const facing = sign(st.faceAcc || 0);
          out.push({
            type: 'down', by: id,
            phi: +st.maxPhi.toFixed(0), hipDrop: +st.maxDrop.toFixed(2),
            speed: +st.speed.toFixed(2), durationMs: Math.round(P.t - st.startT),
            lean, backwards: facingOk ? facing * lean < 0 : null,
          });
        }
      } else if (st.state === 'down') {
        if (phi < c.upPhi && hipDrop < 0.3) {
          st.state = 'up';
          out.push({ type: 'up', by: id });
        }
      }
    }
    return out;
  }
}
