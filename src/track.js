import { dist2, mid, clamp } from './util.js';
import { similarity, blendSig } from './appearance.js';

// Raw frame: { t (ms), aspect, poses: [{ lm: [{x,y,z,visibility}], world: [{x,y,z}] }], sigs?: [colour signature per pose] }
// Output people use image coordinates in "image-height units" (x multiplied by aspect) so
// distances are isotropic. Fighters are labelled 'a' (RED) / 'b' (BLUE).
//
// Who is who is decided by two kinds of evidence, combined:
//   * position: sticky nearest-centre matching (and, on the very first frame, left = RED unless `flip`)
//   * clothing colour (when `appearance` is on and signatures are supplied): each fighter's torso colours are learned
//     while they are apart, then keep the labels right through clinches and crossings, where position cannot.
// A camera with no colour reference of its own can borrow `globalRef` (another camera's learned colours), so every camera
// uses the same RED / BLUE without being told which side it is on.

const APP_MIN = 0.15; // colour evidence below this (|difference of similarities|) is ignored as ambiguous

function prep(pose, aspect) {
  const lm = pose.lm.map((p) => ({ x: p.x * aspect, y: p.y, z: p.z || 0, v: p.visibility ?? 1 }));
  const world = pose.world ? pose.world.map((p) => ({ x: p.x, y: p.y, z: p.z })) : null;
  const P = { lm, world };
  summarize(P);
  return P;
}

function summarize(P) {
  P.sh = mid(P.lm[11], P.lm[12]);
  P.hip = mid(P.lm[23], P.lm[24]);
  P.c = P.hip;
  P.torso = Math.max(0.02, dist2(P.sh, P.hip));
}

function blend(prev, cur, a) {
  return { x: a * cur.x + (1 - a) * prev.x, y: a * cur.y + (1 - a) * prev.y, z: a * (cur.z || 0) + (1 - a) * (prev.z || 0), v: cur.v };
}

export class Tracker {
  constructor(cfg) {
    this.cfg = cfg;
    this.prev = { a: null, b: null };
    this.lastSeen = { a: -1e9, b: -1e9 };
    this.hist = { a: [], b: [] };
    this.histMs = 3500;
    this.flip = false; // true: this camera sees the fighters the other way round (RED is on the right / far side)
    this.appearance = true; // use clothing colour as well as position
    this.ref = { a: null, b: null }; // learned colour signatures of RED / BLUE (object identity is kept so other cameras can share it)
    this.globalRef = null; // another camera's `ref`, used until this camera has learned its own
    this.info = { src: 'position', margin: 0 }; // how the last two-fighter frame was labelled
  }

  reset() {
    this.prev = { a: null, b: null };
    this.lastSeen = { a: -1e9, b: -1e9 };
    this.hist = { a: [], b: [] };
    this.ref.a = this.ref.b = null;
    this.info = { src: 'position', margin: 0 };
  }

  update(raw) {
    const t = raw.t;
    const dets = raw.poses.slice(0, 2).map((p) => prep(p, raw.aspect));
    const sigs = raw.sigs ? raw.sigs.slice(0, 2) : [];

    for (const id of ['a', 'b']) {
      if (t - this.lastSeen[id] > 1500) this.prev[id] = null; // stale track
    }

    // colour references: this camera's own once learned, otherwise another camera's
    const refA = this.ref.a || this.globalRef?.a || null;
    const refB = this.ref.b || this.globalRef?.b || null;
    const sigsOk = this.appearance && sigs.length === dets.length && sigs.every(Boolean);
    const haveRef = sigsOk && refA && refB;

    const assign = { a: null, b: null };
    let swap = false, total = 0;
    if (dets.length >= 2) {
      const [d0, d1] = dets;
      const pa = this.prev.a, pb = this.prev.b;
      // position evidence: > 0 favours d0 = RED / d1 = BLUE, < 0 the opposite
      let pos;
      if (pa && pb) pos = (dist2(d0.c, pb.c) + dist2(d1.c, pa.c) - dist2(d0.c, pa.c) - dist2(d1.c, pb.c)) / ((d0.torso + d1.torso) / 2);
      else if (pa) pos = (dist2(d1.c, pa.c) - dist2(d0.c, pa.c)) / d0.torso;
      else if (pb) pos = (dist2(d0.c, pb.c) - dist2(d1.c, pb.c)) / d0.torso;
      else pos = (d0.c.x < d1.c.x ? 1 : -1) * (this.flip ? -1 : 1);
      pos = clamp(pos, -2, 2);
      // colour evidence: > 0 when d0 looks like RED's clothes and d1 like BLUE's
      let app = 0;
      if (haveRef) app = similarity(sigs[0], refA) + similarity(sigs[1], refB) - similarity(sigs[0], refB) - similarity(sigs[1], refA);
      const useColour = haveRef && Math.abs(app) >= APP_MIN;
      total = useColour ? 1.5 * app + 0.5 * pos : pos;
      swap = total < 0;
      this.info = { src: useColour ? 'colour' : 'position', margin: useColour ? Math.abs(app) : 0 };
      assign.a = swap ? d1 : d0;
      assign.b = swap ? d0 : d1;
    } else if (dets.length === 1) {
      const d = dets[0];
      const pa = this.prev.a, pb = this.prev.b;
      let id;
      const diff = haveRef ? similarity(sigs[0], refA) - similarity(sigs[0], refB) : 0;
      if (haveRef && Math.abs(diff) >= 2 * APP_MIN) id = diff > 0 ? 'a' : 'b'; // clothes clearly say who it is
      else if (pa && pb) id = dist2(d.c, pa.c) <= dist2(d.c, pb.c) ? 'a' : 'b';
      else if (pa) id = 'a';
      else if (pb) id = 'b';
      else id = (d.c.x < 0.5 * raw.aspect) !== this.flip ? 'a' : 'b'; // solo: left half = RED, right half = BLUE (flipped per camera)
      assign[id] = d;
    }

    // learn the colours while the fighters are apart and the labelling is not in doubt
    if (sigsOk && dets.length === 2 && dist2(dets[0].c, dets[1].c) > 1.3 * Math.max(dets[0].torso, dets[1].torso) && (!haveRef || Math.abs(total) >= 0.5)) {
      const sa = swap ? sigs[1] : sigs[0], sb = swap ? sigs[0] : sigs[1];
      this.ref.a = blendSig(this.ref.a, sa, this.ref.a ? 0.03 : 1);
      this.ref.b = blendSig(this.ref.b, sb, this.ref.b ? 0.03 : 1);
    }

    const people = {};
    for (const id of ['a', 'b']) {
      const d = assign[id];
      if (!d) {
        people[id] = null;
        continue;
      }
      const pv = this.prev[id];
      if (pv && dist2(pv.c, d.c) < 0.6) {
        const al = this.cfg.smoothing;
        d.lm = d.lm.map((p, i) => blend(pv.lm[i], p, al));
        if (d.world && pv.world) d.world = d.world.map((p, i) => blend(pv.world[i], p, al));
        summarize(d);
      }
      d.id = id;
      d.t = t;
      this.prev[id] = d;
      this.lastSeen[id] = t;
      people[id] = d;
      const h = this.hist[id];
      h.push(d);
      while (h.length && t - h[0].t > this.histMs) h.shift();
    }
    return { t, aspect: raw.aspect, people, hist: this.hist };
  }
}
