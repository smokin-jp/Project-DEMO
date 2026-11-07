// Pure logic for judging one strike from several cameras (no DOM, unit-tested in Node).
//
//  * StrikeCoordinator: each camera runs its own pose tracker and may flag the same punch/kick. Triggers that agree on who
//    struck, with what, and when (cameras share one clock) are merged into ONE strike, which is then judged by the CLIP
//    backend from every camera's clip at once.
//  * EventDeduper: falls, throws and holdings are detected per camera, so with several cameras the same event arrives
//    more than once and must only count once.

export const POST_MS = 300; // post-roll after the strike peak that every camera's clip must contain

export class StrikeCoordinator {
  /** windowMs: triggers whose peaks are this close are the same strike; waitMs: how long to wait for other cameras */
  constructor({ windowMs = 350, waitMs = 0 } = {}) {
    this.windowMs = windowMs;
    this.waitMs = waitMs;
    this.pending = [];
  }

  reset() {
    this.pending = [];
  }

  /** register a candidate from one camera. Returns {cluster, created}. */
  add(camId, e, now, hint = null) {
    const strike = e.strike || 'punch';
    let c = this.pending.find((p) => p.by === e.by && p.strike === strike && Math.abs(p.peakT - e.peakT) <= this.windowMs);
    const created = !c;
    if (!c) {
      c = { by: e.by, strike, hand: e.hand, peakT: e.peakT, speed: e.speed, extension: e.extension, at: e.at, atCam: camId, hint, firstNow: now, cams: new Set(), marker: null };
      this.pending.push(c);
    }
    c.cams.add(camId);
    if (!created && e.speed > c.speed) {
      // the camera that saw it fastest (usually side-on) provides the reference peak time, limb side and impact point
      Object.assign(c, { speed: e.speed, hand: e.hand, peakT: e.peakT, extension: e.extension, at: e.at, atCam: camId });
      if (hint) c.hint = hint;
    }
    return { cluster: c, created };
  }

  /** strikes ready to be judged: post-roll captured and the other cameras had time to report */
  due(now) {
    const ready = this.pending.filter((c) => now >= Math.max(c.peakT + POST_MS, c.firstNow + this.waitMs));
    if (ready.length) this.pending = this.pending.filter((c) => !ready.includes(c));
    return ready;
  }
}

const WINDOWS = { down: 1500, up: 1500, throw: 1500, holding: 4000, headbutt: 2500, leg_grab: 3000, leg_hold: 3000, low_head: 3000, strike_landed: 600, kick_landed: 600 };

export class EventDeduper {
  constructor() {
    this.seen = new Map();
  }

  reset() {
    this.seen.clear();
  }

  /** true if this event is new, false if another camera already reported it just now */
  accept(e, now) {
    const w = WINDOWS[e.type] ?? 1000;
    const key = `${e.type}:${e.by}:${e.strike || ''}`;
    const last = this.seen.get(key);
    if (last !== undefined && now - last < w) return false;
    this.seen.set(key, now);
    return true;
  }
}
