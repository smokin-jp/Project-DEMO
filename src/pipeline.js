import { Tracker } from './track.js';
import { PunchDetector } from './detect/punch.js';
import { KickDetector } from './detect/kick.js';
import { FallDetector } from './detect/fall.js';
import { ThrowDetector } from './detect/throw.js';
import { HoldDetector } from './detect/hold.js';
import { AttackDetector } from './detect/attack.js';
import { FoulDetector } from './detect/foul.js';
import { cloneConfig } from './config.js';

// A leg swing right after a detected Osoto-gari reap is the throw, not a kick: such kick candidates are held back
// until the throw resolves (then dropped) or the hold expires (then released as a real kick).
const REAP_HOLD_MS = 1300;
const EARLY_RELEASE_MS = 500;

// raw pose frames in -> referee events out
export class Pipeline {
  constructor(cfg = cloneConfig()) {
    this.cfg = cfg;
    this.tracker = new Tracker(cfg);
    this.punch = new PunchDetector(cfg);
    this.kick = new KickDetector(cfg);
    this.fall = new FallDetector(cfg);
    this.throw = new ThrowDetector(cfg);
    this.hold = new HoldDetector(cfg);
    this.attack = new AttackDetector(cfg);
    this.foul = new FoulDetector(cfg);
    this.held = [];
  }

  reset() {
    this.tracker.reset();
    for (const d of [this.punch, this.kick, this.fall, this.throw, this.hold, this.attack, this.foul]) d.reset();
    this.held = [];
  }

  process(raw) {
    const fr = this.tracker.update(raw);
    const events = [];
    events.push(...this.punch.update(fr));
    this.throw.scan(fr); // before the kick detector so fresh reaps are visible to it
    const bundles = this.kick.update(fr);
    const falls = this.fall.update(fr);
    const downs = falls.filter((e) => e.type === 'down');
    const throws = this.throw.resolve(fr, downs); // flags downs as thrown

    for (const b of bundles) {
      const reap = this.throw.reaps.find((r) => r.by === b.by && fr.t - r.t < this.cfg.reap.linkMs);
      if (reap) this.held.push({ ...b, since: fr.t, until: reap.t + REAP_HOLD_MS });
      else events.push(...b.events);
    }
    for (const h of this.held) if (throws.some((t) => t.by === h.by)) h.drop = true;
    this.held = this.held.filter((h) => {
      if (h.drop) return false;
      // released early when the opponent is still upright 0.5 s on: a real throw would have them tipping over by then
      const opp = this.fall.debug[h.by === 'a' ? 'b' : 'a'];
      const upright = opp && opp.phi < 20 && opp.hipDrop < 0.15;
      if (fr.t >= h.until || (fr.t - h.since >= EARLY_RELEASE_MS && upright)) {
        events.push(...h.events);
        return false;
      }
      return true;
    });

    events.push(...throws, ...falls, ...this.hold.update(fr), ...this.attack.update(fr), ...this.foul.update(fr));
    const debug = {
      punch: this.punch.debug,
      kick: this.kick.debug,
      fall: this.fall.debug,
      throw: this.throw.debug,
    };
    return { t: fr.t, people: fr.people, events, debug };
  }
}
