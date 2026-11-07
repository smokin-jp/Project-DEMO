// Sport-agnostic match engine. A ruleset (src/rules/*.js) supplies the variants, the foul table
// and handle()/onTimeUp() logic; the engine owns clock, status, penalties, log and notifications.
//
// Detector events (from pipeline.js) are interpreted *by the ruleset*, so the same physical
// action means different things per sport: e.g. a thrown opponent is a Judo score but a
// boxing / kickboxing foul (throwing); a landed punch is a boxing point but a
// disqualification in Judo.

export class Match {
  constructor(rules, variantId) {
    this.rules = rules;
    this.listeners = new Set();
    this.setup(variantId ?? rules.defaultVariant);
  }

  setup(variantId) {
    this.variantId = variantId in this.rules.variants ? variantId : this.rules.defaultVariant;
    this.cfg = this.rules.variants[this.variantId];
    this.status = 'ready'; // ready | running | paused | rest | counting | ended
    this.round = 1;
    this.now = 0;
    this.golden = false;
    this.clock = this.cfg.roundSec * 1000;
    this.sides = { a: this.newSide('RED'), b: this.newSide('BLUE') };
    this.log = [];
    this.winner = null;
    this.method = null;
    this.timers = [];
    this.nextTimer = 1;
    this.curAuto = false;
    this.startedAt = Date.now(); // wall-clock start of this bout (the report only includes strikes judged since then)
    this.undoStack = []; // state before each penalty, so a referee can withdraw a wrong automatic call
    this.rules.init?.(this);
    this.emit({ kind: 'state' });
  }

  newSide(name) {
    return { name, penalties: [], ...this.rules.sideDefaults() };
  }

  reset() {
    this.setup(this.variantId);
  }

  on(fn) {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  emit(ev) {
    for (const fn of this.listeners) fn(ev, this);
  }

  other(s) {
    return s === 'a' ? 'b' : 'a';
  }

  clockText() {
    const ms = Math.max(0, this.clock);
    const sec = this.golden ? Math.floor(ms / 1000) : Math.ceil(ms / 1000);
    return `${Math.floor(sec / 60)}:${String(sec % 60).padStart(2, '0')}`;
  }

  /** announce a call: logged, shown as a banner and (optionally) spoken */
  say(text, { kind = 'info', side = null, big = null, speak = null, auto = this.curAuto, evidence = null } = {}) {
    const entry = { t: this.now, clock: this.clockText(), round: this.round, text, kind, side, auto, evidence };
    this.log.unshift(entry);
    if (this.log.length > 200) this.log.pop();
    this.emit({ kind: 'call', entry, big, speak: speak ?? (big ? text : null) });
  }

  start() {
    if (this.status === 'ready' || this.status === 'paused') {
      const first = this.status === 'ready';
      this.status = 'running';
      this.say(first ? `${this.rules.name}: match started (${this.cfg.label})` : 'Resumed', { kind: 'info', big: first ? 'FIGHT' : null });
      this.emit({ kind: 'state' });
    }
  }

  pause() {
    if (this.status === 'running') {
      this.status = 'paused';
      this.say('Stop — referee break', { kind: 'info', big: 'BREAK' });
      this.emit({ kind: 'state' });
    }
  }

  later(ms, fn) {
    const id = this.nextTimer++;
    this.timers.push({ id, at: this.now + ms, fn });
    return id;
  }

  cancel(id) {
    this.timers = this.timers.filter((t) => t.id !== id);
  }

  tick(dt) {
    this.now += dt;
    if (this.timers.length) {
      const due = this.timers.filter((t) => t.at <= this.now);
      this.timers = this.timers.filter((t) => t.at > this.now);
      due.forEach((t) => t.fn(this));
    }
    if (this.status === 'running') {
      if (this.golden) this.clock += dt;
      else {
        this.clock -= dt;
        if (this.clock <= 0) {
          this.clock = 0;
          this.rules.onTimeUp(this);
        }
      }
      if (this.status === 'running') this.autoCall(() => this.rules.onTick?.(this, dt)); // time-based rules (e.g. judo passivity)
    } else if (this.status === 'rest') {
      this.clock -= dt;
      if (this.clock <= 0) {
        this.clock = 0;
        this.rules.onRestOver?.(this);
      }
    } else if (this.status === 'counting') {
      this.rules.onCountTick?.(this, dt);
    } else {
      return;
    }
    this.emit({ kind: 'tick' });
  }

  /** feed a detector event or a manual referee action */
  handle(ev) {
    if (this.status === 'ended' || this.status === 'ready') return;
    this.curAuto = !!ev.auto; // calls made while handling an AI event are tagged "AI"
    try {
      this.rules.handle(ev, this);
    } finally {
      this.curAuto = false;
    }
    this.emit({ kind: 'state' });
  }

  penalize(side, foulId, ev = {}) {
    const f = this.rules.fouls[foulId];
    if (!f) return;
    const s = this.sides[side];
    let action;
    if (f.direct) action = f.direct;
    else {
      const ladder = this.cfg.ladder;
      const n = s.penalties.filter((p) => !p.direct).length;
      action = ladder[Math.min(n, ladder.length - 1)];
    }
    this.undoStack.push({ side, foul: foulId, label: f.label, action, snap: this.snapshot() });
    if (this.undoStack.length > 20) this.undoStack.shift();
    s.penalties.push({ foul: foulId, label: f.label, action, direct: !!f.direct, t: this.now });
    this.rules.applyPenalty(action, side, foulId, f, this, ev);
  }

  snapshot() {
    return JSON.parse(JSON.stringify({ sides: this.sides, status: this.status, winner: this.winner, method: this.method, count: this.count ?? null }));
  }

  /** the most recent penalty (that can still be withdrawn), or null */
  lastPenalty() {
    const u = this.undoStack[this.undoStack.length - 1];
    return u ? { side: u.side, foul: u.foul, label: u.label, action: u.action } : null;
  }

  /** withdraw the most recent penalty: restores scores, penalties and (if it ended the bout) the bout itself */
  undoLastPenalty() {
    const u = this.undoStack.pop();
    if (!u) return false;
    const s = u.snap;
    this.sides = s.sides;
    this.status = s.status;
    this.winner = s.winner;
    this.method = s.method;
    this.count = s.count;
    this.say(`Penalty withdrawn by the referee: ${u.label} (${this.sides[u.side].name})`, { kind: 'info', side: u.side, auto: false });
    this.emit({ kind: 'state' });
    return true;
  }

  /** run `fn` so that any call it makes is tagged "AI" (used for rules that fire on their own from the clock) */
  autoCall(fn) {
    const prev = this.curAuto;
    this.curAuto = true;
    try { fn(); } finally { this.curAuto = prev; }
  }

  end(winner, method, text, speak) {
    this.status = 'ended';
    this.winner = winner;
    this.method = method;
    this.say(text, { kind: 'result', side: winner, big: method, speak: speak ?? text });
    this.emit({ kind: 'state' });
  }
}
