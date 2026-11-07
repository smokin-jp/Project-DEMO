import test from 'node:test';
import assert from 'node:assert/strict';
import { Pipeline } from '../src/pipeline.js';
import { Match } from '../src/engine.js';
import { RULESETS } from '../src/rules/index.js';
import { defaultSpec, keyframes, makeFrame, mergeSpec } from './fixtures/skeleton.js';

// Pose-geometry fouls (src/detect/foul.js): headbutt, leg grab / hold, ducking below the belt.

const A0 = defaultSpec(-0.5, 1);
const B0 = defaultSpec(0.5, -1);

/** run a scripted pair through the whole pipeline, return every event */
function run(at, seconds, { fps = 30 } = {}) {
  const pl = new Pipeline();
  const events = [];
  for (let i = 0; i <= seconds * fps; i++) {
    const t = i / fps;
    events.push(...pl.process(makeFrame(at(t), t * 1000, { noise: 0 })).events);
  }
  return events;
}
const of = (events, type) => events.filter((e) => e.type === type);

// ------------------------------------------------------------------------------------------------ headbutt
test('a fast head thrust that meets the opponent\'s head is a headbutt, by the fighter who thrust', () => {
  const at = keyframes(A0, B0, [{ t: 0 }, { t: 1.0 }, { t: 1.15, a: { head: { f: 0.45 }, ang: 10 } }, { t: 1.6, a: { head: { f: 0.45 }, ang: 10 } }, { t: 2.2 }]);
  const hb = of(run(at, 2.2), 'headbutt');
  assert.equal(hb.length, 1, 'one call for one butt');
  assert.deepEqual([hb[0].by, hb[0].against], ['a', 'b']);
});

test('walking into each other, a slow lean-in, a thrust that stops short, and a bare punch are not headbutts', () => {
  const walk = keyframes(A0, B0, [{ t: 0 }, { t: 1.0 }, { t: 1.6, a: { hip: { x: -0.25 } }, b: { hip: { x: 0.25 } } }, { t: 2.4 }]);
  assert.equal(of(run(walk, 2.4), 'headbutt').length, 0, 'closing by walking: heads move with the bodies, no neck thrust');
  const slow = keyframes(A0, B0, [{ t: 0 }, { t: 1.0 }, { t: 2.2, a: { head: { f: 0.45 }, ang: 10 } }, { t: 3 }]);
  assert.equal(of(run(slow, 3), 'headbutt').length, 0, 'a slow lean is not a thrust');
  const short = keyframes(A0, B0, [{ t: 0 }, { t: 1.0 }, { t: 1.15, a: { head: { f: 0.25 } } }, { t: 2 }]);
  assert.equal(of(run(short, 2), 'headbutt').length, 0, 'a thrust that never reaches the other head');
  const punch = keyframes(A0, B0, [{ t: 0 }, { t: 1.0 }, { t: 1.12, a: { lw: { f: 0.64, u: 0.15 }, ang: 8 } }, { t: 1.5 }]);
  assert.equal(of(run(punch, 1.5), 'headbutt').length, 0, 'a jab is not a headbutt');
});

test('someone already on the mat is never called for a headbutt', () => {
  const down = mergeSpec(A0, { ang: 80, hip: { x: -0.3, y: 0.6 } });
  const at = keyframes(down, B0, [{ t: 0 }, { t: 1.0 }, { t: 1.15, a: { head: { f: 0.45 } } }, { t: 2 }]);
  assert.equal(of(run(at, 2), 'headbutt').length, 0);
});

// ------------------------------------------------------------------------------------------------ legs
// A bends right over and keeps a hand on B's thigh (below the belt line)
const GRAB = { hip: { x: -0.1, y: 0.05 }, ang: 70, lw: { f: 0.365, u: -0.079, z: -0.12 } };

test('a hand kept on the opponent\'s leg: leg_grab after 0.7 s, leg_hold after 2.5 s, each once', () => {
  const at = keyframes(A0, B0, [{ t: 0 }, { t: 1.0 }, { t: 1.4, a: GRAB }, { t: 5.5 }]);
  const ev = run(at, 5.5);
  const g = of(ev, 'leg_grab'), h = of(ev, 'leg_hold');
  assert.equal(g.length, 1);
  assert.equal(h.length, 1);
  assert.deepEqual([g[0].by, g[0].against], ['a', 'b']);
  assert.ok(g[0].ms >= 700 && g[0].ms < 1200, `grab called after ${g[0].ms} ms`);
  assert.ok(h[0].ms >= 2500, `hold called after ${h[0].ms} ms`);
});

test('a brief touch of the leg (a caught kick) is a grab but never a hold; hands on the chest are neither', () => {
  const brief = keyframes(A0, B0, [{ t: 0 }, { t: 1.0 }, { t: 1.3, a: GRAB }, { t: 2.4, a: GRAB }, { t: 2.8, a: mergeSpec(A0, {}) }, { t: 4 }]);
  const ev = run(brief, 4);
  assert.equal(of(ev, 'leg_hold').length, 0);
  const guard = keyframes(A0, B0, [{ t: 0 }, { t: 4 }]);
  const ev2 = run(guard, 4);
  assert.equal(of(ev2, 'leg_grab').length + of(ev2, 'leg_hold').length, 0, 'normal guard');
});

test('ducking: head kept below the opponent\'s belt line for 0.7 s is called; standing is not', () => {
  const duck = keyframes(A0, B0, [{ t: 0 }, { t: 1.0 }, { t: 1.4, a: { hip: { x: -0.3, y: 0.45 }, ang: 60 } }, { t: 3.5 }]);
  const lo = of(run(duck, 3.5), 'low_head');
  assert.equal(lo.length, 1);
  assert.equal(lo[0].by, 'a');
  assert.equal(of(run(keyframes(A0, B0, [{ t: 0 }, { t: 3 }]), 3), 'low_head').length, 0);
});

// ------------------------------------------------------------------------------------------------ rules
const started = (sport, variant) => { const m = new Match(RULESETS[sport], variant); m.start(); return m; };
const fouls = (m, side = 'a') => m.sides[side].penalties.map((p) => p.foul);

test('each sport turns the detections into its own penalty and ignores the ones that mean nothing there', () => {
  const ev = (type) => ({ type, by: 'a', against: 'b', auto: true });
  const box = started('boxing', 'amateur');
  for (const t of ['headbutt', 'low_head', 'leg_grab', 'leg_hold']) box.handle(ev(t));
  assert.deepEqual(fouls(box), ['headbutt', 'ducking_low'], 'boxing: headbutt and ducking low; leg holds are not a boxing foul');

  const kb = started('kickboxing', 'unified');
  for (const t of ['headbutt', 'low_head', 'leg_grab', 'leg_hold']) kb.handle(ev(t));
  assert.deepEqual(fouls(kb), ['headbutt', 'leg_hold'], 'kickboxing: headbutt and a held leg');

  const ju = started('judo', 'demo');
  for (const t of ['headbutt', 'low_head', 'leg_hold', 'leg_grab']) ju.handle(ev(t));
  assert.deepEqual(fouls(ju), ['leg_grab'], 'judo: only the leg grab (shido)');
  assert.equal(ju.sides.a.shido, 1);
});

test('an AI foul call can be withdrawn with Undo', () => {
  const m = started('judo', 'demo');
  m.handle({ type: 'leg_grab', by: 'a', against: 'b', auto: true });
  assert.equal(m.sides.a.shido, 1);
  assert.ok(m.undoLastPenalty());
  assert.equal(m.sides.a.shido, 0);
});

// ------------------------------------------------------------------------------------------------ judo: a reap is not a kick
const crit = (value, confidence = 0.9, detail = {}) => ({ value, confidence, detail });
const kickVerdict = (loc = 'torso', over = {}) => ({
  strike: 'kick', strike_check: crit(true), thrower: crit('a'), intended_target: crit('b'), contact: crit(true), contact_location: crit(loc),
  legal_area: crit(true), legal_glove_part: crit(true), genuine_punch: crit(true, 0.8, { classification: 'clean' }),
  scored: true, review: false, confidence: 0.7, foul: null, summary: '', ...over,
});
const kickEv = (loc, over) => ({ type: 'strike_verdict', by: 'a', against: 'b', verdict: kickVerdict(loc, over), auto: true });
const hmFor = (m) => m.sides.a.penalties.some((p) => p.foul === 'kicking');

test('judo: a real kick to the body with no fall is still hansoku-make, after a short wait', () => {
  const m = started('judo', 'demo');
  m.handle(kickEv('torso'));
  assert.equal(hmFor(m), false, 'not called instantly: it waits to see whether this was a throw');
  m.tick(3100);
  assert.equal(hmFor(m), true);
  assert.equal(m.status, 'ended');
});

test('judo: Osoto-gari style leg action is NOT hansoku-make (opponent goes down after it, before the clip is judged, or on the legs)', () => {
  const after = started('judo', 'demo'); // verdict first, fall a moment later
  after.handle(kickEv('torso'));
  after.tick(1200);
  after.handle({ type: 'down', by: 'b', auto: true });
  after.tick(3500);
  assert.equal(hmFor(after), false);
  assert.equal(after.status, 'running');

  const before = started('judo', 'demo'); // the fall was already seen when the slow verdict arrives
  before.handle({ type: 'down', by: 'b', auto: true });
  before.tick(1500);
  before.handle(kickEv('torso'));
  before.tick(3500);
  assert.equal(hmFor(before), false);

  const legs = started('judo', 'demo'); // reaping leg hits the opponent's legs: a sweep
  legs.handle(kickEv('legs'));
  legs.tick(3500);
  assert.equal(hmFor(legs), false);
  const knee = started('judo', 'demo');
  knee.handle(kickEv('knee_joint'));
  knee.tick(3500);
  assert.equal(hmFor(knee), false);
});

test('judo: a leg swing that does not touch the opponent, or a step, is not hansoku-make; a scored throw clears it too', () => {
  const miss = started('judo', 'demo');
  miss.handle(kickEv('none', { contact: crit(false) }));
  miss.tick(3500);
  assert.equal(hmFor(miss), false, 'a missed reap');
  const step = started('judo', 'demo');
  step.handle(kickEv('torso', { strike_check: crit(false) }));
  step.tick(3500);
  assert.equal(hmFor(step), false, 'not a real kick');
  const thrown = started('judo', 'demo');
  thrown.handle(kickEv('torso'));
  thrown.tick(500);
  thrown.handle({ type: 'throw', by: 'a', against: 'b', technique: 'Osoto-gari', grade: 'waza-ari', auto: true });
  thrown.tick(3500);
  assert.equal(hmFor(thrown), false);
  assert.equal(thrown.sides.a.wazaari, 1);
});

test('judo, backend offline: the geometry kick fallback follows the same rules', () => {
  const ev = (target) => ({ type: 'kick_landed', strike: 'kick', by: 'a', against: 'b', target, auto: true });
  const body = started('judo', 'demo');
  body.handle(ev('torso'));
  body.tick(3100);
  assert.equal(hmFor(body), true);
  const leg = started('judo', 'demo');
  leg.handle(ev('leg'));
  leg.tick(3100);
  assert.equal(hmFor(leg), false);
  const thrown = started('judo', 'demo');
  thrown.handle(ev('torso'));
  thrown.handle({ type: 'down', by: 'b', auto: true });
  thrown.tick(3100);
  assert.equal(hmFor(thrown), false);
});

test('end to end: a judo leg grab in the video becomes a shido on the scoreboard', () => {
  const pl = new Pipeline();
  const m = started('judo', 'demo');
  const at = keyframes(A0, B0, [{ t: 0 }, { t: 1.0 }, { t: 1.4, a: GRAB }, { t: 3 }]);
  for (let i = 0; i <= 90; i++) {
    const t = i / 30;
    for (const e of pl.process(makeFrame(at(t), t * 1000, { noise: 0 })).events) m.handle({ ...e, auto: true });
  }
  assert.equal(m.sides.a.shido, 1);
  assert.equal(m.sides.b.shido, 0);
});
