import test from 'node:test';
import assert from 'node:assert/strict';
import { Match } from '../src/engine.js';
import { RULESETS } from '../src/rules/index.js';
import { Pipeline } from '../src/pipeline.js';
import { SCRIPTS, renderScript } from './fixtures/skeleton.js';

const crit = (value, confidence = 0.9, detail = {}) => ({ value, confidence, detail });
const verdict = (over = {}) => ({
  strike: 'punch', strike_check: null, thrower: crit('a'), intended_target: crit('b'), contact: crit(true), contact_location: crit('head'),
  legal_area: crit(true), legal_glove_part: crit(true, 0.8, { part: 'knuckle_part' }), genuine_punch: crit(true, 0.8, { classification: 'clean' }),
  scored: true, review: false, confidence: 0.7, foul: null, summary: '', ...over,
});
const kick = (over = {}) => verdict({ strike: 'kick', strike_check: crit(true, 0.9), legal_glove_part: crit(true, 0.8, { part: 'foot_or_shin' }), ...over });
// a verdict for a strike whose peak happened at detection-clock time `clock`
const sv = (v, clock, by = 'a') => ({ type: 'strike_verdict', by, against: by === 'a' ? 'b' : 'a', verdict: v, auto: true, clock });
const down = (by, clock) => ({ type: 'down', by, clock, auto: true });
const up = (by, clock) => ({ type: 'up', by, clock, auto: true });
const started = (sport, variant) => { const m = new Match(RULESETS[sport], variant); m.start(); return m; };
const fouls = (m, side = 'a') => m.sides[side].penalties.map((p) => p.foul);

/* ---------------- hitting a downed fighter ---------------- */
test('boxing: a strike AFTER the fighter is down is a foul; the punch that knocked him down is not', () => {
  const m = started('boxing', 'amateur');
  m.handle(sv(verdict(), 1000)); // the knockdown punch (peak 1000)
  m.tick(500);
  m.handle(down('b', 1900)); // he is on the mat by 1900 -> 8-count
  assert.equal(m.status, 'counting');
  // that first punch is only judged now, a second or so later: it happened BEFORE he went down, so no foul
  m.handle(sv(verdict(), 1000));
  assert.deepEqual(fouls(m), []);
  // a second punch, thrown at 2600 while he is down, is a foul and does not score
  const before = m.sides.a.punchesTotal;
  m.handle(sv(verdict(), 2600));
  assert.deepEqual(fouls(m), ['hitting_down']);
  assert.equal(m.sides.a.punchesTotal, before);
  // once he is up again, strikes count normally
  m.handle(up('b', 3000));
  m.tick(9000);
  assert.equal(m.status, 'running');
  m.handle(sv(verdict(), 4000));
  assert.equal(m.sides.a.punchesTotal, before + 1);
});

test('boxing: a punch that misses a downed fighter is not a foul; a strike during the fall (before he settles) is not either', () => {
  const m = started('boxing', 'amateur');
  m.handle(down('b', 2000));
  m.handle(sv(verdict({ contact: crit(false), scored: false }), 3000)); // thrown at him but missed
  assert.deepEqual(fouls(m), []);
  m.handle(sv(verdict(), 2100)); // within 150 ms of him settling: still part of the same exchange
  assert.deepEqual(fouls(m), []);
});

test('boxing: geometry-only fallback also catches a strike on a downed fighter', () => {
  const m = started('boxing', 'amateur');
  m.handle(down('b', 1000));
  m.handle({ type: 'strike_landed', by: 'a', against: 'b', target: 'head', peakT: 2000, auto: true, clock: 2300 });
  assert.deepEqual(fouls(m), ['hitting_down']);
});

test('kickboxing: kicks and knees on a downed fighter are fouls too (and the knockdown kick is not)', () => {
  const m = started('kickboxing', 'unified');
  m.handle(sv(kick(), 1000));
  m.handle(down('b', 1900));
  m.handle(sv(kick(), 1000)); // the kick that put him down, judged late: fine
  assert.deepEqual(fouls(m), []);
  m.handle(sv(kick({ strike: 'knee' }), 2600)); // a knee while he is down
  assert.deepEqual(fouls(m), ['hitting_down']);
  // a step is not a strike: never a foul
  m.handle(sv(kick({ strike_check: crit(false, 0.9) }), 2700));
  assert.deepEqual(fouls(m), ['hitting_down']);
});

/* ---------------- judo passivity (non-combativity, 30 s) ---------------- */
const beat = (m, ms) => { for (let t = 0; t < ms; t += 500) { m.handle({ type: 'tracking', auto: true }); m.tick(500); } };

test('judo: 30 s with no attack by either fighter = shido for both; attacks reset the timer', () => {
  const m = started('judo', 'demo');
  beat(m, 29000);
  assert.deepEqual([fouls(m).length, fouls(m, 'b').length], [0, 0]);
  beat(m, 2000);
  assert.deepEqual(fouls(m), ['passivity']);
  assert.deepEqual(fouls(m, 'b'), ['passivity']);
  assert.equal(m.sides.a.shido, 1);
  assert.ok(m.log.some((l) => l.auto && /SHIDO 1/.test(l.text)), 'called by the AI');

  // RED keeps attacking, BLUE does not: only BLUE is penalised
  const m2 = started('judo', 'demo');
  for (let i = 0; i < 70; i++) {
    if (i % 20 === 0) m2.handle({ type: 'attack', by: 'a', kind: 'step', auto: true });
    beat(m2, 500);
  }
  assert.deepEqual(fouls(m2), []);
  assert.deepEqual(fouls(m2, 'b'), ['passivity']);
});

test('judo: the passivity timer only runs while both fighters are standing and in view, and restarts after a throw', () => {
  const m = started('judo', 'demo');
  beat(m, 20000);
  m.tick(60000 > 0 ? 40000 : 0); // ground work / nobody in view for 40 s: no tracking events -> timer paused
  assert.deepEqual(fouls(m), []);
  beat(m, 5000);
  assert.deepEqual(fouls(m), [], 'still only 25 s of standing time');

  const t = started('judo', 'demo');
  beat(t, 25000);
  t.handle({ type: 'throw', by: 'a', against: 'b', technique: 'Osoto-gari', grade: 'yuko', auto: true });
  beat(t, 25000);
  assert.deepEqual(fouls(t), [], 'a throw counts as engaging: the 30 s start again');

  const never = started('judo', 'demo');
  never.tick(120000); // no detections at all: nothing is judged
  assert.deepEqual(fouls(never), []);
});

test('judo: three passivity calls in a row end the bout (3rd shido = hansoku-make)', () => {
  const m = started('judo', 'demo');
  beat(m, 31000);
  beat(m, 31000);
  beat(m, 31000);
  assert.equal(m.status, 'ended');
  assert.equal(m.method, 'HANSOKU-MAKE');
});

/* ---------------- the attack / tracking detector ---------------- */
function events(script, opts = {}) {
  const pl = new Pipeline();
  let s = 3;
  const rnd = () => ((s = (s * 1664525 + 1013904223) % 4294967296) / 4294967296);
  const out = [];
  for (const f of renderScript(script, { rnd, ...opts })) for (const e of pl.process(f).events) out.push({ ...e, at: f.t });
  return out;
}

test('attack detector: a step in is an attack; standing, shuffling feet and jitter are not; tracking beats while both stand', () => {
  const osoto = events(SCRIPTS.osoto);
  const atk = osoto.filter((e) => e.type === 'attack');
  assert.ok(atk.length >= 1 && atk[0].by === 'a', 'tori stepping in is an attack');
  assert.ok(osoto.filter((e) => e.type === 'tracking').length >= 3, 'heartbeat while they stand');
  assert.equal(events(SCRIPTS.walk).filter((e) => e.type === 'attack').length, 0, 'shuffling feet in place is not an attack');
  assert.equal(events(SCRIPTS.boxing).filter((e) => e.type === 'attack' && e.by === 'b').length, 0, 'the fighter who does not move never attacks');
  // once uke is on the mat the standing-contest heartbeat stops
  const last = osoto.filter((e) => e.type === 'tracking').pop();
  assert.ok(last.at < 2600, 'no tracking after the fall (at ' + Math.round(last.at) + ' ms)');
});

/* ---------------- undo last penalty ---------------- */
test('undo: withdraws the latest penalty and restores the score and the bout', () => {
  const m = started('boxing', 'amateur');
  m.handle({ type: 'foul', by: 'a', foul: 'holding' }); // caution
  m.handle({ type: 'foul', by: 'a', foul: 'holding' }); // warning (-1)
  assert.equal(m.sides.a.deductTotal, 1);
  assert.deepEqual(m.lastPenalty(), { side: 'a', foul: 'holding', label: 'Holding / clinching', action: 'warning' });
  assert.equal(m.undoLastPenalty(), true);
  assert.equal(m.sides.a.deductTotal, 0);
  assert.equal(fouls(m).length, 1);
  assert.match(m.log[0].text, /withdrawn/);
  assert.equal(m.undoLastPenalty(), true); // and the caution
  assert.equal(fouls(m).length, 0);
  assert.equal(m.undoLastPenalty(), false, 'nothing left to undo');
});

test('undo: a wrong disqualification can be reversed and the bout continues', () => {
  const m = started('judo', 'demo');
  m.handle({ type: 'foul', by: 'a', foul: 'strike' }); // direct hansoku-make
  assert.equal(m.status, 'ended');
  assert.equal(m.winner, 'b');
  m.undoLastPenalty();
  assert.equal(m.status, 'running');
  assert.equal(m.winner, null);
  assert.equal(m.sides.a.shido, 0);
  assert.equal(fouls(m).length, 0);
  m.handle({ type: 'throw', by: 'a', against: 'b', technique: 'Osoto-gari', grade: 'ippon', auto: true });
  assert.equal(m.winner, 'a', 'play resumed normally');
});

test('undo: a shido from the passivity timer can be withdrawn (the timer keeps its own count)', () => {
  const m = started('judo', 'demo');
  beat(m, 31000);
  assert.equal(m.sides.a.shido, 1);
  m.undoLastPenalty(); // BLUE's shido
  m.undoLastPenalty(); // RED's shido
  assert.deepEqual([m.sides.a.shido, m.sides.b.shido], [0, 0]);
});
