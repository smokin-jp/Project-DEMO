import test from 'node:test';
import assert from 'node:assert/strict';
import { Match } from '../src/engine.js';
import { RULESETS } from '../src/rules/index.js';
import { Pipeline } from '../src/pipeline.js';
import { defaultSpec, makeFrame, mergeSpec } from './fixtures/skeleton.js';

// Which penalties can the system call by itself (from a detection), and does each one really reach the scoreboard?
// Everything else is referee-override only. This test pins that list so it cannot silently drift.

const crit = (value, confidence = 0.9, detail = {}) => ({ value, confidence, detail });
const verdict = (over = {}) => ({
  strike: 'punch', strike_check: null, thrower: crit('a'), intended_target: crit('b'), contact: crit(true), contact_location: crit('head'),
  legal_area: crit(true), legal_glove_part: crit(true, 0.8, { part: 'knuckle_part' }), genuine_punch: crit(true, 0.8, { classification: 'clean' }),
  scored: true, review: false, confidence: 0.7, foul: null, summary: '', ...over,
});
const kick = (over = {}) => verdict({ strike: 'kick', strike_check: crit(true, 0.9), legal_glove_part: crit(true, 0.8, { part: 'foot_or_shin' }), ...over });
const sv = (v) => ({ type: 'strike_verdict', by: 'a', against: 'b', verdict: v, auto: true });
const started = (sport, variant) => { const m = new Match(RULESETS[sport], variant); m.start(); return m; };
const fouls = (m) => m.sides.a.penalties.map((p) => p.foul);

// [sport, variant, penalty id, how a detection produces it]
const AUTO = [
  ['boxing', 'amateur', 'low_blow', (m) => m.handle(sv(verdict({ scored: false, foul: 'low_blow' })))],
  ['boxing', 'amateur', 'elbow', (m) => m.handle(sv(verdict({ scored: false, foul: 'elbow' })))],
  ['boxing', 'amateur', 'open_glove', (m) => m.handle(sv(verdict({ scored: false, foul: 'open_glove' })))],
  ['boxing', 'amateur', 'back_head', (m) => m.handle(sv(verdict({ scored: false, foul: 'back_head' })))],
  ['boxing', 'amateur', 'hit_after_bell', (m) => { m.pause(); m.handle(sv(verdict())); }],
  ['boxing', 'amateur', 'holding', (m) => m.handle({ type: 'holding', by: 'a', auto: true })],
  ['boxing', 'amateur', 'wrestling', (m) => m.handle({ type: 'throw', by: 'a', against: 'b', technique: 'Osoto-gari', grade: 'ippon', auto: true })],
  ['boxing', 'amateur', 'kicking', (m) => m.handle(sv(kick()))],
  ['boxing', 'amateur', 'hitting_down', (m) => { m.handle({ type: 'down', by: 'b', clock: 1000, auto: true }); m.handle({ ...sv(verdict()), clock: 2000 }); }],
  ['boxing', 'amateur', 'headbutt', (m) => m.handle({ type: 'headbutt', by: 'a', auto: true })],
  ['boxing', 'amateur', 'ducking_low', (m) => m.handle({ type: 'low_head', by: 'a', auto: true })],
  ['judo', 'demo', 'leg_grab', (m) => m.handle({ type: 'leg_grab', by: 'a', auto: true })],
  ['kickboxing', 'unified', 'headbutt', (m) => m.handle({ type: 'headbutt', by: 'a', auto: true })],
  ['kickboxing', 'unified', 'leg_hold', (m) => m.handle({ type: 'leg_hold', by: 'a', auto: true })],
  ['judo', 'demo', 'strike', (m) => m.handle(sv(verdict()))],
  ['judo', 'demo', 'kicking', (m) => { m.handle(sv(kick())); m.tick(3100); }], // waits to see whether the opponent goes down (a throw)
  ['judo', 'demo', 'passivity', (m) => { for (let i = 0; i < 64; i++) { m.handle({ type: 'tracking', auto: true }); m.tick(500); } }],
  ['kickboxing', 'unified', 'low_blow', (m) => m.handle(sv(verdict({ scored: false, foul: 'low_blow' })))],
  ['kickboxing', 'unified', 'elbow', (m) => m.handle(sv(verdict({ scored: false, foul: 'elbow' })))],
  ['kickboxing', 'unified', 'open_glove', (m) => m.handle(sv(verdict({ scored: false, foul: 'open_glove' })))],
  ['kickboxing', 'unified', 'back_head', (m) => m.handle(sv(verdict({ scored: false, foul: 'back_head' })))],
  ['kickboxing', 'unified', 'knee_joint_kick', (m) => m.handle(sv(kick({ scored: false, foul: 'knee_joint_kick' })))],
  ['kickboxing', 'wako_fc', 'kick_below_waist', (m) => m.handle(sv(kick({ scored: false, foul: 'kick_below_waist' })))],
  ['kickboxing', 'wako_fc', 'illegal_knee', (m) => m.handle(sv(kick({ strike: 'knee' })))],
  ['kickboxing', 'unified', 'hit_after_bell', (m) => { m.pause(); m.handle(sv(kick())); }],
  ['kickboxing', 'unified', 'holding', (m) => m.handle({ type: 'holding', by: 'a', auto: true })],
  ['kickboxing', 'unified', 'hitting_down', (m) => { m.handle({ type: 'down', by: 'b', clock: 1000, auto: true }); m.handle({ ...sv(kick()), clock: 2000 }); }],
  ['kickboxing', 'unified', 'wrestling', (m) => m.handle({ type: 'throw', by: 'a', against: 'b', technique: 'Osoto-gari', grade: 'ippon', auto: true })],
];

for (const [sport, variant, id, trigger] of AUTO) {
  test(`AUTO ${sport}: ${id} is called from a detection and reaches the scoreboard`, () => {
    const m = started(sport, variant);
    trigger(m);
    assert.ok(fouls(m).includes(id), `${id} not recorded (got: ${fouls(m).join(',') || 'nothing'})`);
  });
}

test('the penalties NOT in that list really are manual-only (no detection can produce them)', () => {
  const autoIds = (sport) => new Set(AUTO.filter((a) => a[0] === sport).map((a) => a[2]));
  const manual = {};
  for (const [sport, rules] of Object.entries(RULESETS)) manual[sport] = Object.keys(rules.fouls).filter((f) => !autoIds(sport).has(f));
  assert.deepEqual(manual.boxing.sort(), ['flagrant', 'mouthguard', 'unsportsmanlike']);
  assert.deepEqual(manual.judo.sort(), ['dangerous_throw', 'defensive', 'disregard', 'false_attack', 'illegal_grip', 'stepping_out']);
  assert.deepEqual(manual.kickboxing.sort(), ['abusive', 'delay', 'disobey', 'flagrant', 'ropes', 'throat', 'thumbing', 'timidity', 'unsportsmanlike']);
  // ...and each of them still works when the referee issues it by hand
  for (const [sport, ids] of Object.entries(manual)) {
    for (const id of ids) {
      const m = started(sport, sport === 'boxing' ? 'amateur' : sport === 'judo' ? 'demo' : 'unified');
      m.handle({ type: 'foul', by: 'a', foul: id });
      assert.equal(m.sides.a.penalties.length, 1, `${sport}/${id} not accepted from the override panel`);
    }
  }
});

test('the holding detector fires for a sustained clinch (and not for fighters at range)', () => {
  const run = (gapX) => {
    const pl = new Pipeline();
    const events = [];
    for (let i = 0; i < 130; i++) {
      const t = (i / 30) * 1000;
      events.push(...pl.process(makeFrame({ a: mergeSpec(defaultSpec(-gapX, 1)), b: mergeSpec(defaultSpec(gapX, -1)) }, t, { noise: 0 })).events);
    }
    return events.filter((e) => e.type === 'holding');
  };
  const clinch = run(0.2); // hips 0.4 m apart (< 1.1 torso lengths) for 4+ s
  assert.equal(clinch.length, 1, 'one holding call for one clinch');
  assert.ok(clinch[0].ms >= 2800);
  assert.equal(run(0.45).length, 0, 'fighters 0.9 m apart are not holding');
});
