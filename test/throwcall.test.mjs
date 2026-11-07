import test from 'node:test';
import assert from 'node:assert/strict';
import { gradeThrow, decideThrow, THROW_MIN_CONFIDENCE } from '../src/throwgrade.js';
import { cloneConfig } from '../src/config.js';
import { Match } from '../src/engine.js';
import { RULESETS } from '../src/rules/index.js';

const cfg = cloneConfig().reap;
const down = (over = {}) => ({ type: 'down', by: 'b', phi: 85, speed: 3, durationMs: 700, backwards: true, ...over });
const result = (technique = 'Uchi-mata', confidence = 0.8) => ({ technique, confidence, probs: { [technique]: confidence } });

test('landing grades: flat and fast = ippon, less = waza-ari, seated = yuko, landing face-down = one level lower', () => {
  assert.equal(gradeThrow(down(), cfg), 'ippon');
  assert.equal(gradeThrow(down({ speed: 0.8 }), cfg), 'waza-ari'); // on the back but without force
  assert.equal(gradeThrow(down({ phi: 40, speed: 2 }), cfg), 'yuko');
  assert.equal(gradeThrow(down({ backwards: false }), cfg), 'waza-ari');
  assert.equal(gradeThrow(down({ phi: 60, backwards: false }), cfg), 'yuko');
});

test('a confident model answer for a real-looking throw becomes a scored throw by the OTHER fighter', () => {
  const d = decideThrow({ down: down({ by: 'b' }), minPair: 1.2, result: result('Uchi-mata', 0.82), gradeCfg: cfg });
  assert.equal(d.kind, 'throw');
  assert.deepEqual([d.event.by, d.event.against, d.event.technique, d.event.grade, d.event.via], ['a', 'b', 'Uchi-mata', 'ippon', 'video-model']);
  assert.match(d.note, /Uchi-mata \(82%\)/);
  assert.equal(decideThrow({ down: down({ by: 'a' }), minPair: null, result: result(), gradeCfg: cfg }).event.by, 'b');
});

test('things that are not throws are never scored, whatever the model says (it has no "not a throw" class)', () => {
  const base = { result: result('Osoto-gari', 0.99), gradeCfg: cfg };
  assert.equal(decideThrow({ ...base, down: down({ thrown: true }), minPair: 1 }).kind, 'ignore'); // the reap rule already scored it
  assert.equal(decideThrow({ ...base, down: down(), minPair: 5 }).kind, 'ignore'); // fighters far apart: a slip or a knockdown
  assert.equal(decideThrow({ ...base, down: down({ speed: 0.4 }), minPair: 1 }).kind, 'ignore'); // gentle sit-down
  assert.equal(decideThrow({ ...base, down: down({ phi: 30 }), minPair: 1 }).kind, 'ignore'); // not flat enough
  assert.equal(decideThrow({ down: down(), minPair: 1, result: null, gradeCfg: cfg }).kind, 'ignore'); // no model
});

test('an unsure model scores nothing but says what it saw; both fighters never tracked together still counts as possible', () => {
  const u = decideThrow({ down: down(), minPair: 1, result: result('O-goshi', THROW_MIN_CONFIDENCE - 0.05), gradeCfg: cfg });
  assert.equal(u.kind, 'unsure');
  assert.match(u.note, /nothing scored/);
  assert.equal(decideThrow({ down: down(), minPair: null, result: result('O-goshi', 0.7), gradeCfg: cfg }).kind, 'throw');
});

test('end to end in the judo rules: the video model\'s throw scores, names the technique, and a waza-ari pair ends the bout', () => {
  const m = new Match(RULESETS.judo, 'demo');
  m.start();
  const d = decideThrow({ down: down({ speed: 1.1 }), minPair: 1, result: result('Ippon-seoi-nage', 0.7), gradeCfg: cfg }); // firm but below ippon speed (1.3)
  m.handle(d.event);
  assert.match(m.log[0].text, /WAZA-ARI \(Ippon-seoi-nage\) to RED/);
  m.handle(d.event);
  assert.equal(m.status, 'ended');
  assert.equal(m.method, 'IPPON'); // waza-ari awasete ippon
});
