import { Pipeline } from '../src/pipeline.js';
import { SCRIPTS, renderScript, mergeSpec, defaultSpec, makeFrame } from './fixtures/skeleton.js';
import test from 'node:test';
import assert from 'node:assert/strict';

// deterministic noise
function lcg(seed = 1) {
  let s = seed;
  return () => ((s = (s * 1664525 + 1013904223) % 4294967296) / 4294967296);
}

function run(script, opts = {}) {
  const pl = new Pipeline();
  const frames = renderScript(script, { rnd: lcg(7), ...opts });
  const evs = [];
  for (const f of frames) for (const e of pl.process(f).events) evs.push({ ...e, at: f.t });
  return evs;
}

const of = (evs, type) => evs.filter((e) => e.type === type);

test('boxing: jab->head, body shot->torso, low punch->low', () => {
  const evs = run(SCRIPTS.boxing);
  const landed = of(evs, 'strike_landed');
  console.log('boxing landed:', landed.map((e) => `${e.by}:${e.target}@${e.at}`).join(' | '));
  assert.equal(landed.length, 3);
  assert.deepEqual(landed.map((e) => e.target), ['head', 'torso', 'low']);
  assert.ok(landed.every((e) => e.by === 'a' && e.against === 'b' && !e.virtual));
  assert.equal(of(evs, 'throw').length, 0);
});

test('boxing solo (opponent not visible): virtual opponent scoring', () => {
  const evs = run(SCRIPTS.boxing, { only: ['a'] });
  const landed = of(evs, 'strike_landed');
  console.log('solo landed:', landed.map((e) => e.target).join(','));
  assert.equal(landed.length, 3);
  assert.ok(landed.every((e) => e.virtual));
});

test('judo: osoto-gari detected as ippon by Red on Blue', () => {
  const evs = run(SCRIPTS.osoto);
  const th = of(evs, 'throw');
  console.log('throw:', JSON.stringify(th), 'downs:', JSON.stringify(of(evs, 'down')));
  assert.equal(th.length, 1);
  assert.equal(th[0].by, 'a');
  assert.equal(th[0].against, 'b');
  assert.equal(th[0].technique, 'Osoto-gari');
  assert.equal(th[0].grade, 'ippon');
  assert.equal(of(evs, 'strike_landed').length, 0);
});

test('judo: a fall WITHOUT a reap is a plain "down", not a throw', () => {
  // remove the reap by pinning tori's rear ankle
  const base = SCRIPTS.osoto;
  const script = {
    ...base,
    at: (t) => {
      const s = base.at(t);
      s.a = mergeSpec(s.a, { ra: { x: -0.15, y: 0.86 }, la: { x: 0.15, y: 0.86 } });
      return s;
    },
  };
  const evs = run(script);
  assert.equal(of(evs, 'throw').length, 0);
  assert.equal(of(evs, 'down').length, 1);
});

test('idle fighters standing still produce no events', () => {
  const pl = new Pipeline();
  const evs = [];
  const rnd = lcg(3);
  for (let i = 0; i < 150; i++) {
    const t = (i / 30) * 1000;
    evs.push(...pl.process(makeFrame({ a: defaultSpec(-0.4, 1), b: defaultSpec(0.4, -1) }, t, { rnd })).events);
  }
  assert.deepEqual(evs.filter((e) => e.type !== 'holding' && e.type !== 'tracking'), []); // the heartbeat is expected; jitter must never look like an attack
});

/* ---------------- kicks and knees ---------------- */
test('kickboxing: a body kick is detected once and lands on Blue\'s torso', () => {
  const evs = run(SCRIPTS.kick);
  const thrown = of(evs, 'kick_thrown');
  console.log('kick:', thrown.map((e) => `${e.by} ${e.strike} ${e.leg} v=${e.speed}`).join(' | '), '->', of(evs, 'kick_landed').map((e) => e.target).join(','));
  assert.equal(thrown.length, 1);
  assert.equal(thrown[0].by, 'a');
  assert.equal(thrown[0].strike, 'kick');
  assert.equal(thrown[0].leg, 'right');
  const landed = of(evs, 'kick_landed');
  assert.equal(landed.length, 1);
  assert.equal(landed[0].target, 'torso');
  assert.equal(of(evs, 'strike_landed').length, 0); // no punch events
});

test('kickboxing: a knee strike is detected as a knee, not a kick', () => {
  const evs = run(SCRIPTS.knee);
  const thrown = of(evs, 'kick_thrown');
  console.log('knee:', thrown.map((e) => `${e.by} ${e.strike} v=${e.speed}`).join(' | '));
  assert.equal(thrown.length, 1);
  assert.equal(thrown[0].strike, 'knee');
});

test('walking / shuffling never registers as a kick', () => {
  assert.equal(of(run(SCRIPTS.walk), 'kick_thrown').length, 0);
});

test('punching and throwing produce no kick events (reaping leg and flailing legs are not kicks)', () => {
  assert.equal(of(run(SCRIPTS.boxing), 'kick_thrown').length, 0);
  assert.equal(of(run(SCRIPTS.osoto), 'kick_thrown').length, 0);
  const base = SCRIPTS.osoto;
  const noReap = { ...base, at: (t) => { const s = base.at(t); s.a = mergeSpec(s.a, { ra: { x: -0.15, y: 0.86 }, la: { x: 0.15, y: 0.86 } }); return s; } };
  assert.equal(of(run(noReap), 'kick_thrown').length, 0); // uke's legs fly up as he falls
});
