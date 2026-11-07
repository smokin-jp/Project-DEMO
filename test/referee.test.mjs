import { Pipeline } from '../src/pipeline.js';
import { Match } from '../src/engine.js';
import { RULESETS } from '../src/rules/index.js';
import { SCRIPTS, renderScript } from './fixtures/skeleton.js';
import test from 'node:test';
import assert from 'node:assert/strict';

function lcg(seed = 1) {
  let s = seed;
  return () => ((s = (s * 1664525 + 1013904223) % 4294967296) / 4294967296);
}

/** run a script through detectors -> a referee for the given sport */
function officiate(sport, scriptName, { variant, only, extraSeconds = 0 } = {}) {
  const m = new Match(RULESETS[sport], variant ?? 'demo');
  m.start();
  const pl = new Pipeline();
  const script = SCRIPTS[scriptName];
  const frames = renderScript(script, { rnd: lcg(7), only });
  let prevT = 0;
  for (const f of frames) {
    m.tick(f.t - prevT);
    prevT = f.t;
    for (const e of pl.process(f).events) m.handle({ ...e, auto: true });
  }
  for (let i = 0; i < extraSeconds * 30; i++) m.tick(33.3);
  return m;
}

const texts = (m) => m.log.map((l) => l.text).reverse();

test('boxing: jab + body shot score, low blow is cautioned', () => {
  const m = officiate('boxing', 'boxing');
  console.log(texts(m).join('\n'));
  assert.equal(m.sides.a.punchesTotal, 2);
  assert.equal(m.sides.a.penalties.length, 1);
  assert.equal(m.sides.a.penalties[0].foul, 'low_blow');
  assert.equal(m.sides.a.penalties[0].action, 'caution');
  assert.equal(m.status, 'running');
});

test('boxing: three illegal acts escalate to DQ', () => {
  const m = new Match(RULESETS.boxing, 'amateur');
  m.start();
  for (let i = 0; i < 3; i++) m.handle({ type: 'foul', by: 'b', foul: 'holding' });
  assert.equal(m.status, 'running');
  m.handle({ type: 'foul', by: 'b', foul: 'low_blow' });
  assert.equal(m.status, 'ended');
  assert.equal(m.method, 'DQ');
  assert.equal(m.winner, 'a');
});

test('boxing: flagrant foul = immediate DQ', () => {
  const m = new Match(RULESETS.boxing, 'amateur');
  m.start();
  m.handle({ type: 'foul', by: 'a', foul: 'flagrant' });
  assert.equal(m.winner, 'b');
});

test('boxing: rounds, rest, 10-point-must scoring and decision', () => {
  const m = new Match(RULESETS.boxing, 'demo');
  m.start();
  for (let i = 0; i < 5; i++) m.handle({ type: 'strike_landed', by: 'a', against: 'b', target: 'head' });
  m.handle({ type: 'strike_landed', by: 'b', against: 'a', target: 'torso' });
  m.tick(46000);
  assert.equal(m.status, 'rest');
  assert.deepEqual([m.sides.a.score, m.sides.b.score], [10, 9]);
  m.handle({ type: 'strike_landed', by: 'a', against: 'b', target: 'head' }); // after the bell
  assert.equal(m.sides.a.penalties[0].foul, 'hit_after_bell');
  m.tick(9000);
  assert.equal(m.status, 'running');
  assert.equal(m.round, 2);
  m.handle({ type: 'strike_landed', by: 'b', against: 'a', target: 'head' });
  m.handle({ type: 'strike_landed', by: 'a', against: 'b', target: 'head' });
  m.handle({ type: 'strike_landed', by: 'a', against: 'b', target: 'head' });
  m.tick(46000);
  assert.equal(m.status, 'ended');
  assert.equal(m.method, 'DECISION');
  assert.equal(m.winner, 'a'); // 10-9 + 10-9
  assert.deepEqual([m.sides.a.score, m.sides.b.score], [20, 18]);
});

test('boxing: knockdown -> mandatory 8 count -> back up -> boxing resumes', () => {
  const m = new Match(RULESETS.boxing, 'amateur');
  m.start();
  m.handle({ type: 'strike_landed', by: 'a', against: 'b', target: 'head' });
  m.tick(500);
  m.handle({ type: 'down', by: 'b' });
  assert.equal(m.status, 'counting');
  m.tick(2000);
  m.handle({ type: 'up', by: 'b' });
  m.tick(5000);
  assert.equal(m.status, 'counting'); // still < 8
  m.tick(1500);
  assert.equal(m.status, 'running');
  assert.equal(m.sides.b.kdTotal, 1);
});

test('boxing: not up by 10 = KO; slip without a blow = no count', () => {
  const m = new Match(RULESETS.boxing, 'amateur');
  m.start();
  m.handle({ type: 'down', by: 'a' }); // slip
  assert.equal(m.status, 'running');
  m.handle({ type: 'strike_landed', by: 'b', against: 'a', target: 'head' });
  m.handle({ type: 'down', by: 'a' });
  m.tick(10500);
  assert.equal(m.method, 'KO');
  assert.equal(m.winner, 'b');
});

test('judo: osoto-gari ippon ends the match for Red', () => {
  const m = officiate('judo', 'osoto');
  console.log(texts(m).join('\n'));
  assert.equal(m.status, 'ended');
  assert.equal(m.method, 'IPPON');
  assert.equal(m.winner, 'a');
});

test('same throw in kickboxing = throwing foul (legal only in judo)', () => {
  const m = officiate('kickboxing', 'osoto', { variant: 'unified' });
  console.log(texts(m).join('\n'));
  assert.ok(m.sides.a.penalties.some((p) => p.foul === 'wrestling'));
  assert.equal(m.sides.a.punchesTotal, 0);
});

test('same throw in boxing = wrestling foul', () => {
  const m = officiate('boxing', 'osoto', { variant: 'amateur' });
  console.log(texts(m).join('\n'));
  assert.ok(m.sides.a.penalties.some((p) => p.foul === 'wrestling'));
  assert.equal(m.sides.a.punchesTotal, 0);
});

test('a punch in judo is a direct disqualification (hansoku-make)', () => {
  const m = officiate('judo', 'boxing');
  assert.equal(m.method, 'HANSOKU-MAKE');
  assert.equal(m.winner, 'b');
});

test('kickboxing: punches (jab/body/low blow) are scored like boxing', () => {
  const m = officiate('kickboxing', 'boxing', { variant: 'unified' });
  assert.equal(m.sides.a.punchesTotal, 2);
  assert.equal(m.sides.a.penalties[0].foul, 'low_blow');
  assert.equal(m.sides.a.penalties[0].action, 'caution'); // unified ladder starts with a caution
});

test('judo: 3 shidos = hansoku-make; two waza-ari = ippon; golden score', () => {
  let m = new Match(RULESETS.judo, 'demo');
  m.start();
  for (let i = 0; i < 3; i++) m.handle({ type: 'foul', by: 'a', foul: 'passivity' });
  assert.equal(m.method, 'HANSOKU-MAKE');
  assert.equal(m.winner, 'b');

  m = new Match(RULESETS.judo, 'demo');
  m.start();
  m.handle({ type: 'throw', by: 'b', against: 'a', technique: 'Osoto-gari', grade: 'waza-ari' });
  assert.equal(m.status, 'running');
  m.handle({ type: 'throw', by: 'b', against: 'a', technique: 'Osoto-gari', grade: 'waza-ari' });
  assert.equal(m.method, 'IPPON');

  m = new Match(RULESETS.judo, 'demo');
  m.start();
  m.tick(91000);
  assert.equal(m.golden, true);
  m.handle({ type: 'throw', by: 'a', against: 'b', technique: 'Osoto-gari', grade: 'yuko' });
  assert.equal(m.method, 'GOLDEN SCORE');
});

test('kickboxing: kicks / knees / leg kicks are awarded and variant rules apply', () => {
  const m = new Match(RULESETS.kickboxing, 'unified');
  m.start();
  m.handle({ type: 'score', by: 'a', kind: 'kick' });
  m.handle({ type: 'score', by: 'a', kind: 'leg_kick' });
  m.handle({ type: 'score', by: 'a', kind: 'knee' });
  assert.equal(m.sides.a.punchesTotal, 3);
  assert.match(m.log[0].text, /knee strike/);

  // WAKO Full Contact: above the waist only, no knees -> those are fouls, not points
  const fc = new Match(RULESETS.kickboxing, 'wako_fc');
  fc.start();
  fc.handle({ type: 'score', by: 'b', kind: 'leg_kick' });
  fc.handle({ type: 'score', by: 'b', kind: 'knee' });
  assert.equal(fc.sides.b.punchesTotal, 0);
  assert.deepEqual(fc.sides.b.penalties.map((p) => p.foul), ['kick_below_waist', 'illegal_knee']);

  // Low Kick: leg kicks are legal
  const lk = new Match(RULESETS.kickboxing, 'wako_lk');
  lk.start();
  lk.handle({ type: 'score', by: 'a', kind: 'leg_kick' });
  assert.equal(lk.sides.a.punchesTotal, 1);
});

test('kickboxing: foul ladder, flagrant DQ, 3 knockdowns = stoppage, pro draw', () => {
  const m = new Match(RULESETS.kickboxing, 'unified');
  m.start();
  m.handle({ type: 'foul', by: 'a', foul: 'elbow' });
  assert.equal(m.sides.a.deductTotal, 0); // warning only
  m.handle({ type: 'foul', by: 'a', foul: 'elbow' });
  assert.equal(m.sides.a.deductTotal, 1); // -1
  m.handle({ type: 'foul', by: 'a', foul: 'knee_joint_kick' });
  m.handle({ type: 'foul', by: 'a', foul: 'timidity' });
  assert.equal(m.method, 'DQ');
  assert.equal(m.winner, 'b');

  const f = new Match(RULESETS.kickboxing, 'unified');
  f.start();
  f.handle({ type: 'foul', by: 'b', foul: 'flagrant' });
  assert.equal(f.winner, 'a');

  const k = new Match(RULESETS.kickboxing, 'unified');
  k.start();
  for (let i = 0; i < 2; i++) {
    k.handle({ type: 'score', by: 'a', kind: 'knockdown' }); // referee-called knockdown of RED
    assert.equal(k.status, 'counting');
    k.handle({ type: 'up', by: 'a' });
    k.tick(8500); // mandatory 8-count, then fighting resumes
    assert.equal(k.status, 'running');
  }
  k.handle({ type: 'score', by: 'a', kind: 'knockdown' });
  assert.equal(k.method, 'RSC');
  assert.equal(k.winner, 'b');

  const d = new Match(RULESETS.kickboxing, 'demo');
  d.start();
  d.tick(46000);
  d.tick(9000);
  d.tick(46000);
  assert.equal(d.method, 'DRAW'); // pro/demo: level = draw
});

/* ---------------- CLIP backend verdicts ---------------- */
const crit = (value, confidence = 0.9, detail = {}) => ({ value, confidence, detail });
function verdict(over = {}) {
  return {
    thrower: crit('a'), intended_target: crit('b'), contact: crit(true), contact_location: crit('head'),
    legal_area: crit(true), legal_glove_part: crit(true, 0.8, { part: 'knuckle_part' }),
    genuine_punch: crit(true, 0.8, { classification: 'clean' }),
    strike: 'punch', strike_check: null,
    scored: true, review: false, confidence: 0.72, foul: null, summary: 'RED right hand -> BLUE: clean punch to the head (72%).', ...over,
  };
}
const pv = (v, by = 'a') => ({ type: 'strike_verdict', by, against: by === 'a' ? 'b' : 'a', verdict: v, auto: true });

test('boxing: backend verdict scoring / foul / review / no-score', () => {
  const m = new Match(RULESETS.boxing, 'amateur');
  m.start();
  m.handle(pv(verdict()));
  assert.equal(m.sides.a.punchesTotal, 1);
  assert.ok(m.log[0].auto && /clean punch to the head/.test(m.log[0].text));

  // blocked: contact but not a scoring punch -> no point
  m.handle(pv(verdict({ scored: false, genuine_punch: crit(false, 0.7, { classification: 'block' }), summary: 'blocked' })));
  assert.equal(m.sides.a.punchesTotal, 1);

  // borderline -> review, no point, no foul
  m.handle(pv(verdict({ scored: false, review: true })));
  assert.equal(m.sides.a.punchesTotal, 1);
  assert.match(m.log[0].text, /REVIEW/);

  // referee awards the reviewed punch
  m.handle({ type: 'score', by: 'a', kind: 'punch' });
  assert.equal(m.sides.a.punchesTotal, 2);

  // illegal contact -> foul from the verdict, not a point
  m.handle(pv(verdict({ scored: false, foul: 'low_blow', contact_location: crit('below_belt') })));
  assert.equal(m.sides.a.punchesTotal, 2);
  assert.equal(m.sides.a.penalties[0].foul, 'low_blow');
});

test('boxing: contact after the bell / during a break is a foul', () => {
  const m = new Match(RULESETS.boxing, 'amateur');
  m.start();
  m.pause();
  m.handle(pv(verdict()));
  assert.equal(m.sides.a.penalties[0].foul, 'hit_after_bell');
  assert.equal(m.sides.a.punchesTotal, 0);
});

test('kickboxing: backend verdicts behave like boxing (score, foul, review)', () => {
  const m = new Match(RULESETS.kickboxing, 'unified');
  m.start();
  m.handle(pv(verdict()));
  assert.equal(m.sides.a.punchesTotal, 1);
  m.handle(pv(verdict({ scored: false, foul: 'back_head', contact_location: crit('back') })));
  assert.equal(m.sides.a.penalties[0].foul, 'back_head');
  m.handle(pv(verdict({ scored: false, review: true })));
  assert.match(m.log[0].text, /REVIEW/);
});

test('judo: a punch that connects is a disqualification, a miss is not', () => {
  for (const [sport, method] of [['judo', 'HANSOKU-MAKE']]) {
    const miss = new Match(RULESETS[sport], 'demo');
    miss.start();
    miss.handle(pv(verdict({ contact: crit(false), genuine_punch: crit(false, 0.9, { classification: 'miss' }), scored: false })));
    assert.equal(miss.status, 'running', sport);
    const hit = new Match(RULESETS[sport], 'demo');
    hit.start();
    hit.handle(pv(verdict()));
    assert.equal(hit.method, method, sport);
    assert.equal(hit.winner, 'b', sport);
  }
});

/* ---------------- kicks and knees: DQ in boxing / judo, points in kickboxing ---------------- */
const kv = (over = {}) => verdict({
  strike: 'kick', strike_check: crit(true, 0.85, { p_real: 0.85 }), contact_location: crit('head'),
  legal_glove_part: crit(true, 0.8, { part: 'foot_or_shin' }), summary: 'RED right kick -> BLUE: clean kick to the head (72%).', ...over,
});
const kickLanded = (by = 'a', over = {}) => ({ type: 'kick_landed', strike: 'kick', by, against: by === 'a' ? 'b' : 'a', target: 'torso', auto: true, ...over });

test('BOXING: a confirmed kick or knee is a disqualification; a step is ignored', () => {
  for (const strike of ['kick', 'knee']) {
    const m = new Match(RULESETS.boxing, 'amateur');
    m.start();
    m.handle(pv(kv({ strike })));
    assert.equal(m.status, 'ended', strike);
    assert.equal(m.method, 'DQ', strike);
    assert.equal(m.winner, 'b', strike);
    assert.equal(m.sides.a.punchesTotal, 0);
  }
  const step = new Match(RULESETS.boxing, 'amateur');
  step.start();
  step.handle(pv(kv({ strike_check: crit(false, 0.9), scored: false, summary: 'not a real kick' })));
  assert.equal(step.status, 'running');
  assert.match(step.log[0].text, /not a real kick/);
  // backend offline: the geometry-only fallback still disqualifies a kick that lands
  const off = new Match(RULESETS.boxing, 'amateur');
  off.start();
  off.handle(kickLanded('b'));
  assert.equal(off.method, 'DQ');
  assert.equal(off.winner, 'a');
});

test('JUDO: a confirmed kick or knee is hansoku-make; a step is ignored', () => {
  for (const strike of ['kick', 'knee']) {
    const m = new Match(RULESETS.judo, 'demo');
    m.start();
    m.handle(pv(kv({ strike })));
    m.tick(3100); // judo waits to see whether the opponent goes down (a throw) before calling a kick
    assert.equal(m.method, 'HANSOKU-MAKE', strike);
    assert.equal(m.winner, 'b', strike);
  }
  const step = new Match(RULESETS.judo, 'demo');
  step.start();
  step.handle(pv(kv({ strike_check: crit(false, 0.9) })));
  assert.equal(step.status, 'running');
  const off = new Match(RULESETS.judo, 'demo');
  off.start();
  off.handle(kickLanded('a'));
  off.tick(3100);
  assert.equal(off.method, 'HANSOKU-MAKE');
});

test('KICKBOXING: a clean kick / leg kick / knee earns points, a step earns nothing', () => {
  const m = new Match(RULESETS.kickboxing, 'unified');
  m.start();
  m.handle(pv(kv()));
  assert.equal(m.sides.a.punchesTotal, 1);
  assert.match(m.log[0].text, /clean kick to the head/);
  m.handle(pv(kv({ contact_location: crit('legs') })));
  assert.match(m.log[0].text, /clean leg kick to the legs/);
  m.handle(pv(kv({ strike: 'knee', contact_location: crit('torso'), legal_glove_part: crit(true, 0.8, { part: 'knee' }) })));
  assert.match(m.log[0].text, /clean knee strike to the torso/);
  assert.equal(m.sides.a.punchesTotal, 3);
  assert.equal(m.status, 'running');

  m.handle(pv(kv({ strike_check: crit(false, 0.9), scored: false })));
  assert.equal(m.sides.a.punchesTotal, 3);
  // blocked / missed kicks do not score
  m.handle(pv(kv({ scored: false, genuine_punch: crit(false, 0.7, { classification: 'block' }) })));
  assert.equal(m.sides.a.punchesTotal, 3);
  // borderline kick is flagged for review
  m.handle(pv(kv({ scored: false, review: true })));
  assert.match(m.log[0].text, /REVIEW/);
});

test('KICKBOXING: kick fouls, variant rules (no leg kicks / knees in Full Contact), after-the-bell, offline fallback', () => {
  const m = new Match(RULESETS.kickboxing, 'unified');
  m.start();
  m.handle(pv(kv({ scored: false, foul: 'knee_joint_kick', contact_location: crit('knee_joint') })));
  assert.equal(m.sides.a.penalties[0].foul, 'knee_joint_kick');
  assert.equal(m.sides.a.punchesTotal, 0);

  const fc = new Match(RULESETS.kickboxing, 'wako_fc');
  fc.start();
  fc.handle(pv(kv({ strike: 'knee' })));           // knees are illegal in Full Contact (foul, not a point)
  assert.equal(fc.sides.a.penalties[0].foul, 'illegal_knee');
  fc.handle(pv(kv({ scored: false, foul: 'kick_below_waist', contact_location: crit('legs') })));
  assert.equal(fc.sides.a.penalties[1].foul, 'kick_below_waist');
  assert.equal(fc.sides.a.punchesTotal, 0);

  // legal target zones are sent to the backend per variant
  assert.deepEqual(RULESETS.kickboxing.variants.wako_fc.legalZones, ['head', 'torso']);
  assert.deepEqual(RULESETS.kickboxing.variants.unified.legalZones, ['head', 'torso', 'legs']);
  assert.equal(RULESETS.boxing.variants.amateur.legalZones, undefined); // boxing/judo default to head + torso

  const rest = new Match(RULESETS.kickboxing, 'demo');
  rest.start();
  rest.pause();
  rest.handle(pv(kv()));
  assert.equal(rest.sides.a.penalties[0].foul, 'hit_after_bell');

  // backend offline: geometry-only scoring
  const off = new Match(RULESETS.kickboxing, 'unified');
  off.start();
  off.handle(kickLanded('a', { target: 'head' }));
  off.handle(kickLanded('a', { target: 'leg' }));
  off.handle(kickLanded('a', { strike: 'knee', target: 'torso' }));
  assert.equal(off.sides.a.punchesTotal, 3);
  const offFc = new Match(RULESETS.kickboxing, 'wako_fc');
  offFc.start();
  offFc.handle(kickLanded('a', { target: 'leg' }));
  assert.equal(offFc.sides.a.penalties[0].foul, 'kick_below_waist');
});

test('the same kick is a DQ in boxing and judo, but a score in kickboxing', () => {
  const out = {};
  for (const sport of ['boxing', 'judo', 'kickboxing']) {
    const m = new Match(RULESETS[sport]);
    m.start();
    m.handle(pv(kv()));
    m.tick(3100); // judo's kick call waits for a possible throw
    out[sport] = m.status === 'ended' ? `${m.method} (${m.winner})` : `running, ${m.sides.a.punchesTotal} pt`;
  }
  assert.deepEqual(out, { boxing: 'DQ (b)', judo: 'HANSOKU-MAKE (b)', kickboxing: 'running, 1 pt' });
});
