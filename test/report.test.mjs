import test from 'node:test';
import assert from 'node:assert/strict';
import { Match } from '../src/engine.js';
import { RULESETS } from '../src/rules/index.js';
import { buildReportHtml } from '../src/report.js';

const crit = (value, confidence = 0.9, detail = {}) => ({ value, confidence, detail });
const verdict = (over = {}) => ({
  strike: 'punch', strike_check: null, thrower: crit('a', 0.99, { name: 'RED' }), intended_target: crit('b', 0.9, { name: 'BLUE' }), contact: crit(true),
  contact_location: crit('head'), legal_area: crit(true), legal_glove_part: crit(true, 0.8, { part: 'knuckle_part' }),
  genuine_punch: crit(true, 0.8, { classification: 'clean' }), scored: true, review: false, confidence: 0.72, foul: null,
  summary: 'RED right punch -> BLUE: clean punch to the head (72%).', crops: { frame: 'QUJD', impact: 'REVG' }, views: [], ...over,
});
const review = (id, v, extra = {}) => ({ id, by: 'a', hand: 'right', strike: v.strike, clock: '2:41', round: 1, v, status: null, at: Date.now(), ...extra });

test('report of a finished boxing bout contains the result, scorecard, penalties, timeline and the judged strikes', () => {
  const m = new Match(RULESETS.boxing, 'demo');
  m.start();
  m.handle({ type: 'strike_verdict', by: 'a', against: 'b', verdict: verdict(), auto: true });
  m.handle({ type: 'foul', by: 'b', foul: 'holding' });
  m.tick(46000); // end of round 1 -> scorecard
  const html = buildReportHtml({ match: m, reviews: [review(1, verdict())] });
  assert.match(html, /^<!doctype html>/);
  assert.match(html, /Boxing/);
  assert.match(html, /Bout not finished/);
  assert.match(html, /Scorecard \(10-point must\)/);
  assert.match(html, /<td>1<\/td><td><b>10<\/b>/); // round 1: RED 10
  assert.match(html, /Holding \/ clinching/); // BLUE's penalty
  assert.match(html, /clean punch to the head/);
  assert.match(html, /class="ai">AI</); // the AI-tagged call
  assert.match(html, /1 Who threw it\?/);
  assert.match(html, /data:image\/jpeg;base64,QUJD/); // screenshots are embedded
  assert.match(html, /human referee's decision is final/);
});

test('report states the winner when the bout is over (and works for every sport)', () => {
  const j = new Match(RULESETS.judo, 'demo');
  j.start();
  j.handle({ type: 'throw', by: 'a', against: 'b', technique: 'Osoto-gari', grade: 'ippon', auto: true });
  const html = buildReportHtml({ match: j, reviews: [] });
  assert.match(html, /RED wins by IPPON/);
  assert.match(html, /waza-ari 0 · yuko 0 · shido 0/);
  assert.match(html, /No strikes were judged/);
  for (const sport of ['boxing', 'kickboxing', 'judo']) {
    const m = new Match(RULESETS[sport]);
    m.start();
    assert.match(buildReportHtml({ match: m, reviews: [] }), /<h1>AI Referee: match report<\/h1>/);
  }
});

test('report shows multi-camera detail, kick reviews and the referee decision on a review', () => {
  const v = verdict({
    strike: 'kick', strike_check: crit(true, 0.9), scored: false, review: true, summary: 'borderline',
    views: [{ camera: 'cam1', quality: 0.92, available: true, p_contact: 0.65, location: 'head' }, { camera: 'cam2', quality: 0.3, available: true, p_contact: 0.2, location: 'none' }],
    contact: crit(false, 0.6, { veto_by_good_view: true }), crops: { cam1_frame: 'QQ==', cam2_frame: 'Qg==' },
  });
  const m = new Match(RULESETS.kickboxing, 'unified');
  m.start();
  const html = buildReportHtml({ match: m, reviews: [review(7, v, { status: 'awarded' })] });
  assert.match(html, /Real kick\? \(not a step\)/);
  assert.match(html, /Cameras \(each view weighted/);
  assert.match(html, /overruled an apparent hit from another angle/);
  assert.match(html, /cam1 frame/);
  assert.match(html, /Referee decision on this review: <b>awarded<\/b>/);
  assert.match(html, /needs referee review/);
});

test('report escapes anything that came from outside (names, summaries, log text), so it cannot inject markup', () => {
  const m = new Match(RULESETS.boxing, 'demo');
  m.start();
  m.say('<script>alert("log")</script>', { kind: 'info' });
  const v = verdict({ summary: '<img src=x onerror=alert(1)>', crops: { 'x"><script>bad()</script>': 'QUJD' } });
  v.thrower.detail.name = '<b onmouseover=1>RED</b>';
  const html = buildReportHtml({ match: m, reviews: [review(1, v, { clock: '<i>', hand: '"><script>h()</script>' })] });
  assert.ok(!/<script>(alert|bad|h)\(/.test(html), 'no raw script tags from data');
  assert.ok(!/<img src=x onerror/.test(html), 'no raw injected image tag');
  assert.match(html, /&lt;script&gt;alert\(&quot;log&quot;\)&lt;\/script&gt;/);
  assert.ok((html.match(/<script/g) || []).length === 0, 'the report contains no script elements at all except the print button handler');
});

test('the match remembers when it started (the report only lists strikes judged in the current bout)', () => {
  const before = Date.now();
  const m = new Match(RULESETS.boxing);
  assert.ok(m.startedAt >= before);
  m.reset();
  assert.ok(m.startedAt >= before);
});
