import test from 'node:test';
import assert from 'node:assert/strict';
import { bodyOf, plausibleSecond, personBox, Confirmer } from '../src/second.js';

// 33-landmark skeleton of a standing person, hips at (cx, cy), torso length t (image fractions, aspect 1 for simplicity)
function skeleton(cx, cy, t = 0.25, over = {}) {
  const lm = Array.from({ length: 33 }, () => ({ x: cx, y: cy, z: 0, visibility: 0.9 }));
  const set = (i, x, y, v = 0.9) => (lm[i] = { x, y, z: 0, visibility: v });
  set(0, cx, cy - 1.3 * t);
  set(11, cx - 0.12 * t, cy - t); set(12, cx + 0.12 * t, cy - t);
  set(23, cx - 0.1 * t, cy); set(24, cx + 0.1 * t, cy);
  set(25, cx - 0.1 * t, cy + 0.9 * t); set(26, cx + 0.1 * t, cy + 0.9 * t);
  set(27, cx - 0.1 * t, cy + 1.8 * t); set(28, cx + 0.1 * t, cy + 1.8 * t);
  for (const [i, v] of Object.entries(over)) lm[i] = { ...lm[i], ...v };
  return lm;
}
const body = (lm, aspect = 1) => bodyOf(lm, aspect);
const first = body(skeleton(0.4, 0.5));

test('a second skeleton of similar size standing next to the first is plausible', () => {
  assert.equal(plausibleSecond(body(skeleton(0.6, 0.5)), first), true);
  assert.equal(plausibleSecond(body(skeleton(0.55, 0.55, 0.2)), first), true, 'a bit smaller, partly overlapping');
});

test('rejects the false positives seen on real footage: banner in the corner, tiny or huge figures, the same body again', () => {
  assert.equal(plausibleSecond(body(skeleton(0.05, 0.08, 0.05)), first), false, 'tiny skeleton far away in a corner');
  const small = body(skeleton(0.2, 0.5, 0.1));
  assert.equal(plausibleSecond(body(skeleton(0.6, 0.5, 0.1)), small), true, 'four torso lengths away is still a fight');
  assert.equal(plausibleSecond(body(skeleton(0.95, 0.5, 0.1)), small), false, 'too far from the first fighter (> 5 torso lengths)');
  assert.equal(plausibleSecond(body(skeleton(0.6, 0.5, 0.6)), first), false, 'more than twice the size');
  assert.equal(plausibleSecond(body(skeleton(0.55, 0.25, 0.25)), first), false, 'a spectator in the crowd: hips a full torso above the fighters\' hips');
  assert.equal(plausibleSecond(body(skeleton(0.55, 0.62, 0.25)), first), true, 'a fighter a little lower (crouching or thrown) is fine');
  assert.equal(plausibleSecond(body(skeleton(0.48, 0.28, 0.25)), first), true, 'a fighter lifted in a throw is higher but overlaps the first one');
  assert.equal(plausibleSecond(body(skeleton(0.41, 0.5)), first), false, 'the first fighter detected again');
});

test('rejects poorly supported or cut-off skeletons', () => {
  const weak = skeleton(0.6, 0.5, 0.25, { 11: { visibility: 0.3 }, 12: { visibility: 0.3 }, 23: { visibility: 0.5 } });
  assert.equal(plausibleSecond(body(weak), first), false, 'shoulders and hips not well supported');
  assert.equal(plausibleSecond(body(skeleton(0.6, 1.05)), first), false, 'hips below the bottom of the frame');
  const noKnees = skeleton(0.6, 0.5, 0.25, { 25: { visibility: 0.1 }, 26: { visibility: 0.1 } });
  assert.equal(plausibleSecond(body(noKnees), first), false, 'no knee visible at all');
  assert.equal(plausibleSecond(body(skeleton(0.6, 0.5, 0.25, { 25: { visibility: 0.9 }, 26: { visibility: 0.1 } })), first), true, 'one knee is enough (the other may be hidden)');
});

test('aspect ratio is respected (widescreen distances are not squashed)', () => {
  const a = skeleton(0.2, 0.5, 0.1), b = skeleton(0.65, 0.5, 0.1); // 0.45 of the width apart
  assert.equal(plausibleSecond(bodyOf(b, 1), bodyOf(a, 1)), true, 'square picture: 4.5 torso lengths apart');
  assert.equal(plausibleSecond(bodyOf(b, 16 / 9), bodyOf(a, 16 / 9)), false, '16:9 picture: the same gap is 8 torso lengths');
});

test('personBox covers the visible joints with a margin and clamps to the frame', () => {
  const b = personBox(skeleton(0.5, 0.5), 0.2);
  assert.ok(b.x0 < 0.5 - 0.1 * 0.25 && b.x1 > 0.5 + 0.1 * 0.25);
  assert.ok(b.y0 < 0.5 - 1.3 * 0.25 && b.y1 > 0.5 + 1.8 * 0.25 - 0.001);
  const edge = personBox(skeleton(0.02, 0.5), 0.5);
  assert.equal(edge.x0, 0);
  assert.equal(personBox(skeleton(0.5, 0.5).map((p) => ({ ...p, visibility: 0.1 }))), null, 'nothing visible: no box');
});

test('a recovered skeleton is believed only when seen twice in a row at the same place', () => {
  const c = new Confirmer(800, 1.5);
  const a = body(skeleton(0.6, 0.5));
  assert.equal(c.see(0, a), false, 'first sighting: not yet');
  assert.equal(c.see(150, a), true, 'repeated at once');
  assert.equal(c.see(300, body(skeleton(0.62, 0.5))), true, 'small drift is fine');
  assert.equal(c.see(450, body(skeleton(0.1, 0.5))), false, 'jumped to another place: starts over');
  assert.equal(c.see(2000, body(skeleton(0.1, 0.5))), false, 'too long since the last sighting');
  c.reset();
  assert.equal(c.see(2100, a), false);
});
