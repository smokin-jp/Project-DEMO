import test from 'node:test';
import assert from 'node:assert/strict';
import { signature, similarity, HUE_BINS, SIG_LEN, swatch } from '../src/appearance.js';
import { Tracker } from '../src/track.js';
import { cloneConfig } from '../src/config.js';

/* ---- helpers ---- */
const mkLm = (cx, vis = 1) => {
  // aspect 1; shoulders y .35, hips y .55 (torso .2)
  const lm = Array.from({ length: 33 }, () => ({ x: cx, y: 0.45, z: 0, visibility: vis }));
  lm[11] = { x: cx - 0.03, y: 0.35, z: 0, visibility: vis };
  lm[12] = { x: cx + 0.03, y: 0.35, z: 0, visibility: vis };
  lm[23] = { x: cx - 0.03, y: 0.55, z: 0, visibility: vis };
  lm[24] = { x: cx + 0.03, y: 0.55, z: 0, visibility: vis };
  return lm;
};
const pose = (cx) => ({ lm: mkLm(cx), world: mkLm(cx).map((p) => ({ x: p.x, y: p.y, z: 0 })) });
const oneHot = (bin, rgb) => { const h = new Float32Array(SIG_LEN); h[bin] = 1; return { h, rgb }; };
const RED = oneHot(0, [200, 30, 30]), BLUE = oneHot(8, [30, 40, 200]), BLACK = oneHot(HUE_BINS, [10, 10, 10]);
const frame = (t, people) => ({ t, aspect: 1, poses: people.map((p) => pose(p.x)), sigs: people.map((p) => p.sig) });
const newTracker = () => new Tracker(cloneConfig());

/** feed `n` separated frames so the tracker learns who wears what (RED fighter on the left, BLUE on the right) */
function learn(tr, n = 3) {
  for (let i = 0; i < n; i++) tr.update(frame(i * 33, [{ x: 0.3, sig: RED }, { x: 0.7, sig: BLUE }]));
}

/* ---- colour signatures from pixels ---- */
function paint(W, H, rects, bg = [90, 90, 90]) {
  const d = new Uint8ClampedArray(W * H * 4);
  for (let i = 0; i < W * H; i++) d.set([...bg, 255], i * 4);
  for (const { x0, x1, y0, y1, rgb } of rects) for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) d.set([...rgb, 255], (y * W + x) * 4);
  return d;
}

test('torso colours are read from the pixels: red vs blue are very different, same clothes match, black and white have their own bins', () => {
  const W = 200, H = 200;
  const img = paint(W, H, [
    { x0: 40, x1: 80, y0: 60, y1: 120, rgb: [210, 30, 30] }, // fighter on the left, red top
    { x0: 120, x1: 160, y0: 60, y1: 120, rgb: [30, 50, 210] }, // fighter on the right, blue top
  ]);
  const red = signature(img, W, H, mkLm(0.3)), blue = signature(img, W, H, mkLm(0.7));
  assert.ok(red && blue);
  assert.ok(similarity(red, blue) < 0.2, 'red and blue should share almost nothing');
  assert.ok(similarity(red, red) > 0.99);
  assert.ok(red.h[0] + red.h[1] + red.h[HUE_BINS - 1] > 0.8, 'red lives in the hue bins around 0 degrees');
  assert.ok(blue.h[6] + blue.h[7] + blue.h[8] + blue.h[9] > 0.9, 'blue lives around 210-270 degrees');
  assert.match(swatch(red), /^rgb\(2\d\d,/);

  const mono = paint(W, H, [{ x0: 40, x1: 80, y0: 60, y1: 120, rgb: [15, 15, 15] }, { x0: 120, x1: 160, y0: 60, y1: 120, rgb: [245, 245, 245] }]);
  const black = signature(mono, W, H, mkLm(0.3)), white = signature(mono, W, H, mkLm(0.7));
  assert.ok(black.h[HUE_BINS] > 0.9 && white.h[HUE_BINS + 2] > 0.9);
  assert.ok(similarity(black, white) < 0.1);
});

test('no signature when the torso cannot be read (joints hidden)', () => {
  const img = paint(100, 100, []);
  assert.equal(signature(img, 100, 100, mkLm(0.5, 0.1)), null);
});

/* ---- identity through swaps, clinches and crossings ---- */
test('RED stays RED when the fighters swap sides abruptly (position alone gets this wrong)', () => {
  const swapped = [{ x: 0.7, sig: RED }, { x: 0.3, sig: BLUE }]; // the red-clothed fighter is now on the right
  const colour = newTracker();
  learn(colour);
  const r = colour.update(frame(200, swapped));
  assert.ok(r.people.a.c.x > r.people.b.c.x, 'RED (red clothes) is the fighter now on the right');
  assert.equal(colour.info.src, 'colour');

  const posOnly = newTracker();
  posOnly.appearance = false;
  learn(posOnly);
  const p = posOnly.update(frame(200, swapped));
  assert.ok(p.people.a.c.x < p.people.b.c.x, 'position-only tracking hands RED to whoever is nearest: the wrong fighter');
});

test('RED stays RED through a clinch and a crossing', () => {
  const tr = newTracker();
  learn(tr);
  const ax = [0.3, 0.38, 0.46, 0.5, 0.51, 0.52, 0.6, 0.7, 0.8], bx = [0.7, 0.62, 0.54, 0.5, 0.5, 0.49, 0.4, 0.3, 0.2];
  let last;
  ax.forEach((x, i) => (last = tr.update(frame(300 + i * 33, [{ x, sig: RED }, { x: bx[i], sig: BLUE }]))));
  assert.ok(last.people.a.c.x > last.people.b.c.x, 'after crossing, RED (red clothes) is on the right');
});

test('a fighter alone in view is identified by clothes, not by which half of the picture they are in', () => {
  const tr = newTracker();
  learn(tr);
  tr.reset(); // new bout: references cleared...
  assert.equal(tr.ref.a, null);
  learn(tr);
  const r = tr.update(frame(500, [{ x: 0.2, sig: BLUE }])); // BLUE standing on the left half
  assert.equal(r.people.a, null);
  assert.ok(r.people.b);
});

test('similar clothes: colour is ignored and position (and the flip option) decide', () => {
  const tr = newTracker();
  for (let i = 0; i < 3; i++) tr.update(frame(i * 33, [{ x: 0.3, sig: BLACK }, { x: 0.7, sig: BLACK }]));
  const r = tr.update(frame(200, [{ x: 0.3, sig: BLACK }, { x: 0.7, sig: BLACK }]));
  assert.equal(tr.info.src, 'position');
  assert.ok(r.people.a.c.x < r.people.b.c.x);
  const flipped = newTracker();
  flipped.flip = true;
  const f = flipped.update(frame(0, [{ x: 0.3, sig: BLACK }, { x: 0.7, sig: BLACK }]));
  assert.ok(f.people.a.c.x > f.people.b.c.x, 'the manual "RED on the right" option still works');
});

/* ---- several cameras ---- */
test('a second camera takes RED / BLUE from the first camera by clothing colour, whichever side it stands on', () => {
  const cam1 = newTracker();
  learn(cam1); // camera 1: RED (red clothes) on the left
  const cam2 = newTracker(); // the opposite side: sees the fighters the other way round, and is NOT told to flip
  cam2.globalRef = cam1.ref;
  const r = cam2.update(frame(0, [{ x: 0.3, sig: BLUE }, { x: 0.7, sig: RED }]));
  assert.ok(r.people.a.c.x > r.people.b.c.x, 'camera 2 also calls the red-clothed fighter RED (now on the right)');
  assert.equal(cam2.info.src, 'colour');
  // and it keeps its own reference afterwards, even if camera 1 is reset
  cam2.update(frame(33, [{ x: 0.3, sig: BLUE }, { x: 0.7, sig: RED }]));
  cam2.update(frame(66, [{ x: 0.3, sig: BLUE }, { x: 0.7, sig: RED }]));
  assert.ok(cam2.ref.a && cam2.ref.b);
  cam1.reset();
  assert.ok(cam1.ref && cam1.ref.a === null, 'reset empties the shared reference object, it does not replace it');
  const again = cam2.update(frame(99, [{ x: 0.3, sig: BLUE }, { x: 0.7, sig: RED }]));
  assert.ok(again.people.a.c.x > again.people.b.c.x);
});

test('a camera that cannot learn colours yet (no reference anywhere) behaves exactly as before', () => {
  const tr = newTracker();
  const r = tr.update({ t: 0, aspect: 1, poses: [pose(0.3), pose(0.7)] }); // no signatures at all
  assert.ok(r.people.a.c.x < r.people.b.c.x);
  assert.equal(tr.info.src, 'position');
});
