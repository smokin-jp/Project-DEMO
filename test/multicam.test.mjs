import test from 'node:test';
import assert from 'node:assert/strict';
import { StrikeCoordinator, EventDeduper, POST_MS } from '../src/multicam.js';
import { Tracker } from '../src/track.js';
import { cloneConfig } from '../src/config.js';

const thrown = (over = {}) => ({ type: 'punch_thrown', strike: 'punch', by: 'a', hand: 'right', peakT: 1000, speed: 3, extension: 0.4, at: { x: 0.5, y: 0.3 }, ...over });

test('the same punch seen by two cameras becomes ONE strike, judged once both had time to report', () => {
  const co = new StrikeCoordinator({ waitMs: 700 });
  const r1 = co.add('cam1', thrown({ speed: 3.0, peakT: 1000 }), 1100);
  const r2 = co.add('cam2', thrown({ speed: 4.5, peakT: 1060, at: { x: 0.2, y: 0.4 } }), 1250);
  assert.equal(r1.created, true);
  assert.equal(r2.created, false);
  assert.equal(co.pending.length, 1);
  const c = co.pending[0];
  assert.deepEqual([...c.cams].sort(), ['cam1', 'cam2']);
  assert.equal(c.speed, 4.5); // the camera that saw it fastest provides the reference peak
  assert.equal(c.peakT, 1060);
  assert.equal(c.atCam, 'cam2');
  assert.deepEqual(co.due(1500), []); // post-roll done (1360) but still waiting for other cameras (1100 + 700)
  assert.equal(co.due(1800).length, 1);
  assert.equal(co.pending.length, 0);
});

test('different fighters, strike types or moments are separate strikes; one camera waits only for the post-roll', () => {
  const co = new StrikeCoordinator({ waitMs: 0 });
  co.add('cam1', thrown({ by: 'a' }), 1000);
  co.add('cam1', thrown({ by: 'b' }), 1000);
  co.add('cam1', thrown({ strike: 'kick', type: 'kick_thrown' }), 1000);
  co.add('cam1', thrown({ peakT: 2000 }), 2100);
  assert.equal(co.pending.length, 4);
  assert.equal(co.due(1000 + POST_MS - 1).length, 0);
  assert.equal(co.due(1000 + POST_MS).length, 3); // single camera: judged as soon as the post-roll is in
});

test('events detected by several cameras only count once', () => {
  const d = new EventDeduper();
  assert.equal(d.accept({ type: 'throw', by: 'a' }, 1000), true);
  assert.equal(d.accept({ type: 'throw', by: 'a' }, 1200), false); // second camera, same throw
  assert.equal(d.accept({ type: 'throw', by: 'b' }, 1200), true); // different fighter
  assert.equal(d.accept({ type: 'down', by: 'b' }, 1200), true);
  assert.equal(d.accept({ type: 'down', by: 'b' }, 1900), false);
  assert.equal(d.accept({ type: 'down', by: 'b' }, 2800), true); // a later, separate fall
  assert.equal(d.accept({ type: 'kick_landed', by: 'a', strike: 'kick' }, 0), true);
  assert.equal(d.accept({ type: 'kick_landed', by: 'a', strike: 'knee' }, 100), true); // different strike type
  assert.equal(d.accept({ type: 'kick_landed', by: 'a', strike: 'kick' }, 300), false);
});

test('a camera on the opposite side can flip which fighter is RED', () => {
  const lm = (x) => Array.from({ length: 33 }, () => ({ x, y: 0.5, z: 0, visibility: 1 }));
  const pose = (x) => ({ lm: lm(x), world: lm(x) });
  const left = pose(0.25), right = pose(0.75);
  const normal = new Tracker(cloneConfig());
  const flipped = new Tracker(cloneConfig());
  flipped.flip = true;
  const rn = normal.update({ t: 0, aspect: 1, poses: [left, right] });
  const rf = flipped.update({ t: 0, aspect: 1, poses: [left, right] });
  assert.ok(rn.people.a.c.x < rn.people.b.c.x); // normally RED is the fighter on the left
  assert.ok(rf.people.a.c.x > rf.people.b.c.x); // flipped: RED is on the right
  // a lone fighter: left half = RED, unless flipped
  assert.ok(new Tracker(cloneConfig()).update({ t: 0, aspect: 1, poses: [left] }).people.a);
  const f2 = new Tracker(cloneConfig());
  f2.flip = true;
  assert.ok(f2.update({ t: 0, aspect: 1, poses: [left] }).people.b);
});
