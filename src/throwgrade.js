// Grading and deciding judo throws. Pure functions (no DOM), unit-tested in Node.

/** How good was the landing? From the fall measured by FallDetector (torso angle, shoulder speed, duration, facing). */
export function gradeThrow(down, c) {
  let grade;
  if (down.phi >= c.ipponPhi && down.speed >= c.ipponSpeed && down.durationMs <= c.ipponDurMs) grade = 'ippon';
  else if (down.phi >= c.wazaPhi) grade = 'waza-ari';
  else grade = 'yuko';
  if (down.backwards === false) grade = grade === 'ippon' ? 'waza-ari' : 'yuko'; // landed facing forward: one level lower
  return grade;
}

export const THROW_MIN_CONFIDENCE = 0.5; // below this the video model's guess is not trusted
const MAX_PAIR_DIST = 3.0; // torso lengths: the fighters must have been close (when we could see both)
const MIN_FALL_SPEED = 1.0; // torso lengths / s of shoulder drop
const MIN_FALL_PHI = 55; // degrees of torso tilt

/**
 * A fighter went down in judo and no leg-reap rule explained it. Should the throw-type model's answer become a scored throw?
 * There is no "not a throw" class in the model, so everything that suggests this was NOT a throw is checked here first.
 * @param down    the 'down' event ({by, phi, speed, durationMs, backwards, thrown})
 * @param minPair smallest distance between the two fighters (torso lengths) in the last ~2 s, or null if both were never seen together
 * @param result  backend answer {technique, confidence, probs} or null
 */
export function decideThrow({ down, minPair, result, gradeCfg, minConfidence = THROW_MIN_CONFIDENCE }) {
  if (down.thrown) return { kind: 'ignore', reason: 'already explained as a throw' };
  if (minPair != null && minPair > MAX_PAIR_DIST) return { kind: 'ignore', reason: 'the fighters were not close: a slip or a knockdown, not a throw' };
  if (down.speed < MIN_FALL_SPEED || down.phi < MIN_FALL_PHI) return { kind: 'ignore', reason: 'too gentle a fall for a throw' };
  if (!result) return { kind: 'ignore', reason: 'no throw model available' };
  if (result.confidence < minConfidence) {
    return { kind: 'unsure', note: `Fall next to the opponent, but the video model is not sure which throw it was (best guess ${result.technique} ${Math.round(result.confidence * 100)}%): nothing scored` };
  }
  const against = down.by;
  return {
    kind: 'throw',
    event: {
      type: 'throw', by: against === 'a' ? 'b' : 'a', against, technique: result.technique,
      grade: gradeThrow(down, gradeCfg), confidence: result.confidence, via: 'video-model', auto: true,
    },
    note: `Throw recognised by the video model: ${result.technique} (${Math.round(result.confidence * 100)}%)`,
  };
}
