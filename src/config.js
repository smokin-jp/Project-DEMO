// All detector thresholds in one place so they can be tuned live from the UI.
// Distances in "torso lengths" (shoulder-mid -> hip-mid, image space) unless marked (m) = metres (MediaPipe world landmarks).
export const DEFAULTS = {
  smoothing: 0.65, // EMA alpha on landmarks (1 = no smoothing)

  punch: {
    startRate: 1.2,      // m/s: arm-reach growth that starts a punch candidate
    minSpeed: 1.8,       // m/s: peak wrist speed relative to shoulder
    minExtension: 0.16,  // m: reach gained during the punch
    minReachRatio: 0.80, // peak reach / arm length (arm nearly straight)
    cooldownMs: 350,
    landMargin: 0.25,    // torso lengths of slack around opponent target zones
    soloVirtualOpponent: true, // no opponent in frame -> score against an imaginary one
  },

  kick: {                // kick / knee candidates (the CLIP backend then confirms: real kick vs a step)
    startSpeed: 1.6,     // m/s: ankle/knee speed relative to the hip that starts a candidate
    minSpeed: 2.2,       // m/s: peak speed
    minLift: 0.45,       // torso lengths the ankle must rise above its standing height (kick)
    minKneeLift: 0.35,   // torso lengths the knee must rise above its standing height (knee strike)
    minTravel: 0.40,     // m: ankle travel relative to the hip (kick)
    kickKneeAngle: 120,  // deg: the leg must be at least this straight at the peak of a kick
    cooldownMs: 500,
  },

  attack: {              // judo: attempts to engage (feeds the 30 s non-combativity rule)
    stepSpeed: 1.0,      // torso lengths/s the hips close towards the opponent
    legSpeed: 4.0,       // torso lengths/s a foot swings towards the opponent (relative to the hip): well above ordinary stepping
    engageDist: 4.0,     // torso lengths: only count it when the fighters are within a few body lengths of each other
    cooldownMs: 700,
  },

  fall: {
    minPhi: 50,          // deg from vertical torso must reach
    minHipDrop: 0.45,    // torso lengths hips must drop below standing baseline
    minShoulderDrop: 0.5,
    windowMs: 1000,
    settleMs: 350,
    upPhi: 35,
  },

  reap: {                // Osoto-gari leg reap
    maxPairDist: 2.4,    // torso lengths between fighters' hips
    minSpeed: 3.2,       // torso lengths/s, ankle speed towards the opponent (relative to own hip)
    reachDist: 0.95,     // torso lengths: ankle must pass this close to opponent's legs
    linkMs: 1600,        // fall must follow the reap within this window
    ipponPhi: 70,
    ipponSpeed: 1.3,     // torso lengths/s shoulder drop
    ipponDurMs: 1100,
    wazaPhi: 55,
  },

  foul: {                // pose-geometry fouls: headbutt, leg grab / hold, ducking below the belt
    maxTilt: 60,         // deg: a fighter already on the mat is not judged by these
    windowMs: 150,       // headbutt: movement is measured over about this long
    headContact: 1.0,    // torso lengths between the two noses when the heads meet
    minClose: 0.35,      // torso lengths the heads must close during the window (about 2.3 torso/s)
    minThrust: 0.28,     // torso lengths the head moves towards the opponent relative to the attacker's own shoulders
    minHeadVsBody: 1.8,  // the head must travel this many times further than the shoulders (a lunge or lean carries both)
    minShare: 0.60,      // share of the closing that is the attacker's own movement (not the opponent walking onto him)
    belowBelt: 0.20,     // torso lengths below the opponent's hip line a wrist must be to count as "below the belt"
    legTouch: 0.20,      // torso lengths from wrist to the opponent's thigh / shin segment that counts as gripping
    legGrabMs: 700,      // judo: grip kept this long = leg grab (shido)
    legHoldMs: 2500,     // kickboxing: a leg held this long (catching a kick briefly is legal)
    engageDist: 2.2,     // torso lengths between the fighters' hips
    duckBelow: 0.05,     // torso lengths below the opponent's hip line the head must be to count as ducking below the belt
    duckMaxTilt: 65,     // deg: bent right over is ducking; flat on the mat is not
    duckDist: 2.0,
    duckMs: 700,
    cooldownMs: 2500,
  },

  hold: {                // clinch / holding
    maxDist: 1.1,        // torso lengths between hips
    ms: 2800,
    cooldownMs: 4000,
  },
};

export function cloneConfig() {
  return JSON.parse(JSON.stringify(DEFAULTS));
}
