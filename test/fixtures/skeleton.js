// Synthetic two-fighter skeleton generator. Produces the same "raw frame" shape that the
// MediaPipe wrapper produces, so the whole pipeline (tracker -> detectors -> referee) can be
// exercised without a camera: used for unit tests AND as a stage-fright-proof fallback demo.
//
// World units are metres, x right, y down, z depth (orthographic side-on camera).

const L = { torso: 0.5, ua: 0.3, fa: 0.27, th: 0.45, sh: 0.43 };
const CAM_H = 3.0; // metres visible across image height
const rad = (d) => (d * Math.PI) / 180;

const V = (x = 0, y = 0, z = 0) => ({ x, y, z });
const add = (a, b) => V(a.x + b.x, a.y + b.y, a.z + b.z);
const sub = (a, b) => V(a.x - b.x, a.y - b.y, a.z - b.z);
const mul = (a, k) => V(a.x * k, a.y * k, a.z * k);
const dot = (a, b) => a.x * b.x + a.y * b.y + a.z * b.z;
const len = (a) => Math.hypot(a.x, a.y, a.z);
const norm = (a) => mul(a, 1 / (len(a) || 1));

function ik(S, T, l1, l2, pole) {
  const d = sub(T, S);
  let Ld = len(d);
  const dir = norm(d);
  Ld = Math.min(Ld, l1 + l2 - 1e-4);
  const a = (l1 * l1 - l2 * l2 + Ld * Ld) / (2 * Ld);
  const h = Math.sqrt(Math.max(l1 * l1 - a * a, 0));
  const pp = norm(sub(pole, mul(dir, dot(pole, dir))));
  return { elbow: add(add(S, mul(dir, a)), mul(pp, h)), end: add(S, mul(dir, Ld)) };
}

export function defaultSpec(hipX, face) {
  return {
    hip: { x: hipX, y: 0 },
    ang: 0,
    face,
    // wrists in torso frame relative to mid-shoulder: f = forward, u = up, z = depth
    lw: { f: 0.2, u: 0.08, z: -0.12 },
    rw: { f: 0.16, u: 0.0, z: 0.12 },
    head: { f: 0, u: 0 }, // head offset from its normal place (neck thrust / duck), torso frame
    // ankles relative to hip (world axes)
    la: { x: face * 0.15, y: 0.86, z: -0.09 },
    ra: { x: -face * 0.15, y: 0.86, z: 0.09 },
  };
}

export function mergeSpec(base, over = {}) {
  const o = JSON.parse(JSON.stringify(base));
  for (const k of Object.keys(over)) {
    if (typeof over[k] === 'object' && over[k] !== null) o[k] = { ...o[k], ...over[k] };
    else o[k] = over[k];
  }
  return o;
}

function lerp(a, b, u) {
  if (typeof a === 'number') return a + (b - a) * u;
  if (a && typeof a === 'object') {
    const o = {};
    for (const k of Object.keys(a)) o[k] = lerp(a[k], b[k], u);
    return o;
  }
  return a;
}

const smooth = (u) => u * u * (3 - 2 * u);

// keys: [{t, a:{...overrides}, b:{...overrides}}]; overrides accumulate (each key patches the previous)
export function keyframes(baseA, baseB, keys) {
  let sa = baseA, sb = baseB;
  const full = keys.map((k) => {
    sa = mergeSpec(sa, k.a);
    sb = mergeSpec(sb, k.b);
    return { t: k.t, a: sa, b: sb };
  });
  return (t) => {
    if (t <= full[0].t) return { a: full[0].a, b: full[0].b };
    for (let i = 1; i < full.length; i++) {
      if (t <= full[i].t) {
        const u = smooth((t - full[i - 1].t) / (full[i].t - full[i - 1].t));
        return { a: { ...lerp(full[i - 1].a, full[i].a, u), face: full[0].a.face }, b: { ...lerp(full[i - 1].b, full[i].b, u), face: full[0].b.face } };
      }
    }
    return { a: full[full.length - 1].a, b: full[full.length - 1].b };
  };
}

export function buildPerson(spec) {
  const a = rad(spec.ang);
  const up = V(Math.sin(a), -Math.cos(a), 0);
  const fwd = mul(V(Math.cos(a), Math.sin(a), 0), spec.face);
  const H = V(spec.hip.x, spec.hip.y, 0);
  const M = add(H, mul(up, L.torso));
  const P = new Array(33);

  const LH = add(H, V(0, 0, -0.09)), RH = add(H, V(0, 0, 0.09));
  const LS = add(M, V(0, 0, -0.1)), RS = add(M, V(0, 0, 0.1));
  const wrist = (w) => add(add(add(M, mul(fwd, w.f)), mul(up, w.u)), V(0, 0, w.z));

  const arm = (S, w, pz) => ik(S, wrist(w), L.ua, L.fa, V(0, 1, pz * 0.4));
  const lArm = arm(LS, spec.lw, -1);
  const rArm = arm(RS, spec.rw, 1);
  const leg = (Hj, an, side) => ik(Hj, add(H, V(an.x, an.y, an.z)), L.th, L.sh, V(spec.face, 0.1, side * 0.1));
  const lLeg = leg(LH, spec.la, -1);
  const rLeg = leg(RH, spec.ra, 1);

  const hd = spec.head || { f: 0, u: 0 };
  const hOff = add(mul(up, hd.u), mul(fwd, hd.f));
  const nose = add(add(add(M, mul(up, 0.26)), mul(fwd, 0.09)), hOff);
  const lEar = add(add(add(M, mul(up, 0.24)), V(0, 0, -0.08)), hOff);
  const rEar = add(add(add(M, mul(up, 0.24)), V(0, 0, 0.08)), hOff);

  for (let i = 0; i < 33; i++) P[i] = nose;
  P[0] = nose; P[7] = lEar; P[8] = rEar;
  P[11] = LS; P[12] = RS;
  P[13] = lArm.elbow; P[14] = rArm.elbow;
  P[15] = lArm.end; P[16] = rArm.end;
  P[17] = P[19] = P[21] = lArm.end; P[18] = P[20] = P[22] = rArm.end;
  P[23] = LH; P[24] = RH;
  P[25] = lLeg.elbow; P[26] = rLeg.elbow;
  P[27] = lLeg.end; P[28] = rLeg.end;
  P[29] = P[31] = lLeg.end; P[30] = P[32] = rLeg.end;
  return P;
}

// pts (world metres) -> {lm (normalised image), world (hip-centred)}
export function project(pts, aspect, noise = 0, rnd = Math.random) {
  const g = () => (noise ? (rnd() + rnd() + rnd() - 1.5) * 2 * noise : 0);
  const hipMid = V((pts[23].x + pts[24].x) / 2, (pts[23].y + pts[24].y) / 2, (pts[23].z + pts[24].z) / 2);
  return {
    lm: pts.map((p) => ({
      x: (0.5 * aspect + p.x / CAM_H) / aspect + g() / aspect,
      y: 0.42 + p.y / CAM_H + g(),
      z: p.z / CAM_H,
      visibility: 1,
    })),
    world: pts.map((p) => ({ x: p.x - hipMid.x + g() * 3, y: p.y - hipMid.y + g() * 3, z: p.z - hipMid.z + g() * 3 })),
  };
}

export function makeFrame(specs, t, { aspect = 16 / 9, noise = 0.0015, rnd, only } = {}) {
  const poses = [];
  for (const id of ['a', 'b']) {
    if (only && !only.includes(id)) continue;
    poses.push(project(buildPerson(specs[id]), aspect, noise, rnd));
  }
  return { t, aspect, poses };
}

/* ------------------------------ scripted demos ------------------------------ */

const A0 = defaultSpec(-0.4, 1);
const B0 = defaultSpec(0.4, -1);

// helper: punch segment keys
function punchKeys(t0, hand, { u = 0.15, f = 0.64 } = {}) {
  const k = hand === 'L' ? 'lw' : 'rw';
  const home = hand === 'L' ? { f: 0.2, u: 0.08, z: -0.12 } : { f: 0.16, u: 0, z: 0.12 };
  return [
    { t: t0, a: { [k]: home, ang: 0 } },
    { t: t0 + 0.12, a: { [k]: { f: f, u, z: home.z }, ang: 8 } },
    { t: t0 + 0.22, a: { [k]: { f: f, u, z: home.z }, ang: 8 } },
    { t: t0 + 0.5, a: { [k]: home, ang: 0 } },
  ];
}

export const SCRIPTS = {
  // Boxing: Red lands a jab to the head, a cross to the body, then throws an illegal low blow.
  boxing: {
    label: 'Boxing: jab, body shot, then a LOW BLOW',
    sport: 'boxing',
    duration: 6.0,
    at: keyframes(A0, B0, [
      { t: 0 },
      { t: 0.8 },
      ...punchKeys(0.8, 'L', { u: 0.15 }),
      { t: 1.9 },
      ...punchKeys(1.9, 'R', { u: -0.2 }),
      { t: 3.2 },
      // illegal: Red lunges in, bends and drives a shot at Blue's belt line
      { t: 3.2 },
      { t: 3.32, a: { hip: { x: -0.05 }, ang: 30, lw: { f: 0.4, u: -0.4, z: -0.12 } } },
      { t: 3.55, a: { hip: { x: -0.05 }, ang: 30, lw: { f: 0.4, u: -0.4, z: -0.12 } } },
      { t: 3.9, a: { hip: { x: -0.4 }, ang: 0, lw: { f: 0.2, u: 0.08, z: -0.12 } } },
      { t: 6.0 },
    ]),
  },


  // Kickboxing: Red throws a roundhouse-style body kick at Blue.
  kick: {
    label: 'Kickboxing: Red body kick',
    sport: 'kickboxing',
    duration: 3.0,
    at: keyframes(A0, B0, [
      { t: 0 },
      { t: 0.8 },
      { t: 0.95, a: { ra: { x: 0.15, y: 0.35 }, ang: -4 } }, // chamber: knee up
      { t: 1.12, a: { ra: { x: 0.8, y: -0.1 }, ang: -12 } }, // snap: leg whips out into Blue's body
      { t: 1.24, a: { ra: { x: 0.8, y: -0.1 }, ang: -12 } },
      { t: 1.6, a: { ra: { x: -0.15, y: 0.86 }, ang: 0 } },
      { t: 3.0 },
    ]),
  },

  // Kickboxing: Red drives a knee into Blue.
  knee: {
    label: 'Kickboxing: Red knee strike',
    sport: 'kickboxing',
    duration: 3.0,
    at: keyframes(mergeSpec(A0, { hip: { x: -0.2 } }), B0, [
      { t: 0 },
      { t: 0.8 },
      { t: 0.97, a: { ra: { x: 0.12, y: 0.12 }, ang: 4 } }, // knee driven up and forward, leg folded
      { t: 1.15, a: { ra: { x: 0.12, y: 0.12 }, ang: 4 } },
      { t: 1.6, a: { ra: { x: -0.15, y: 0.86 }, ang: 0 } },
      { t: 3.0 },
    ]),
  },

  // Nobody strikes: Red just walks / shuffles (feet swing ~0.2 m, lift ~6 cm). Must never register as a kick.
  walk: {
    label: 'Walking',
    sport: 'kickboxing',
    duration: 4.0,
    at(t) {
      const ph = Math.sin(2 * Math.PI * 1.0 * t);
      return {
        a: mergeSpec(A0, { la: { x: 0.15 + 0.2 * ph, y: 0.86 - 0.06 * Math.max(0, ph) }, ra: { x: -0.15 - 0.2 * ph, y: 0.86 - 0.06 * Math.max(0, -ph) } }),
        b: B0,
      };
    },
  },

  // Judo: Red (tori) executes Osoto-gari, Blue (uke) goes over backwards.
  osoto: {
    label: 'Judo: Red throws Osoto-gari',
    sport: 'judo',
    duration: 5.0,
    at: keyframes(
      mergeSpec(A0, { hip: { x: -0.55 }, lw: { f: 0.38, u: 0.0, z: -0.1 }, rw: { f: 0.38, u: 0.0, z: 0.1 } }),
      mergeSpec(B0, { hip: { x: 0.55 }, lw: { f: 0.38, u: 0.0, z: -0.1 }, rw: { f: 0.38, u: 0.0, z: 0.1 } }),
      [
        { t: 0 },
        { t: 1.0 },
        // kuzushi: tori steps in, uke is pulled onto his heels
        { t: 1.45, a: { hip: { x: -0.18 }, ang: 6, la: { x: 0.2, y: 0.86 }, ra: { x: -0.25, y: 0.86 } }, b: { hip: { x: 0.45 }, ang: 14 } },
        // tori winds up the reaping leg
        { t: 1.62, a: { ang: 2, ra: { x: -0.45, y: 0.6 } }, b: { ang: 16 } },
        // THE REAP: leg whips forward and up through uke's legs
        { t: 1.78, a: { ang: 22, ra: { x: 0.65, y: 0.25 } }, b: { ang: 20, la: { x: 0.15, y: 0.7 }, ra: { x: 0.2, y: 0.55 } } },
        { t: 1.92, a: { ang: 28, ra: { x: 0.8, y: 0.1 } }, b: { hip: { x: 0.5, y: 0.2 }, ang: 50, la: { x: -0.5, y: -0.4 }, ra: { x: -0.45, y: -0.5 } } },
        // uke lands flat on his back
        { t: 2.3, a: { ang: 25, ra: { x: 0.6, y: 0.4 } }, b: { hip: { x: 0.62, y: 0.78 }, ang: 88, la: { x: -0.85, y: -0.05 }, ra: { x: -0.85, y: 0.05 }, lw: { f: 0.3, u: -0.2 }, rw: { f: 0.3, u: -0.2 } } },
        { t: 2.9, a: { ang: 6, ra: { x: -0.1, y: 0.86 } }, b: {} },
        { t: 5.0 },
      ]
    ),
  },
};

/** Run a script through a callback at a fixed frame rate. Returns frames (array) */
export function renderScript(script, { fps = 30, noise = 0.0015, rnd, only } = {}) {
  const frames = [];
  const n = Math.round(script.duration * fps);
  for (let i = 0; i <= n; i++) {
    const t = i / fps;
    frames.push(makeFrame(script.at(t), t * 1000, { noise, rnd, only }));
  }
  return frames;
}
