// Clothing-colour signatures: lets the tracker keep each fighter's RED / BLUE identity through clinches and crossings, and
// lets several cameras agree on who is who without being told which side each camera is on.
//
// signature = a small colour histogram of the fighter's torso (hue bins for coloured pixels, plus dark / grey / bright
// bins for black, grey and white clothing) and the mean colour (shown as a swatch in the UI).
// Two histograms are compared with the Bhattacharyya coefficient (1 = same colours, 0 = nothing in common).

export const HUE_BINS = 12;
export const SIG_LEN = HUE_BINS + 3; // + dark, grey, bright

function rgb2hsv(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b), d = mx - mn;
  let h = 0;
  if (d > 0) {
    if (mx === r) h = ((g - b) / d) % 6;
    else if (mx === g) h = (b - r) / d + 2;
    else h = (r - g) / d + 4;
    h *= 60;
    if (h < 0) h += 360;
  }
  return [h, mx === 0 ? 0 : d / mx, mx];
}

/** pixel box of the torso (chest and stomach, away from arms and background) for landmarks in image fractions (0..1) */
export function torsoBox(lm, W, H) {
  for (const i of [11, 12, 23, 24]) if (!lm[i] || (lm[i].visibility ?? 1) < 0.3) return null;
  const xs = [lm[11].x, lm[12].x, lm[23].x, lm[24].x];
  const top = (lm[11].y + lm[12].y) / 2, bottom = (lm[23].y + lm[24].y) / 2;
  const L = bottom - top;
  if (!(L > 0.02)) return null;
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
  // side-on a body has almost no shoulder width: never narrower than 0.3 torso lengths
  const halfW = Math.max(((Math.max(...xs) - Math.min(...xs)) / 2) * 0.8, 0.3 * L * (H / W));
  const x0 = Math.max(0, Math.floor((cx - halfW) * W)), x1 = Math.min(W, Math.ceil((cx + halfW) * W));
  const y0 = Math.max(0, Math.floor((top + 0.12 * L) * H)), y1 = Math.min(H, Math.ceil((bottom - 0.08 * L) * H));
  return x1 - x0 >= 3 && y1 - y0 >= 3 ? { x0, x1, y0, y1 } : null;
}

/** colour signature of one person from RGBA pixel data (W x H) and their landmarks; null when the torso is not readable */
export function signature(data, W, H, lm) {
  const box = torsoBox(lm, W, H);
  if (!box) return null;
  const h = new Float32Array(SIG_LEN);
  let n = 0, R = 0, G = 0, B = 0;
  for (let y = box.y0; y < box.y1; y += 2) {
    for (let x = box.x0; x < box.x1; x += 2) {
      const o = (y * W + x) * 4;
      const r = data[o], g = data[o + 1], b = data[o + 2];
      R += r; G += g; B += b; n++;
      const [hue, s, v] = rgb2hsv(r, g, b);
      if (s > 0.25 && v > 0.2) {
        // spread a little over neighbouring hue bins so a small hue shift (lighting) does not change the bin entirely
        const pos = (hue / 360) * HUE_BINS;
        const b0 = Math.floor(pos) % HUE_BINS;
        h[(b0 + HUE_BINS - 1) % HUE_BINS] += 0.25;
        h[b0] += 0.5;
        h[(b0 + 1) % HUE_BINS] += 0.25;
      } else h[HUE_BINS + (v < 0.25 ? 0 : v > 0.7 ? 2 : 1)] += 1;
    }
  }
  if (n < 20) return null;
  let sum = 0;
  for (let i = 0; i < SIG_LEN; i++) sum += h[i];
  for (let i = 0; i < SIG_LEN; i++) h[i] /= sum;
  return { h, rgb: [R / n, G / n, B / n] };
}

/** Bhattacharyya coefficient of two signatures: 1 = same colours, 0 = nothing in common */
export function similarity(a, b) {
  let s = 0;
  for (let i = 0; i < SIG_LEN; i++) s += Math.sqrt(a.h[i] * b.h[i]);
  return s;
}

/** running reference: moves `ref` towards `sig` (alpha 1 = replace). Returns the new reference. */
export function blendSig(ref, sig, alpha) {
  if (!ref) return { h: Float32Array.from(sig.h), rgb: [...sig.rgb] };
  for (let i = 0; i < SIG_LEN; i++) ref.h[i] = ref.h[i] * (1 - alpha) + sig.h[i] * alpha;
  for (let i = 0; i < 3; i++) ref.rgb[i] = ref.rgb[i] * (1 - alpha) + sig.rgb[i] * alpha;
  return ref;
}

export const swatch = (sig) => (sig ? `rgb(${sig.rgb.map((v) => Math.round(v)).join(',')})` : null);
