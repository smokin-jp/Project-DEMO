// MediaPipe BlazePose landmark indices used throughout: 0 nose, 7/8 ears, 11/12 shoulders, 13/14 elbows, 15/16 wrists,
// 23/24 hips, 25/26 knees, 27/28 ankles.
export const POSE_CONNECTIONS = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24],
  [23, 25], [25, 27], [24, 26], [26, 28],
  [0, 7], [0, 8], [7, 8],
];

export const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
export const dist2 = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
export const dist3 = (a, b) => Math.hypot(a.x - b.x, a.y - b.y, (a.z || 0) - (b.z || 0));
export const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: ((a.z || 0) + (b.z || 0)) / 2 });
export const sign = (v) => (v > 0 ? 1 : v < 0 ? -1 : 0);

// angle at vertex b (degrees), 3D
export function angle3(a, b, c) {
  const v1 = [a.x - b.x, a.y - b.y, (a.z || 0) - (b.z || 0)];
  const v2 = [c.x - b.x, c.y - b.y, (c.z || 0) - (b.z || 0)];
  const dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
  const n = Math.hypot(...v1) * Math.hypot(...v2);
  if (n < 1e-9) return 0;
  return (Math.acos(clamp(dot / n, -1, 1)) * 180) / Math.PI;
}

export const other = (id) => (id === 'a' ? 'b' : 'a');
