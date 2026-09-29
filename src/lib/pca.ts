import type { Matrix } from "../model/types";

function powerIteration(c: Matrix, iters = 300): number[] {
  let v = c.map((_, i) => 1 / Math.sqrt(c.length + i));
  for (let k = 0; k < iters; k++) {
    const w = c.map((row) => row.reduce((s, x, j) => s + x * v[j], 0));
    const n = Math.hypot(...w) || 1;
    v = w.map((x) => x / n);
  }
  const big = v.reduce((b, x) => (Math.abs(x) > Math.abs(b) ? x : b), 0);
  return big < 0 ? v.map((x) => -x) : v;
}

/** Projects the rows of `x` onto their top two principal components. */
export function pca2(x: Matrix): { coords: [number, number][]; explained: [number, number] } {
  const n = x.length;
  const dim = x[0].length;
  const mean = x[0].map((_, j) => x.reduce((s, r) => s + r[j], 0) / n);
  const xc = x.map((r) => r.map((v, j) => v - mean[j]));
  const cov = Array.from({ length: dim }, (_, a) =>
    Array.from({ length: dim }, (_, b) => xc.reduce((s, r) => s + r[a] * r[b], 0) / n),
  );
  const total = cov.reduce((s, r, i) => s + r[i], 0);
  const v1 = powerIteration(cov);
  const l1 = v1.reduce((s, x, i) => s + x * cov[i].reduce((t, c, j) => t + c * v1[j], 0), 0);
  const deflated = cov.map((r, a) => r.map((c, b) => c - l1 * v1[a] * v1[b]));
  const v2 = powerIteration(deflated);
  const l2 = v2.reduce((s, x, i) => s + x * deflated[i].reduce((t, c, j) => t + c * v2[j], 0), 0);
  const coords = xc.map((r) => [r.reduce((s, v, j) => s + v * v1[j], 0), r.reduce((s, v, j) => s + v * v2[j], 0)] as [number, number]);
  return { coords, explained: [l1 / total, l2 / total] };
}
