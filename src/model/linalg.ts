import type { Matrix, Vector } from "./types";

export function matmul(a: Matrix, b: Matrix): Matrix {
  const n = b.length;
  const m = b[0].length;
  return a.map((row) => {
    if (row.length !== n) throw new Error(`matmul shape mismatch: ${row.length} vs ${n}`);
    const out = new Array<number>(m).fill(0);
    for (let k = 0; k < n; k++) {
      const x = row[k];
      const bk = b[k];
      for (let j = 0; j < m; j++) out[j] += x * bk[j];
    }
    return out;
  });
}

export function addRowVector(a: Matrix, v: Vector): Matrix {
  return a.map((row) => row.map((x, j) => x + v[j]));
}

export function add(a: Matrix, b: Matrix): Matrix {
  return a.map((row, i) => row.map((x, j) => x + b[i][j]));
}

export function transpose(a: Matrix): Matrix {
  return a[0].map((_, j) => a.map((row) => row[j]));
}

export function linear(x: Matrix, w: Matrix, b: Vector): Matrix {
  return addRowVector(matmul(x, w), b);
}

export function sliceCols(a: Matrix, start: number, end: number): Matrix {
  return a.map((row) => row.slice(start, end));
}

export function concatCols(parts: Matrix[]): Matrix {
  return parts[0].map((_, i) => parts.flatMap((p) => p[i]));
}

export function dot(a: Vector, b: Vector): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export const GELU_C = Math.sqrt(2 / Math.PI);

/** GPT-2's tanh approximation of GELU. */
export function gelu(u: number): number {
  return 0.5 * u * (1 + Math.tanh(GELU_C * (u + 0.044715 * u ** 3)));
}

export function argmax(v: Vector): number {
  let best = 0;
  for (let i = 1; i < v.length; i++) if (v[i] > v[best]) best = i;
  return best;
}
