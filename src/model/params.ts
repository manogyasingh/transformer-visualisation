import type { Matrix, ModelConfig, ModelParams, Vector } from "./types";

type Leaf = Matrix | Vector;

function isVector(x: unknown): x is Vector {
  return Array.isArray(x) && (x.length === 0 || typeof x[0] === "number");
}

function isMatrix(x: unknown): x is Matrix {
  return Array.isArray(x) && x.length > 0 && isVector(x[0]);
}

/**
 * Visits every parameter row in `p` together with the matching rows of `others`
 * (which must have the same structure). Vectors are visited as a single row.
 */
export function forEachRow(
  p: ModelParams,
  others: ModelParams[],
  fn: (row: Vector, otherRows: Vector[], inMatrix: boolean) => void,
): void {
  const walk = (a: unknown, bs: unknown[]) => {
    if (isVector(a)) fn(a, bs as Vector[], false);
    else if (isMatrix(a)) a.forEach((row, i) => fn(row, (bs as Matrix[]).map((b) => b[i]), true));
    else if (Array.isArray(a)) a.forEach((x, i) => walk(x, bs.map((b) => (b as unknown[])[i])));
    else if (a && typeof a === "object")
      for (const k of Object.keys(a)) walk((a as Record<string, unknown>)[k], bs.map((b) => (b as Record<string, unknown>)[k]));
  };
  walk(p, others);
}

function deepCopy<T>(x: T): T {
  if (Array.isArray(x)) return (typeof x[0] === "number" ? x.slice() : x.map(deepCopy)) as T;
  if (x && typeof x === "object") {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(x)) out[k] = deepCopy(v);
    return out as T;
  }
  return x;
}

export function cloneParams(p: ModelParams): ModelParams {
  return deepCopy(p);
}

export function zerosLike(p: ModelParams): ModelParams {
  const z = cloneParams(p);
  forEachRow(z, [], (row) => row.fill(0));
  return z;
}

/** acc += scale * g, in place. */
export function addScaled(acc: ModelParams, g: ModelParams, scale: number): void {
  forEachRow(acc, [g], (row, [gr]) => {
    for (let i = 0; i < row.length; i++) row[i] += scale * gr[i];
  });
}

export function gradNorm(g: ModelParams): number {
  let s = 0;
  forEachRow(g, [], (row) => row.forEach((x) => (s += x * x)));
  return Math.sqrt(s);
}

export interface ParamRef {
  id: string;
  /** TeX label */
  label: string;
  get: (p: ModelParams) => Leaf;
}

/** Every parameter tensor, in a fixed order, with a TeX label. */
export function paramList(cfg: ModelConfig): ParamRef[] {
  const list: ParamRef[] = [
    { id: "wte", label: "W_E", get: (p) => p.wte },
    { id: "wpe", label: "W_P", get: (p) => p.wpe },
  ];
  for (let l = 0; l < cfg.nLayers; l++) {
    const s = `^{(${l + 1})}`;
    const L = (p: ModelParams) => p.layers[l];
    list.push(
      { id: `b${l + 1}.ln1.gamma`, label: `\\gamma_1${s}`, get: (p) => L(p).ln1.gamma },
      { id: `b${l + 1}.ln1.beta`, label: `\\beta_1${s}`, get: (p) => L(p).ln1.beta },
      { id: `b${l + 1}.wq`, label: `W_Q${s}`, get: (p) => L(p).attn.wq },
      { id: `b${l + 1}.bq`, label: `b_Q${s}`, get: (p) => L(p).attn.bq },
      { id: `b${l + 1}.wk`, label: `W_K${s}`, get: (p) => L(p).attn.wk },
      { id: `b${l + 1}.bk`, label: `b_K${s}`, get: (p) => L(p).attn.bk },
      { id: `b${l + 1}.wv`, label: `W_V${s}`, get: (p) => L(p).attn.wv },
      { id: `b${l + 1}.bv`, label: `b_V${s}`, get: (p) => L(p).attn.bv },
      { id: `b${l + 1}.wo`, label: `W_O${s}`, get: (p) => L(p).attn.wo },
      { id: `b${l + 1}.bo`, label: `b_O${s}`, get: (p) => L(p).attn.bo },
      { id: `b${l + 1}.ln2.gamma`, label: `\\gamma_2${s}`, get: (p) => L(p).ln2.gamma },
      { id: `b${l + 1}.ln2.beta`, label: `\\beta_2${s}`, get: (p) => L(p).ln2.beta },
      { id: `b${l + 1}.w1`, label: `W_1${s}`, get: (p) => L(p).mlp.w1 },
      { id: `b${l + 1}.b1`, label: `b_1${s}`, get: (p) => L(p).mlp.b1 },
      { id: `b${l + 1}.w2`, label: `W_2${s}`, get: (p) => L(p).mlp.w2 },
      { id: `b${l + 1}.b2`, label: `b_2${s}`, get: (p) => L(p).mlp.b2 },
    );
  }
  list.push(
    { id: "lnf.gamma", label: "\\gamma_f", get: (p) => p.lnf.gamma },
    { id: "lnf.beta", label: "\\beta_f", get: (p) => p.lnf.beta },
  );
  return list;
}

/** Reads/writes element (i, j) of a parameter; vectors ignore i. */
export function getElement(leaf: Leaf, i: number, j: number): number {
  return isMatrix(leaf) ? leaf[i][j] : (leaf as Vector)[j];
}

export function setElement(leaf: Leaf, i: number, j: number, value: number): void {
  if (isMatrix(leaf)) leaf[i][j] = value;
  else (leaf as Vector)[j] = value;
}

export function asMatrix(leaf: Leaf): Matrix {
  return isMatrix(leaf) ? leaf : [leaf as Vector];
}

export function leafIsMatrix(leaf: Leaf): boolean {
  return isMatrix(leaf);
}
