import type { Example } from "../model/backward";
import { cloneParams, forEachRow } from "../model/params";
import type { Matrix, ModelConfig, ModelParams, TinyGpt } from "../model/types";

export interface TrainerSettings {
  lr: number;
  totalSteps: number;
  batchSize: number;
  seed: number;
}

export const DEFAULT_TRAINER_SETTINGS: TrainerSettings = { lr: 0.01, totalSteps: 2000, batchSize: 4, seed: 1 };

export const ADAM = { beta1: 0.9, beta2: 0.99, eps: 1e-8, weightDecay: 1e-3 };

export const INIT_STD = { embedding: 0.3 };

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function normalSampler(rand: () => number) {
  return (std: number) => std * Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());
}

/** Same scheme as the NumPy script: N(0, 0.3²) embeddings, N(0, 1/fan_in) weights, zero biases, γ = 1, β = 0. */
export function initParams(cfg: ModelConfig, seed: number): ModelParams {
  const normal = normalSampler(mulberry32(seed));
  const mat = (r: number, c: number, std: number): Matrix =>
    Array.from({ length: r }, () => Array.from({ length: c }, () => normal(std)));
  const zeros = (n: number) => new Array<number>(n).fill(0);
  const ones = (n: number) => new Array<number>(n).fill(1);
  const d = cfg.dModel;
  const f = cfg.dFF;
  return {
    wte: mat(cfg.vocab.length, d, INIT_STD.embedding),
    wpe: mat(cfg.nCtx, d, INIT_STD.embedding),
    layers: Array.from({ length: cfg.nLayers }, () => ({
      ln1: { gamma: ones(d), beta: zeros(d) },
      attn: {
        wq: mat(d, d, 1 / Math.sqrt(d)),
        bq: zeros(d),
        wk: mat(d, d, 1 / Math.sqrt(d)),
        bk: zeros(d),
        wv: mat(d, d, 1 / Math.sqrt(d)),
        bv: zeros(d),
        wo: mat(d, d, 1 / Math.sqrt(d)),
        bo: zeros(d),
      },
      ln2: { gamma: ones(d), beta: zeros(d) },
      mlp: { w1: mat(d, f, 1 / Math.sqrt(d)), b1: zeros(f), w2: mat(f, d, 1 / Math.sqrt(f)), b2: zeros(d) },
    })),
    lnf: { gamma: ones(d), beta: zeros(d) },
  };
}

/** Cosine decay from lr down to 5% of lr; `step` is the 1-based index of the update. */
export function lrAt(step: number, s: TrainerSettings): number {
  const t = Math.min(step, s.totalSteps) / s.totalSteps;
  return s.lr * (0.05 + 0.95 * 0.5 * (1 + Math.cos(Math.PI * t)));
}

export interface AdamElement {
  m: number;
  v: number;
  mHat: number;
  vHat: number;
  decay: number;
  update: number;
  pNew: number;
}

/** One Adam(W) update of a single number. `t` is the 1-based step; decay applies to weight matrices only. */
export function adamElement(
  p: number,
  g: number,
  m: number,
  v: number,
  t: number,
  lr: number,
  decays: boolean,
  eps = ADAM.eps,
): AdamElement {
  const mNew = ADAM.beta1 * m + (1 - ADAM.beta1) * g;
  const vNew = ADAM.beta2 * v + (1 - ADAM.beta2) * g * g;
  const mHat = mNew / (1 - ADAM.beta1 ** t);
  const vHat = vNew / (1 - ADAM.beta2 ** t);
  const decay = decays ? ADAM.weightDecay * p : 0;
  const update = lr * (mHat / (Math.sqrt(vHat) + eps) + decay);
  return { m: mNew, v: vNew, mHat, vHat, decay, update, pNew: p - update };
}

export interface OptimizerState {
  params: ModelParams;
  m: ModelParams;
  v: ModelParams;
  /** Number of updates applied so far. */
  step: number;
}

export function adamUpdate(state: OptimizerState, grads: ModelParams, s: TrainerSettings): OptimizerState {
  const params = cloneParams(state.params);
  const m = cloneParams(state.m);
  const v = cloneParams(state.v);
  const t = state.step + 1;
  const lr = lrAt(t, s);
  forEachRow(params, [m, v, grads], (row, [mr, vr, gr], inMatrix) => {
    for (let i = 0; i < row.length; i++) {
      const r = adamElement(row[i], gr[i], mr[i], vr[i], t, lr, inMatrix);
      row[i] = r.pNew;
      mr[i] = r.m;
      vr[i] = r.v;
    }
  });
  return { params, m, v, step: t };
}

/** The training corpus, one example per distinct sentence, weighted by its number of copies. */
export function corpusExamples(model: TinyGpt): (Example & { text: string })[] {
  return model.training.corpus.map(({ text, count }) => {
    const ids = text.split(" ").map((w) => model.config.vocab.indexOf(w));
    return { text, ids: ids.slice(0, -1), targets: ids.slice(1), weight: count };
  });
}

/** Draws sentences with probability proportional to their count (with replacement). */
export function sampleBatch(rand: () => number, examples: Example[], size: number): Example[] {
  const total = examples.reduce((s, e) => s + e.weight, 0);
  return Array.from({ length: size }, () => {
    let r = rand() * total;
    const e = examples.find((x) => (r -= x.weight) < 0) ?? examples[examples.length - 1];
    return { ids: e.ids, targets: e.targets, weight: 1 };
  });
}
