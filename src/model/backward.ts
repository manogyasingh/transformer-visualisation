import { runForward } from "./forward";
import { GELU_C, matmul, transpose } from "./linalg";
import { addScaled, zerosLike } from "./params";
import type {
  BlockParams,
  BlockTrace,
  ForwardTrace,
  LayerNormTrace,
  Matrix,
  ModelConfig,
  ModelParams,
  Vector,
} from "./types";

export interface LayerNormGrad {
  dOut: Matrix;
  /** dOut ⊙ γ */
  dXhat: Matrix;
  /** mean_j dXhat_ij */
  meanDXhat: Vector;
  /** mean_j dXhat_ij · xhat_ij */
  meanDXhatXhat: Vector;
  dX: Matrix;
  dGamma: Vector;
  dBeta: Vector;
}

export interface LinearGrad {
  /** Gradient arriving at the layer's output. */
  dOut: Matrix;
  dX: Matrix;
  dW: Matrix;
  db: Vector;
}

export interface HeadGrad {
  dZ: Matrix;
  dA: Matrix;
  dV: Matrix;
  /** Σ_j A_ij dA_ij */
  rowDot: Vector;
  /** Gradient w.r.t. the (masked, scaled) scores. */
  dS: Matrix;
  dQ: Matrix;
  dK: Matrix;
}

export interface BlockGrad {
  /** dL/dX^(ℓ), arriving from above. */
  dOut: Matrix;
  mlpDown: LinearGrad;
  geluDeriv: Matrix;
  dU: Matrix;
  mlpUp: LinearGrad;
  ln2: LayerNormGrad;
  /** dL/dR = dOut (skip path) + ln2.dX (MLP path) */
  dR: Matrix;
  attnProj: LinearGrad;
  heads: HeadGrad[];
  q: LinearGrad;
  k: LinearGrad;
  v: LinearGrad;
  /** dL/dY = q.dX + k.dX + v.dX */
  dY: Matrix;
  ln1: LayerNormGrad;
  /** dL/dX^(ℓ-1) = dR (skip path) + ln1.dX (attention path) */
  dIn: Matrix;
}

export interface TrainTrace {
  forward: ForwardTrace;
  targets: number[];
  /** Each position's loss is multiplied by this (1/T for a mean over one sentence). */
  scale: number;
  probs: Matrix;
  nll: Vector;
  loss: number;
  dLogits: Matrix;
  dF: Matrix;
  /** W_E gradient from its use as the unembedding matrix. */
  dWteOut: Matrix;
  lnf: LayerNormGrad;
  blocks: BlockGrad[];
  dX0: Matrix;
  /** W_E gradient from its use as the embedding lookup table. */
  dWteIn: Matrix;
  dWpe: Matrix;
  grads: ModelParams;
}

const add = (a: Matrix, b: Matrix): Matrix => a.map((r, i) => r.map((x, j) => x + b[i][j]));
const colSum = (a: Matrix): Vector => a[0].map((_, j) => a.reduce((s, r) => s + r[j], 0));

export function geluDerivative(u: number): number {
  const t = Math.tanh(GELU_C * (u + 0.044715 * u ** 3));
  return 0.5 * (1 + t) + 0.5 * u * (1 - t * t) * GELU_C * (1 + 3 * 0.044715 * u * u);
}

function layerNormBackward(dOut: Matrix, ln: LayerNormTrace): LayerNormGrad {
  const d = dOut[0].length;
  const dXhat = dOut.map((r) => r.map((g, j) => g * ln.gamma[j]));
  const meanDXhat = dXhat.map((r) => r.reduce((s, x) => s + x, 0) / d);
  const meanDXhatXhat = dXhat.map((r, i) => r.reduce((s, x, j) => s + x * ln.normalized[i][j], 0) / d);
  const dX = dXhat.map((r, i) =>
    r.map((x, j) => ln.rstd[i] * (x - meanDXhat[i] - ln.normalized[i][j] * meanDXhatXhat[i])),
  );
  const dGamma = ln.gamma.map((_, j) => dOut.reduce((s, r, i) => s + r[j] * ln.normalized[i][j], 0));
  return { dOut, dXhat, meanDXhat, meanDXhatXhat, dX, dGamma, dBeta: colSum(dOut) };
}

function linearBackward(dOut: Matrix, x: Matrix, w: Matrix): LinearGrad {
  return { dOut, dX: matmul(dOut, transpose(w)), dW: matmul(transpose(x), dOut), db: colSum(dOut) };
}

function blockBackward(dOut: Matrix, b: BlockTrace, p: BlockParams, cfg: ModelConfig): BlockGrad {
  const mlpDown = linearBackward(dOut, b.act, p.mlp.w2);
  const geluDeriv = b.up.map((r) => r.map(geluDerivative));
  const dU = mlpDown.dX.map((r, i) => r.map((g, j) => g * geluDeriv[i][j]));
  const mlpUp = linearBackward(dU, b.ln2.output, p.mlp.w1);
  const ln2 = layerNormBackward(mlpUp.dX, b.ln2);
  const dR = add(dOut, ln2.dX);

  const attnProj = linearBackward(dR, b.concat, p.attn.wo);
  const scale = Math.sqrt(cfg.dHead);
  const heads: HeadGrad[] = b.heads.map((h, hi) => {
    const dZ = attnProj.dX.map((r) => r.slice(hi * cfg.dHead, (hi + 1) * cfg.dHead));
    const dA = matmul(dZ, transpose(h.v));
    const dV = matmul(transpose(h.weights), dZ);
    const rowDot = dA.map((r, i) => r.reduce((s, x, j) => s + x * h.weights[i][j], 0));
    const dS = dA.map((r, i) => r.map((x, j) => h.weights[i][j] * (x - rowDot[i])));
    const dQ = matmul(dS, h.k).map((r) => r.map((x) => x / scale));
    const dK = matmul(transpose(dS), h.q).map((r) => r.map((x) => x / scale));
    return { dZ, dA, dV, rowDot, dS, dQ, dK };
  });
  const merge = (key: "dQ" | "dK" | "dV") => heads[0][key].map((_, i) => heads.flatMap((h) => h[key][i]));
  const y = b.ln1.output;
  const q = linearBackward(merge("dQ"), y, p.attn.wq);
  const k = linearBackward(merge("dK"), y, p.attn.wk);
  const v = linearBackward(merge("dV"), y, p.attn.wv);
  const dY = add(add(q.dX, k.dX), v.dX);
  const ln1 = layerNormBackward(dY, b.ln1);
  return { dOut, mlpDown, geluDeriv, dU, mlpUp, ln2, dR, attnProj, heads, q, k, v, dY, ln1, dIn: add(dR, ln1.dX) };
}

export interface FinalBackward {
  lnf: LayerNormGrad;
  blocks: BlockGrad[];
  dX0: Matrix;
  /** W_E gradient from its use as the embedding lookup table. */
  dWteIn: Matrix;
  dWpe: Matrix;
  /** Gradients of every weight; `wte` holds only the embedding-lookup part. */
  grads: ModelParams;
}

/**
 * Backpropagates dF, the gradient arriving at the final LayerNorm's output, down to every weight.
 * Scalar heads (reward and value models) start here; language-model losses start at the logits.
 */
export function backwardFromFinal(
  model: { config: ModelConfig; params: ModelParams },
  forward: ForwardTrace,
  dF: Matrix,
): FinalBackward {
  const { config: cfg, params } = model;
  const lnf = layerNormBackward(dF, forward.lnf);
  const blocks: BlockGrad[] = new Array(cfg.nLayers);
  let d = lnf.dX;
  for (let l = cfg.nLayers - 1; l >= 0; l--) {
    blocks[l] = blockBackward(d, forward.blocks[l], params.layers[l], cfg);
    d = blocks[l].dIn;
  }
  const dX0 = d;
  const ids = forward.ids;
  const dWteIn = params.wte.map((r) => r.map(() => 0));
  ids.forEach((id, i) => dX0[i].forEach((g, j) => (dWteIn[id][j] += g)));
  const dWpe = params.wpe.map((r, i) => (i < ids.length ? dX0[i].slice() : r.map(() => 0)));

  const grads: ModelParams = {
    wte: dWteIn,
    wpe: dWpe,
    layers: blocks.map((g) => ({
      ln1: { gamma: g.ln1.dGamma, beta: g.ln1.dBeta },
      attn: {
        wq: g.q.dW,
        bq: g.q.db,
        wk: g.k.dW,
        bk: g.k.db,
        wv: g.v.dW,
        bv: g.v.db,
        wo: g.attnProj.dW,
        bo: g.attnProj.db,
      },
      ln2: { gamma: g.ln2.dGamma, beta: g.ln2.dBeta },
      mlp: { w1: g.mlpUp.dW, b1: g.mlpUp.db, w2: g.mlpDown.dW, b2: g.mlpDown.db },
    })),
    lnf: { gamma: lnf.dGamma, beta: lnf.dBeta },
  };
  return { lnf, blocks, dX0, dWteIn, dWpe, grads };
}

export interface LogitsBackward extends FinalBackward {
  dF: Matrix;
  /** W_E gradient from its use as the unembedding matrix. */
  dWteOut: Matrix;
}

/** Backpropagates a gradient on the logits (T × V) to every weight, including both uses of the tied W_E. */
export function backwardFromLogits(
  model: { config: ModelConfig; params: ModelParams },
  forward: ForwardTrace,
  dLogits: Matrix,
): LogitsBackward {
  const dF = matmul(dLogits, model.params.wte);
  const dWteOut = matmul(transpose(dLogits), forward.lnf.output);
  const back = backwardFromFinal(model, forward, dF);
  return { ...back, dF, dWteOut, grads: { ...back.grads, wte: add(dWteOut, back.dWteIn) } };
}

export function softmaxRows(logits: Matrix): Matrix {
  return logits.map((z) => {
    const m = Math.max(...z);
    const e = z.map((v) => Math.exp(v - m));
    const s = e.reduce((a, c) => a + c, 0);
    return e.map((v) => v / s);
  });
}

/**
 * Forward and backward pass for one sequence, recording every gradient.
 * The loss is scale · Σ_i −log p(targets_i); scale defaults to 1/T (mean over positions).
 */
export function forwardBackward(
  model: { config: ModelConfig; params: ModelParams },
  ids: number[],
  targets: number[],
  scale = 1 / ids.length,
): TrainTrace {
  const forward = runForward(model, ids);
  const probs = softmaxRows(forward.logitsAll);
  const nll = targets.map((t, i) => -Math.log(probs[i][t]));
  const loss = scale * nll.reduce((a, c) => a + c, 0);
  const dLogits = probs.map((r, i) => r.map((p, v) => (p - (v === targets[i] ? 1 : 0)) * scale));
  const back = backwardFromLogits(model, forward, dLogits);
  return { forward, targets, scale, probs, nll, loss, dLogits, ...back };
}

export interface Example {
  ids: number[];
  targets: number[];
  weight: number;
}

/** Weighted mean cross-entropy over every target token in the batch, and its gradient. */
export function batchLossAndGrads(
  model: { config: ModelConfig; params: ModelParams },
  batch: Example[],
): { loss: number; grads: ModelParams } {
  const nTokens = batch.reduce((s, e) => s + e.weight * e.targets.length, 0);
  const grads = zerosLike(model.params);
  let loss = 0;
  for (const e of batch) {
    const tr = forwardBackward(model, e.ids, e.targets, e.weight / nTokens);
    loss += tr.loss;
    addScaled(grads, tr.grads, 1);
  }
  return { loss, grads };
}

/** Loss only (no gradients), e.g. for finite differences or evaluation. */
export function batchLoss(model: { config: ModelConfig; params: ModelParams }, batch: Example[]): number {
  const nTokens = batch.reduce((s, e) => s + e.weight * e.targets.length, 0);
  let loss = 0;
  for (const e of batch) {
    const logits = runForward(model, e.ids).logitsAll;
    e.targets.forEach((t, i) => {
      const z = logits[i];
      const m = Math.max(...z);
      const lse = m + Math.log(z.reduce((s, v) => s + Math.exp(v - m), 0));
      loss += (e.weight / nTokens) * (lse - z[t]);
    });
  }
  return loss;
}
