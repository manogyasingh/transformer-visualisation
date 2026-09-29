import {
  add,
  concatCols,
  dot,
  gelu,
  linear,
  matmul,
  sliceCols,
  transpose,
} from "./linalg";
import type {
  BlockParams,
  BlockTrace,
  ForwardTrace,
  HeadTrace,
  LayerNormParams,
  LayerNormTrace,
  Matrix,
  ModelConfig,
  TinyGpt,
} from "./types";

function layerNorm(x: Matrix, p: LayerNormParams, eps: number): LayerNormTrace {
  const d = x[0].length;
  const mean = x.map((row) => row.reduce((s, v) => s + v, 0) / d);
  const variance = x.map((row, i) => row.reduce((s, v) => s + (v - mean[i]) ** 2, 0) / d);
  const rstd = variance.map((v) => 1 / Math.sqrt(v + eps));
  const normalized = x.map((row, i) => row.map((v) => (v - mean[i]) * rstd[i]));
  const output = normalized.map((row) => row.map((v, j) => p.gamma[j] * v + p.beta[j]));
  return { input: x, mean, variance, rstd, normalized, gamma: p.gamma, beta: p.beta, output };
}

function attentionHead(q: Matrix, k: Matrix, v: Matrix, dHead: number): HeadTrace {
  const T = q.length;
  const scale = Math.sqrt(dHead);
  const dots = q.map((qi) => k.map((kj) => dot(qi, kj)));
  const scores = dots.map((row) => row.map((s) => s / scale));
  const masked = scores.map((row, i) => row.map((s, j) => (j <= i ? s : -Infinity)));
  const rowMax = masked.map((row) => Math.max(...row));
  const exps = masked.map((row, i) => row.map((s) => Math.exp(s - rowMax[i])));
  const rowSum = exps.map((row) => row.reduce((a, b) => a + b, 0));
  const weights = exps.map((row, i) => row.map((e) => e / rowSum[i]));
  const output = matmul(weights, v);
  if (weights.length !== T) throw new Error("attention shape mismatch");
  return { q, k, v, dots, scores, masked, rowMax, exps, rowSum, weights, output };
}

function block(x: Matrix, p: BlockParams, cfg: ModelConfig): BlockTrace {
  const ln1 = layerNorm(x, p.ln1, cfg.lnEps);
  const q = linear(ln1.output, p.attn.wq, p.attn.bq);
  const k = linear(ln1.output, p.attn.wk, p.attn.bk);
  const v = linear(ln1.output, p.attn.wv, p.attn.bv);
  const heads: HeadTrace[] = [];
  for (let h = 0; h < cfg.nHeads; h++) {
    const lo = h * cfg.dHead;
    const hi = lo + cfg.dHead;
    heads.push(attentionHead(sliceCols(q, lo, hi), sliceCols(k, lo, hi), sliceCols(v, lo, hi), cfg.dHead));
  }
  const concat = concatCols(heads.map((h) => h.output));
  const attnOut = linear(concat, p.attn.wo, p.attn.bo);
  const resid1 = add(x, attnOut);
  const ln2 = layerNorm(resid1, p.ln2, cfg.lnEps);
  const up = linear(ln2.output, p.mlp.w1, p.mlp.b1);
  const act = up.map((row) => row.map(gelu));
  const down = linear(act, p.mlp.w2, p.mlp.b2);
  const resid2 = add(resid1, down);
  return { input: x, ln1, q, k, v, heads, concat, attnOut, resid1, ln2, up, act, down, resid2 };
}

/** Runs the full forward pass and records every intermediate value. */
export function runForward(model: TinyGpt, ids: number[]): ForwardTrace {
  const { config: cfg, params } = model;
  if (ids.length === 0) throw new Error("need at least one token");
  if (ids.length > cfg.nCtx) throw new Error(`context length is ${cfg.nCtx}`);

  const tokEmb = ids.map((id) => params.wte[id].slice());
  const posEmb = ids.map((_, i) => params.wpe[i].slice());
  const h0 = add(tokEmb, posEmb);

  const blocks: BlockTrace[] = [];
  let h = h0;
  for (const layer of params.layers) {
    const b = block(h, layer, cfg);
    blocks.push(b);
    h = b.resid2;
  }
  const lnf = layerNorm(h, params.lnf, cfg.lnEps);
  const logitsAll = matmul(lnf.output, transpose(params.wte));

  return {
    ids,
    tokens: ids.map((id) => cfg.vocab[id]),
    tokEmb,
    posEmb,
    h0,
    blocks,
    lnf,
    logitsAll,
    logits: logitsAll[logitsAll.length - 1],
  };
}
