export type Vector = number[];
/** Row-major: matrix[row][col]. Rows are tokens, columns are features. */
export type Matrix = number[][];

export interface ModelConfig {
  vocab: string[];
  nCtx: number;
  dModel: number;
  nHeads: number;
  dHead: number;
  dFF: number;
  nLayers: number;
  lnEps: number;
}

export interface LayerNormParams {
  gamma: Vector;
  beta: Vector;
}

/** All weight matrices are stored as (in × out), so y = x·W + b with x a row vector. */
export interface BlockParams {
  ln1: LayerNormParams;
  attn: {
    wq: Matrix;
    bq: Vector;
    wk: Matrix;
    bk: Vector;
    wv: Matrix;
    bv: Vector;
    wo: Matrix;
    bo: Vector;
  };
  ln2: LayerNormParams;
  mlp: {
    w1: Matrix;
    b1: Vector;
    w2: Matrix;
    b2: Vector;
  };
}

export interface ModelParams {
  wte: Matrix;
  wpe: Matrix;
  layers: BlockParams[];
  lnf: LayerNormParams;
}

export interface TrainingInfo {
  corpus: { text: string; count: number }[];
  presetPrompts: string[];
  steps: number;
  finalLoss: number;
  roundedLoss: number;
  optimalLoss: number;
}

export interface TinyGpt {
  config: ModelConfig;
  params: ModelParams;
  training: TrainingInfo;
  reference: { prompt: string; ids: number[]; logits: Vector }[];
  referenceGrads: { sentence: string; loss: number; grads: ModelParams };
}

export interface LayerNormTrace {
  input: Matrix;
  mean: Vector;
  variance: Vector;
  /** 1 / sqrt(variance + eps) */
  rstd: Vector;
  normalized: Matrix;
  gamma: Vector;
  beta: Vector;
  output: Matrix;
}

export interface HeadTrace {
  q: Matrix;
  k: Matrix;
  v: Matrix;
  /** q_i · k_j before scaling */
  dots: Matrix;
  /** dots / sqrt(d_head) */
  scores: Matrix;
  /** scores with -Infinity above the diagonal */
  masked: Matrix;
  rowMax: Vector;
  /** exp(masked - rowMax) */
  exps: Matrix;
  rowSum: Vector;
  weights: Matrix;
  output: Matrix;
}

export interface BlockTrace {
  input: Matrix;
  ln1: LayerNormTrace;
  q: Matrix;
  k: Matrix;
  v: Matrix;
  heads: HeadTrace[];
  concat: Matrix;
  attnOut: Matrix;
  resid1: Matrix;
  ln2: LayerNormTrace;
  up: Matrix;
  act: Matrix;
  down: Matrix;
  resid2: Matrix;
}

export interface ForwardTrace {
  ids: number[];
  tokens: string[];
  tokEmb: Matrix;
  posEmb: Matrix;
  h0: Matrix;
  blocks: BlockTrace[];
  lnf: LayerNormTrace;
  /** Logits for every position (T × V); only the last row is used to generate. */
  logitsAll: Matrix;
  logits: Vector;
}
