import { backwardFromFinal, backwardFromLogits, batchLossAndGrads, softmaxRows, type Example } from "../model/backward";
import { runForward } from "../model/forward";
import { cloneParams, forEachRow, zerosLike } from "../model/params";
import { labelOf, scalarOutputs, type PreferenceLabel } from "../model/reward-model";
import { decode, DEFAULT_DECODING, type SamplingResult } from "../model/sampling";
import type { Matrix, ModelConfig, ModelParams, ScalarModelParams, Vector } from "../model/types";
import { adamElement, mulberry32, sampleBatch } from "../training/trainer";

/**
 * One iteration of PPO for RLHF, as in Ziegler et al. (2019), Stiennon et al. (2020) and InstructGPT
 * (Ouyang et al., 2022): sample responses, score them with the reward model, add a per-token KL penalty
 * to the reference model, estimate advantages with GAE, then take a few epochs of clipped updates of the
 * policy and of a separate value model. Like InstructGPT's PPO-ptx, each policy update also mixes in the
 * gradient of the pretraining loss on a few corpus sentences.
 */

export interface PpoSettings {
  /** KL penalty coefficient β. */
  beta: number;
  /** PPO clip range ε for the probability ratio. */
  clip: number;
  /** Clip range for the value model's change. */
  valueClip: number;
  gamma: number;
  lambda: number;
  /** Optimisation epochs over each batch of rollouts (one minibatch = the whole batch). */
  epochs: number;
  samplesPerPrompt: number;
  lr: number;
  valueLr: number;
  /** Adam's ε (Huang et al., 2024 use 1e-5 rather than the usual 1e-8). */
  adamEps: number;
  /** γ, the weight of the pretraining loss mixed into every policy update (PPO-ptx); 0 is plain PPO. */
  ptxCoef: number;
  /** Corpus sentences in each epoch's pretraining minibatch. */
  ptxBatch: number;
  /** Score given to a response that hits the length limit without a full stop. */
  truncatedScore: number;
  totalIterations: number;
  seed: number;
}

export const DEFAULT_PPO_SETTINGS: PpoSettings = {
  beta: 0.2,
  clip: 0.2,
  valueClip: 0.2,
  gamma: 1,
  lambda: 0.95,
  epochs: 4,
  samplesPerPrompt: 2,
  lr: 5e-4,
  valueLr: 1e-3,
  adamEps: 1e-5,
  ptxCoef: 0.3,
  ptxBatch: 4,
  truncatedScore: -1,
  totalIterations: 200,
  seed: 1,
};

export interface PpoModels {
  cfg: ModelConfig;
  /** π_ref: frozen copy of the policy's starting weights. */
  reference: ModelParams;
  /** Config of the reward and value models (one more position than the policy). */
  scalarCfg: ModelConfig;
  reward: ScalarModelParams;
  prompts: number[][];
  eos: number;
  /** The simulated labeler's hidden utility, which the reward model only approximates. */
  utility: Record<PreferenceLabel, number>;
  /** Pretraining sentences for the PPO-ptx term, weighted by their number of copies. */
  corpus: Example[];
}

export interface AdamState<P> {
  params: P;
  m: P;
  v: P;
  /** Number of updates applied so far. */
  step: number;
}

export interface PpoState {
  iteration: number;
  policy: AdamState<ModelParams>;
  value: AdamState<ScalarModelParams>;
}

function freshAdam<P extends object>(params: P): AdamState<P> {
  return { params: cloneParams(params), m: zerosLike(params), v: zerosLike(params), step: 0 };
}

/** The policy starts as the reference model and the value model as the reward model. */
export function initialState(models: PpoModels): PpoState {
  return { iteration: 0, policy: freshAdam(models.reference), value: freshAdam(models.reward) };
}

/** Adam with a constant learning rate and no weight decay. */
export function adamStep<P extends object>(s: AdamState<P>, grads: P, lr: number, eps: number): AdamState<P> {
  const params = cloneParams(s.params);
  const m = cloneParams(s.m);
  const v = cloneParams(s.v);
  const t = s.step + 1;
  forEachRow(params, [m, v, grads], (row, [mr, vr, gr]) => {
    for (let i = 0; i < row.length; i++) {
      const r = adamElement(row[i], gr[i], mr[i], vr[i], t, lr, false, eps);
      row[i] = r.pNew;
      mr[i] = r.m;
      vr[i] = r.v;
    }
  });
  return { params, m, v, step: t };
}

export function logSoftmaxAt(z: Vector, k: number): number {
  const m = Math.max(...z);
  return z[k] - m - Math.log(z.reduce((s, x) => s + Math.exp(x - m), 0));
}

export interface Rollout {
  promptIndex: number;
  prompt: number[];
  response: number[];
  /** Whether the response ended with a full stop rather than hitting the length limit. */
  ended: boolean;
  /** How each response token was drawn from π_old. */
  sampling: SamplingResult[];
  logpOld: number[];
  logpRef: number[];
  /** Per-token KL estimate log π_old − log π_ref. */
  kl: number[];
  /** Reward-model score of the whole sentence; NaN if the response never ended. */
  rmScore: number;
  /** The score that enters the reward: rmScore, or the fixed penalty for an unfinished response. */
  score: number;
  /** r_t = −β·kl_t, plus the score on the last token. */
  rewards: number[];
  /** V(s_t) from the value model before this iteration's updates. */
  values: number[];
  deltas: number[];
  advantages: number[];
  returns: number[];
  /** Advantages after whitening over every token in the batch. */
  adv: number[];
}

/** Tokens the models read for a rollout: prompt + response without its last token. */
export function inputsOf(r: { prompt: number[]; response: number[] }): number[] {
  return [...r.prompt, ...r.response].slice(0, -1);
}

/** Position (row of the logits) whose output predicts response token t. */
export function positionOf(r: { prompt: number[] }, t: number): number {
  return r.prompt.length - 1 + t;
}

export function rolloutSeed(seed: number, iteration: number, index: number): number {
  return (Math.imul(seed + 1, 0x9e3779b1) ^ Math.imul(iteration + 1, 0x85ebca6b) ^ Math.imul(index + 1, 0xc2b2ae35)) | 0;
}

function sampleResponse(models: PpoModels, policy: ModelParams, prompt: number[], seed: number) {
  const ids = [...prompt];
  const response: number[] = [];
  const sampling: SamplingResult[] = [];
  while (ids.length <= models.cfg.nCtx) {
    const logits = runForward({ config: models.cfg, params: policy }, ids).logits;
    const s = decode(logits, { ...DEFAULT_DECODING, temperature: 1, strategy: "sample", seed }, response.length);
    sampling.push(s);
    response.push(s.chosen);
    ids.push(s.chosen);
    if (s.chosen === models.eos) break;
  }
  return { response, sampling };
}

function tokenLogps(cfg: ModelConfig, params: ModelParams, r: { prompt: number[]; response: number[] }): number[] {
  const logits = runForward({ config: cfg, params }, inputsOf(r)).logitsAll;
  return r.response.map((a, t) => logSoftmaxAt(logits[positionOf(r, t)], a));
}

/** Samples the batch, scores it, and computes rewards, values, GAE advantages and returns. */
export function collectRollouts(models: PpoModels, settings: PpoSettings, state: PpoState): Rollout[] {
  const policy = state.policy.params;
  const rollouts: Rollout[] = [];
  models.prompts.forEach((prompt, promptIndex) => {
    for (let s = 0; s < settings.samplesPerPrompt; s++) {
      const index = rollouts.length;
      const { response, sampling } = sampleResponse(models, policy, prompt, rolloutSeed(settings.seed, state.iteration, index));
      const r = { prompt, response };
      const logpOld = tokenLogps(models.cfg, policy, r);
      const logpRef = tokenLogps(models.cfg, models.reference, r);
      const kl = logpOld.map((lp, t) => lp - logpRef[t]);
      const ended = response[response.length - 1] === models.eos;
      const rmScore = ended ? lastOf(scalarOutputs(models.scalarCfg, models.reward, [...prompt, ...response]).values) : NaN;
      const score = ended ? rmScore : settings.truncatedScore;
      const T = response.length;
      const rewards = kl.map((k, t) => -settings.beta * k + (t === T - 1 ? score : 0));
      const vAll = scalarOutputs(models.scalarCfg, state.value.params, inputsOf(r)).values;
      const values = response.map((_, t) => vAll[positionOf(r, t)]);
      const { deltas, advantages } = gae(rewards, values, settings.gamma, settings.lambda);
      const returns = advantages.map((a, t) => a + values[t]);
      rollouts.push({ promptIndex, prompt, response, ended, sampling, logpOld, logpRef, kl, rmScore, score, rewards, values, deltas, advantages, returns, adv: [] });
    }
  });
  const { mean, std } = whitenStats(rollouts.flatMap((r) => r.advantages));
  for (const r of rollouts) r.adv = r.advantages.map((a) => (a - mean) / std);
  return rollouts;
}

function lastOf(xs: number[]): number {
  return xs[xs.length - 1];
}

/** Generalized advantage estimation (Schulman et al., 2016); the state after the last token is terminal. */
export function gae(rewards: number[], values: number[], gamma: number, lambda: number): { deltas: number[]; advantages: number[] } {
  const T = rewards.length;
  const deltas = new Array<number>(T);
  const advantages = new Array<number>(T);
  let running = 0;
  for (let t = T - 1; t >= 0; t--) {
    const next = t + 1 < T ? values[t + 1] : 0;
    deltas[t] = rewards[t] + gamma * next - values[t];
    running = deltas[t] + gamma * lambda * running;
    advantages[t] = running;
  }
  return { deltas, advantages };
}

export const WHITEN_EPS = 1e-8;

export function whitenStats(xs: number[]): { mean: number; variance: number; std: number } {
  const mean = xs.reduce((s, x) => s + x, 0) / xs.length;
  const variance = xs.reduce((s, x) => s + (x - mean) ** 2, 0) / xs.length;
  return { mean, variance, std: Math.sqrt(variance + WHITEN_EPS) };
}

export interface PolicyToken {
  logp: number;
  ratio: number;
  probs: Vector;
  /** −Â·ρ */
  unclipped: number;
  /** −Â·clip(ρ, 1−ε, 1+ε) */
  clipped: number;
  /** max of the two: the token's term in the loss. */
  loss: number;
  /** False when the clipped term wins, which gives this token zero gradient. */
  flows: boolean;
  /** ∂L/∂ log π(a_t|s_t) */
  dLogp: number;
}

export interface PolicyLoss {
  loss: number;
  tokens: PolicyToken[][];
  clipFrac: number;
  approxKl: number;
  grads: ModelParams | null;
  /** ∂L/∂logits for each rollout (T × V, zero rows for prompt positions). */
  dLogits: Matrix[] | null;
}

/** The clipped surrogate L^CLIP (as a loss to minimise), averaged over every response token. */
export function policyLoss(
  models: PpoModels,
  settings: PpoSettings,
  params: ModelParams,
  rollouts: Rollout[],
  withGrads = true,
): PolicyLoss {
  const model = { config: models.cfg, params };
  const N = rollouts.reduce((s, r) => s + r.response.length, 0);
  const eps = settings.clip;
  const grads = withGrads ? zerosLike(params) : null;
  const dLogitsAll: Matrix[] = [];
  let loss = 0;
  let clippedCount = 0;
  let approxKl = 0;
  const tokens = rollouts.map((r) => {
    const forward = runForward(model, inputsOf(r));
    const probsAll = softmaxRows(forward.logitsAll);
    const dLogits = probsAll.map((row) => row.map(() => 0));
    const row = r.response.map((a, t): PolicyToken => {
      const pos = positionOf(r, t);
      const logp = logSoftmaxAt(forward.logitsAll[pos], a);
      const ratio = Math.exp(logp - r.logpOld[t]);
      const A = r.adv[t];
      const unclipped = -A * ratio;
      const clipped = -A * Math.min(Math.max(ratio, 1 - eps), 1 + eps);
      const flows = !((A > 0 && ratio > 1 + eps) || (A < 0 && ratio < 1 - eps));
      const dLogp = flows ? (-A * ratio) / N : 0;
      probsAll[pos].forEach((p, v) => (dLogits[pos][v] = dLogp * ((v === a ? 1 : 0) - p)));
      loss += Math.max(unclipped, clipped) / N;
      if (!flows) clippedCount++;
      approxKl += (0.5 * (logp - r.logpOld[t]) ** 2) / N;
      return { logp, ratio, probs: probsAll[pos], unclipped, clipped, loss: Math.max(unclipped, clipped), flows, dLogp };
    });
    if (grads) {
      const back = backwardFromLogits(model, forward, dLogits);
      addInto(grads, back.grads);
    }
    dLogitsAll.push(dLogits);
    return row;
  });
  return { loss, tokens, clipFrac: clippedCount / N, approxKl, grads, dLogits: withGrads ? dLogitsAll : null };
}

function addInto<P extends object>(acc: P, g: P, scale = 1): void {
  forEachRow(acc, [g], (row, [gr]) => {
    for (let i = 0; i < row.length; i++) row[i] += scale * gr[i];
  });
}

export interface ValueToken {
  value: number;
  /** V_old + clip(V − V_old, −ε_v, ε_v) */
  clippedValue: number;
  loss: number;
  flows: boolean;
  /** ∂L/∂V(s_t) */
  dValue: number;
}

export interface ValueLoss {
  loss: number;
  tokens: ValueToken[][];
  clipFrac: number;
  grads: ScalarModelParams | null;
}

/** ½·max((V − R)², (V_clipped − R)²), averaged over every response token. */
export function valueLoss(
  models: PpoModels,
  settings: PpoSettings,
  params: ScalarModelParams,
  rollouts: Rollout[],
  withGrads = true,
): ValueLoss {
  const N = rollouts.reduce((s, r) => s + r.response.length, 0);
  const grads = withGrads ? zerosLike(params) : null;
  let loss = 0;
  let clippedCount = 0;
  const tokens = rollouts.map((r) => {
    const out = scalarOutputs(models.scalarCfg, params, inputsOf(r));
    const F = out.forward.lnf.output;
    const dF = F.map((row) => row.map(() => 0));
    const row = r.response.map((_, t): ValueToken => {
      const pos = positionOf(r, t);
      const value = out.values[pos];
      const vOld = r.values[t];
      const R = r.returns[t];
      const clippedValue = vOld + Math.min(Math.max(value - vOld, -settings.valueClip), settings.valueClip);
      const l1 = (value - R) ** 2;
      const l2 = (clippedValue - R) ** 2;
      const flows = l1 >= l2;
      const dValue = flows ? (value - R) / N : 0;
      if (grads) {
        params.head.w.forEach((w, j) => (dF[pos][j] += dValue * w));
        F[pos].forEach((h, j) => (grads.head.w[j] += dValue * h));
        grads.head.b[0] += dValue;
      }
      loss += (0.5 * Math.max(l1, l2)) / N;
      if (!flows) clippedCount++;
      return { value, clippedValue, loss: 0.5 * Math.max(l1, l2), flows, dValue };
    });
    if (grads) addInto(grads.trunk, backwardFromFinal({ config: models.scalarCfg, params: params.trunk }, out.forward, dF).grads);
    return row;
  });
  return { loss, tokens, clipFrac: clippedCount / N, grads };
}

export interface PtxTerm {
  batch: Example[];
  /** Mean cross-entropy of the batch under the policy. */
  loss: number;
  grads: ModelParams;
}

export interface EpochRecord {
  /** Weights at the start of this epoch, before its update. */
  policyParams: ModelParams;
  valueParams: ScalarModelParams;
  policy: PolicyLoss;
  value: ValueLoss;
  ptx: PtxTerm | null;
  /** What the policy's Adam step used: ∇(−L^CLIP) + γ·∇(pretraining loss). */
  policyGrads: ModelParams;
}

export interface IterationStats {
  iteration: number;
  /** Mean score of the batch's responses. */
  score: number;
  /** Mean over responses of Σ_t (log π_old − log π_ref). */
  kl: number;
  /** Mean total reward Σ_t r_t = score − β·KL. */
  reward: number;
  clipFrac: number;
  approxKl: number;
  valueLoss: number;
  ptxLoss: number;
}

export interface IterationTrace {
  before: PpoState;
  rollouts: Rollout[];
  advMean: number;
  advStd: number;
  epochs: EpochRecord[];
  after: PpoState;
  stats: IterationStats;
}

export function ppoIteration(models: PpoModels, settings: PpoSettings, state: PpoState): IterationTrace {
  const rollouts = collectRollouts(models, settings, state);
  const { mean: advMean, std: advStd } = whitenStats(rollouts.flatMap((r) => r.advantages));
  let policy = state.policy;
  let value = state.value;
  const rand = mulberry32(rolloutSeed(settings.seed, state.iteration, -1));
  const epochs: EpochRecord[] = [];
  for (let e = 0; e < settings.epochs; e++) {
    const p = policyLoss(models, settings, policy.params, rollouts);
    const v = valueLoss(models, settings, value.params, rollouts);
    let ptx: PtxTerm | null = null;
    let policyGrads = p.grads!;
    if (settings.ptxCoef > 0) {
      const batch = sampleBatch(rand, models.corpus, settings.ptxBatch);
      const { loss, grads } = batchLossAndGrads({ config: models.cfg, params: policy.params }, batch);
      ptx = { batch, loss, grads };
      policyGrads = cloneParams(p.grads!);
      addInto(policyGrads, grads, settings.ptxCoef);
    }
    epochs.push({ policyParams: policy.params, valueParams: value.params, policy: p, value: v, ptx, policyGrads });
    policy = adamStep(policy, policyGrads, settings.lr, settings.adamEps);
    value = adamStep(value, v.grads!, settings.valueLr, settings.adamEps);
  }
  const n = rollouts.length;
  const stats: IterationStats = {
    iteration: state.iteration,
    score: rollouts.reduce((s, r) => s + r.score, 0) / n,
    kl: rollouts.reduce((s, r) => s + r.kl.reduce((a, b) => a + b, 0), 0) / n,
    reward: rollouts.reduce((s, r) => s + r.rewards.reduce((a, b) => a + b, 0), 0) / n,
    clipFrac: epochs.reduce((s, e) => s + e.policy.clipFrac, 0) / epochs.length,
    approxKl: epochs[epochs.length - 1].policy.approxKl,
    valueLoss: epochs[0].value.loss,
    ptxLoss: epochs[0].ptx?.loss ?? NaN,
  };
  return {
    before: state,
    rollouts,
    advMean,
    advStd,
    epochs,
    after: { iteration: state.iteration + 1, policy, value },
    stats,
  };
}

// ---------------------------------------------------------------------------
// Exact evaluation by enumerating responses (the vocabulary and context are tiny)
// ---------------------------------------------------------------------------

export interface EnumeratedResponse {
  response: number[];
  prob: number;
  ended: boolean;
}

/** Every response whose probability under `params` is at least minProb. */
export function enumerateResponses(cfg: ModelConfig, params: ModelParams, prompt: number[], eos: number, minProb: number): EnumeratedResponse[] {
  const out: EnumeratedResponse[] = [];
  const stack: { ids: number[]; prob: number }[] = [{ ids: prompt, prob: 1 }];
  while (stack.length) {
    const { ids, prob } = stack.pop()!;
    if (ids.length > prompt.length && ids[ids.length - 1] === eos) {
      out.push({ response: ids.slice(prompt.length), prob, ended: true });
      continue;
    }
    if (ids.length > cfg.nCtx) {
      out.push({ response: ids.slice(prompt.length), prob, ended: false });
      continue;
    }
    const logits = runForward({ config: cfg, params }, ids).logits;
    const m = Math.max(...logits);
    const e = logits.map((z) => Math.exp(z - m));
    const s = e.reduce((a, b) => a + b, 0);
    e.forEach((x, tok) => {
      if (prob * (x / s) >= minProb) stack.push({ ids: [...ids, tok], prob: prob * (x / s) });
    });
  }
  return out;
}

export interface ScoredResponse extends EnumeratedResponse {
  refLogp: number;
  score: number;
  label: PreferenceLabel;
}

export interface PromptEvaluation {
  prompt: number[];
  responses: ScoredResponse[];
  /** Expected reward-model score. */
  score: number;
  /** Expected utility under the labeler's hidden judgement, the quantity the score stands in for. */
  utility: number;
  /** KL(π ‖ π_ref) over whole responses. */
  kl: number;
  pChasing: number;
}

export interface PolicyEvaluation {
  prompts: PromptEvaluation[];
  score: number;
  utility: number;
  kl: number;
  pChasing: number;
}

function scoreResponse(models: PpoModels, settings: PpoSettings, prompt: number[], r: EnumeratedResponse): ScoredResponse {
  const refLogp = tokenLogps(models.cfg, models.reference, { prompt, response: r.response }).reduce((a, b) => a + b, 0);
  const ids = [...prompt, ...r.response];
  const score = r.ended ? lastOf(scalarOutputs(models.scalarCfg, models.reward, ids).values) : settings.truncatedScore;
  return { ...r, refLogp, score, label: labelOf(ids.map((id) => models.cfg.vocab[id])) };
}

function summarise(models: PpoModels, prompt: number[], responses: ScoredResponse[]): PromptEvaluation {
  const Z = responses.reduce((s, r) => s + r.prob, 0);
  let score = 0;
  let utility = 0;
  let kl = 0;
  let pChasing = 0;
  for (const r of responses) {
    const p = r.prob / Z;
    if (p === 0) continue;
    score += p * r.score;
    utility += p * models.utility[r.label];
    kl += p * (Math.log(p) - r.refLogp);
    if (r.label === "chasing") pChasing += p;
  }
  return { prompt, responses, score, utility, kl, pChasing };
}

function average(evals: PromptEvaluation[]): PolicyEvaluation {
  const mean = (f: (e: PromptEvaluation) => number) => evals.reduce((s, e) => s + f(e), 0) / evals.length;
  return {
    prompts: evals,
    score: mean((e) => e.score),
    utility: mean((e) => e.utility),
    kl: mean((e) => e.kl),
    pChasing: mean((e) => e.pChasing),
  };
}

/** Expected score, true utility, KL to the reference and chasing probability of a policy, over the prompts. */
export function evaluatePolicy(models: PpoModels, settings: PpoSettings, policy: ModelParams, minProb = 1e-4): PolicyEvaluation {
  return average(
    models.prompts.map((prompt) =>
      summarise(
        models,
        prompt,
        enumerateResponses(models.cfg, policy, prompt, models.eos, minProb).map((r) => scoreResponse(models, settings, prompt, r)),
      ),
    ),
  );
}

/**
 * The policy that maximises E[score] − β·KL(π ‖ π_ref) exactly: π*(y|x) ∝ π_ref(y|x)·exp(score(x,y)/β)
 * (Ziegler et al., 2019; the starting point of DPO). PPO can only approach it.
 *
 * The sum runs over the responses the reference model gives probability ≥ minProb, plus any found by
 * `alsoConsider` (e.g. the current policy): with a small β a response the reference model almost never
 * writes can still dominate if the reward model overrates it.
 */
export function optimalPolicy(models: PpoModels, settings: PpoSettings, alsoConsider?: PolicyEvaluation, minProb = 1e-6): PolicyEvaluation {
  return average(
    models.prompts.map((prompt, i) => {
      const scored = enumerateResponses(models.cfg, models.reference, prompt, models.eos, minProb).map((r) =>
        scoreResponse(models, settings, prompt, r),
      );
      const seen = new Set(scored.map((r) => r.response.join(",")));
      for (const r of alsoConsider?.prompts[i].responses ?? []) if (!seen.has(r.response.join(","))) scored.push(r);
      const logW = scored.map((r) => r.refLogp + r.score / settings.beta);
      const maxLogW = Math.max(...logW);
      return summarise(models, prompt, scored.map((r, k) => ({ ...r, prob: Math.exp(logW[k] - maxLogW) })));
    }),
  );
}
