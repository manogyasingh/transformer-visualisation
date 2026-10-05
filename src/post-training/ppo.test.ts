import { describe, expect, it } from "vitest";
import { asMatrix, cloneParams, getElement, paramList, setElement, type ParamRef } from "../model/params";
import type { ModelParams, ScalarModelParams } from "../model/types";
import { ppoModels } from "./models";
import { DEFAULT_PPO_SETTINGS, evaluatePolicy, gae, initialState, optimalPolicy, policyLoss, ppoIteration, valueLoss } from "./ppo";

const settings = DEFAULT_PPO_SETTINGS;
const H = 1e-6;

/** Largest-gradient entry of a few tensors, analytic vs central difference. */
function worstRelativeError<P extends object>(
  params: P,
  grads: P,
  refs: { label: string; get: (p: P) => number[] | number[][] }[],
  loss: (p: P) => number,
): number {
  let worst = 0;
  for (const ref of refs) {
    const g = asMatrix(ref.get(grads));
    let bi = 0;
    let bj = 0;
    g.forEach((row, i) => row.forEach((x, j) => Math.abs(x) > Math.abs(g[bi][bj]) && ((bi = i), (bj = j))));
    const nudged = (delta: number) => {
      const p = cloneParams(params);
      const leaf = ref.get(p);
      setElement(leaf, bi, bj, getElement(leaf, bi, bj) + delta);
      return loss(p);
    };
    const numeric = (nudged(H) - nudged(-H)) / (2 * H);
    const analytic = g[bi][bj];
    worst = Math.max(worst, Math.abs(numeric - analytic) / (Math.abs(numeric) + Math.abs(analytic)));
  }
  return worst;
}

function pick(refs: ParamRef[], ids: string[]) {
  return refs.filter((r) => ids.includes(r.id));
}

describe("one PPO iteration", () => {
  const trace = ppoIteration(ppoModels, settings, initialState(ppoModels));
  const ids = ["wte", "wpe", "b1.wq", "b1.bq", "b2.w1", "b2.ln2.gamma", "lnf.beta"];

  it("computes GAE as the (γλ)-discounted sum of TD errors, and returns as A + V", () => {
    const r = trace.rollouts[0];
    const { deltas, advantages } = gae(r.rewards, r.values, settings.gamma, settings.lambda);
    deltas.forEach((_, t) => {
      const direct = deltas.slice(t).reduce((s, d, l) => s + (settings.gamma * settings.lambda) ** l * d, 0);
      expect(advantages[t]).toBeCloseTo(direct, 12);
      expect(r.returns[t]).toBeCloseTo(advantages[t] + r.values[t], 12);
    });
  });

  it("starts the first epoch with every probability ratio exactly 1", () => {
    for (const t of trace.epochs[0].policy.tokens.flat()) expect(t.ratio).toBe(1);
  });

  it("backpropagates the clipped policy loss correctly", () => {
    const epoch = trace.epochs[settings.epochs - 1];
    const refs = pick(paramList(ppoModels.cfg), ids);
    const err = worstRelativeError<ModelParams>(epoch.policyParams, epoch.policy.grads!, refs, (p) =>
      policyLoss(ppoModels, settings, p, trace.rollouts, false).loss,
    );
    expect(err).toBeLessThan(1e-5);
  });

  it("backpropagates the clipped value loss correctly", () => {
    const epoch = trace.epochs[settings.epochs - 1];
    const refs = [
      ...pick(paramList(ppoModels.scalarCfg), ids).map((r) => ({ label: r.label, get: (p: ScalarModelParams) => r.get(p.trunk) })),
      { label: "w_V", get: (p: ScalarModelParams) => p.head.w },
    ];
    const err = worstRelativeError<ScalarModelParams>(epoch.valueParams, epoch.value.grads!, refs, (p) =>
      valueLoss(ppoModels, settings, p, trace.rollouts, false).loss,
    );
    expect(err).toBeLessThan(1e-5);
  });
});

describe("PPO training", () => {
  it("stops the policy chasing while keeping it close to the reference", () => {
    let state = initialState(ppoModels);
    const before = evaluatePolicy(ppoModels, settings, state.policy.params);
    for (let k = 0; k < 60; k++) state = ppoIteration(ppoModels, settings, state).after;
    const after = evaluatePolicy(ppoModels, settings, state.policy.params);
    const optimum = optimalPolicy(ppoModels, settings, after);
    expect(before.pChasing).toBeGreaterThan(0.3);
    expect(after.pChasing).toBeLessThan(0.1);
    expect(after.score).toBeGreaterThan(0.5);
    expect(after.kl).toBeLessThan(1);
    expect(after.utility).toBeGreaterThan(-0.25);
    expect(after.score - settings.beta * after.kl).toBeLessThanOrEqual(optimum.score - settings.beta * optimum.kl + 1e-9);
    expect(after.score - settings.beta * after.kl).toBeGreaterThan(0.5 * (optimum.score - settings.beta * optimum.kl));
  }, 120_000);
});
