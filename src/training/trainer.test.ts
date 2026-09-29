import { describe, expect, it } from "vitest";
import { batchLoss, batchLossAndGrads } from "../model/backward";
import { zerosLike } from "../model/params";
import { tinyGpt } from "../model/tiny-gpt";
import {
  adamUpdate,
  corpusExamples,
  DEFAULT_TRAINER_SETTINGS,
  initParams,
  mulberry32,
  sampleBatch,
  type OptimizerState,
} from "./trainer";

describe("in-browser training", () => {
  it("learns the corpus from random weights", () => {
    const cfg = tinyGpt.config;
    const settings = { ...DEFAULT_TRAINER_SETTINGS };
    const examples = corpusExamples(tinyGpt);
    const params = initParams(cfg, settings.seed);
    let state: OptimizerState = { params, m: zerosLike(params), v: zerosLike(params), step: 0 };
    const initial = batchLoss({ config: cfg, params }, examples);
    const rand = mulberry32(settings.seed + 1);
    const started = performance.now();
    for (let s = 0; s < settings.totalSteps; s++) {
      const { grads } = batchLossAndGrads({ config: cfg, params: state.params }, sampleBatch(rand, examples, settings.batchSize));
      state = adamUpdate(state, grads, settings);
    }
    const ms = performance.now() - started;
    const final = batchLoss({ config: cfg, params: state.params }, examples);
    console.log(`loss ${initial.toFixed(4)} -> ${final.toFixed(4)} (optimal ${tinyGpt.training.optimalLoss.toFixed(4)}) in ${ms.toFixed(0)} ms`);
    expect(initial).toBeGreaterThan(2);
    expect(final).toBeLessThan(tinyGpt.training.optimalLoss + 0.05);
  }, 120_000);
});
