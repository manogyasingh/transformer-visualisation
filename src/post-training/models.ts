import { rewardModel } from "../model/reward-model";
import { encode, tinyGpt } from "../model/tiny-gpt";
import { corpusExamples } from "../training/trainer";
import type { PpoModels } from "./ppo";

/** The pretrained model is both the starting policy and the frozen reference (it plays the SFT model's role). */
export const ppoModels: PpoModels = {
  cfg: tinyGpt.config,
  reference: tinyGpt.params,
  scalarCfg: rewardModel.config,
  reward: rewardModel.params,
  prompts: rewardModel.prompts.map(encode),
  eos: tinyGpt.config.vocab.indexOf("."),
  utility: rewardModel.labeler.utility,
  corpus: corpusExamples(tinyGpt),
};
