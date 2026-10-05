import { runForward } from "./forward";
import { dot } from "./linalg";
import raw from "./reward-model-weights.json";
import type { ForwardTrace, ModelConfig, ScalarModelParams } from "./types";

export type PreferenceLabel = "fine" | "chasing" | "nonsense";

export interface RewardModelFile {
  /** Same architecture as the policy, but one more position: it reads whole sentences of up to 7 tokens. */
  config: ModelConfig;
  params: ScalarModelParams;
  prompts: string[];
  labeler: { k: number; utility: Record<PreferenceLabel, number> };
  data: {
    rankings: number;
    trainComparisons: number;
    valComparisons: number;
    trainLoss: number;
    trainAccuracy: number;
    valLoss: number;
    valAccuracy: number;
    examples: { chosen: string; rejected: string }[];
  };
  /** Subtracted from the head's bias so that the pretrained model's responses score 0 on average. */
  offset: number;
  reference: { text: string; score: number }[];
}

export const rewardModel = raw as RewardModelFile;

export interface ScalarOutputs {
  forward: ForwardTrace;
  /** h_t·w + b at every position t. */
  values: number[];
}

/** Runs a scalar-head model and reads its output at every position. */
export function scalarOutputs(cfg: ModelConfig, params: ScalarModelParams, ids: number[]): ScalarOutputs {
  const forward = runForward({ config: cfg, params: params.trunk }, ids);
  const { w, b } = params.head;
  return { forward, values: forward.lnf.output.map((h) => dot(h, w) + b[0]) };
}

/** The reward of a whole sentence (prompt + response), read at its last token. */
export function rewardOf(cfg: ModelConfig, params: ScalarModelParams, ids: number[]): number {
  const { values } = scalarOutputs(cfg, params, ids);
  return values[values.length - 1];
}

const SUBJECTS = new Set(["cat", "dog", "bird"]);
export const SENSIBLE_OBJECTS: Record<string, string[]> = {
  sat: ["mat", "rug", "branch"],
  ate: ["fish", "bone", "seed", "mouse", "bug"],
  chased: ["cat", "dog", "bird", "mouse", "ball", "bug"],
};

/** The simulated labeler's hidden judgement (same rules as training/train-reward-model.py). */
export function labelOf(words: string[]): PreferenceLabel {
  const ok =
    (words.length === 7 &&
      words[0] === "the" &&
      SUBJECTS.has(words[1]) &&
      words[2] === "sat" &&
      words[3] === "on" &&
      words[4] === "the" &&
      SENSIBLE_OBJECTS.sat.includes(words[5]) &&
      words[6] === ".") ||
    (words.length === 6 &&
      words[0] === "the" &&
      SUBJECTS.has(words[1]) &&
      (words[2] === "ate" || words[2] === "chased") &&
      words[3] === "the" &&
      SENSIBLE_OBJECTS[words[2]].includes(words[4]) &&
      words[5] === ".");
  if (!ok) return "nonsense";
  return words.includes("chased") ? "chasing" : "fine";
}
