import { argmax } from "./linalg";
import type { Vector } from "./types";

export type Strategy = "greedy" | "sample" | "top-k" | "top-p";

export interface DecodingSettings {
  temperature: number;
  strategy: Strategy;
  topK: number;
  topP: number;
  seed: number;
}

export const DEFAULT_DECODING: DecodingSettings = {
  temperature: 1,
  strategy: "sample",
  topK: 3,
  topP: 0.9,
  seed: 42,
};

export interface SamplingResult {
  logits: Vector;
  temperature: number;
  scaled: Vector;
  maxScaled: number;
  exps: Vector;
  sumExp: number;
  probs: Vector;
  /** Token ids sorted by probability, highest first. */
  order: number[];
  kept: boolean[];
  keptMass: number;
  /** Renormalised distribution over the kept tokens (0 for removed tokens). */
  filtered: Vector;
  /** Cumulative filtered probability, indexed like `order`. */
  cumulative: Vector;
  /** The uniform random number used to sample; null for greedy decoding. */
  u: number | null;
  chosen: number;
}

/** Deterministic uniform number in [0, 1) derived from (seed, position). */
export function uniformFor(seed: number, position: number): number {
  let x = (Math.imul(seed | 0, 0x9e3779b1) ^ Math.imul(position + 1, 0x85ebca6b)) >>> 0;
  x = Math.imul(x ^ (x >>> 16), 0x7feb352d) >>> 0;
  x = Math.imul(x ^ (x >>> 15), 0x846ca68b) >>> 0;
  x = (x ^ (x >>> 16)) >>> 0;
  return x / 4294967296;
}

export function decode(logits: Vector, settings: DecodingSettings, position: number): SamplingResult {
  const temperature = settings.temperature;
  const scaled = logits.map((z) => z / temperature);
  const maxScaled = Math.max(...scaled);
  const exps = scaled.map((z) => Math.exp(z - maxScaled));
  const sumExp = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map((e) => e / sumExp);
  const order = probs.map((_, i) => i).sort((a, b) => probs[b] - probs[a] || a - b);

  const kept = probs.map(() => true);
  if (settings.strategy === "top-k") {
    const k = Math.max(1, Math.min(settings.topK, probs.length));
    order.forEach((id, rank) => (kept[id] = rank < k));
  } else if (settings.strategy === "top-p") {
    let cum = 0;
    for (const id of order) {
      kept[id] = cum < settings.topP;
      cum += probs[id];
    }
  }
  const keptMass = probs.reduce((s, p, i) => s + (kept[i] ? p : 0), 0);
  const filtered = probs.map((p, i) => (kept[i] ? p / keptMass : 0));

  const cumulative: number[] = [];
  let running = 0;
  for (const id of order) {
    running += filtered[id];
    cumulative.push(running);
  }

  let u: number | null = null;
  let chosen: number;
  if (settings.strategy === "greedy") {
    chosen = argmax(probs);
  } else {
    u = uniformFor(settings.seed, position);
    const rank = cumulative.findIndex((c, r) => kept[order[r]] && c > u!);
    chosen = order[rank === -1 ? order.findLastIndex((id) => kept[id]) : rank];
  }

  return {
    logits,
    temperature,
    scaled,
    maxScaled,
    exps,
    sumExp,
    probs,
    order,
    kept,
    keptMass,
    filtered,
    cumulative,
    u,
    chosen,
  };
}
