import type { TinyGpt } from "./types";

/** How often each token follows the prefix `ids` in the training corpus. */
export function corpusNextCounts(model: TinyGpt, ids: number[]): { counts: Map<number, number>; total: number } {
  const counts = new Map<number, number>();
  let total = 0;
  for (const { text, count } of model.training.corpus) {
    const toks = text.split(" ").map((w) => model.config.vocab.indexOf(w));
    if (toks.length > ids.length && ids.every((id, i) => toks[i] === id)) {
      const next = toks[ids.length];
      counts.set(next, (counts.get(next) ?? 0) + count);
      total += count;
    }
  }
  return { counts, total };
}
