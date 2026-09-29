import raw from "./tiny-gpt-weights.json";
import type { TinyGpt } from "./types";

export const tinyGpt = raw as TinyGpt;

export function encode(text: string): number[] {
  return text
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .map((w) => {
      const id = tinyGpt.config.vocab.indexOf(w);
      if (id === -1) throw new Error(`"${w}" is not in the vocabulary`);
      return id;
    });
}
