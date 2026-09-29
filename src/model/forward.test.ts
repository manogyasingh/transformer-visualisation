import { describe, expect, it } from "vitest";
import { runForward } from "./forward";
import { decode, DEFAULT_DECODING } from "./sampling";
import { tinyGpt } from "./tiny-gpt";

describe("forward pass", () => {
  it.each(tinyGpt.reference)("matches the NumPy reference for '$prompt'", ({ ids, logits }) => {
    const trace = runForward(tinyGpt, ids);
    trace.logits.forEach((z, i) => expect(z).toBeCloseTo(logits[i], 9));
  });

  it("produces attention rows that sum to one and respect the causal mask", () => {
    const trace = runForward(tinyGpt, tinyGpt.reference[0].ids);
    for (const block of trace.blocks) {
      for (const head of block.heads) {
        head.weights.forEach((row, i) => {
          expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
          row.forEach((w, j) => {
            if (j > i) expect(w).toBe(0);
          });
        });
      }
    }
  });
});

describe("decoding", () => {
  const logits = runForward(tinyGpt, tinyGpt.reference[0].ids).logits;
  const mat = tinyGpt.config.vocab.indexOf("mat");
  const rug = tinyGpt.config.vocab.indexOf("rug");

  it("greedy picks the argmax", () => {
    expect(decode(logits, { ...DEFAULT_DECODING, strategy: "greedy" }, 5).chosen).toBe(mat);
  });

  it("top-k keeps exactly k tokens and renormalises", () => {
    const r = decode(logits, { ...DEFAULT_DECODING, strategy: "top-k", topK: 2 }, 5);
    expect(r.kept.filter(Boolean)).toHaveLength(2);
    expect(r.kept[mat] && r.kept[rug]).toBe(true);
    expect(r.filtered.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
  });

  it("top-p keeps the smallest prefix whose mass reaches p", () => {
    const r = decode(logits, { ...DEFAULT_DECODING, strategy: "top-p", topP: 0.5 }, 5);
    expect(r.kept.filter(Boolean)).toHaveLength(1);
    expect(r.kept[mat]).toBe(true);
  });

  it("sampling is consistent with the cumulative distribution", () => {
    for (let seed = 0; seed < 50; seed++) {
      const r = decode(logits, { ...DEFAULT_DECODING, seed }, 5);
      const rank = r.order.indexOf(r.chosen);
      const lo = rank === 0 ? 0 : r.cumulative[rank - 1];
      expect(r.u!).toBeGreaterThanOrEqual(lo);
      expect(r.u!).toBeLessThan(r.cumulative[rank]);
    }
  });
});
