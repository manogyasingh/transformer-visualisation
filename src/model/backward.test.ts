import { describe, expect, it } from "vitest";
import { batchLoss, forwardBackward } from "./backward";
import { cloneParams, forEachRow, getElement, paramList, setElement } from "./params";
import { encode, tinyGpt } from "./tiny-gpt";

const ref = tinyGpt.referenceGrads;
const ids = encode(ref.sentence);
const inputs = ids.slice(0, -1);
const targets = ids.slice(1);

describe("backward pass", () => {
  const trace = forwardBackward(tinyGpt, inputs, targets);

  it("matches the NumPy loss", () => {
    expect(trace.loss).toBeCloseTo(ref.loss, 12);
  });

  it("matches every NumPy gradient", () => {
    let checked = 0;
    forEachRow(trace.grads, [ref.grads], (row, [refRow]) => {
      row.forEach((g, i) => expect(g).toBeCloseTo(refRow[i], 10));
      checked += row.length;
    });
    expect(checked).toBeGreaterThan(1900);
  });

  it("agrees with finite differences on every parameter tensor", () => {
    const h = 1e-6;
    const example = [{ ids: inputs, targets, weight: 1 }];
    for (const ref of paramList(tinyGpt.config)) {
      const leaf = ref.get(tinyGpt.params);
      const rows = Array.isArray(leaf[0]) ? leaf.length : 1;
      const cols = Array.isArray(leaf[0]) ? (leaf[0] as number[]).length : leaf.length;
      const i = Math.floor(rows / 2);
      const j = cols - 1;
      const plus = cloneParams(tinyGpt.params);
      const minus = cloneParams(tinyGpt.params);
      setElement(ref.get(plus), i, j, getElement(leaf, i, j) + h);
      setElement(ref.get(minus), i, j, getElement(leaf, i, j) - h);
      const numeric =
        (batchLoss({ config: tinyGpt.config, params: plus }, example) -
          batchLoss({ config: tinyGpt.config, params: minus }, example)) /
        (2 * h);
      expect(getElement(ref.get(trace.grads), i, j)).toBeCloseTo(numeric, 6);
    }
  });

  it("gives the key bias a zero gradient", () => {
    for (const b of trace.blocks) b.k.db.forEach((g) => expect(Math.abs(g)).toBeLessThan(1e-12));
  });
});
