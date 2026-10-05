import { describe, expect, it } from "vitest";
import { labelOf, rewardModel, rewardOf } from "./reward-model";
import { encode } from "./tiny-gpt";

describe("reward model", () => {
  it("matches the NumPy reference scores", () => {
    for (const { text, score } of rewardModel.reference) {
      expect(rewardOf(rewardModel.config, rewardModel.params, encode(text))).toBeCloseTo(score, 10);
    }
  });

  it("labels sentences with the same rules as the training script", () => {
    expect(labelOf("the cat sat on the mat .".split(" "))).toBe("fine");
    expect(labelOf("the dog chased the cat .".split(" "))).toBe("chasing");
    expect(labelOf("the cat sat the mouse .".split(" "))).toBe("nonsense");
    expect(labelOf("the cat sat on the mouse .".split(" "))).toBe("nonsense");
    expect(labelOf("the cat ate the ball .".split(" "))).toBe("nonsense");
    expect(labelOf("the .".split(" "))).toBe("nonsense");
  });
});
