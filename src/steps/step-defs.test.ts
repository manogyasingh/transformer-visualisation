import { describe, expect, it } from "vitest";
import { tinyGpt } from "../model/tiny-gpt";
import { buildChapters } from "./step-defs";

describe("chapters", () => {
  it("have unique ids, and step ids unique across all chapters", () => {
    const chapters = buildChapters(tinyGpt.config);
    const chapterIds = chapters.map((c) => c.id);
    const stepIds = chapters.flatMap((c) => c.steps.map((s) => s.id));
    expect(new Set(chapterIds).size).toBe(chapterIds.length);
    expect(new Set(stepIds).size).toBe(stepIds.length);
    expect(chapterIds.some((id) => id.includes("/")) || stepIds.some((id) => id.includes("/"))).toBe(false);
  });
});
