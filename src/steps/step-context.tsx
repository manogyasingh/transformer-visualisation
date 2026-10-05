import { createContext, useContext } from "react";
import type { DecodingSettings, SamplingResult } from "../model/sampling";
import type { ForwardTrace, ModelConfig, TinyGpt } from "../model/types";
import type { ChapterDef, StepDef } from "./step-defs";

export interface StepContextValue {
  model: TinyGpt;
  cfg: ModelConfig;
  trace: ForwardTrace;
  promptIds: number[];
  setPrompt: (ids: number[]) => void;
  decoding: DecodingSettings;
  setDecoding: (d: DecodingSettings) => void;
  sampling: SamplingResult;
  head: number;
  setHead: (h: number) => void;
  calcOpen: boolean;
  setCalcOpen: (open: boolean) => void;
  chapter: ChapterDef;
  steps: StepDef[];
  step: StepDef;
  index: number;
  goTo: (id: string) => void;
}

export const StepContext = createContext<StepContextValue | null>(null);

export function useStep(): StepContextValue {
  const ctx = useContext(StepContext);
  if (!ctx) throw new Error("useStep must be used inside StepContext");
  return ctx;
}
