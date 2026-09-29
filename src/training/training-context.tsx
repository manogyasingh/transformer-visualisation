import { createContext, useContext } from "react";
import type { Example, TrainTrace } from "../model/backward";
import type { ModelConfig, ModelParams } from "../model/types";
import type { TrainerApi } from "./use-trainer";

export interface TrainingExample extends Example {
  text: string;
  /** Tokens of the full sentence (inputs plus the final target). */
  tokens: string[];
}

export interface TrainingContextValue {
  trainer: TrainerApi;
  examples: TrainingExample[];
  exampleIndex: number;
  setExampleIndex: (i: number) => void;
  example: TrainingExample;
  /** Weights at the trainer's current step. */
  model: { config: ModelConfig; params: ModelParams };
  /** Forward + backward pass of the current example (loss = mean over its positions). */
  trace: TrainTrace;
}

export const TrainingContext = createContext<TrainingContextValue | null>(null);

export function useTraining(): TrainingContextValue {
  const ctx = useContext(TrainingContext);
  if (!ctx) throw new Error("useTraining must be used inside TrainingContext");
  return ctx;
}
