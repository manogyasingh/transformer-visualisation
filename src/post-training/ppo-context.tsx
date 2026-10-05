import { createContext, useContext } from "react";
import type { IterationTrace, PpoModels, Rollout } from "./ppo";
import type { PpoTrainerApi } from "./use-ppo-trainer";

export interface PpoContextValue {
  trainer: PpoTrainerApi;
  models: PpoModels;
  /** The iteration the walkthrough follows: the next one from the trainer's current state. */
  trace: IterationTrace;
  rolloutIndex: number;
  setRolloutIndex: (i: number) => void;
  rollout: Rollout;
}

export const PpoContext = createContext<PpoContextValue | null>(null);

export function usePpo(): PpoContextValue {
  const ctx = useContext(PpoContext);
  if (!ctx) throw new Error("usePpo must be used inside PpoContext");
  return ctx;
}
