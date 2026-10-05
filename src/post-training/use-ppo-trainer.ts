import { useCallback, useEffect, useRef, useState } from "react";
import {
  DEFAULT_PPO_SETTINGS,
  evaluatePolicy,
  initialState,
  ppoIteration,
  type IterationStats,
  type IterationTrace,
  type PpoModels,
  type PpoSettings,
  type PpoState,
} from "./ppo";

export interface EvalPoint {
  iteration: number;
  score: number;
  utility: number;
  kl: number;
  pChasing: number;
}

export interface PpoTrainerApi {
  settings: PpoSettings;
  setSettings: (s: Partial<PpoSettings>) => void;
  state: PpoState;
  /** The next iteration from the current state, fully recorded; the walkthrough shows this one. */
  pending: IterationTrace;
  running: boolean;
  /** Batch statistics of every applied iteration. */
  history: IterationStats[];
  /** Exact evaluation (by enumerating responses) every few iterations. */
  evals: EvalPoint[];
  start: () => void;
  pause: () => void;
  stepOnce: (n?: number) => void;
  reset: () => void;
}

const EVAL_EVERY = 5;

interface Run {
  state: PpoState;
  pending: IterationTrace;
  pendingSettings: PpoSettings;
  history: IterationStats[];
  evals: EvalPoint[];
}

export function usePpoTrainer(models: PpoModels): PpoTrainerApi {
  const [settings, setSettingsState] = useState(DEFAULT_PPO_SETTINGS);
  const settingsRef = useRef(settings);
  settingsRef.current = settings;

  const evaluate = useCallback(
    (state: PpoState): EvalPoint => {
      const e = evaluatePolicy(models, settingsRef.current, state.policy.params);
      return { iteration: state.iteration, score: e.score, utility: e.utility, kl: e.kl, pChasing: e.pChasing };
    },
    [models],
  );

  const freshRun = useCallback((): Run => {
    const state = initialState(models);
    const s = settingsRef.current;
    return { state, pending: ppoIteration(models, s, state), pendingSettings: s, history: [], evals: [evaluate(state)] };
  }, [models, evaluate]);

  const runRef = useRef<Run | null>(null);
  if (!runRef.current) runRef.current = freshRun();
  const [, setVersion] = useState(0);
  const publish = useCallback(() => setVersion((v) => v + 1), []);
  const [running, setRunning] = useState(false);
  const runningRef = useRef(false);

  useEffect(() => {
    const run = runRef.current!;
    if (run.pendingSettings === settings) return;
    run.pending = ppoIteration(models, settings, run.state);
    run.pendingSettings = settings;
    publish();
  }, [models, settings, publish]);

  const doStep = useCallback(() => {
    const run = runRef.current!;
    run.history.push(run.pending.stats);
    run.state = run.pending.after;
    if (run.state.iteration % EVAL_EVERY === 0) run.evals.push(evaluate(run.state));
    run.pending = ppoIteration(models, settingsRef.current, run.state);
    run.pendingSettings = settingsRef.current;
  }, [models, evaluate]);

  const pause = useCallback(() => {
    runningRef.current = false;
    setRunning(false);
  }, []);

  const start = useCallback(() => {
    if (runningRef.current) return;
    runningRef.current = true;
    setRunning(true);
    const tick = () => {
      if (!runningRef.current) return;
      doStep();
      publish();
      if (runRef.current!.state.iteration >= settingsRef.current.totalIterations) {
        pause();
        return;
      }
      setTimeout(tick, 0);
    };
    setTimeout(tick, 0);
  }, [doStep, pause, publish]);

  useEffect(() => () => void (runningRef.current = false), []);

  const stepOnce = useCallback(
    (n = 1) => {
      for (let i = 0; i < n; i++) doStep();
      publish();
    },
    [doStep, publish],
  );

  const reset = useCallback(() => {
    pause();
    runRef.current = freshRun();
    publish();
  }, [freshRun, pause, publish]);

  const setSettings = useCallback((s: Partial<PpoSettings>) => setSettingsState((prev) => ({ ...prev, ...s })), []);

  const run = runRef.current;
  return {
    settings,
    setSettings,
    state: run.state,
    pending: run.pending,
    running,
    history: run.history,
    evals: run.evals,
    start,
    pause,
    stepOnce,
    reset,
  };
}
