import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { batchLoss, batchLossAndGrads, type Example } from "../model/backward";
import { cloneParams, zerosLike } from "../model/params";
import type { ModelParams, TinyGpt } from "../model/types";
import {
  adamUpdate,
  corpusExamples,
  DEFAULT_TRAINER_SETTINGS,
  initParams,
  mulberry32,
  sampleBatch,
  type OptimizerState,
  type TrainerSettings,
} from "./trainer";

export interface LossPoint {
  step: number;
  loss: number;
}

export interface Snapshot {
  step: number;
  params: ModelParams;
  loss: number;
}

export interface TrainerApi {
  settings: TrainerSettings;
  setSettings: (s: Partial<TrainerSettings>) => void;
  state: OptimizerState;
  running: boolean;
  /** Loss over the whole corpus, recorded every few steps. */
  corpusHistory: LossPoint[];
  /** Loss of each minibatch. */
  batchHistory: LossPoint[];
  snapshots: Snapshot[];
  /** Whether the current run started from random weights or from the shipped trained weights. */
  origin: "init" | "trained";
  start: () => void;
  pause: () => void;
  stepOnce: (n?: number) => void;
  reset: (seed?: number) => void;
  loadTrained: () => void;
  applyGradients: (grads: ModelParams, batchLossValue: number) => void;
}

const FRAME_BUDGET_MS = 40;
const CORPUS_EVAL_EVERY = 10;

function isSnapshotStep(step: number): boolean {
  if (step <= 3) return true;
  const marks = [5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1000];
  return marks.includes(step) || (step > 1000 && step % 250 === 0);
}

interface Run {
  state: OptimizerState;
  rand: () => number;
  corpusHistory: LossPoint[];
  batchHistory: LossPoint[];
  snapshots: Snapshot[];
  origin: "init" | "trained";
}

export function useTrainer(model: TinyGpt): TrainerApi {
  const cfg = model.config;
  const examples: Example[] = useMemo(() => corpusExamples(model), [model]);
  const [settings, setSettingsState] = useState(DEFAULT_TRAINER_SETTINGS);
  const settingsRef = useRef(settings);
  settingsRef.current = settings;

  const corpusLoss = useCallback((params: ModelParams) => batchLoss({ config: cfg, params }, examples), [cfg, examples]);

  const freshRun = useCallback(
    (params: ModelParams, step: number, seed: number, origin: Run["origin"]): Run => {
      const loss = corpusLoss(params);
      return {
        state: { params, m: zerosLike(params), v: zerosLike(params), step },
        rand: mulberry32(seed * 7919 + 17),
        corpusHistory: [{ step, loss }],
        batchHistory: [],
        snapshots: [{ step, params, loss }],
        origin,
      };
    },
    [corpusLoss],
  );

  const runRef = useRef<Run | null>(null);
  if (!runRef.current) runRef.current = freshRun(initParams(cfg, settings.seed), 0, settings.seed, "init");
  const [, setVersion] = useState(0);
  const publish = useCallback(() => setVersion((v) => v + 1), []);
  const [running, setRunning] = useState(false);
  const runningRef = useRef(false);

  const record = useCallback(
    (run: Run, batchLossValue: number) => {
      const { step, params } = run.state;
      run.batchHistory.push({ step, loss: batchLossValue });
      const snap = isSnapshotStep(step);
      if (snap || step % CORPUS_EVAL_EVERY === 0) {
        const loss = corpusLoss(params);
        run.corpusHistory.push({ step, loss });
        if (snap) run.snapshots.push({ step, params, loss });
      }
    },
    [corpusLoss],
  );

  const doStep = useCallback(() => {
    const run = runRef.current!;
    const s = settingsRef.current;
    const batch = sampleBatch(run.rand, examples, s.batchSize);
    const { loss, grads } = batchLossAndGrads({ config: cfg, params: run.state.params }, batch);
    run.state = adamUpdate(run.state, grads, s);
    record(run, loss);
  }, [cfg, examples, record]);

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
      const t0 = performance.now();
      const total = settingsRef.current.totalSteps;
      while (performance.now() - t0 < FRAME_BUDGET_MS && runRef.current!.state.step < total) doStep();
      publish();
      if (runRef.current!.state.step >= total) {
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

  const reset = useCallback(
    (seed?: number) => {
      pause();
      const s = seed ?? settingsRef.current.seed;
      if (seed !== undefined) setSettingsState((prev) => ({ ...prev, seed }));
      runRef.current = freshRun(initParams(cfg, s), 0, s, "init");
      publish();
    },
    [cfg, freshRun, pause, publish],
  );

  const loadTrained = useCallback(() => {
    pause();
    const s = settingsRef.current;
    runRef.current = freshRun(cloneParams(model.params), s.totalSteps, s.seed, "trained");
    publish();
  }, [freshRun, model.params, pause, publish]);

  const applyGradients = useCallback(
    (grads: ModelParams, batchLossValue: number) => {
      const run = runRef.current!;
      run.state = adamUpdate(run.state, grads, settingsRef.current);
      record(run, batchLossValue);
      publish();
    },
    [publish, record],
  );

  const setSettings = useCallback((s: Partial<TrainerSettings>) => setSettingsState((prev) => ({ ...prev, ...s })), []);

  const run = runRef.current;
  return {
    settings,
    setSettings,
    state: run.state,
    running,
    corpusHistory: run.corpusHistory,
    batchHistory: run.batchHistory,
    snapshots: run.snapshots,
    origin: run.origin,
    start,
    pause,
    stepOnce,
    reset,
    loadTrained,
    applyGradients,
  };
}
