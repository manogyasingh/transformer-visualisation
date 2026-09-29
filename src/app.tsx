import { useCallback, useEffect, useMemo, useState, type ComponentType } from "react";
import { PromptEditor } from "./components/prompt-editor";
import { Sidebar } from "./components/sidebar";
import { forwardBackward } from "./model/backward";
import { runForward } from "./model/forward";
import { decode, DEFAULT_DECODING } from "./model/sampling";
import { encode, tinyGpt } from "./model/tiny-gpt";
import {
  AttnProjStep,
  GeluStep,
  Ln1Step,
  Ln2Step,
  MaskStep,
  MlpDownStep,
  MlpUpStep,
  QkvStep,
  Resid1Step,
  Resid2Step,
  ScoresStep,
  SoftmaxStep,
  SplitHeadsStep,
  WeightedSumStep,
} from "./steps/block-steps";
import { OverviewStep, PositionEmbeddingStep, TokenEmbeddingStep, TokenizeStep } from "./steps/input-steps";
import {
  FinalLnStep,
  LogitsStep,
  ResultStep,
  SampleStep,
  SoftmaxOutStep,
  TemperatureStep,
} from "./steps/output-steps";
import { StepContext, type StepContextValue } from "./steps/step-context";
import { buildSteps, buildTrainingSteps, type StepDef } from "./steps/step-defs";
import { BLOCK_BACKWARD_COMPONENTS } from "./steps/training/block-backward-steps";
import { TDynamicsStep } from "./steps/training/dynamics-step";
import { TDataStep, TForwardStep, TInitStep, TLossStep, TOverviewStep, TProbsStep } from "./steps/training/intro-steps";
import { TDLogitsStep, TEmbedStep, TLnfStep, TUnembedStep } from "./steps/training/output-backward-steps";
import { TAdamStep, TAfterStep, TGradCheckStep } from "./steps/training/update-steps";
import { corpusExamples } from "./training/trainer";
import { TrainingContext, type TrainingContextValue, type TrainingExample } from "./training/training-context";
import { useTrainer } from "./training/use-trainer";

const STEP_COMPONENTS: Record<string, ComponentType> = {
  overview: OverviewStep,
  tokenize: TokenizeStep,
  "token-embedding": TokenEmbeddingStep,
  "position-embedding": PositionEmbeddingStep,
  ln1: Ln1Step,
  qkv: QkvStep,
  "split-heads": SplitHeadsStep,
  scores: ScoresStep,
  mask: MaskStep,
  softmax: SoftmaxStep,
  "weighted-sum": WeightedSumStep,
  "attn-proj": AttnProjStep,
  resid1: Resid1Step,
  ln2: Ln2Step,
  "mlp-up": MlpUpStep,
  gelu: GeluStep,
  "mlp-down": MlpDownStep,
  resid2: Resid2Step,
  "final-ln": FinalLnStep,
  logits: LogitsStep,
  temperature: TemperatureStep,
  "softmax-out": SoftmaxOutStep,
  sample: SampleStep,
  result: ResultStep,
  "t-overview": TOverviewStep,
  "t-data": TDataStep,
  "t-init": TInitStep,
  "t-forward": TForwardStep,
  "t-probs": TProbsStep,
  "t-loss": TLossStep,
  "t-dlogits": TDLogitsStep,
  "t-unembed": TUnembedStep,
  "t-lnf": TLnfStep,
  ...BLOCK_BACKWARD_COMPONENTS,
  "t-embed": TEmbedStep,
  "t-grad-check": TGradCheckStep,
  "t-adam": TAdamStep,
  "t-after": TAfterStep,
  "t-dynamics": TDynamicsStep,
};

type Mode = "inference" | "training";
const TRAIN_PREFIX = "train/";

function isTyping(target: EventTarget | null): boolean {
  return target instanceof HTMLElement && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName);
}

function navName(step: StepDef): string {
  return step.layer !== undefined ? `${step.section}: ${step.navLabel}` : step.navLabel;
}

function parseHash(infSteps: StepDef[], trainSteps: StepDef[]): { mode: Mode; index: number } | null {
  const h = window.location.hash.slice(1);
  if (h.startsWith(TRAIN_PREFIX) || h === "train") {
    const i = trainSteps.findIndex((s) => s.id === h.slice(TRAIN_PREFIX.length));
    return { mode: "training", index: Math.max(0, i) };
  }
  const i = infSteps.findIndex((s) => s.id === h);
  return i >= 0 ? { mode: "inference", index: i } : null;
}

export function App() {
  const model = tinyGpt;
  const cfg = model.config;
  const infSteps = useMemo(() => buildSteps(cfg), [cfg]);
  const trainSteps = useMemo(() => buildTrainingSteps(cfg), [cfg]);
  const initial = useMemo(() => parseHash(infSteps, trainSteps), [infSteps, trainSteps]);

  const [mode, setMode] = useState<Mode>(initial?.mode ?? "inference");
  const [infIndex, setInfIndex] = useState(initial?.mode === "inference" ? initial.index : 0);
  const [trainIndex, setTrainIndex] = useState(initial?.mode === "training" ? initial.index : 0);
  const steps = mode === "training" ? trainSteps : infSteps;
  const index = mode === "training" ? trainIndex : infIndex;
  const setIndex = mode === "training" ? setTrainIndex : setInfIndex;

  const [promptIds, setPromptIds] = useState(() => encode(model.training.presetPrompts[0]));
  const [decoding, setDecoding] = useState(DEFAULT_DECODING);
  const [head, setHead] = useState(0);
  const [calcOpen, setCalcOpen] = useState(true);
  const [editing, setEditing] = useState(false);

  const trace = useMemo(() => runForward(model, promptIds), [model, promptIds]);
  const sampling = useMemo(
    () => decode(trace.logits, decoding, promptIds.length),
    [trace, decoding, promptIds.length],
  );

  const trainer = useTrainer(model);
  const examples: TrainingExample[] = useMemo(
    () =>
      corpusExamples(model).map((e) => ({
        ...e,
        tokens: [...e.ids, e.targets[e.targets.length - 1]].map((id) => cfg.vocab[id]),
      })),
    [model, cfg],
  );
  const [exampleIndex, setExampleIndex] = useState(0);
  const example = examples[exampleIndex];
  const trainModel = useMemo(() => ({ config: cfg, params: trainer.state.params }), [cfg, trainer.state.params]);
  const trainTrace = useMemo(
    () => forwardBackward(trainModel, example.ids, example.targets),
    [trainModel, example],
  );

  const step = steps[index];

  const goTo = useCallback(
    (id: string) => {
      const t = trainSteps.findIndex((s) => s.id === id);
      if (t >= 0) {
        setMode("training");
        setTrainIndex(t);
        return;
      }
      const i = infSteps.findIndex((s) => s.id === id);
      if (i >= 0) {
        setMode("inference");
        setInfIndex(i);
      }
    },
    [infSteps, trainSteps],
  );

  useEffect(() => {
    window.history.replaceState(null, "", `#${mode === "training" ? TRAIN_PREFIX : ""}${step.id}`);
  }, [mode, step.id]);

  useEffect(() => {
    const onHash = () => {
      const parsed = parseHash(infSteps, trainSteps);
      if (!parsed) return;
      setMode(parsed.mode);
      (parsed.mode === "training" ? setTrainIndex : setInfIndex)(parsed.index);
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, [infSteps, trainSteps]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (isTyping(e.target) || e.metaKey || e.ctrlKey || e.altKey) return;
      const n = steps.length;
      if (e.key === "ArrowRight") setIndex((i) => Math.min(i + 1, n - 1));
      else if (e.key === "ArrowLeft") setIndex((i) => Math.max(i - 1, 0));
      else if (e.key === "Home") setIndex(0);
      else if (e.key === "End") setIndex(n - 1);
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [steps.length, setIndex]);

  const ctx: StepContextValue = {
    model,
    cfg,
    trace,
    promptIds,
    setPrompt: setPromptIds,
    decoding,
    setDecoding,
    sampling,
    head,
    setHead,
    calcOpen,
    setCalcOpen,
    steps,
    step,
    index,
    goTo,
  };

  const trainingCtx: TrainingContextValue = {
    trainer,
    examples,
    exampleIndex,
    setExampleIndex,
    example,
    model: trainModel,
    trace: trainTrace,
  };

  const StepComponent = STEP_COMPONENTS[step.kind];
  const revealed = mode === "inference" && index >= infSteps.findIndex((s) => s.kind === "sample");
  const prev = steps[index - 1];
  const next = steps[index + 1];
  const lastLoss = trainer.corpusHistory[trainer.corpusHistory.length - 1]?.loss;

  return (
    <StepContext.Provider value={ctx}>
      <TrainingContext.Provider value={trainingCtx}>
        <div className="app">
          <header className="topbar">
            <div className="brand">
              <div className="brand-title">Transformer dry run</div>
              <div className="brand-sub">a tiny GPT, every number exact</div>
            </div>
            <div className="tabs mode-tabs">
              <button className={mode === "inference" ? "tab active" : "tab"} onClick={() => setMode("inference")}>
                Inference: generate one token
              </button>
              <button className={mode === "training" ? "tab active" : "tab"} onClick={() => setMode("training")}>
                Pretraining: one training step
              </button>
            </div>
            {mode === "inference" ? (
              <div className="prompt-display">
                <span className="prompt-label">prompt</span>
                {trace.tokens.map((t, i) => (
                  <span key={i} className="token-chip">
                    {t}
                  </span>
                ))}
                <span
                  className={revealed ? "token-chip new" : "token-chip ghost"}
                  title={revealed ? "the sampled next token" : "revealed at the sampling step"}
                >
                  {revealed ? cfg.vocab[sampling.chosen] : "?"}
                </span>
                <button className="small-btn" onClick={() => setEditing((e) => !e)}>
                  {editing ? "Close" : "Change prompt"}
                </button>
              </div>
            ) : (
              <div className="prompt-display">
                <span className="prompt-label">training sentence</span>
                <select
                  className="sentence-select"
                  value={exampleIndex}
                  onChange={(e) => setExampleIndex(Number(e.target.value))}
                >
                  {examples.map((e, i) => (
                    <option key={e.text} value={i}>
                      {e.text}
                    </option>
                  ))}
                </select>
                <span className="trainer-status" title="the model's current training step and loss on the whole corpus">
                  step {trainer.state.step} · loss {lastLoss?.toFixed(3)}
                </span>
                <button className="small-btn" onClick={trainer.running ? trainer.pause : trainer.start}>
                  {trainer.running ? "❚❚ Pause" : "▶ Train"}
                </button>
              </div>
            )}
          </header>
          {editing && mode === "inference" && (
            <PromptEditor
              cfg={cfg}
              ids={promptIds}
              presets={model.training.presetPrompts}
              onApply={setPromptIds}
              onClose={() => setEditing(false)}
            />
          )}
          <div className="main">
            <Sidebar
              steps={steps}
              current={index}
              onSelect={setIndex}
              T={mode === "training" ? example.ids.length : promptIds.length}
              cfg={cfg}
            />
            <div className="content">
              <StepComponent key={`${mode}|${step.id}|${mode === "training" ? exampleIndex : promptIds.join(",")}`} />
              <footer className="footer-nav">
                <button className="nav-btn" disabled={!prev} onClick={() => setIndex(index - 1)}>
                  ← {prev ? navName(prev) : ""}
                </button>
                <div className="progress" title={`step ${index} of ${steps.length - 1}`}>
                  <div className="progress-fill" style={{ width: `${(index / (steps.length - 1)) * 100}%` }} />
                </div>
                <button className="nav-btn primary" disabled={!next} onClick={() => setIndex(index + 1)}>
                  {next ? `${navName(next)} →` : "End of the walkthrough"}
                </button>
              </footer>
            </div>
          </div>
        </div>
      </TrainingContext.Provider>
    </StepContext.Provider>
  );
}
