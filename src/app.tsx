import { useCallback, useEffect, useMemo, useState, type ComponentType } from "react";
import { PromptEditor } from "./components/prompt-editor";
import { Sidebar } from "./components/sidebar";
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
import { buildSteps, type StepKind } from "./steps/step-defs";

const STEP_COMPONENTS: Record<StepKind, ComponentType> = {
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
};

function isTyping(target: EventTarget | null): boolean {
  return target instanceof HTMLElement && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName);
}

function navName(step: { navLabel: string; layer?: number }): string {
  return step.layer !== undefined ? `Block ${step.layer + 1}: ${step.navLabel}` : step.navLabel;
}

export function App() {
  const model = tinyGpt;
  const cfg = model.config;
  const steps = useMemo(() => buildSteps(cfg), [cfg]);
  const [promptIds, setPromptIds] = useState(() => encode(model.training.presetPrompts[0]));
  const [decoding, setDecoding] = useState(DEFAULT_DECODING);
  const [head, setHead] = useState(0);
  const [calcOpen, setCalcOpen] = useState(true);
  const [editing, setEditing] = useState(false);
  const [index, setIndex] = useState(() => {
    const i = steps.findIndex((s) => s.id === window.location.hash.slice(1));
    return i >= 0 ? i : 0;
  });

  const trace = useMemo(() => runForward(model, promptIds), [model, promptIds]);
  const sampling = useMemo(
    () => decode(trace.logits, decoding, promptIds.length),
    [trace, decoding, promptIds.length],
  );
  const step = steps[index];

  const goTo = useCallback(
    (id: string) => {
      const i = steps.findIndex((s) => s.id === id);
      if (i >= 0) setIndex(i);
    },
    [steps],
  );

  useEffect(() => {
    window.history.replaceState(null, "", `#${steps[index].id}`);
  }, [steps, index]);

  useEffect(() => {
    const onHash = () => goTo(window.location.hash.slice(1));
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, [goTo]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (isTyping(e.target) || e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === "ArrowRight") setIndex((i) => Math.min(i + 1, steps.length - 1));
      else if (e.key === "ArrowLeft") setIndex((i) => Math.max(i - 1, 0));
      else if (e.key === "Home") setIndex(0);
      else if (e.key === "End") setIndex(steps.length - 1);
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [steps.length]);

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

  const StepComponent = STEP_COMPONENTS[step.kind];
  const revealed = index >= steps.findIndex((s) => s.kind === "sample");
  const prev = steps[index - 1];
  const next = steps[index + 1];

  return (
    <StepContext.Provider value={ctx}>
      <div className="app">
        <header className="topbar">
          <div className="brand">
            <div className="brand-title">Transformer dry run</div>
            <div className="brand-sub">one token through a tiny GPT, every number exact</div>
          </div>
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
        </header>
        {editing && (
          <PromptEditor
            cfg={cfg}
            ids={promptIds}
            presets={model.training.presetPrompts}
            onApply={setPromptIds}
            onClose={() => setEditing(false)}
          />
        )}
        <div className="main">
          <Sidebar steps={steps} current={index} onSelect={setIndex} T={promptIds.length} cfg={cfg} />
          <div className="content">
            <StepComponent key={`${step.id}|${promptIds.join(",")}`} />
            <footer className="footer-nav">
              <button className="nav-btn" disabled={!prev} onClick={() => setIndex(index - 1)}>
                ← {prev ? navName(prev) : ""}
              </button>
              <div className="progress" title={`step ${index} of ${steps.length - 1}`}>
                <div className="progress-fill" style={{ width: `${(index / (steps.length - 1)) * 100}%` }} />
              </div>
              <button className="nav-btn primary" disabled={!next} onClick={() => setIndex(index + 1)}>
                {next ? `${navName(next)} →` : "End of the dry run"}
              </button>
            </footer>
          </div>
        </div>
      </div>
    </StepContext.Provider>
  );
}
