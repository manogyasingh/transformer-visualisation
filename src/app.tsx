import { useCallback, useEffect, useMemo, useState, type ComponentType, type ReactNode } from "react";
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
import { ppoModels } from "./post-training/models";
import { PpoContext, type PpoContextValue } from "./post-training/ppo-context";
import { usePpoTrainer } from "./post-training/use-ppo-trainer";
import { PGaeStep, PValuesStep, PWhitenStep } from "./steps/post-training/advantage-steps";
import { PDynamicsStep } from "./steps/post-training/dynamics-step";
import { POverviewStep, PRewardModelStep } from "./steps/post-training/intro-steps";
import { PLogprobsStep, PRewardsStep, PRolloutStep, PScoreStep } from "./steps/post-training/rollout-steps";
import { PAfterStep, PEpochsStep, PPolicyGradStep, PPtxStep, PRatioStep, PValueLossStep } from "./steps/post-training/update-steps";
import { StepContext, type StepContextValue } from "./steps/step-context";
import { buildChapters, type ChapterDef, type StepDef } from "./steps/step-defs";
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
  "p-overview": POverviewStep,
  "p-reward-model": PRewardModelStep,
  "p-rollout": PRolloutStep,
  "p-logprobs": PLogprobsStep,
  "p-score": PScoreStep,
  "p-rewards": PRewardsStep,
  "p-values": PValuesStep,
  "p-gae": PGaeStep,
  "p-whiten": PWhitenStep,
  "p-ratio": PRatioStep,
  "p-policy-grad": PPolicyGradStep,
  "p-ptx": PPtxStep,
  "p-value-loss": PValueLossStep,
  "p-epochs": PEpochsStep,
  "p-after": PAfterStep,
  "p-dynamics": PDynamicsStep,
};

interface Loc {
  chapter: number;
  index: number;
}

function isTyping(target: EventTarget | null): boolean {
  return target instanceof HTMLElement && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName);
}

function navName(step: StepDef): string {
  return step.layer !== undefined ? `${step.section}: ${step.navLabel}` : step.navLabel;
}

function findStep(chapters: ChapterDef[], id: string): Loc | null {
  for (let c = 0; c < chapters.length; c++) {
    const i = chapters[c].steps.findIndex((s) => s.id === id);
    if (i >= 0) return { chapter: c, index: i };
  }
  return null;
}

/** Reads `#<chapter id>/<step id>`; a bare step id (`#tokenize`, `#train/t-loss`) is looked up in every chapter. */
function parseHash(chapters: ChapterDef[]): Loc | null {
  const [chapterId, stepId = ""] = window.location.hash.slice(1).split("/");
  const c = chapters.findIndex((ch) => ch.id === chapterId);
  if (c >= 0) return { chapter: c, index: Math.max(0, chapters[c].steps.findIndex((s) => s.id === stepId)) };
  return findStep(chapters, stepId || chapterId);
}

/** The step before or after `loc`, running on into the neighbouring chapter at either end. */
function neighbour(chapters: ChapterDef[], { chapter, index }: Loc, dir: 1 | -1): Loc | null {
  const i = index + dir;
  if (i >= 0 && i < chapters[chapter].steps.length) return { chapter, index: i };
  const c = chapter + dir;
  if (c < 0 || c >= chapters.length) return null;
  return { chapter: c, index: dir > 0 ? 0 : chapters[c].steps.length - 1 };
}

export function App() {
  const model = tinyGpt;
  const cfg = model.config;
  const chapters = useMemo(() => buildChapters(cfg), [cfg]);
  const initial = useMemo(() => parseHash(chapters), [chapters]);

  const [chapterIndex, setChapterIndex] = useState(initial?.chapter ?? 0);
  const [positions, setPositions] = useState(() => chapters.map((_, c) => (c === initial?.chapter ? initial.index : 0)));
  const chapter = chapters[chapterIndex];
  const steps = chapter.steps;
  const index = positions[chapterIndex];
  const show = useCallback(({ chapter: c, index: i }: Loc) => {
    setChapterIndex(c);
    setPositions((p) => p.map((x, k) => (k === c ? i : x)));
  }, []);

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

  const ppoTrainer = usePpoTrainer(ppoModels);
  const ppoTrace = ppoTrainer.pending;
  const [rolloutChoice, setRolloutIndex] = useState(0);
  const rolloutIndex = Math.min(rolloutChoice, ppoTrace.rollouts.length - 1);

  const step = steps[index];

  const goTo = useCallback(
    (id: string) => {
      const loc = findStep(chapters, id);
      if (loc) show(loc);
    },
    [chapters, show],
  );

  useEffect(() => {
    window.history.replaceState(null, "", `#${chapter.id}/${step.id}`);
  }, [chapter.id, step.id]);

  useEffect(() => {
    const onHash = () => {
      const parsed = parseHash(chapters);
      if (parsed) show(parsed);
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, [chapters, show]);

  const prevLoc = neighbour(chapters, { chapter: chapterIndex, index }, -1);
  const nextLoc = neighbour(chapters, { chapter: chapterIndex, index }, 1);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (isTyping(e.target) || e.metaKey || e.ctrlKey || e.altKey) return;
      const here = { chapter: chapterIndex, index };
      let to: Loc | null;
      if (e.key === "ArrowRight") to = neighbour(chapters, here, 1);
      else if (e.key === "ArrowLeft") to = neighbour(chapters, here, -1);
      else if (e.key === "Home") to = { chapter: chapterIndex, index: 0 };
      else if (e.key === "End") to = { chapter: chapterIndex, index: steps.length - 1 };
      else return;
      e.preventDefault();
      if (to) show(to);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [chapters, chapterIndex, index, steps.length, show]);

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
    chapter,
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

  const ppoCtx: PpoContextValue = {
    trainer: ppoTrainer,
    models: ppoModels,
    trace: ppoTrace,
    rolloutIndex,
    setRolloutIndex,
    rollout: ppoTrace.rollouts[rolloutIndex],
  };

  const StepComponent = STEP_COMPONENTS[step.kind];
  const words = (ids: number[]) => ids.map((id) => cfg.vocab[id]).join(" ");
  const lastEval = ppoTrainer.evals[ppoTrainer.evals.length - 1];
  const sidebarT: Record<string, number> = {
    inference: promptIds.length,
    pretraining: example.ids.length,
    "post-training": ppoTrace.rollouts.reduce((s, r) => s + r.response.length, 0),
  };
  const stepKeys: Record<string, string> = { inference: promptIds.join(","), pretraining: String(exampleIndex) };
  const revealed = chapter.id === "inference" && index >= steps.findIndex((s) => s.kind === "sample");
  const lastLoss = trainer.corpusHistory[trainer.corpusHistory.length - 1]?.loss;
  const locName = (loc: Loc) => {
    const target = chapters[loc.chapter];
    const name = navName(target.steps[loc.index]);
    return loc.chapter === chapterIndex ? name : `${target.title}: ${name}`;
  };

  const chapterBar: Record<string, ReactNode> = {
    inference: (
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
    ),
    pretraining: (
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
    ),
    "post-training": (
      <div className="prompt-display">
        <span className="prompt-label">response</span>
        <select className="sentence-select" value={rolloutIndex} onChange={(e) => setRolloutIndex(Number(e.target.value))}>
          {ppoTrace.rollouts.map((r, i) => (
            <option key={i} value={i}>
              {i + 1}. {words(r.prompt)} → {words(r.response)}
            </option>
          ))}
        </select>
        <span className="trainer-status" title="the policy's PPO iteration, and its exact expected score and KL to the reference at the last evaluation">
          iteration {ppoTrainer.state.iteration} · score {lastEval?.score.toFixed(3)} · KL {lastEval?.kl.toFixed(3)}
        </span>
        <button className="small-btn" onClick={ppoTrainer.running ? ppoTrainer.pause : ppoTrainer.start}>
          {ppoTrainer.running ? "❚❚ Pause" : "▶ Run PPO"}
        </button>
      </div>
    ),
  };

  return (
    <StepContext.Provider value={ctx}>
      <TrainingContext.Provider value={trainingCtx}>
        <PpoContext.Provider value={ppoCtx}>
          <div className="app">
            <header className="topbar">
              <div className="brand">
                <div className="brand-title">Transformer dry run</div>
                <div className="brand-sub">a tiny GPT, every number exact</div>
              </div>
              {chapterBar[chapter.id]}
            </header>
            {editing && chapter.id === "inference" && (
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
                chapters={chapters}
                chapterIndex={chapterIndex}
                positions={positions}
                onSelectChapter={setChapterIndex}
                onSelectStep={(i) => show({ chapter: chapterIndex, index: i })}
                T={sidebarT[chapter.id] ?? 0}
                cfg={cfg}
              />
              <div className="content">
                <StepComponent key={`${step.id}|${stepKeys[chapter.id] ?? ""}`} />
                <footer className="footer-nav">
                  <button className="nav-btn" disabled={!prevLoc} onClick={() => prevLoc && show(prevLoc)}>
                    ← {prevLoc ? locName(prevLoc) : ""}
                  </button>
                  <div className="progress" title={`step ${index} of ${steps.length - 1}`}>
                    <div className="progress-fill" style={{ width: `${(index / (steps.length - 1)) * 100}%` }} />
                  </div>
                  <button className="nav-btn primary" disabled={!nextLoc} onClick={() => nextLoc && show(nextLoc)}>
                    {nextLoc ? `${locName(nextLoc)} →` : "End of the walkthrough"}
                  </button>
                </footer>
              </div>
            </div>
          </div>
        </PpoContext.Provider>
      </TrainingContext.Provider>
    </StepContext.Provider>
  );
}
