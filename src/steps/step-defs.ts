import type { ModelConfig } from "../model/types";

export type BlockStepKind =
  | "ln1"
  | "qkv"
  | "split-heads"
  | "scores"
  | "mask"
  | "softmax"
  | "weighted-sum"
  | "attn-proj"
  | "resid1"
  | "ln2"
  | "mlp-up"
  | "gelu"
  | "mlp-down"
  | "resid2";

export type StepKind =
  | "overview"
  | "tokenize"
  | "token-embedding"
  | "position-embedding"
  | BlockStepKind
  | "final-ln"
  | "logits"
  | "temperature"
  | "softmax-out"
  | "sample"
  | "result";

export type StepGroup = "start" | "input" | "block" | "output";

export interface StepDef {
  id: string;
  kind: string;
  group?: StepGroup;
  /** Sidebar section and optional subsection. */
  section: string;
  subsection?: string;
  /** 0-based block index for block steps. */
  layer?: number;
  sub?: "attention" | "mlp";
  navLabel: string;
  title: string;
  shape: (T: number, cfg: ModelConfig) => string;
}

const BLOCK_STEPS: {
  kind: BlockStepKind;
  sub: "attention" | "mlp";
  navLabel: string;
  title: string;
  shape: (T: number, cfg: ModelConfig) => string;
}[] = [
  { kind: "ln1", sub: "attention", navLabel: "LayerNorm 1", title: "LayerNorm before attention", shape: (T, c) => `${T}×${c.dModel}` },
  { kind: "qkv", sub: "attention", navLabel: "Queries, keys, values", title: "Project to queries, keys and values", shape: (T, c) => `3 × ${T}×${c.dModel}` },
  { kind: "split-heads", sub: "attention", navLabel: "Split into heads", title: "Split into attention heads", shape: (T, c) => `${c.nHeads} × ${T}×${c.dHead}` },
  { kind: "scores", sub: "attention", navLabel: "Attention scores", title: "Attention scores: scaled dot products", shape: (T, c) => `${c.nHeads} × ${T}×${T}` },
  { kind: "mask", sub: "attention", navLabel: "Causal mask", title: "Causal mask: no peeking at the future", shape: (T, c) => `${c.nHeads} × ${T}×${T}` },
  { kind: "softmax", sub: "attention", navLabel: "Softmax", title: "Softmax: scores become attention weights", shape: (T, c) => `${c.nHeads} × ${T}×${T}` },
  { kind: "weighted-sum", sub: "attention", navLabel: "Weighted sum of values", title: "Mix the value vectors", shape: (T, c) => `${c.nHeads} × ${T}×${c.dHead}` },
  { kind: "attn-proj", sub: "attention", navLabel: "Concat + output proj.", title: "Concatenate heads and project", shape: (T, c) => `${T}×${c.dModel}` },
  { kind: "resid1", sub: "attention", navLabel: "Residual add", title: "Residual connection around attention", shape: (T, c) => `${T}×${c.dModel}` },
  { kind: "ln2", sub: "mlp", navLabel: "LayerNorm 2", title: "LayerNorm before the MLP", shape: (T, c) => `${T}×${c.dModel}` },
  { kind: "mlp-up", sub: "mlp", navLabel: "Expand (W₁)", title: "MLP: expand to the hidden layer", shape: (T, c) => `${T}×${c.dFF}` },
  { kind: "gelu", sub: "mlp", navLabel: "GELU", title: "MLP: GELU non-linearity", shape: (T, c) => `${T}×${c.dFF}` },
  { kind: "mlp-down", sub: "mlp", navLabel: "Project (W₂)", title: "MLP: project back down", shape: (T, c) => `${T}×${c.dModel}` },
  { kind: "resid2", sub: "mlp", navLabel: "Residual add", title: "Residual connection around the MLP", shape: (T, c) => `${T}×${c.dModel}` },
];

export function buildSteps(cfg: ModelConfig): StepDef[] {
  const start = { group: "start" as const, section: "Start" };
  const input = { group: "input" as const, section: "Input" };
  const output = { group: "output" as const, section: "Output" };
  const steps: StepDef[] = [
    { ...start, id: "overview", kind: "overview", navLabel: "Overview", title: "One token through a tiny GPT", shape: () => "" },
    { ...input, id: "tokenize", kind: "tokenize", navLabel: "Tokenize", title: "Tokenization: text becomes integers", shape: (T) => `${T} ids` },
    { ...input, id: "token-embedding", kind: "token-embedding", navLabel: "Token embedding", title: "Token embedding lookup", shape: (T, c) => `${T}×${c.dModel}` },
    { ...input, id: "position-embedding", kind: "position-embedding", navLabel: "Position embedding", title: "Add position embeddings", shape: (T, c) => `${T}×${c.dModel}` },
  ];
  for (let layer = 0; layer < cfg.nLayers; layer++) {
    for (const s of BLOCK_STEPS) {
      steps.push({
        ...s,
        id: `block-${layer + 1}-${s.kind}`,
        group: "block",
        layer,
        section: `Block ${layer + 1}`,
        subsection: s.sub === "mlp" ? "MLP" : "Attention",
      });
    }
  }
  steps.push(
    { ...output, id: "final-ln", kind: "final-ln", navLabel: "Final LayerNorm", title: "Final LayerNorm", shape: (T, c) => `${T}×${c.dModel}` },
    { ...output, id: "logits", kind: "logits", navLabel: "Logits (unembed)", title: "Unembedding: a score for every token in the vocabulary", shape: (_, c) => `${c.vocab.length}` },
    { ...output, id: "temperature", kind: "temperature", navLabel: "Temperature", title: "Temperature scaling", shape: (_, c) => `${c.vocab.length}` },
    { ...output, id: "softmax-out", kind: "softmax-out", navLabel: "Softmax", title: "Softmax: logits become probabilities", shape: (_, c) => `${c.vocab.length}` },
    { ...output, id: "sample", kind: "sample", navLabel: "Sample", title: "Decoding: pick the next token", shape: () => "1 id" },
    { ...output, id: "result", kind: "result", navLabel: "Result", title: "The new token, and what happens next", shape: () => "" },
  );
  return steps;
}

export type BackwardBlockKind =
  | "tb-resid-out"
  | "tb-mlp-down"
  | "tb-gelu"
  | "tb-mlp-up"
  | "tb-ln2"
  | "tb-resid-mid"
  | "tb-attn-proj"
  | "tb-weighted-sum"
  | "tb-softmax"
  | "tb-scores"
  | "tb-qkv"
  | "tb-ln1"
  | "tb-resid-in";

const BACKWARD_BLOCK_STEPS: { kind: BackwardBlockKind; sub: "MLP" | "Attention"; navLabel: string; title: string }[] = [
  { kind: "tb-resid-out", sub: "MLP", navLabel: "Residual: copy gradient", title: "Gradient enters the block: the residual copies it" },
  { kind: "tb-mlp-down", sub: "MLP", navLabel: "Project (W₂)", title: "Backward through the MLP down-projection" },
  { kind: "tb-gelu", sub: "MLP", navLabel: "GELU", title: "Backward through GELU" },
  { kind: "tb-mlp-up", sub: "MLP", navLabel: "Expand (W₁)", title: "Backward through the MLP up-projection" },
  { kind: "tb-ln2", sub: "MLP", navLabel: "LayerNorm 2", title: "Backward through LayerNorm 2" },
  { kind: "tb-resid-mid", sub: "MLP", navLabel: "Residual: add gradients", title: "Residual: the two gradient paths add up" },
  { kind: "tb-attn-proj", sub: "Attention", navLabel: "Output projection", title: "Backward through the attention output projection" },
  { kind: "tb-weighted-sum", sub: "Attention", navLabel: "Weighted sum of values", title: "Backward through the weighted sum of values" },
  { kind: "tb-softmax", sub: "Attention", navLabel: "Softmax", title: "Backward through the attention softmax" },
  { kind: "tb-scores", sub: "Attention", navLabel: "Scores", title: "Backward through the attention scores" },
  { kind: "tb-qkv", sub: "Attention", navLabel: "Q, K, V projections", title: "Backward through the query, key and value projections" },
  { kind: "tb-ln1", sub: "Attention", navLabel: "LayerNorm 1", title: "Backward through LayerNorm 1" },
  { kind: "tb-resid-in", sub: "Attention", navLabel: "Residual: add gradients", title: "Residual: gradient leaves the block" },
];

export function buildTrainingSteps(cfg: ModelConfig): StepDef[] {
  const T = (t: number, c: ModelConfig) => `${t}×${c.dModel}`;
  const step = (section: string, id: string, navLabel: string, title: string, shape: StepDef["shape"] = () => ""): StepDef => ({
    id,
    kind: id,
    section,
    navLabel,
    title,
    shape,
  });
  const steps: StepDef[] = [
    step("Start", "t-overview", "Overview", "Pretraining: learning to predict the next token"),
    step("Data & initialisation", "t-data", "Inputs and targets", "Training data: every position is a prediction task", (t) => `${t} targets`),
    step("Data & initialisation", "t-init", "Random initialisation", "Where training starts: random weights"),
    step("Forward pass & loss", "t-forward", "Forward pass", "Forward pass: logits at every position", (t, c) => `${t}×${c.vocab.length}`),
    step("Forward pass & loss", "t-probs", "Softmax", "Softmax: a distribution at every position", (t, c) => `${t}×${c.vocab.length}`),
    step("Forward pass & loss", "t-loss", "Cross-entropy loss", "Cross-entropy: how surprised was the model?", () => "1"),
    step("Backward: output", "t-dlogits", "Gradient of the logits", "Backpropagation starts: the gradient of the logits", (t, c) => `${t}×${c.vocab.length}`),
    step("Backward: output", "t-unembed", "Unembedding", "Backward through the unembedding", T),
    step("Backward: output", "t-lnf", "Final LayerNorm", "Backward through the final LayerNorm", T),
  ];
  for (let layer = cfg.nLayers - 1; layer >= 0; layer--) {
    for (const s of BACKWARD_BLOCK_STEPS) {
      steps.push({
        id: `t-block-${layer + 1}-${s.kind.slice(3)}`,
        kind: s.kind,
        layer,
        section: `Backward: block ${layer + 1}`,
        subsection: s.sub,
        navLabel: s.navLabel,
        title: s.title,
        shape: T,
      });
    }
  }
  steps.push(
    step("Backward: embeddings", "t-embed", "Embeddings", "Backward into the embedding tables", (_, c) => `${c.vocab.length}×${c.dModel}`),
    step("Update", "t-grad-check", "Gradient check", "Checking a gradient numerically"),
    step("Update", "t-adam", "Adam update", "The Adam update: turning gradients into new weights"),
    step("Update", "t-after", "After the update", "Did it help? The loss before and after one step"),
    step("Training over time", "t-dynamics", "Watch it learn", "Training over time: from random weights to a language model"),
  );
  return steps;
}

/** In the post-training chapter the sidebar's T is the number of response tokens in the batch. */
export function buildPostTrainingSteps(): StepDef[] {
  const step = (section: string, id: string, navLabel: string, title: string, shape: StepDef["shape"] = () => ""): StepDef => ({
    id,
    kind: id,
    section,
    navLabel,
    title,
    shape,
  });
  const perToken: StepDef["shape"] = (n) => `${n} tokens`;
  return [
    step("Start", "p-overview", "Overview", "Post-training: one iteration of RLHF with PPO"),
    step("Reward model", "p-reward-model", "Learning from comparisons", "The reward model: learning what labelers prefer"),
    step("Rollout", "p-rollout", "Sample responses", "Rollouts: the policy answers every prompt", () => "8 responses"),
    step("Rollout", "p-logprobs", "Policy vs reference", "Log-probabilities under the policy and the reference", perToken),
    step("Rewards", "p-score", "Reward-model score", "The reward model scores each response", () => "8 scores"),
    step("Rewards", "p-rewards", "Per-token rewards", "Per-token rewards: a KL penalty on every token, the score at the end", perToken),
    step("Advantages", "p-values", "Value estimates", "The value model predicts the return", perToken),
    step("Advantages", "p-gae", "GAE", "Generalized advantage estimation", perToken),
    step("Advantages", "p-whiten", "Whitening", "Whitening the advantages", perToken),
    step("PPO update", "p-ratio", "Clipped objective", "The probability ratio and the clipped objective", perToken),
    step("PPO update", "p-policy-grad", "Gradient at the logits", "From the objective to the logits", (_, c) => `T×${c.vocab.length}`),
    step("PPO update", "p-ptx", "Pretraining mix", "The pretraining mix: InstructGPT's PPO-ptx"),
    step("PPO update", "p-value-loss", "Value loss", "Training the value model", perToken),
    step("PPO update", "p-epochs", "Epochs", "Several epochs on the same batch"),
    step("PPO update", "p-after", "After the iteration", "Did it help? The policy before and after one iteration"),
    step("Training over time", "p-dynamics", "Watch it learn", "Many iterations: alignment, the KL leash, and reward hacking"),
  ];
}

export interface ChapterDef {
  /** First part of the URL hash, `#<chapter id>/<step id>`. */
  id: string;
  title: string;
  subtitle: string;
  /** Step ids must be unique across all chapters: `goTo` and old-style links look a step up by id alone. */
  steps: StepDef[];
}

export function buildChapters(cfg: ModelConfig): ChapterDef[] {
  return [
    { id: "inference", title: "Inference", subtitle: "Generate one token", steps: buildSteps(cfg) },
    { id: "pretraining", title: "Pretraining", subtitle: "One training step", steps: buildTrainingSteps(cfg) },
    { id: "post-training", title: "Post-training", subtitle: "One PPO iteration (RLHF)", steps: buildPostTrainingSteps() },
  ];
}
