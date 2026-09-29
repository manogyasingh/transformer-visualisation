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
  kind: StepKind;
  group: StepGroup;
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
  const steps: StepDef[] = [
    { id: "overview", kind: "overview", group: "start", navLabel: "Overview", title: "One token through a tiny GPT", shape: () => "" },
    { id: "tokenize", kind: "tokenize", group: "input", navLabel: "Tokenize", title: "Tokenization: text becomes integers", shape: (T) => `${T} ids` },
    { id: "token-embedding", kind: "token-embedding", group: "input", navLabel: "Token embedding", title: "Token embedding lookup", shape: (T, c) => `${T}×${c.dModel}` },
    { id: "position-embedding", kind: "position-embedding", group: "input", navLabel: "Position embedding", title: "Add position embeddings", shape: (T, c) => `${T}×${c.dModel}` },
  ];
  for (let layer = 0; layer < cfg.nLayers; layer++) {
    for (const s of BLOCK_STEPS) {
      steps.push({ ...s, id: `block-${layer + 1}-${s.kind}`, group: "block", layer });
    }
  }
  steps.push(
    { id: "final-ln", kind: "final-ln", group: "output", navLabel: "Final LayerNorm", title: "Final LayerNorm", shape: (T, c) => `${T}×${c.dModel}` },
    { id: "logits", kind: "logits", group: "output", navLabel: "Logits (unembed)", title: "Unembedding: a score for every token in the vocabulary", shape: (_, c) => `${c.vocab.length}` },
    { id: "temperature", kind: "temperature", group: "output", navLabel: "Temperature", title: "Temperature scaling", shape: (_, c) => `${c.vocab.length}` },
    { id: "softmax-out", kind: "softmax-out", group: "output", navLabel: "Softmax", title: "Softmax: logits become probabilities", shape: (_, c) => `${c.vocab.length}` },
    { id: "sample", kind: "sample", group: "output", navLabel: "Sample", title: "Decoding: pick the next token", shape: () => "1 id" },
    { id: "result", kind: "result", group: "output", navLabel: "Result", title: "The new token, and what happens next", shape: () => "" },
  );
  return steps;
}
