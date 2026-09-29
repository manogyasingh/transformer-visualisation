import { useState } from "react";
import type { ModelConfig } from "../model/types";

export function PromptEditor({
  cfg,
  ids,
  presets,
  onApply,
  onClose,
}: {
  cfg: ModelConfig;
  ids: number[];
  presets: string[];
  onApply: (ids: number[]) => void;
  onClose: () => void;
}) {
  const [draft, setDraft] = useState(ids);
  const full = draft.length >= cfg.nCtx;
  const apply = (next: number[]) => {
    onApply(next);
    onClose();
  };

  return (
    <div className="prompt-editor">
      <div className="pe-section">
        <div className="pe-title">Presets</div>
        <div className="pe-presets">
          {presets.map((p) => (
            <button key={p} className="small-btn" onClick={() => apply(p.split(" ").map((w) => cfg.vocab.indexOf(w)))}>
              {p}
            </button>
          ))}
        </div>
      </div>
      <div className="pe-section">
        <div className="pe-title">
          Or build your own ({draft.length}/{cfg.nCtx} tokens, from the {cfg.vocab.length}-word vocabulary)
        </div>
        <div className="pe-draft">
          {draft.map((id, i) => (
            <span key={i} className="token-chip">
              {cfg.vocab[id]}
            </span>
          ))}
          {draft.length === 0 && <span className="muted">click words below…</span>}
          <button className="small-btn" disabled={draft.length === 0} onClick={() => setDraft(draft.slice(0, -1))}>
            ⌫ remove last
          </button>
          <button className="small-btn" disabled={draft.length === 0} onClick={() => setDraft([])}>
            clear
          </button>
        </div>
        <div className="pe-vocab">
          {cfg.vocab.map((w, id) => (
            <button key={w} className="vocab-btn" disabled={full} onClick={() => setDraft([...draft, id])}>
              {w}
            </button>
          ))}
        </div>
      </div>
      <div className="pe-actions">
        <button className="primary-btn" disabled={draft.length === 0} onClick={() => apply(draft)}>
          Use this prompt
        </button>
        <button className="small-btn" onClick={onClose}>
          Cancel
        </button>
      </div>
    </div>
  );
}
