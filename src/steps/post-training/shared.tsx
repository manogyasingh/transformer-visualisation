import { Fragment, useState, type ReactNode } from "react";
import type { Segment } from "../../components/charts";
import { Tabs } from "../../components/layout";
import { cellColors, type ColorScale } from "../../lib/colors";
import { fmt } from "../../lib/format";
import type { SamplingResult } from "../../model/sampling";
import { usePpo } from "../../post-training/ppo-context";
import type { Rollout } from "../../post-training/ppo";
import { useStep } from "../step-context";

export function useWords() {
  const { models } = usePpo();
  return (ids: number[]) => ids.map((id) => models.cfg.vocab[id]).join(" ");
}

export interface TokenFocus {
  /** Rollout index. */
  i: number;
  /** Response token index. */
  t: number;
}

/** Hovered token, defaulting to the last token of the selected rollout; stays valid when the batch changes. */
export function useTokenFocus(initial?: (r: Rollout) => number): [TokenFocus, (f: TokenFocus) => void] {
  const { trace, rolloutIndex } = usePpo();
  const [focus, setFocus] = useState<TokenFocus | null>(null);
  const i = Math.min(focus?.i ?? rolloutIndex, trace.rollouts.length - 1);
  const r = trace.rollouts[i];
  const t = Math.min(focus?.t ?? (initial ? initial(r) : r.response.length - 1), r.response.length - 1);
  return [{ i, t }, setFocus];
}

export function TokenGrid({
  value,
  dp = 3,
  scale = "diverging",
  center = 0,
  maxAbs,
  focus,
  onFocus,
  dim,
  trailing,
}: {
  value: (r: Rollout, i: number, t: number) => number;
  dp?: number;
  scale?: ColorScale;
  /** Colours show the distance from this value (e.g. 1 for probability ratios). */
  center?: number;
  maxAbs?: number;
  focus?: TokenFocus;
  onFocus?: (f: TokenFocus) => void;
  /** Tokens to grey out, e.g. those whose gradient is clipped to zero. */
  dim?: (r: Rollout, i: number, t: number) => boolean;
  trailing?: { header: ReactNode; cell: (r: Rollout, i: number) => ReactNode };
}) {
  const { trace, models, rolloutIndex, setRolloutIndex } = usePpo();
  const vocab = models.cfg.vocab;
  const all = trace.rollouts.flatMap((r, i) => r.response.map((_, t) => Math.abs(value(r, i, t) - center)));
  const m = maxAbs ?? Math.max(1e-12, ...all.filter(Number.isFinite));
  return (
    <table className="token-grid">
      <thead>
        <tr>
          <th>#</th>
          <th>prompt</th>
          <th className="tg-left">response</th>
          {trailing && <th>{trailing.header}</th>}
        </tr>
      </thead>
      <tbody>
        {trace.rollouts.map((r, i) => (
          <tr key={i} className={i === rolloutIndex ? "current" : ""} onClick={() => setRolloutIndex(i)}>
            <td className="tg-index">{i + 1}</td>
            <td className="tg-prompt">{r.prompt.map((id) => vocab[id]).join(" ")}</td>
            <td className="tg-response">
              {r.response.map((a, t) => {
                const v = value(r, i, t);
                const { background, color } = cellColors(v - center, scale, m);
                const cls = ["tg-token", focus?.i === i && focus?.t === t ? "focus" : "", dim?.(r, i, t) ? "dim" : ""].join(" ");
                return (
                  <span key={t} className={cls} onMouseEnter={onFocus ? () => onFocus({ i, t }) : undefined}>
                    <span className="tg-word">{vocab[a]}</span>
                    <span className="tg-value" style={{ background, color }}>
                      {fmt(v, dp)}
                    </span>
                  </span>
                );
              })}
              {!r.ended && <span className="tg-cut" title="hit the length limit without a full stop">✂</span>}
            </td>
            {trailing && <td className="tg-trailing">{trailing.cell(r, i)}</td>}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function EpochTabs({ epoch, onChange }: { epoch: number; onChange: (e: number) => void }) {
  const { trainer } = usePpo();
  return (
    <Tabs
      value={epoch}
      onChange={onChange}
      options={Array.from({ length: trainer.settings.epochs }, (_, e) => ({ value: e, label: `Epoch ${e + 1}` }))}
    />
  );
}

export function PpoControls({ compact = false }: { compact?: boolean }) {
  const { trainer } = usePpo();
  const { state, settings, running } = trainer;
  const last = trainer.evals[trainer.evals.length - 1];
  return (
    <div className="trainer-controls">
      <button
        className="primary-btn"
        onClick={running ? trainer.pause : trainer.start}
        disabled={!running && state.iteration >= settings.totalIterations}
      >
        {running ? "❚❚ Pause" : state.iteration === 0 ? "▶ Run PPO" : "▶ Continue"}
      </button>
      <button className="small-btn" disabled={running} onClick={() => trainer.stepOnce(1)}>
        +1 iteration
      </button>
      <button className="small-btn" disabled={running} onClick={() => trainer.stepOnce(10)}>
        +10
      </button>
      <button className="small-btn" onClick={trainer.reset}>
        Reset to the pretrained policy
      </button>
      {!compact && last && (
        <span className="trainer-readout">
          iteration <strong>{state.iteration}</strong> / {settings.totalIterations} · at iteration {last.iteration}: expected score{" "}
          <strong>{last.score.toFixed(3)}</strong>, KL to reference <strong>{last.kl.toFixed(3)}</strong>, P(chasing){" "}
          <strong>{last.pChasing.toFixed(3)}</strong>
        </span>
      )}
    </div>
  );
}

/** One segment of [0, 1) per token, as in the inference chapter's sampling step. */
export function samplingSegments(s: SamplingResult, vocab: string[]): Segment[] {
  const segments: Segment[] = [];
  s.order.forEach((id, r) => {
    if (s.filtered[id] === 0) return;
    segments.push({ key: id, label: vocab[id], from: s.cumulative[r] - s.filtered[id], to: s.cumulative[r], chosen: id === s.chosen });
  });
  return segments;
}

/** The iteration as a flow of clickable stages, grouped by sidebar section. */
export function PpoDiagram() {
  const { steps, goTo, step: current } = useStep();
  const sections: { title: string; ids: string[] }[] = [];
  for (const s of steps) {
    const last = sections[sections.length - 1];
    if (last && last.title === s.section) last.ids.push(s.id);
    else sections.push({ title: s.section, ids: [s.id] });
  }
  const notes: Record<string, string> = {
    "Reward model": "trained once, before RL; frozen during it",
    "Training over time": "",
  };
  return (
    <div className="arch">
      {sections.slice(1).map((sec, n) => (
        <Fragment key={sec.title}>
          {n > 0 && <div className="arch-down">{sec.title === "Training over time" ? "↺ θ_old ← θ, sample a new batch, repeat" : "↓"}</div>}
          <div className={sec.title === "PPO update" ? "arch-block backward" : "arch-block"}>
            <div className="arch-block-title">
              {sec.title}
              {notes[sec.title] && <span className="arch-note"> · {notes[sec.title]}</span>}
            </div>
            <div className="arch-flow">
              {sec.ids.map((id, k) => {
                const s = steps.find((x) => x.id === id)!;
                return (
                  <Fragment key={id}>
                    {k > 0 && <span className="arch-arrow">→</span>}
                    <button className={`arch-chip${current.id === id ? " current" : ""}`} onClick={() => goTo(id)}>
                      {s.navLabel}
                    </button>
                  </Fragment>
                );
              })}
            </div>
          </div>
        </Fragment>
      ))}
    </div>
  );
}
