import { Fragment } from "react";
import { useStep } from "../steps/step-context";
import type { StepDef } from "../steps/step-defs";

function Chip({ step, label }: { step: StepDef; label?: string }) {
  const { goTo, step: current } = useStep();
  return (
    <button className={`arch-chip${current.id === step.id ? " current" : ""}`} onClick={() => goTo(step.id)}>
      {label ?? step.navLabel}
    </button>
  );
}

function Flow({ steps }: { steps: StepDef[] }) {
  return (
    <div className="arch-flow">
      {steps.map((s, i) => (
        <Fragment key={s.id}>
          {i > 0 && <span className="arch-arrow">→</span>}
          <Chip step={s} label={s.kind.startsWith("resid") ? "⊕ residual" : undefined} />
        </Fragment>
      ))}
    </div>
  );
}

export function ArchitectureDiagram() {
  const { steps, cfg, trace } = useStep();
  const byId = (id: string) => steps.find((s) => s.id === id)!;
  const blocks = Array.from({ length: cfg.nLayers }, (_, l) => steps.filter((s) => s.layer === l));

  return (
    <div className="arch">
      <div className="arch-io">“{trace.tokens.join(" ")}”</div>
      <div className="arch-down">↓</div>
      <Flow steps={[byId("tokenize"), byId("token-embedding"), byId("position-embedding")]} />
      <div className="arch-down">↓ residual stream X⁽⁰⁾</div>
      {blocks.map((b, l) => (
        <Fragment key={l}>
          <div className="arch-block">
            <div className="arch-block-title">Block {l + 1}</div>
            <div className="arch-sub">
              <span className="arch-sub-title">Attention</span>
              <Flow steps={b.filter((s) => s.sub === "attention")} />
            </div>
            <div className="arch-sub">
              <span className="arch-sub-title">MLP</span>
              <Flow steps={b.filter((s) => s.sub === "mlp")} />
            </div>
          </div>
          <div className="arch-down">↓ X⁽{l + 1}⁾</div>
        </Fragment>
      ))}
      <Flow steps={["final-ln", "logits", "temperature", "softmax-out", "sample"].map(byId)} />
      <div className="arch-down">↓</div>
      <Chip step={byId("result")} label="next token" />
    </div>
  );
}
