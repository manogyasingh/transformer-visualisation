import type { ReactNode } from "react";
import { useStep } from "../steps/step-context";

function crumb(group: string, layer: number | undefined, sub: string | undefined): string {
  if (group === "start") return "Start";
  if (group === "input") return "Input";
  if (group === "output") return "Output";
  return `Block ${(layer ?? 0) + 1} › ${sub === "mlp" ? "MLP" : "Attention"}`;
}

export function StepLayout({
  explain,
  formula,
  deeper,
  calc,
  children,
}: {
  explain: ReactNode;
  formula?: ReactNode;
  deeper?: ReactNode;
  calc?: ReactNode;
  children?: ReactNode;
}) {
  const { step, index, steps } = useStep();
  return (
    <div className="step">
      <div className="step-scroll">
        <header className="step-header">
          <div className="step-crumb">
            {crumb(step.group, step.layer, step.sub)} · step {index} of {steps.length - 1}
          </div>
          <h1>{step.title}</h1>
        </header>
        <section className={formula ? "step-explain with-formula" : "step-explain"}>
          <div className="prose">{explain}</div>
          {formula && <div className="formula-box">{formula}</div>}
        </section>
        {deeper && (
          <details className="deeper">
            <summary>Go deeper</summary>
            <div className="prose">{deeper}</div>
          </details>
        )}
        {children && <section className="step-viz">{children}</section>}
      </div>
      {calc && <section className="calc-panel">{calc}</section>}
    </div>
  );
}
