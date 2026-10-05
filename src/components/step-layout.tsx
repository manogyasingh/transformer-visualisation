import type { ReactNode } from "react";
import { useStep } from "../steps/step-context";

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
  const { chapter, step, index, steps } = useStep();
  return (
    <div className="step">
      <div className="step-scroll">
        <header className="step-header">
          <div className="step-crumb">
            {chapter.title} › {step.section}
            {step.subsection ? ` › ${step.subsection}` : ""} · step {index} of {steps.length - 1}
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
