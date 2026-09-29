import type { ReactNode } from "react";
import { useStep } from "../steps/step-context";
import { Tex } from "./tex";

const DENSE_TEX_LENGTH = 1400;

export function CalcPanel({ target, children }: { target: ReactNode; children: ReactNode }) {
  const { calcOpen, setCalcOpen } = useStep();
  return (
    <div className="calc">
      <div className="calc-head">
        <button className="calc-toggle" onClick={() => setCalcOpen(!calcOpen)} title="Show or hide the calculation">
          {calcOpen ? "▾" : "▸"}
        </button>
        <span className="calc-title">Exact calculation</span>
        <span className="calc-target">{target}</span>
        <span className="calc-hint">Hover any cell above to see how it was computed</span>
      </div>
      {calcOpen && <div className="calc-body">{children}</div>}
    </div>
  );
}

export function CalcLine({ label, tex, children }: { label?: ReactNode; tex?: string; children?: ReactNode }) {
  const dense = (tex?.length ?? 0) > DENSE_TEX_LENGTH;
  return (
    <div className="calc-line">
      <div className="calc-label">{label}</div>
      <div className={dense ? "calc-expr dense" : "calc-expr"}>
        {tex && <Tex>{tex}</Tex>}
        {children}
      </div>
    </div>
  );
}

export function CalcNote({ children }: { children: ReactNode }) {
  return <div className="calc-note">{children}</div>;
}
