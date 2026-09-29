import { useState, type ReactNode } from "react";
import { Tabs } from "../components/layout";
import { HEAD_COLORS } from "../lib/colors";
import type { ColGroup } from "../components/matrix-view";
import { useStep } from "./step-context";

export function tokenRowLabels(tokens: string[]): ReactNode[] {
  return tokens.map((t, i) => (
    <span className="rl" key={i}>
      <span className="rl-pos">{i}</span>
      <span className="rl-tok">{t}</span>
    </span>
  ));
}

export function indexLabels(n: number): ReactNode[] {
  return Array.from({ length: n }, (_, i) => (
    <span className="rl" key={i}>
      <span className="rl-pos">{i}</span>
    </span>
  ));
}

export function tokenColLabels(tokens: string[]): ReactNode[] {
  return tokens.map((t, i) => (
    <span key={i} title={`position ${i}`}>
      {t}
    </span>
  ));
}

export interface Cell {
  i: number;
  j: number;
  /** Optional index of the summation term to emphasise (e.g. the k in Σ_k A_ik B_kj). */
  k?: number;
}

export function useCell(initial: Cell) {
  return useState<Cell>(initial);
}

export function HeadTabs() {
  const { cfg, head, setHead } = useStep();
  return (
    <Tabs
      value={head}
      onChange={setHead}
      options={Array.from({ length: cfg.nHeads }, (_, h) => ({
        value: h,
        label: (
          <span>
            <span className="head-dot" style={{ background: HEAD_COLORS[h] }} />
            Head {h + 1}
          </span>
        ),
      }))}
    />
  );
}

export function headColGroups(nHeads: number, dHead: number): ColGroup[] {
  return Array.from({ length: nHeads }, (_, h) => ({
    start: h * dHead,
    end: (h + 1) * dHead,
    label: `head ${h + 1}`,
    color: HEAD_COLORS[h],
  }));
}

/** Formats a token for prose: a monospace chip. */
export function Tok({ children }: { children: ReactNode }) {
  return <span className="tok-inline">{children}</span>;
}

export function lastIndex<T>(xs: T[]): number {
  return xs.length - 1;
}
