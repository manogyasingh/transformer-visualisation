import type { ReactNode } from "react";
import { Tex } from "./tex";

/**
 * Classic matrix-multiplication layout: B sits above the result C and A sits to
 * its left, so row i of A and column j of B line up with cell (i, j) of C.
 */
export function MatMulView({
  a,
  b,
  bias,
  c,
  after,
  corner,
}: {
  a: ReactNode;
  b: ReactNode;
  bias?: ReactNode;
  c: ReactNode;
  after?: ReactNode;
  corner?: ReactNode;
}) {
  return (
    <div className="matmul">
      <div className="mm-corner" style={{ gridRow: bias ? "1 / 3" : "1" }}>
        {corner}
      </div>
      <div className="mm-b">{b}</div>
      {bias && <div className="mm-bias">{bias}</div>}
      <div className="mm-a" style={{ gridRow: bias ? 3 : 2 }}>
        {a}
      </div>
      <div className="mm-c" style={{ gridRow: bias ? 3 : 2 }}>
        {c}
      </div>
      {after && (
        <div className="mm-after" style={{ gridRow: bias ? 3 : 2 }}>
          {after}
        </div>
      )}
    </div>
  );
}

/** Matrices laid out left to right, bottom-aligned so equal-height rows line up. */
export function EqRow({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={`eq-row ${className ?? ""}`}>{children}</div>;
}

export function Op({ children, tex }: { children?: ReactNode; tex?: string }) {
  return <div className="op">{tex ? <Tex>{tex}</Tex> : children}</div>;
}

export function Tabs<T extends string | number>({
  value,
  options,
  onChange,
}: {
  value: T;
  options: { value: T; label: ReactNode }[];
  onChange: (v: T) => void;
}) {
  return (
    <div className="tabs" role="tablist">
      {options.map((o) => (
        <button
          key={String(o.value)}
          role="tab"
          aria-selected={o.value === value}
          className={o.value === value ? "tab active" : "tab"}
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function VizCaption({ children }: { children: ReactNode }) {
  return <p className="viz-caption">{children}</p>;
}
