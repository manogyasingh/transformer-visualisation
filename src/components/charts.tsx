import type { ReactNode } from "react";
import { fmt } from "../lib/format";
import { gelu } from "../model/linalg";
import type { Matrix } from "../model/types";

export function AttentionArcs({
  tokens,
  weights,
  row,
  onRow,
  color,
}: {
  tokens: string[];
  weights: Matrix;
  row: number;
  onRow: (i: number) => void;
  color: string;
}) {
  const T = tokens.length;
  const rowH = 30;
  const top = 26;
  const keyX = 128;
  const queryX = 232;
  const yOf = (i: number) => top + i * rowH + rowH / 2;
  return (
    <svg className="arcs" width={330} height={top + T * rowH + 6}>
      <text x={keyX} y={14} textAnchor="end" className="arcs-head">
        keys j
      </text>
      <text x={queryX} y={14} className="arcs-head">
        query i
      </text>
      {tokens.map((t, j) => {
        const w = weights[row][j];
        const masked = j > row;
        return (
          <g key={`k${j}`}>
            {!masked && (
              <line
                x1={keyX + 6}
                y1={yOf(j)}
                x2={queryX - 8}
                y2={yOf(row)}
                stroke={color}
                strokeWidth={1 + 10 * w}
                strokeOpacity={0.15 + 0.85 * w}
                strokeLinecap="round"
              />
            )}
            <text x={8} y={yOf(j) + 4} className="arcs-weight">
              {masked ? "masked" : fmt(w, 3)}
            </text>
            <text x={keyX} y={yOf(j) + 4} textAnchor="end" className={masked ? "arcs-tok masked" : "arcs-tok"}>
              {t}
            </text>
          </g>
        );
      })}
      {tokens.map((t, i) => (
        <text
          key={`q${i}`}
          x={queryX}
          y={yOf(i) + 4}
          className={i === row ? "arcs-tok active" : "arcs-tok clickable"}
          onMouseEnter={() => onRow(i)}
        >
          {i === T - 1 ? `${t} ▸` : t}
        </text>
      ))}
    </svg>
  );
}

export function GeluPlot({ values, focus }: { values: number[]; focus: number }) {
  const W = 340;
  const H = 240;
  const pad = 30;
  const lo = Math.min(-4, Math.floor(Math.min(...values)) - 0.5);
  const hi = Math.max(3, Math.ceil(Math.max(...values)) + 0.5);
  const yLo = -0.6;
  const yHi = Math.max(1, gelu(hi));
  const sx = (x: number) => pad + ((x - lo) / (hi - lo)) * (W - 2 * pad);
  const sy = (y: number) => H - pad - ((y - yLo) / (yHi - yLo)) * (H - 2 * pad);
  const curve: string[] = [];
  const relu: string[] = [];
  for (let k = 0; k <= 200; k++) {
    const x = lo + ((hi - lo) * k) / 200;
    curve.push(`${k ? "L" : "M"}${sx(x).toFixed(1)},${sy(gelu(x)).toFixed(1)}`);
    relu.push(`${k ? "L" : "M"}${sx(x).toFixed(1)},${sy(Math.max(0, x)).toFixed(1)}`);
  }
  const fu = values[focus];
  return (
    <svg className="gelu-plot" width={W} height={H}>
      <line x1={pad} x2={W - pad} y1={sy(0)} y2={sy(0)} className="axis" />
      <line x1={sx(0)} x2={sx(0)} y1={pad / 2} y2={H - pad} className="axis" />
      {[Math.ceil(lo), -2, -1, 1, 2, Math.floor(hi)]
        .filter((v, i, a) => a.indexOf(v) === i && v !== 0 && v >= lo && v <= hi)
        .map((v) => (
          <text key={v} x={sx(v)} y={sy(0) + 14} textAnchor="middle" className="tick">
            {v}
          </text>
        ))}
      <path d={relu.join(" ")} className="relu-line" />
      <path d={curve.join(" ")} className="gelu-line" />
      {values.map((u, j) => (
        <circle key={j} cx={sx(u)} cy={sy(gelu(u))} r={j === focus ? 0 : 3} className="gelu-dot" />
      ))}
      <circle cx={sx(fu)} cy={sy(gelu(fu))} r={6} className="gelu-focus" />
      <text x={W - pad} y={pad} textAnchor="end" className="gelu-label">
        u = {fmt(fu, 3)} → GELU(u) = {fmt(gelu(fu), 3)}
      </text>
      <text x={W - pad} y={H - 6} textAnchor="end" className="legend">
        — GELU &nbsp; - - ReLU
      </text>
    </svg>
  );
}

export interface BarItem {
  key: number;
  label: ReactNode;
  value: number;
  highlight?: boolean;
  muted?: boolean;
  focus?: boolean;
}

export function BarList({
  items,
  min,
  max,
  dp = 2,
  color = "#ea580c",
  negColor = "#2563eb",
  onHover,
  width = 240,
}: {
  items: BarItem[];
  min: number;
  max: number;
  dp?: number;
  color?: string;
  negColor?: string;
  onHover?: (key: number) => void;
  width?: number;
}) {
  const span = max - min || 1;
  const zero = ((0 - min) / span) * width;
  return (
    <div className="bar-list">
      {items.map((it) => {
        const x = ((it.value - min) / span) * width;
        const left = Math.min(zero, x);
        const w = Math.max(1, Math.abs(x - zero));
        return (
          <div
            key={it.key}
            className={`bar-row${it.highlight ? " highlight" : ""}${it.muted ? " muted" : ""}${it.focus ? " focus" : ""}`}
            onMouseEnter={onHover ? () => onHover(it.key) : undefined}
          >
            <div className="bar-label">{it.label}</div>
            <div className="bar-track" style={{ width }}>
              {min < 0 && <div className="bar-zero" style={{ left: zero }} />}
              <div
                className="bar-fill"
                style={{ left, width: w, background: it.value >= 0 ? color : negColor }}
              />
            </div>
            <div className="bar-value">{fmt(it.value, dp)}</div>
          </div>
        );
      })}
    </div>
  );
}

export interface Segment {
  key: number;
  label: string;
  from: number;
  to: number;
  chosen: boolean;
}

export function NumberLine({ segments, u }: { segments: Segment[]; u: number | null }) {
  return (
    <div className="number-line">
      <div className="nl-bar">
        {segments.map((s, i) => (
          <div
            key={s.key}
            className={`nl-seg${s.chosen ? " chosen" : ""}`}
            style={{ left: `${s.from * 100}%`, width: `${(s.to - s.from) * 100}%`, opacity: 1 - (i % 2) * 0.12 }}
            title={`${s.label}: [${s.from.toFixed(4)}, ${s.to.toFixed(4)})`}
          >
            {s.to - s.from > 0.06 && <span>{s.label}</span>}
          </div>
        ))}
        {u !== null && (
          <div className="nl-u" style={{ left: `${u * 100}%` }}>
            <span>u = {u.toFixed(4)}</span>
          </div>
        )}
      </div>
      <div className="nl-ticks">
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <span key={t} style={{ left: `${t * 100}%` }}>
            {t}
          </span>
        ))}
      </div>
    </div>
  );
}

export function TemperatureCurve({
  logits,
  labels,
  tau,
  ids,
  colors,
}: {
  logits: number[];
  labels: string[];
  tau: number;
  ids: number[];
  colors: string[];
}) {
  const W = 380;
  const H = 220;
  const pad = 32;
  const tLo = 0.05;
  const tHi = 2.5;
  const sx = (t: number) => pad + ((t - tLo) / (tHi - tLo)) * (W - 2 * pad);
  const sy = (p: number) => H - pad - p * (H - 2 * pad);
  const probsAt = (t: number) => {
    const z = logits.map((l) => l / t);
    const m = Math.max(...z);
    const e = z.map((v) => Math.exp(v - m));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
  };
  const paths = ids.map(() => [] as string[]);
  for (let k = 0; k <= 160; k++) {
    const t = tLo + ((tHi - tLo) * k) / 160;
    const p = probsAt(t);
    ids.forEach((id, n) => paths[n].push(`${k ? "L" : "M"}${sx(t).toFixed(1)},${sy(p[id]).toFixed(1)}`));
  }
  const now = probsAt(tau);
  return (
    <svg className="temp-curve" width={W} height={H}>
      <line x1={pad} x2={W - pad} y1={sy(0)} y2={sy(0)} className="axis" />
      <line x1={pad} x2={pad} y1={sy(1)} y2={sy(0)} className="axis" />
      {[0, 0.5, 1].map((p) => (
        <text key={p} x={pad - 6} y={sy(p) + 4} textAnchor="end" className="tick">
          {p}
        </text>
      ))}
      {[0.5, 1, 1.5, 2, 2.5].map((t) => (
        <text key={t} x={sx(t)} y={sy(0) + 14} textAnchor="middle" className="tick">
          {t}
        </text>
      ))}
      <text x={W - pad} y={H - 4} textAnchor="end" className="tick">
        temperature τ
      </text>
      <line x1={sx(tau)} x2={sx(tau)} y1={sy(1)} y2={sy(0)} className="tau-marker" />
      {ids.map((id, n) => (
        <g key={id}>
          <path d={paths[n].join(" ")} fill="none" stroke={colors[n]} strokeWidth={2} />
          <circle cx={sx(tau)} cy={sy(now[id])} r={4} fill={colors[n]} />
          <text x={W - pad} y={pad + 4 + n * 15} textAnchor="end" fill={colors[n]} className="legend">
            p({labels[n]}) = {now[id].toFixed(3)}
          </text>
        </g>
      ))}
    </svg>
  );
}
