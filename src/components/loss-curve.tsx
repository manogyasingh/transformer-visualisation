import type { LossPoint } from "../training/use-trainer";

export function LossCurve({
  corpus,
  batch,
  optimal,
  maxStep,
  marker,
}: {
  corpus: LossPoint[];
  batch: LossPoint[];
  optimal: number;
  maxStep: number;
  marker?: number;
}) {
  const W = 600;
  const H = 240;
  const pad = { l: 40, r: 12, t: 12, b: 28 };
  const xMax = Math.max(maxStep, corpus[corpus.length - 1]?.step ?? 1, 1);
  const xMin = Math.min(corpus[0]?.step ?? 0, xMax - 1);
  const yMax = Math.max(3, ...corpus.map((p) => p.loss));
  const sx = (s: number) => pad.l + ((s - xMin) / (xMax - xMin)) * (W - pad.l - pad.r);
  const sy = (l: number) => pad.t + (1 - Math.min(l, yMax) / yMax) * (H - pad.t - pad.b);
  const path = (pts: LossPoint[]) => pts.map((p, i) => `${i ? "L" : "M"}${sx(p.step).toFixed(1)},${sy(p.loss).toFixed(1)}`).join(" ");
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => Math.round(xMin + f * (xMax - xMin)));
  return (
    <svg className="loss-curve" width={W} height={H}>
      <line x1={pad.l} x2={W - pad.r} y1={sy(0)} y2={sy(0)} className="axis" />
      <line x1={pad.l} x2={pad.l} y1={pad.t} y2={sy(0)} className="axis" />
      {[0, 1, 2, 3].filter((y) => y <= yMax).map((y) => (
        <text key={y} x={pad.l - 6} y={sy(y) + 4} textAnchor="end" className="tick">
          {y}
        </text>
      ))}
      {ticks.map((t) => (
        <text key={t} x={sx(t)} y={H - 8} textAnchor="middle" className="tick">
          {t}
        </text>
      ))}
      <line x1={pad.l} x2={W - pad.r} y1={sy(optimal)} y2={sy(optimal)} className="optimal-line" />
      <text x={W - pad.r} y={sy(optimal) - 5} textAnchor="end" className="legend">
        best possible {optimal.toFixed(3)}
      </text>
      {batch.length > 1 && <path d={path(batch)} className="batch-line" />}
      {corpus.length > 1 && <path d={path(corpus)} className="corpus-line" />}
      {marker !== undefined && <line x1={sx(marker)} x2={sx(marker)} y1={pad.t} y2={sy(0)} className="tau-marker" />}
      <text x={pad.l + 8} y={pad.t + 12} className="legend">
        — loss on the whole corpus · faint: loss of each minibatch
      </text>
    </svg>
  );
}
