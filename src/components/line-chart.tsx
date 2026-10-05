export interface Series {
  label: string;
  color: string;
  points: { x: number; y: number }[];
  dashed?: boolean;
}

export interface Guide {
  y: number;
  label: string;
  color: string;
}

function niceTicks(lo: number, hi: number): number[] {
  const span = hi - lo || 1;
  const step = 10 ** Math.floor(Math.log10(span / 3));
  const unit = [1, 2, 5, 10].map((m) => m * step).find((u) => span / u <= 5) ?? step * 10;
  const ticks: number[] = [];
  for (let v = Math.ceil(lo / unit) * unit; v <= hi + 1e-9; v += unit) ticks.push(Number(v.toFixed(10)));
  return ticks;
}

export function LineChart({
  series,
  xMax,
  yMin,
  yMax,
  guides = [],
  title,
  width = 350,
  height = 210,
}: {
  series: Series[];
  xMax: number;
  yMin: number;
  yMax: number;
  guides?: Guide[];
  title: string;
  width?: number;
  height?: number;
}) {
  const pad = { l: 38, r: 10, t: 22, b: 24 };
  const xHi = Math.max(1, xMax);
  const sx = (x: number) => pad.l + (x / xHi) * (width - pad.l - pad.r);
  const sy = (y: number) => pad.t + (1 - (Math.min(Math.max(y, yMin), yMax) - yMin) / (yMax - yMin)) * (height - pad.t - pad.b);
  const path = (pts: { x: number; y: number }[]) => pts.map((p, i) => `${i ? "L" : "M"}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(" ");
  return (
    <svg className="line-chart" width={width} height={height}>
      <text x={pad.l} y={12} className="line-chart-title">
        {title}
      </text>
      <line x1={pad.l} x2={width - pad.r} y1={sy(yMin)} y2={sy(yMin)} className="axis" />
      <line x1={pad.l} x2={pad.l} y1={pad.t} y2={sy(yMin)} className="axis" />
      {yMin < 0 && yMax > 0 && <line x1={pad.l} x2={width - pad.r} y1={sy(0)} y2={sy(0)} className="axis zero" />}
      {niceTicks(yMin, yMax).map((v) => (
        <text key={v} x={pad.l - 5} y={sy(v) + 3} textAnchor="end" className="tick">
          {v}
        </text>
      ))}
      {[0, 0.5, 1].map((f) => (
        <text key={f} x={sx(f * xHi)} y={height - 8} textAnchor="middle" className="tick">
          {Math.round(f * xHi)}
        </text>
      ))}
      {guides.map((g) => (
        <g key={g.label}>
          <line x1={pad.l} x2={width - pad.r} y1={sy(g.y)} y2={sy(g.y)} stroke={g.color} strokeDasharray="4 4" />
          <text x={width - pad.r} y={sy(g.y) - 4} textAnchor="end" className="legend" style={{ fill: g.color }}>
            {g.label}
          </text>
        </g>
      ))}
      {series.map((s) =>
        s.points.length > 1 ? (
          <path key={s.label} d={path(s.points)} fill="none" stroke={s.color} strokeWidth={2} strokeDasharray={s.dashed ? "5 3" : undefined} />
        ) : null,
      )}
      {series.map((s, k) => (
        <text key={s.label} x={pad.l + 6} y={pad.t + 12 + k * 13} className="legend" style={{ fill: s.color }}>
          — {s.label}
        </text>
      ))}
    </svg>
  );
}
