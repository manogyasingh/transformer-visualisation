export type ColorScale = "diverging" | "sequential";

const POSITIVE = [234, 88, 12];
const NEGATIVE = [37, 99, 235];
const SEQUENTIAL = [109, 40, 217];

export interface CellColors {
  background: string;
  color: string;
}

export function cellColors(value: number, scale: ColorScale, maxAbs: number): CellColors {
  if (!Number.isFinite(value)) return { background: "", color: "" };
  const t = maxAbs > 0 ? Math.min(1, Math.abs(value) / maxAbs) : 0;
  const alpha = 0.06 + 0.84 * t;
  const rgb = scale === "sequential" ? SEQUENTIAL : value >= 0 ? POSITIVE : NEGATIVE;
  const [r, g, b] = rgb.map((c) => Math.round(255 - alpha * (255 - c)));
  return {
    background: `rgb(${r}, ${g}, ${b})`,
    color: alpha > 0.55 ? "#fff" : "#1f2937",
  };
}

export function maxAbsOf(values: number[][]): number {
  let m = 0;
  for (const row of values) for (const v of row) if (Number.isFinite(v)) m = Math.max(m, Math.abs(v));
  return m;
}

export const HEAD_COLORS = ["#0891b2", "#d97706", "#7c3aed", "#db2777"];
