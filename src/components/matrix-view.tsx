import type { ReactNode } from "react";
import { cellColors, maxAbsOf, type ColorScale } from "../lib/colors";
import { fmt } from "../lib/format";
import type { Matrix } from "../model/types";
import { Tex } from "./tex";

export type CellSize = "normal" | "medium" | "compact" | "tiny";

const SIZES: Record<CellSize, { w: number; h: number; font: number }> = {
  normal: { w: 48, h: 24, font: 11.5 },
  medium: { w: 39, h: 22, font: 10 },
  compact: { w: 31, h: 20, font: 8.6 },
  tiny: { w: 16, h: 16, font: 0 },
};
const GAP = 1;

export interface Highlight {
  rows?: number[];
  cols?: number[];
  cells?: [number, number][];
  focus?: [number, number] | null;
}

export interface ColGroup {
  start: number;
  end: number;
  label: string;
  color: string;
}

export interface MatrixViewProps {
  data: Matrix;
  /** TeX label shown above the matrix. */
  label?: string;
  note?: ReactNode;
  showShape?: boolean;
  rowLabels?: ReactNode[];
  colLabels?: ReactNode[] | "index" | "none";
  verticalColLabels?: boolean;
  size?: CellSize;
  dp?: number;
  scale?: ColorScale;
  maxAbs?: number;
  highlight?: Highlight;
  colGroups?: ColGroup[];
  dimRow?: (i: number) => boolean;
  dimCell?: (i: number, j: number) => boolean;
  markRow?: number;
  /** Display values multiplied by a power of ten (shown in the label) so small gradients stay readable. */
  autoScale?: boolean;
  onHover?: (i: number, j: number) => void;
}

export function MatrixView({
  data,
  label,
  note,
  showShape = true,
  rowLabels,
  colLabels = "index",
  verticalColLabels = false,
  size = "normal",
  dp = 2,
  scale = "diverging",
  maxAbs,
  highlight,
  colGroups,
  dimRow,
  dimCell,
  markRow,
  autoScale = false,
  onHover,
}: MatrixViewProps) {
  const rows = data.length;
  const cols = data[0]?.length ?? 0;
  const { w, h, font } = SIZES[size];
  const scaleMax = maxAbs ?? maxAbsOf(data);
  const exponent = autoScale && scaleMax > 0 ? Math.floor(Math.log10(scaleMax)) : 0;
  const shift = exponent < -1 ? exponent : 0;
  const unit = 10 ** shift;
  const stepX = w + GAP;
  const stepY = h + GAP;
  const gridStyle = { gridTemplateColumns: `repeat(${cols}, ${w}px)`, columnGap: GAP };

  const labels: ReactNode[] | null =
    colLabels === "none" ? null : colLabels === "index" ? data[0].map((_, j) => j) : colLabels;

  return (
    <div className={`matrix matrix-${size}`}>
      {(label || note) && (
        <div className="matrix-label">
          {label && <Tex>{label}</Tex>}
          {showShape && (
            <span className="matrix-shape">
              {rows}×{cols}
            </span>
          )}
          {shift !== 0 && (
            <span className="matrix-unit">
              <Tex>{`\\times 10^{${shift}}`}</Tex>
            </span>
          )}
          {note && <span className="matrix-note">{note}</span>}
        </div>
      )}
      {colGroups && (
        <div className="matrix-groups" style={gridStyle}>
          {colGroups.map((g) => (
            <div
              key={g.label}
              className="matrix-group"
              style={{ gridColumn: `${g.start + 1} / ${g.end + 1}`, borderColor: g.color, color: g.color }}
            >
              {g.label}
            </div>
          ))}
        </div>
      )}
      {labels && (
        <div className={`matrix-collabels${verticalColLabels ? " vertical" : ""}`} style={gridStyle}>
          {labels.map((l, j) => (
            <div key={j} className={`matrix-collabel${highlight?.cols?.includes(j) ? " active" : ""}`}>
              {l}
            </div>
          ))}
        </div>
      )}
      <div className="matrix-body" style={{ ...gridStyle, rowGap: GAP }}>
        {rowLabels && (
          <div className="matrix-rowlabels" style={{ gridTemplateRows: `repeat(${rows}, ${h}px)`, rowGap: GAP }}>
            {rowLabels.map((l, i) => (
              <div
                key={i}
                className={`matrix-rowlabel${i === markRow ? " marked" : ""}${highlight?.rows?.includes(i) ? " active" : ""}`}
              >
                {l}
              </div>
            ))}
          </div>
        )}
        {data.map((row, i) =>
          row.map((v, j) => {
            const { background, color } = cellColors(v, scale, scaleMax);
            const dim = dimRow?.(i) || dimCell?.(i, j);
            const cls = [
              "cell",
              v === -Infinity ? "cell-neg-inf" : "",
              dim ? "cell-dim" : "",
              onHover ? "cell-hoverable" : "",
            ].join(" ");
            return (
              <div
                key={`${i}-${j}`}
                className={cls}
                style={{ background, color, height: h, fontSize: font }}
                onMouseEnter={onHover ? () => onHover(i, j) : undefined}
              >
                {font > 0 ? fmt(v / unit, dp) : null}
              </div>
            );
          }),
        )}
        {highlight?.rows?.map((i) => (
          <div
            key={`r${i}`}
            className="hl hl-row"
            style={{ top: i * stepY - 2, left: -2, width: cols * stepX - GAP + 4, height: h + 4 }}
          />
        ))}
        {highlight?.cols?.map((j) => (
          <div
            key={`c${j}`}
            className="hl hl-col"
            style={{ top: -2, left: j * stepX - 2, width: w + 4, height: rows * stepY - GAP + 4 }}
          />
        ))}
        {highlight?.cells?.map(([i, j]) => (
          <div
            key={`x${i}-${j}`}
            className="hl hl-cell"
            style={{ top: i * stepY - 1, left: j * stepX - 1, width: w + 2, height: h + 2 }}
          />
        ))}
        {highlight?.focus && (
          <div
            className="hl hl-focus"
            style={{
              top: highlight.focus[0] * stepY - 2,
              left: highlight.focus[1] * stepX - 2,
              width: w + 4,
              height: h + 4,
            }}
          />
        )}
      </div>
    </div>
  );
}

export function rowVector(v: number[]): Matrix {
  return [v];
}

export function colVector(v: number[]): Matrix {
  return v.map((x) => [x]);
}
