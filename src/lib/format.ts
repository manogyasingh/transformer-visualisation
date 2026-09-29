const MINUS = "\u2212";

export const ROW_COLOR = "#059669";
export const COL_COLOR = "#c026d3";
export const RESULT_COLOR = "#b45309";

function fixed(x: number, dp: number): string {
  const s = x.toFixed(dp);
  return Number(s) === 0 ? (0).toFixed(dp) : s;
}

/** Plain-text number for matrix cells and labels (typographic minus sign). */
export function fmt(x: number, dp = 2): string {
  if (x === -Infinity) return `${MINUS}∞`;
  if (x === Infinity) return "∞";
  const s = fixed(x, dp);
  return s.startsWith("-") ? MINUS + s.slice(1) : s;
}

/** TeX number. */
export function tn(x: number, dp = 4): string {
  if (x === -Infinity) return "-\\infty";
  if (x === Infinity) return "\\infty";
  return fixed(x, dp);
}

/** TeX number that switches to scientific notation for very small magnitudes. */
export function tsci(x: number, dp = 4): string {
  if (x !== 0 && Math.abs(x) < 10 ** -dp) {
    const [m, e] = x.toExponential(2).split("e");
    return `${m}\\times10^{${Number(e)}}`;
  }
  return tn(x, dp);
}

/** TeX number wrapped in parentheses, for use as a factor in a product. */
export function tp(x: number, dp = 4): string {
  return `(${tn(x, dp)})`;
}

/** TeX number as an operand of + or -, parenthesised only when negative. */
export function ta(x: number, dp = 4): string {
  return x < 0 || x === -Infinity ? tp(x, dp) : tn(x, dp);
}

/** A TeX term preceded by its sign, for use inside a sum ("+ 0.12" / "- 0.12"). */
export function tsigned(x: number, dp = 4, first = false): string {
  const s = tn(x, dp);
  if (first) return s;
  return s.startsWith("-") ? `- ${s.slice(1)}` : `+ ${s}`;
}

export function colored(color: string, body: string): string {
  return `\\textcolor{${color}}{${body}}`;
}

export function emphasized(body: string): string {
  return `\\colorbox{#fde68a}{$${body}$}`;
}

export function boxed(x: number, dp = 4): string {
  return `\\boxed{${tn(x, dp)}}`;
}

/** Sum of plain numbers written with proper signs: "0.12 - 0.34 + 0.56". */
export function texSum(values: number[], dp = 4): string {
  return values.map((v, i) => tsigned(v, dp, i === 0)).join(" ");
}

/** TeX number with `sig` significant figures, switching to scientific notation for small values. */
export function tg(x: number, sig = 4): string {
  if (x === 0) return "0";
  if (!Number.isFinite(x)) return tn(x);
  const a = Math.abs(x);
  if (a >= 1e-3 && a < 1e5) return String(Number(x.toPrecision(sig)));
  const [m, e] = x.toExponential(sig - 1).split("e");
  return `${m}\\times10^{${Number(e)}}`;
}

/** Parenthesised factor in gradient notation. */
export function tgp(x: number): string {
  return `(${tg(x)})`;
}

/** Operand of + or - in gradient notation, parenthesised only when negative. */
export function tga(x: number): string {
  return x < 0 ? tgp(x) : tg(x);
}

export function tgSum(values: number[]): string {
  return values
    .map((v, i) => {
      const s = tg(v);
      if (i === 0) return s;
      return s.startsWith("-") ? `- ${s.slice(1)}` : `+ ${s}`;
    })
    .join(" ");
}

export interface DotExpansion {
  /** "(a0)(b0) + (a1)(b1) + ..." */
  factors: string;
  /** "a0*b0 + a1*b1 + ..." evaluated term by term */
  products: string;
  value: number;
}

/**
 * Writes out a dot product term by term, colouring the left factors with the
 * row colour and the right factors with the column colour.
 */
export function dotExpansion(
  a: number[],
  b: number[],
  opts: { emphasize?: number; dp?: number; aColor?: string; bColor?: string; sig?: boolean } = {},
): DotExpansion {
  const dp = opts.dp ?? 4;
  const aColor = opts.aColor ?? ROW_COLOR;
  const bColor = opts.bColor ?? COL_COLOR;
  const factor = opts.sig ? tgp : (x: number) => tp(x, dp);
  const terms = a.map((x, k) => {
    const t = `${colored(aColor, factor(x))}${colored(bColor, factor(b[k]))}`;
    return k === opts.emphasize ? emphasized(t) : t;
  });
  const prods = a.map((x, k) => x * b[k]);
  return {
    factors: terms.join(" + "),
    products: opts.sig ? tgSum(prods) : texSum(prods, dp),
    value: prods.reduce((s, v) => s + v, 0),
  };
}
