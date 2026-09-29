import { useState, type ReactNode } from "react";
import { CalcLine, CalcNote, CalcPanel } from "../../components/calc-panel";
import { EqRow, MatMulView, Op, Tabs } from "../../components/layout";
import { colVector, MatrixView, rowVector, type CellSize, type ColGroup } from "../../components/matrix-view";
import { StepLayout } from "../../components/step-layout";
import { Tex } from "../../components/tex";
import { colored, COL_COLOR, dotExpansion, emphasized, ROW_COLOR, tg, tga, tgp, tgSum } from "../../lib/format";
import type { LayerNormGrad, LinearGrad } from "../../model/backward";
import { transpose } from "../../model/linalg";
import type { LayerNormTrace, Matrix } from "../../model/types";
import { useTraining } from "../../training/training-context";
import { indexLabels, tokenColLabels, tokenRowLabels } from "../shared";

export function sizeFor(n: number): CellSize {
  return n > 16 ? "compact" : "normal";
}

/** δA is shorthand for ∂L/∂A, a matrix with the same shape as A. */
export const d = (name: string) => (name.includes("_") ? `(\\delta ${name})` : `{\\delta ${name}}`);

export function useTokens() {
  const { trace } = useTraining();
  const tokens = trace.forward.tokens;
  return { tokens, T: tokens.length, rowLabels: tokenRowLabels(tokens), colLabels: tokenColLabels(tokens) };
}

export function boxedG(x: number): string {
  return `\\boxed{${tg(x)}}`;
}

export interface LinearNames {
  x: string;
  w: string;
  b: string;
  y: string;
}

export function LinearBackwardView({
  x,
  w,
  grad,
  names,
  intro,
  next,
  xColGroups,
  deeper,
  topControls,
  extraViz,
}: {
  x: Matrix;
  w: Matrix;
  grad: LinearGrad;
  names: LinearNames;
  intro: ReactNode;
  next?: ReactNode;
  xColGroups?: ColGroup[];
  deeper?: ReactNode;
  topControls?: ReactNode;
  extraViz?: ReactNode;
}) {
  const { T, rowLabels, colLabels } = useTokens();
  const [tab, setTab] = useState<"input" | "weight" | "bias">("input");
  const nIn = w.length;
  const nOut = w[0].length;
  const [inCell, setInCell] = useState({ i: T - 1, k: 0, j: undefined as number | undefined });
  const [wCell, setWCell] = useState({ k: 0, j: 0, i: undefined as number | undefined });
  const [bCell, setBCell] = useState({ j: 0 });
  const { x: X, w: W, b: B, y: Y } = names;
  const wT = transpose(w);
  const xT = transpose(x);

  let calc: ReactNode;
  if (tab === "input") {
    const { i, k, j } = inCell;
    const e = dotExpansion(grad.dOut[i], w[k], { sig: true, emphasize: j });
    calc = (
      <CalcPanel target={<Tex>{`${d(X)}_{${i},${k}}`}</Tex>}>
        <CalcLine
          label="chain rule"
          tex={String.raw`${d(X)}_{${i},${k}} = \sum_{j=0}^{${nOut - 1}} \frac{\partial L}{\partial ${Y}_{${i},j}}\frac{\partial ${Y}_{${i},j}}{\partial ${X}_{${i},${k}}} = \sum_{j} ${colored(ROW_COLOR, `${d(Y)}_{${i},j}`)}\,${colored(COL_COLOR, `(${W})_{${k},j}`)}`}
        />
        <CalcLine label="substitute" tex={`= ${e.factors}`} />
        <CalcLine label="multiply" tex={`= ${e.products}`} />
        <CalcLine label="sum" tex={`= ${boxedG(grad.dX[i][k])}`} />
      </CalcPanel>
    );
  } else if (tab === "weight") {
    const { k, j, i } = wCell;
    const e = dotExpansion(xT[k], grad.dOut.map((r) => r[j]), { sig: true, emphasize: i });
    calc = (
      <CalcPanel target={<Tex>{`${d(W)}_{${k},${j}}`}</Tex>}>
        <CalcLine
          label="chain rule"
          tex={String.raw`${d(W)}_{${k},${j}} = \sum_{i=0}^{${T - 1}} \frac{\partial L}{\partial ${Y}_{i,${j}}}\frac{\partial ${Y}_{i,${j}}}{\partial (${W})_{${k},${j}}} = \sum_{i} ${colored(ROW_COLOR, `${X}_{i,${k}}`)}\,${colored(COL_COLOR, `${d(Y)}_{i,${j}}`)}`}
        />
        <CalcLine label="substitute" tex={`= ${e.factors}`} />
        <CalcLine label="multiply" tex={`= ${e.products}`} />
        <CalcLine label="sum" tex={`= ${boxedG(grad.dW[k][j])}`} />
        <CalcNote>The same weight is used at every position, so its gradient sums one contribution per position.</CalcNote>
      </CalcPanel>
    );
  } else {
    const { j } = bCell;
    calc = (
      <CalcPanel target={<Tex>{`${d(B)}_{${j}}`}</Tex>}>
        <CalcLine
          label="chain rule"
          tex={String.raw`${d(B)}_{${j}} = \sum_{i=0}^{${T - 1}} ${d(Y)}_{i,${j}}\cdot\frac{\partial ${Y}_{i,${j}}}{\partial (${B})_{${j}}} = \sum_i ${d(Y)}_{i,${j}}\cdot 1`}
        />
        <CalcLine label="sum" tex={`= ${tgSum(grad.dOut.map((r) => r[j]))} = ${boxedG(grad.db[j])}`} />
      </CalcPanel>
    );
  }

  return (
    <StepLayout
      explain={
        <>
          {intro}
          <p>
            Forward, this layer computed <Tex>{`${Y} = ${X}${W} + ${B}`}</Tex>. Backward, the gradient{" "}
            <Tex>{d(Y)}</Tex> that arrived from above is turned into three gradients by the chain rule: one for the
            input (passed further back), and one each for the weight and the bias (used to update them).
          </p>
          {next}
        </>
      }
      formula={
        <>
          <Tex display>{`${d(X)} = ${d(Y)}\\,${W}^{\\top}`}</Tex>
          <Tex display>{`${d(W)} = ${X}^{\\top}\\,${d(Y)}`}</Tex>
          <Tex display>{`${d(B)} = \\textstyle\\sum_i ${d(Y)}_{i,:}`}</Tex>
        </>
      }
      deeper={deeper}
      calc={calc}
    >
      {topControls}
      <Tabs
        value={tab}
        onChange={setTab}
        options={[
          { value: "input", label: <Tex>{`\\text{input gradient } ${d(X)}`}</Tex> },
          { value: "weight", label: <Tex>{`\\text{weight gradient } ${d(W)}`}</Tex> },
          { value: "bias", label: <Tex>{`\\text{bias gradient } ${d(B)}`}</Tex> },
        ]}
      />
      {tab === "input" && (
        <MatMulView
          corner={
            <div className="mm-hint">
              <Tex>{`${d(X)} = ${d(Y)}\\,${W}^\\top`}</Tex>
            </div>
          }
          b={
            <MatrixView
              data={wT}
              label={`${W}^\\top`}
              rowLabels={indexLabels(nOut)}
              size={sizeFor(nIn)}
              highlight={{ cols: [inCell.k], focus: inCell.j !== undefined ? [inCell.j, inCell.k] : null }}
              onHover={(j, k) => setInCell((c) => ({ ...c, k, j }))}
            />
          }
          a={
            <MatrixView
              data={grad.dOut}
              label={d(Y)}
              rowLabels={rowLabels}
              markRow={T - 1}
              size={sizeFor(nOut)}
              autoScale
              highlight={{ rows: [inCell.i], focus: inCell.j !== undefined ? [inCell.i, inCell.j] : null }}
              onHover={(i, j) => setInCell((c) => ({ ...c, i, j }))}
            />
          }
          c={
            <MatrixView
              data={grad.dX}
              label={d(X)}
              size={sizeFor(nIn)}
              colGroups={xColGroups}
              autoScale
              highlight={{ focus: [inCell.i, inCell.k] }}
              onHover={(i, k) => setInCell({ i, k, j: undefined })}
            />
          }
        />
      )}
      {tab === "weight" && (
        <MatMulView
          corner={
            <div className="mm-hint">
              <Tex>{`${d(W)} = ${X}^\\top ${d(Y)}`}</Tex>
            </div>
          }
          b={
            <MatrixView
              data={grad.dOut}
              label={d(Y)}
              rowLabels={rowLabels}
              size={sizeFor(nOut)}
              autoScale
              highlight={{ cols: [wCell.j], focus: wCell.i !== undefined ? [wCell.i, wCell.j] : null }}
              onHover={(i, j) => setWCell((c) => ({ ...c, j, i }))}
            />
          }
          a={
            <MatrixView
              data={xT}
              label={`${X}^\\top`}
              rowLabels={indexLabels(nIn)}
              colLabels={colLabels}
              highlight={{ rows: [wCell.k], focus: wCell.i !== undefined ? [wCell.k, wCell.i] : null }}
              onHover={(k, i) => setWCell((c) => ({ ...c, k, i }))}
            />
          }
          c={
            <MatrixView
              data={grad.dW}
              label={d(W)}
              size={sizeFor(nOut)}
              autoScale
              highlight={{ focus: [wCell.k, wCell.j] }}
              onHover={(k, j) => setWCell({ k, j, i: undefined })}
            />
          }
        />
      )}
      {tab === "bias" && (
        <EqRow>
          <MatrixView
            data={grad.dOut}
            label={d(Y)}
            rowLabels={rowLabels}
            markRow={T - 1}
            size={sizeFor(nOut)}
            autoScale
            highlight={{ cols: [bCell.j] }}
            onHover={(_, j) => setBCell({ j })}
          />
          <Op>
            <span className="op-text">sum over positions →</span>
          </Op>
          <MatrixView
            data={rowVector(grad.db)}
            label={d(B)}
            size={sizeFor(nOut)}
            autoScale
            highlight={{ focus: [0, bCell.j] }}
            onHover={(_, j) => setBCell({ j })}
          />
        </EqRow>
      )}
      {extraViz}
    </StepLayout>
  );
}

export interface LayerNormNamesB {
  x: string;
  y: string;
  gamma: string;
  beta: string;
}

export function LayerNormBackwardView({
  ln,
  grad,
  names,
  intro,
}: {
  ln: LayerNormTrace;
  grad: LayerNormGrad;
  names: LayerNormNamesB;
  intro: ReactNode;
}) {
  const { T, rowLabels } = useTokens();
  const dm = ln.gamma.length;
  const [tab, setTab] = useState<"input" | "params">("input");
  const [cell, setCell] = useState({ i: T - 1, j: 0 });
  const { i, j } = cell;
  const hover = (i: number, j: number) => setCell({ i, j });
  const hl = { rows: [i], focus: [i, j] as [number, number] };
  const { x: X, y: Y, gamma: G, beta: B } = names;
  const XH = `\\hat{${X}}`;
  const rowTerms = grad.dXhat[i]
    .map((v, k) => {
      const s = tg(v);
      const term = k === 0 ? s : s.startsWith("-") ? `- ${s.slice(1)}` : `+ ${s}`;
      return k === j ? emphasized(term) : term;
    })
    .join(" ");

  const calc =
    tab === "input" ? (
      <CalcPanel
        target={
          <>
            <Tex>{`${d(X)}_{${i},${j}}`}</Tex>
          </>
        }
      >
        <CalcLine
          label="through γ"
          tex={String.raw`${d(XH)}_{${i},${j}} = ${d(Y)}_{${i},${j}}\,(${G})_{${j}} = ${tgp(grad.dOut[i][j])}${tgp(ln.gamma[j])} = ${tg(grad.dXhat[i][j])}`}
        />
        <CalcLine
          label="row mean"
          tex={String.raw`a_{${i}} = \tfrac{1}{${dm}}\textstyle\sum_{j'} ${d(XH)}_{${i},j'} = \tfrac{1}{${dm}}\big(${rowTerms}\big) = ${tg(grad.meanDXhat[i])}`}
        />
        <CalcLine
          label="row mean"
          tex={String.raw`b_{${i}} = \tfrac{1}{${dm}}\textstyle\sum_{j'} ${d(XH)}_{${i},j'}\hat{${X}}_{${i},j'} = ${tg(grad.meanDXhatXhat[i])}`}
        />
        <CalcLine
          label="result"
          tex={String.raw`${d(X)}_{${i},${j}} = \tfrac{1}{\sigma_{${i}}}\big(${d(XH)}_{${i},${j}} - a_{${i}} - \hat{${X}}_{${i},${j}}\, b_{${i}}\big) = ${tg(ln.rstd[i])}\big(${tg(grad.dXhat[i][j])} - ${tgp(grad.meanDXhat[i])} - ${tgp(ln.normalized[i][j])}${tgp(grad.meanDXhatXhat[i])}\big) = ${boxedG(grad.dX[i][j])}`}
        />
      </CalcPanel>
    ) : (
      <CalcPanel
        target={
          <>
            <Tex>{`${d(G)}_{${j}}`}</Tex> and <Tex>{`${d(B)}_{${j}}`}</Tex>
          </>
        }
      >
        <CalcLine
          label="scale"
          tex={String.raw`${d(G)}_{${j}} = \textstyle\sum_i ${colored(ROW_COLOR, `${d(Y)}_{i,${j}}`)}\,${colored(COL_COLOR, `\\hat{${X}}_{i,${j}}`)} = ${dotExpansion(grad.dOut.map((r) => r[j]), ln.normalized.map((r) => r[j]), { sig: true }).factors} = ${boxedG(grad.dGamma[j])}`}
        />
        <CalcLine
          label="shift"
          tex={String.raw`${d(B)}_{${j}} = \textstyle\sum_i ${d(Y)}_{i,${j}} = ${tgSum(grad.dOut.map((r) => r[j]))} = ${boxedG(grad.dBeta[j])}`}
        />
      </CalcPanel>
    );

  return (
    <StepLayout
      explain={
        <>
          {intro}
          <p>
            LayerNorm's backward pass is the least obvious one. Every output of a row depends on every input of that
            row, because the mean <Tex>\mu</Tex> and standard deviation <Tex>\sigma</Tex> are computed from the whole
            row. So each input gradient is the direct term minus two row-wide corrections, <Tex>a_i</Tex> and{" "}
            <Tex>{String.raw`\hat X_{i,j} b_i`}</Tex>. Rows (positions) still never interact.
          </p>
        </>
      }
      formula={
        <>
          <Tex display>{String.raw`${d(XH)} = ${d(Y)} \odot ${G}`}</Tex>
          <Tex display>{String.raw`${d(X)}_{i,j} = \frac{1}{\sigma_i}\Big(${d(XH)}_{i,j} - a_i - \hat{${X}}_{i,j}\,b_i\Big)`}</Tex>
          <Tex display>{String.raw`a_i = \tfrac{1}{d}\textstyle\sum_{j} ${d(XH)}_{i,j},\;\; b_i = \tfrac{1}{d}\sum_{j} ${d(XH)}_{i,j}\hat{${X}}_{i,j}`}</Tex>
          <Tex display>{String.raw`${d(G)} = \textstyle\sum_i ${d(Y)}_{i,:}\odot \hat{${X}}_{i,:},\;\; ${d(B)} = \sum_i ${d(Y)}_{i,:}`}</Tex>
          <div className="formula-caption">
            <Tex>{String.raw`\sigma_i = \sqrt{\sigma_i^2 + \epsilon}`}</Tex> from the forward pass
          </div>
        </>
      }
      deeper={
        <p>
          Where the corrections come from: <Tex>{String.raw`\hat x_k = (x_k - \mu)/\sigma`}</Tex>, with{" "}
          <Tex>{String.raw`\partial\mu/\partial x_j = 1/d`}</Tex> and{" "}
          <Tex>{String.raw`\partial\sigma/\partial x_j = \hat x_j/(d)`}</Tex>. So{" "}
          <Tex>{String.raw`\partial \hat x_k/\partial x_j = \tfrac{1}{\sigma}\big([k{=}j] - \tfrac{1}{d} - \tfrac{1}{d}\hat x_k \hat x_j\big)`}</Tex>
          . Summing <Tex>{String.raw`\delta\hat x_k`}</Tex> times this over k gives the formula. A consequence: the
          gradient of every row sums to zero and is (up to <Tex>\epsilon</Tex>) orthogonal to{" "}
          <Tex>{String.raw`\hat x`}</Tex>, because LayerNorm's output doesn't change if you shift or rescale its
          input.
        </p>
      }
      calc={calc}
    >
      <Tabs
        value={tab}
        onChange={setTab}
        options={[
          { value: "input", label: <Tex>{`\\text{input gradient } ${d(X)}`}</Tex> },
          { value: "params", label: <Tex>{`\\text{parameter gradients } ${d(G)}, ${d(B)}`}</Tex> },
        ]}
      />
      {tab === "input" ? (
        <EqRow>
          <MatrixView data={grad.dOut} label={d(Y)} size="medium" rowLabels={rowLabels} markRow={T - 1} autoScale highlight={hl} onHover={hover} />
          <Op tex={`\\odot ${G}`} />
          <MatrixView data={grad.dXhat} label={d(`\\hat{${X}}`)} size="medium" autoScale highlight={hl} onHover={hover} />
          <MatrixView data={colVector(grad.meanDXhat)} label="a" size="medium" colLabels="none" showShape={false} autoScale highlight={{ rows: [i] }} onHover={(i) => setCell((c) => ({ ...c, i }))} />
          <MatrixView data={colVector(grad.meanDXhatXhat)} label="b" size="medium" colLabels="none" showShape={false} autoScale highlight={{ rows: [i] }} onHover={(i) => setCell((c) => ({ ...c, i }))} />
          <Op tex="\to" />
          <MatrixView data={grad.dX} label={d(X)} size="medium" autoScale highlight={hl} onHover={hover} />
        </EqRow>
      ) : (
        <EqRow>
          <MatrixView data={grad.dOut} label={d(Y)} size="medium" rowLabels={rowLabels} markRow={T - 1} autoScale highlight={{ cols: [j] }} onHover={hover} />
          <MatrixView data={ln.normalized} label={`\\hat{${X}}`} size="medium" highlight={{ cols: [j] }} onHover={hover} />
          <Op tex="\to" />
          <div className="stack">
            <MatrixView data={rowVector(grad.dGamma)} label={d(G)} size="medium" autoScale highlight={{ focus: [0, j] }} onHover={(_, j) => setCell((c) => ({ ...c, j }))} />
            <MatrixView data={rowVector(grad.dBeta)} label={d(B)} size="medium" autoScale highlight={{ focus: [0, j] }} onHover={(_, j) => setCell((c) => ({ ...c, j }))} />
          </div>
        </EqRow>
      )}
    </StepLayout>
  );
}

/** Two gradients meeting at a residual connection: skip + branch = total. */
export function ResidualMergeView({
  skip,
  branch,
  total,
  names,
  intro,
}: {
  skip: Matrix;
  branch: Matrix;
  total: Matrix;
  names: [string, string, string];
  intro: ReactNode;
}) {
  const { T, rowLabels } = useTokens();
  const [cell, setCell] = useState({ i: T - 1, j: 0 });
  const { i, j } = cell;
  const hl = { focus: [i, j] as [number, number] };
  const hover = (i: number, j: number) => setCell({ i, j });
  const [A, B, C] = names;
  return (
    <StepLayout
      explain={intro}
      formula={
        <>
          <Tex display>{`${C} = ${A} + ${B}`}</Tex>
          <div className="formula-caption">
            multivariable chain rule: a value used in two places gets the sum of both gradients
          </div>
        </>
      }
      calc={
        <CalcPanel target={<Tex>{`${C}_{${i},${j}}`}</Tex>}>
          <CalcLine label="add" tex={`${C}_{${i},${j}} = ${tg(skip[i][j])} + ${tga(branch[i][j])} = ${boxedG(total[i][j])}`} />
        </CalcPanel>
      }
    >
      <EqRow>
        <MatrixView data={skip} label={A} rowLabels={rowLabels} markRow={T - 1} autoScale highlight={hl} onHover={hover} />
        <Op tex="+" />
        <MatrixView data={branch} label={B} autoScale highlight={hl} onHover={hover} />
        <Op tex="=" />
        <MatrixView data={total} label={C} autoScale highlight={hl} onHover={hover} />
      </EqRow>
    </StepLayout>
  );
}
