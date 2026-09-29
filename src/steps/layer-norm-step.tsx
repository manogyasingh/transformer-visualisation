import type { ReactNode } from "react";
import { CalcLine, CalcPanel } from "../components/calc-panel";
import { colVector, MatrixView, rowVector } from "../components/matrix-view";
import { StepLayout } from "../components/step-layout";
import { Tex } from "../components/tex";
import { boxed, emphasized, ta, tn, tp, tsigned } from "../lib/format";
import type { LayerNormTrace } from "../model/types";
import { useStep } from "./step-context";
import { tokenRowLabels, useCell } from "./shared";

export interface LayerNormNames {
  input: string;
  output: string;
  gamma: string;
  beta: string;
}

export function LayerNormStep({
  ln,
  names,
  intro,
  deeper,
  onlyLastRowMatters = false,
}: {
  ln: LayerNormTrace;
  names: LayerNormNames;
  intro: ReactNode;
  deeper?: ReactNode;
  onlyLastRowMatters?: boolean;
}) {
  const { trace, cfg } = useStep();
  const T = trace.ids.length;
  const d = cfg.dModel;
  const [cell, setCell] = useCell({ i: T - 1, j: 0 });
  const { i, j } = cell;
  const hover = (i: number, j: number) => setCell({ i, j });
  const hoverRow = (i: number) => setCell((c) => ({ i, j: c.j }));
  const rowHl = { rows: [i], focus: [i, j] as [number, number] };

  const x = ln.input[i];
  const mu = ln.mean[i];
  const variance = ln.variance[i];
  const diffs = x.map((v) => v - mu);
  const sumX = x.map((v, k) => (k === j ? emphasized(tsigned(v, 4, k === 0)) : tsigned(v, 4, k === 0))).join(" ");
  const sumSq = diffs.map((v, k) => (k === j ? emphasized(`${tp(v)}^2`) : `${tp(v)}^2`)).join(" + ");
  const sd = Math.sqrt(variance + cfg.lnEps);
  const { input: X, output: Y, gamma: G, beta: B } = names;

  return (
    <StepLayout
      explain={
        <>
          {intro}
          <p>
            LayerNorm standardises each token's vector <em>on its own</em>: subtract the mean of its {d} numbers,
            divide by their standard deviation, then apply a learned per-feature scale <Tex>\gamma</Tex> and shift{" "}
            <Tex>\beta</Tex>. Rows never interact here.
            {onlyLastRowMatters && " Only the last row feeds the prediction, but every row is normalised."}
          </p>
        </>
      }
      formula={
        <>
          <Tex display>{String.raw`\mu_i = \frac{1}{d}\sum_{j=0}^{d-1} ${X}_{i,j}`}</Tex>
          <Tex display>{String.raw`\sigma_i^2 = \frac{1}{d}\sum_{j=0}^{d-1} (${X}_{i,j}-\mu_i)^2`}</Tex>
          <Tex display>{String.raw`\hat{${X}}_{i,j} = \frac{${X}_{i,j}-\mu_i}{\sqrt{\sigma_i^2+\epsilon}}`}</Tex>
          <Tex display>{String.raw`${Y}_{i,j} = (${G})_j\,\hat{${X}}_{i,j} + (${B})_j`}</Tex>
          <div className="formula-caption">
            d = {d}, <Tex>{String.raw`\epsilon = 10^{-5}`}</Tex>
          </div>
        </>
      }
      deeper={
        deeper ?? (
          <p>
            The variance divides by d, not d − 1 (the “biased” estimator, as in PyTorch), and{" "}
            <Tex>\epsilon</Tex> only matters if a row is nearly constant. After the first two lines every row of{" "}
            <Tex>{String.raw`\hat{${X}}`}</Tex> has mean exactly 0 and variance (almost exactly) 1. GPT-2 is a{" "}
            <em>pre-LN</em> transformer: normalisation is applied to the input of each sub-layer, while the residual
            stream itself is never normalised. That keeps each sub-layer's input at a predictable scale however large
            the stream grows, which makes deep stacks train stably.
          </p>
        )
      }
      calc={
        <CalcPanel
          target={
            <>
              row {i} (<span className="tok-inline">{trace.tokens[i]}</span>), column {j}
            </>
          }
        >
          <CalcLine
            label="mean"
            tex={String.raw`\mu_{${i}} = \tfrac{1}{${d}}\big(${sumX}\big) = \tfrac{1}{${d}}(${tn(x.reduce((a, b) => a + b, 0))}) = ${boxed(mu)}`}
          />
          <CalcLine
            label="variance"
            tex={String.raw`\sigma_{${i}}^2 = \tfrac{1}{${d}}\big(${sumSq}\big) = \tfrac{1}{${d}}(${tn(diffs.reduce((a, b) => a + b * b, 0))}) = ${boxed(variance)}`}
          />
          <CalcLine
            label="normalise"
            tex={String.raw`\hat{${X}}_{${i},${j}} = \dfrac{${tn(x[j])} - ${ta(mu)}}{\sqrt{${tn(variance)} + 0.00001}} = \dfrac{${tn(diffs[j])}}{${tn(sd)}} = ${boxed(ln.normalized[i][j])}`}
          />
          <CalcLine
            label="scale & shift"
            tex={String.raw`${Y}_{${i},${j}} = (${G})_{${j}}\,\hat{${X}}_{${i},${j}} + (${B})_{${j}} = ${tp(ln.gamma[j])}${tp(ln.normalized[i][j])} + ${ta(ln.beta[j])} = ${boxed(ln.output[i][j])}`}
          />
        </CalcPanel>
      }
    >
      <div className="ln-grid">
        <div className="ln-y-params">
          <MatrixView data={rowVector(ln.gamma)} label={G} size="medium" highlight={{ cols: [j] }} onHover={(_, j) => setCell((c) => ({ ...c, j }))} />
          <MatrixView data={rowVector(ln.beta)} label={B} size="medium" colLabels="none" highlight={{ cols: [j] }} onHover={(_, j) => setCell((c) => ({ ...c, j }))} />
        </div>
        <div className="ln-x">
          <MatrixView
            data={ln.input}
            label={X}
            size="medium"
            rowLabels={tokenRowLabels(trace.tokens)}
            markRow={T - 1}
            highlight={rowHl}
            onHover={hover}
          />
        </div>
        <div className="ln-mu">
          <MatrixView data={colVector(ln.mean)} label="\mu" size="medium" colLabels="none" showShape={false} highlight={{ rows: [i] }} onHover={hoverRow} />
        </div>
        <div className="ln-var">
          <MatrixView data={colVector(ln.variance)} label="\sigma^2" size="medium" colLabels="none" showShape={false} highlight={{ rows: [i] }} onHover={hoverRow} />
        </div>
        <div className="ln-xhat">
          <MatrixView data={ln.normalized} label={String.raw`\hat{${X}}`} size="medium" highlight={rowHl} onHover={hover} />
        </div>
        <div className="ln-y">
          <MatrixView data={ln.output} label={Y} size="medium" highlight={rowHl} onHover={hover} />
        </div>
      </div>
    </StepLayout>
  );
}
