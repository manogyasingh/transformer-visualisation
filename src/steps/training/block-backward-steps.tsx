import { useState, type ComponentType } from "react";
import { CalcLine, CalcNote, CalcPanel } from "../../components/calc-panel";
import { EqRow, MatMulView, Op, Tabs } from "../../components/layout";
import { MatrixView } from "../../components/matrix-view";
import { StepLayout } from "../../components/step-layout";
import { Tex } from "../../components/tex";
import { colored, COL_COLOR, dotExpansion, ROW_COLOR, tg, tgp } from "../../lib/format";
import { geluDerivative } from "../../model/backward";
import { GELU_C, transpose } from "../../model/linalg";
import { useTraining } from "../../training/training-context";
import type { BackwardBlockKind } from "../step-defs";
import { useStep } from "../step-context";
import { headColGroups, HeadTabs, indexLabels, Tok } from "../shared";
import {
  boxedG,
  d,
  LayerNormBackwardView,
  LinearBackwardView,
  ResidualMergeView,
  useTokens,
} from "./backward-views";

function useTrainBlock() {
  const { step, cfg } = useStep();
  const tr = useTraining();
  const layer = step.layer ?? 0;
  return {
    cfg,
    layer,
    L: layer + 1,
    last: layer === cfg.nLayers - 1,
    fwd: tr.trace.forward.blocks[layer],
    g: tr.trace.blocks[layer],
    p: tr.model.params.layers[layer],
  };
}

function ResidOutStep() {
  const { L, last, g } = useTrainBlock();
  const { T, rowLabels } = useTokens();
  const [cell, setCell] = useState({ i: T - 1, j: 0 });
  const X = `X^{(${L})}`;
  return (
    <StepLayout
      explain={
        <>
          <p>
            Backpropagation now enters block {L} from the top. The gradient <Tex>{d(X)}</Tex> of the block's
            output came from {last ? "the final LayerNorm" : `block ${L + 1}`}.
          </p>
          <p>
            The block's last operation was the residual addition <Tex>{`${X} = R + D`}</Tex>. Its derivative with
            respect to each term is the identity, so the gradient is <strong>copied unchanged</strong> into both: into
            the MLP branch as <Tex>{d("D")}</Tex>, and along the skip connection towards <Tex>R</Tex>. This
            uninterrupted skip path is why deep residual networks train well: the gradient reaches early layers
            without having to pass through every sub-layer.
          </p>
        </>
      }
      formula={
        <>
          <Tex display>{`${X} = R + D`}</Tex>
          <Tex display>{`${d("D")} = ${d(X)},\\qquad ${d("R")}^{\\text{skip}} = ${d(X)}`}</Tex>
        </>
      }
      calc={
        <CalcPanel target={<Tex>{`${d("D")}_{${cell.i},${cell.j}}`}</Tex>}>
          <CalcLine
            label="chain rule"
            tex={`${d("D")}_{${cell.i},${cell.j}} = ${d(X)}_{${cell.i},${cell.j}}\\cdot\\frac{\\partial (R + D)_{${cell.i},${cell.j}}}{\\partial D_{${cell.i},${cell.j}}} = ${tg(g.dOut[cell.i][cell.j])}\\cdot 1 = ${boxedG(g.dOut[cell.i][cell.j])}`}
          />
        </CalcPanel>
      }
    >
      <EqRow>
        <MatrixView
          data={g.dOut}
          label={d(X)}
          rowLabels={rowLabels}
          markRow={T - 1}
          autoScale
          highlight={{ focus: [cell.i, cell.j] }}
          onHover={(i, j) => setCell({ i, j })}
        />
        <Op>
          <span className="op-text">copied to both branches →</span>
        </Op>
        <MatrixView data={g.dOut} label={d("D")} autoScale highlight={{ focus: [cell.i, cell.j] }} onHover={(i, j) => setCell({ i, j })} />
        <MatrixView data={g.dOut} label={`${d("R")}^{\\text{skip}}`} autoScale highlight={{ focus: [cell.i, cell.j] }} onHover={(i, j) => setCell({ i, j })} />
      </EqRow>
    </StepLayout>
  );
}

function MlpDownStep() {
  const { fwd, g, p } = useTrainBlock();
  return (
    <LinearBackwardView
      x={fwd.act}
      w={p.mlp.w2}
      grad={g.mlpDown}
      names={{ x: "G", w: "W_2", b: "b_2", y: "D" }}
      intro={
        <p>
          First stop on the MLP branch: the down-projection <Tex>{"D = G W_2 + b_2"}</Tex>, where <Tex>G</Tex> is the
          GELU output.
        </p>
      }
    />
  );
}

function GeluStep() {
  const { fwd, g } = useTrainBlock();
  const { T, rowLabels, tokens } = useTokens();
  const [cell, setCell] = useState({ i: T - 1, j: 0 });
  const { i, j } = cell;
  const u = fwd.up[i][j];
  const t = Math.tanh(GELU_C * (u + 0.044715 * u ** 3));
  const hl = { rows: [i], focus: [i, j] as [number, number] };
  const hover = (i: number, j: number) => setCell({ i, j });
  const dead = g.geluDeriv[i].filter((x) => Math.abs(x) < 0.05).length;
  return (
    <StepLayout
      explain={
        <>
          <p>
            GELU acts elementwise, so its backward pass is elementwise too: each gradient is multiplied by the slope
            of GELU at that unit's input, <Tex>{"\\mathrm{GELU}'(U_{i,j})"}</Tex>.
          </p>
          <p>
            Units with a strongly negative input have slope ≈ 0 and block the gradient, so their incoming weights
            barely learn from this example. In row {i} (<Tok>{tokens[i]}</Tok>),{" "}
            {dead} of {g.geluDeriv[i].length} units have |slope| &lt; 0.05. Large positive inputs have slope ≈ 1 and
            pass the gradient through unchanged.
          </p>
        </>
      }
      formula={
        <>
          <Tex display>{"\\delta U = \\delta G \\odot \\mathrm{GELU}'(U)"}</Tex>
          <Tex display>{"\\mathrm{GELU}'(u) = \\tfrac12(1 + t) + \\tfrac12 u (1 - t^2)\\sqrt{2/\\pi}\\,(1 + 3\\cdot 0.044715\\,u^2)"}</Tex>
          <div className="formula-caption">
            with <Tex>{"t = \\tanh\\big(\\sqrt{2/\\pi}(u + 0.044715u^3)\\big)"}</Tex>
          </div>
        </>
      }
      calc={
        <CalcPanel target={<Tex>{`${d("U")}_{${i},${j}}`}</Tex>}>
          <CalcLine label="input" tex={`u = U_{${i},${j}} = ${tg(u)},\\quad t = ${tg(t)}`} />
          <CalcLine
            label="slope"
            tex={`\\mathrm{GELU}'(u) = \\tfrac12(1 + ${tg(t)}) + \\tfrac12${tgp(u)}(1 - ${tgp(t)}^2)(0.7979)(1 + 0.134145\\cdot${tgp(u)}^2) = ${tg(geluDerivative(u))}`}
          />
          <CalcLine
            label="result"
            tex={`${d("U")}_{${i},${j}} = ${d("G")}_{${i},${j}}\\cdot\\mathrm{GELU}'(u) = ${tgp(g.mlpDown.dX[i][j])}${tgp(g.geluDeriv[i][j])} = ${boxedG(g.dU[i][j])}`}
          />
        </CalcPanel>
      }
    >
      <div className="gelu-mats">
        <MatrixView data={g.mlpDown.dX} label={d("G")} rowLabels={rowLabels} markRow={T - 1} size="compact" autoScale highlight={hl} onHover={hover} />
        <div className="arch-down">⊙ multiply elementwise by the slope</div>
        <MatrixView data={g.geluDeriv} label="\mathrm{GELU}'(U)" rowLabels={rowLabels} markRow={T - 1} size="compact" highlight={hl} onHover={hover} />
        <div className="arch-down">=</div>
        <MatrixView data={g.dU} label={d("U")} rowLabels={rowLabels} markRow={T - 1} size="compact" autoScale highlight={hl} onHover={hover} />
      </div>
    </StepLayout>
  );
}

function MlpUpStep() {
  const { fwd, g, p } = useTrainBlock();
  return (
    <LinearBackwardView
      x={fwd.ln2.output}
      w={p.mlp.w1}
      grad={g.mlpUp}
      names={{ x: "N", w: "W_1", b: "b_1", y: "U" }}
      intro={
        <p>
          Next, the up-projection <Tex>{"U = N W_1 + b_1"}</Tex>. Its input gradient <Tex>{d("N")}</Tex> is back to
          width 8.
        </p>
      }
    />
  );
}

function Ln2Step() {
  const { fwd, g } = useTrainBlock();
  return (
    <LayerNormBackwardView
      ln={fwd.ln2}
      grad={g.ln2}
      names={{ x: "R", y: "N", gamma: "\\gamma_2", beta: "\\beta_2" }}
      intro={
        <p>
          The MLP branch began with <Tex>{"N = \\mathrm{LN}_2(R)"}</Tex>. Backpropagating through it gives the
          gradient that reaches <Tex>R</Tex> via the MLP, plus gradients for <Tex>{"\\gamma_2"}</Tex> and{" "}
          <Tex>{"\\beta_2"}</Tex>.
        </p>
      }
    />
  );
}

function ResidMidStep() {
  const { L, g } = useTrainBlock();
  return (
    <ResidualMergeView
      skip={g.dOut}
      branch={g.ln2.dX}
      total={g.dR}
      names={[`${d("R")}^{\\text{skip}}`, `${d("R")}^{\\text{MLP}}`, d("R")]}
      intro={
        <p>
          <Tex>R</Tex> was used twice in the forward pass: it was fed into the MLP branch and also passed straight to{" "}
          <Tex>{`X^{(${L})}`}</Tex> by the skip connection. By the multivariable chain rule, its total gradient is
          the sum of the gradients along both paths.
        </p>
      }
    />
  );
}

function AttnProjStep() {
  const { cfg, fwd, g, p } = useTrainBlock();
  return (
    <LinearBackwardView
      x={fwd.concat}
      w={p.attn.wo}
      grad={g.attnProj}
      names={{ x: "Z", w: "W_O", b: "b_O", y: "O" }}
      xColGroups={headColGroups(cfg.nHeads, cfg.dHead)}
      intro={
        <p>
          Into the attention branch. Because <Tex>{"R = X + O"}</Tex>, the gradient of the attention output is{" "}
          <Tex>{`${d("O")} = ${d("R")}`}</Tex> (another residual copy). The first operation to undo is the output
          projection <Tex>{"O = Z W_O + b_O"}</Tex>.
        </p>
      }
      next={
        <p>
          The input gradient <Tex>{d("Z")}</Tex> is split back into one {cfg.dHead}-column slice per head (the colored
          bars), which is where the per-head backward pass starts.
        </p>
      }
    />
  );
}

function WeightedSumStep() {
  const { g, fwd } = useTrainBlock();
  const { head } = useStep();
  const { T, rowLabels, colLabels } = useTokens();
  const [tab, setTab] = useState<"dA" | "dV">("dA");
  const [cell, setCell] = useState({ i: T - 1, j: 0, k: undefined as number | undefined });
  const hg = g.heads[head];
  const hf = fwd.heads[head];
  const h = head + 1;
  const { i, j, k } = cell;

  const calc =
    tab === "dA" ? (
      <CalcPanel target={<Tex>{`${d(`A^{(${h})}`)}_{${i},${j}}`}</Tex>}>
        <CalcLine
          label="chain rule"
          tex={`${d("A")}_{${i},${j}} = \\sum_c ${colored(ROW_COLOR, `${d("Z")}_{${i},c}`)}\\,${colored(COL_COLOR, `V_{${j},c}`)}`}
        />
        <CalcLine label="substitute" tex={`= ${dotExpansion(hg.dZ[i], hf.v[j], { sig: true, emphasize: k }).factors}`} />
        <CalcLine label="sum" tex={`= ${boxedG(hg.dA[i][j])}`} />
        {j > i && <CalcNote>This entry is computed, but it is multiplied by a zero attention weight in the next step, so it has no effect.</CalcNote>}
      </CalcPanel>
    ) : (
      <CalcPanel target={<Tex>{`${d(`V^{(${h})}`)}_{${i},${j}}`}</Tex>}>
        <CalcLine
          label="chain rule"
          tex={`${d("V")}_{${i},${j}} = \\sum_{i'} ${colored(ROW_COLOR, `A_{i',${i}}`)}\\,${colored(COL_COLOR, `${d("Z")}_{i',${j}}`)}`}
        />
        <CalcLine
          label="substitute"
          tex={`= ${dotExpansion(hf.weights.map((r) => r[i]), hg.dZ.map((r) => r[j]), { sig: true, emphasize: k }).factors}`}
        />
        <CalcLine label="sum" tex={`= ${boxedG(hg.dV[i][j])}`} />
        <CalcNote>
          Position {i}'s value is pulled on by every later position that attended to it: gradient flows backwards in
          time.
        </CalcNote>
      </CalcPanel>
    );

  return (
    <StepLayout
      explain={
        <>
          <p>
            Forward, head {h} computed <Tex>{"Z = AV"}</Tex>. That product has two inputs, so there are two gradients:
            one for the attention weights <Tex>A</Tex> (continuing into the softmax), and one for the values{" "}
            <Tex>V</Tex>.
          </p>
          <p>
            <Tex>{d("V")}</Tex> is where attention sends learning signal <em>backwards in time</em>: the value of an
            early token receives gradient from every later token that looked at it.
          </p>
        </>
      }
      formula={
        <>
          <Tex display>{`${d("A")} = ${d("Z")}\\,V^{\\top}`}</Tex>
          <Tex display>{`${d("V")} = A^{\\top}\\,${d("Z")}`}</Tex>
          <div className="formula-caption">(all per head; head superscripts dropped)</div>
        </>
      }
      calc={calc}
    >
      <HeadTabs />
      <Tabs
        value={tab}
        onChange={(t) => {
          setTab(t);
          setCell({ i: T - 1, j: 0, k: undefined });
        }}
        options={[
          { value: "dA", label: <Tex>{"\\text{gradient of weights } \\delta A"}</Tex> },
          { value: "dV", label: <Tex>{"\\text{gradient of values } \\delta V"}</Tex> },
        ]}
      />
      {tab === "dA" ? (
        <MatMulView
          b={<MatrixView data={transpose(hf.v)} label={`V^{(${h})\\top}`} rowLabels={indexLabels(hf.v[0].length)} colLabels={colLabels} highlight={{ cols: [j] }} onHover={(c, j) => setCell((x) => ({ ...x, j, k: c }))} />}
          a={<MatrixView data={hg.dZ} label={d(`Z^{(${h})}`)} rowLabels={rowLabels} markRow={T - 1} autoScale highlight={{ rows: [i], focus: k !== undefined ? [i, k] : null }} onHover={(i, c) => setCell((x) => ({ ...x, i, k: c }))} />}
          c={<MatrixView data={hg.dA} label={d(`A^{(${h})}`)} colLabels={colLabels} autoScale dimCell={(i, j) => j > i} highlight={{ focus: [i, j] }} onHover={(i, j) => setCell({ i, j, k: undefined })} />}
        />
      ) : (
        <MatMulView
          b={<MatrixView data={hg.dZ} label={d(`Z^{(${h})}`)} rowLabels={rowLabels} autoScale highlight={{ cols: [j], focus: k !== undefined ? [k, j] : null }} onHover={(ii, j) => setCell((x) => ({ ...x, j, k: ii }))} />}
          a={<MatrixView data={transpose(hf.weights)} label={`A^{(${h})\\top}`} rowLabels={rowLabels} colLabels={colLabels} scale="sequential" maxAbs={1} highlight={{ rows: [i], focus: k !== undefined ? [i, k] : null }} onHover={(i, ii) => setCell((x) => ({ ...x, i, k: ii }))} />}
          c={<MatrixView data={hg.dV} label={d(`V^{(${h})}`)} autoScale highlight={{ focus: [i, j] }} onHover={(i, j) => setCell({ i, j, k: undefined })} />}
        />
      )}
    </StepLayout>
  );
}

function SoftmaxBackStep() {
  const { g, fwd } = useTrainBlock();
  const { head } = useStep();
  const { T, rowLabels, colLabels } = useTokens();
  const hg = g.heads[head];
  const hf = fwd.heads[head];
  const h = head + 1;
  const [cell, setCell] = useState({ i: T - 1, j: 0 });
  const { i, j } = cell;
  const hl = { rows: [i], focus: [i, j] as [number, number] };
  const hover = (i: number, j: number) => setCell({ i, j });
  return (
    <StepLayout
      explain={
        <>
          <p>
            Each row of <Tex>A</Tex> is a softmax of the same row of scores, so changing one score changes the whole
            row. The resulting gradient has a neat form: a score is pushed up if its weight's gradient{" "}
            <Tex>{"\\delta A_{i,j}"}</Tex> is larger than the attention-weighted average <Tex>{"r_i"}</Tex> of that
            row, and down otherwise, scaled by the weight itself.
          </p>
          <p>
            Masked entries have <Tex>{"A_{i,j} = 0"}</Tex>, so they get exactly zero gradient: nothing is learned about
            attending to the future.
          </p>
        </>
      }
      formula={
        <>
          <Tex display>{"r_i = \\textstyle\\sum_{j'} A_{i,j'}\\,\\delta A_{i,j'}"}</Tex>
          <Tex display>{`${d("\\tilde S")}_{i,j} = A_{i,j}\\big(\\delta A_{i,j} - r_i\\big)`}</Tex>
          <div className="formula-caption">
            from the softmax Jacobian <Tex>{"\\partial A_{i,j'}/\\partial \\tilde S_{i,j} = A_{i,j'}([j'{=}j] - A_{i,j})"}</Tex>
          </div>
        </>
      }
      calc={
        <CalcPanel target={<Tex>{`${d(`\\tilde S^{(${h})}`)}_{${i},${j}}`}</Tex>}>
          <CalcLine
            label="row average"
            tex={`r_{${i}} = ${dotExpansion(hf.weights[i], hg.dA[i], { sig: true, emphasize: j }).factors} = ${tg(hg.rowDot[i])}`}
          />
          <CalcLine
            label="result"
            tex={`${d("\\tilde S")}_{${i},${j}} = A_{${i},${j}}(\\delta A_{${i},${j}} - r_{${i}}) = ${tgp(hf.weights[i][j])}\\big(${tg(hg.dA[i][j])} - ${tgp(hg.rowDot[i])}\\big) = ${boxedG(hg.dS[i][j])}`}
          />
          {j > i && <CalcNote>Masked position: the attention weight is 0, so the gradient is 0.</CalcNote>}
        </CalcPanel>
      }
    >
      <HeadTabs />
      <EqRow>
        <MatrixView data={hf.weights} label={`A^{(${h})}`} rowLabels={rowLabels} markRow={T - 1} colLabels={colLabels} scale="sequential" maxAbs={1} highlight={hl} onHover={hover} />
        <MatrixView data={hg.dA} label={d(`A^{(${h})}`)} colLabels={colLabels} autoScale dimCell={(i, j) => j > i} highlight={hl} onHover={hover} />
        <Op tex="\to" />
        <MatrixView data={hg.dS} label={d(`\\tilde S^{(${h})}`)} colLabels={colLabels} autoScale highlight={hl} onHover={hover} />
      </EqRow>
    </StepLayout>
  );
}

function ScoresBackStep() {
  const { g, fwd, cfg } = useTrainBlock();
  const { head } = useStep();
  const { T, rowLabels, colLabels } = useTokens();
  const [tab, setTab] = useState<"dQ" | "dK">("dQ");
  const [cell, setCell] = useState({ i: T - 1, j: 0, k: undefined as number | undefined });
  const hg = g.heads[head];
  const hf = fwd.heads[head];
  const h = head + 1;
  const s = Math.sqrt(cfg.dHead);
  const { i, j, k } = cell;
  const isQ = tab === "dQ";
  const left = isQ ? hg.dS : transpose(hg.dS);
  const right = isQ ? hf.k : hf.q;
  const out = isQ ? hg.dQ : hg.dK;
  const e = dotExpansion(left[i], right.map((r) => r[j]), { sig: true, emphasize: k });
  const M = isQ ? "Q" : "K";
  const other = isQ ? "K" : "Q";

  return (
    <StepLayout
      explain={
        <>
          <p>
            Forward, <Tex>{"\\tilde S = QK^\\top/\\sqrt{d_h}"}</Tex> (plus the constant mask). The query gradient
            collects, for each query, the keys it was compared with, weighted by how much each score should change;
            the key gradient does the same for each key, collecting the queries that looked at it.
          </p>
          <p>
            Afterwards, the per-head gradients are concatenated back into T×{cfg.dModel} matrices{" "}
            <Tex>{"\\delta Q, \\delta K, \\delta V"}</Tex>: the reverse of the split into heads.
          </p>
        </>
      }
      formula={
        <>
          <Tex display>{`${d("Q")} = ${d("\\tilde S")}\\,K/\\sqrt{d_h}`}</Tex>
          <Tex display>{`${d("K")} = ${d("\\tilde S")}^{\\top} Q/\\sqrt{d_h}`}</Tex>
        </>
      }
      calc={
        <CalcPanel target={<Tex>{`${d(`${M}^{(${h})}`)}_{${i},${j}}`}</Tex>}>
          <CalcLine
            label="chain rule"
            tex={
              isQ
                ? `${d("Q")}_{${i},${j}} = \\tfrac{1}{\\sqrt{${cfg.dHead}}}\\sum_{j'} ${colored(ROW_COLOR, `${d("\\tilde S")}_{${i},j'}`)}\\,${colored(COL_COLOR, `K_{j',${j}}`)}`
                : `${d("K")}_{${i},${j}} = \\tfrac{1}{\\sqrt{${cfg.dHead}}}\\sum_{i'} ${colored(ROW_COLOR, `${d("\\tilde S")}_{i',${i}}`)}\\,${colored(COL_COLOR, `Q_{i',${j}}`)}`
            }
          />
          <CalcLine label="substitute" tex={`= \\tfrac{1}{${s}}\\big(${e.factors}\\big)`} />
          <CalcLine label="sum" tex={`= \\tfrac{1}{${s}}(${tg(e.value)}) = ${boxedG(out[i][j])}`} />
        </CalcPanel>
      }
    >
      <HeadTabs />
      <Tabs
        value={tab}
        onChange={(t) => {
          setTab(t);
          setCell({ i: T - 1, j: 0, k: undefined });
        }}
        options={[
          { value: "dQ", label: <Tex>{"\\text{query gradient } \\delta Q"}</Tex> },
          { value: "dK", label: <Tex>{"\\text{key gradient } \\delta K"}</Tex> },
        ]}
      />
      <MatMulView
        corner={
          <div className="mm-hint">
            <Tex>{isQ ? `${d("\\tilde S")} K / ${s}` : `${d("\\tilde S")}^\\top Q / ${s}`}</Tex>
          </div>
        }
        b={<MatrixView data={right} label={`${other}^{(${h})}`} rowLabels={rowLabels} highlight={{ cols: [j], focus: k !== undefined ? [k, j] : null }} onHover={(kk, j) => setCell((x) => ({ ...x, j, k: kk }))} />}
        a={<MatrixView data={left} label={isQ ? d(`\\tilde S^{(${h})}`) : `${d(`\\tilde S^{(${h})}`)}^{\\top}`} rowLabels={rowLabels} markRow={T - 1} colLabels={colLabels} autoScale highlight={{ rows: [i], focus: k !== undefined ? [i, k] : null }} onHover={(i, kk) => setCell((x) => ({ ...x, i, k: kk }))} />}
        c={<MatrixView data={out} label={d(`${M}^{(${h})}`)} autoScale highlight={{ focus: [i, j] }} onHover={(i, j) => setCell({ i, j, k: undefined })} />}
      />
    </StepLayout>
  );
}

function QkvBackStep() {
  const { fwd, g, p } = useTrainBlock();
  const { rowLabels, T } = useTokens();
  const [which, setWhich] = useState<"q" | "k" | "v">("q");
  const U = which.toUpperCase();
  const w = { q: p.attn.wq, k: p.attn.wk, v: p.attn.wv }[which];
  return (
    <LinearBackwardView
      key={which}
      x={fwd.ln1.output}
      w={w}
      grad={g[which]}
      names={{ x: "Y", w: `W_${U}`, b: `b_${U}`, y: U }}
      topControls={
        <Tabs
          value={which}
          onChange={setWhich}
          options={[
            { value: "q", label: "Query projection" },
            { value: "k", label: "Key projection" },
            { value: "v", label: "Value projection" },
          ]}
        />
      }
      intro={
        <p>
          <Tex>Y</Tex> fed three projections, so it receives three input gradients, which add up (see the sum at the
          bottom). Pick a projection to see its backward pass.
        </p>
      }
      next={
        which === "k" ? (
          <p>
            Check the bias gradient tab: <Tex>{"\\delta b_K"}</Tex> is zero up to rounding error. Every row of{" "}
            <Tex>{"\\delta\\tilde S"}</Tex> sums to zero (softmax outputs always sum to 1), and summing{" "}
            <Tex>{"\\delta K"}</Tex> over positions gives{" "}
            <Tex>{"\\delta b_K = \\tfrac{1}{\\sqrt{d_h}}\\sum_i Q_{i,:}\\sum_j \\delta\\tilde S_{i,j} = 0"}</Tex> (per
            head).
          </p>
        ) : undefined
      }
      extraViz={
        <div className="qkv-sum">
          <h3 className="viz-title">Total gradient of Y: the three contributions add</h3>
          <EqRow>
            <MatrixView data={g.q.dX} label={`${d("Y")}^{Q}`} rowLabels={rowLabels} markRow={T - 1} size="medium" autoScale />
            <Op tex="+" />
            <MatrixView data={g.k.dX} label={`${d("Y")}^{K}`} size="medium" autoScale />
            <Op tex="+" />
            <MatrixView data={g.v.dX} label={`${d("Y")}^{V}`} size="medium" autoScale />
            <Op tex="=" />
            <MatrixView data={g.dY} label={d("Y")} size="medium" autoScale />
          </EqRow>
        </div>
      }
    />
  );
}

function Ln1BackStep() {
  const { fwd, g } = useTrainBlock();
  return (
    <LayerNormBackwardView
      ln={fwd.ln1}
      grad={g.ln1}
      names={{ x: "X", y: "Y", gamma: "\\gamma_1", beta: "\\beta_1" }}
      intro={
        <p>
          The attention branch began with <Tex>{"Y = \\mathrm{LN}_1(X)"}</Tex>, where <Tex>X</Tex> is the block's
          input. This gives the gradient that reaches <Tex>X</Tex> through attention.
        </p>
      }
    />
  );
}

function ResidInStep() {
  const { L, g } = useTrainBlock();
  const X = `X^{(${L - 1})}`;
  return (
    <ResidualMergeView
      skip={g.dR}
      branch={g.ln1.dX}
      total={g.dIn}
      names={[`${d("R")}`, `${d("X")}^{\\text{attn}}`, d(X)]}
      intro={
        <>
          <p>
            The block's input <Tex>X</Tex> was used by the attention branch and by the skip connection (
            <Tex>{"R = X + O"}</Tex>), so again the two gradients add. The result is the gradient of the block's input{" "}
            <Tex>{X}</Tex>.
          </p>
          <p>
            {L > 1 ? `It flows into block ${L - 1}, where the same thirteen steps repeat with that block's weights.` : "It flows into the embedding tables, the last stop."}
          </p>
        </>
      }
    />
  );
}

export const BLOCK_BACKWARD_COMPONENTS: Record<BackwardBlockKind, ComponentType> = {
  "tb-resid-out": ResidOutStep,
  "tb-mlp-down": MlpDownStep,
  "tb-gelu": GeluStep,
  "tb-mlp-up": MlpUpStep,
  "tb-ln2": Ln2Step,
  "tb-resid-mid": ResidMidStep,
  "tb-attn-proj": AttnProjStep,
  "tb-weighted-sum": WeightedSumStep,
  "tb-softmax": SoftmaxBackStep,
  "tb-scores": ScoresBackStep,
  "tb-qkv": QkvBackStep,
  "tb-ln1": Ln1BackStep,
  "tb-resid-in": ResidInStep,
};
