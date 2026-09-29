import { useState, type ReactNode } from "react";
import { CalcLine, CalcNote, CalcPanel } from "../components/calc-panel";
import { AttentionArcs, GeluPlot } from "../components/charts";
import { EqRow, MatMulView, Op, Tabs } from "../components/layout";
import { MatrixView, type CellSize, type ColGroup } from "../components/matrix-view";
import { StepLayout } from "../components/step-layout";
import { Tex } from "../components/tex";
import { HEAD_COLORS } from "../lib/colors";
import {
  boxed,
  colored,
  COL_COLOR,
  dotExpansion,
  emphasized,
  ROW_COLOR,
  ta,
  tn,
  tp,
  tsigned,
} from "../lib/format";
import { GELU_C, gelu, transpose } from "../model/linalg";
import type { Matrix, Vector } from "../model/types";
import { useStep } from "./step-context";
import { LayerNormStep } from "./layer-norm-step";
import {
  headColGroups,
  HeadTabs,
  indexLabels,
  Tok,
  tokenColLabels,
  tokenRowLabels,
  useCell,
  type Cell,
} from "./shared";

function useBlock() {
  const ctx = useStep();
  const layer = ctx.step.layer ?? 0;
  return {
    ...ctx,
    layer,
    block: ctx.trace.blocks[layer],
    params: ctx.model.params.layers[layer],
    T: ctx.trace.ids.length,
  };
}

// ---------------------------------------------------------------------------
// Linear layers: Y = X W + b
// ---------------------------------------------------------------------------

interface LinearNames {
  x: string;
  w: string;
  b: string;
  y: string;
}

function LinearCalc({
  x,
  w,
  b,
  y,
  names,
  cell,
  extra,
}: {
  x: Matrix;
  w: Matrix;
  b: Vector;
  y: Matrix;
  names: LinearNames;
  cell: Cell;
  extra?: ReactNode;
}) {
  const { i, j, k } = cell;
  const e = dotExpansion(
    x[i],
    w.map((row) => row[j]),
    { emphasize: k },
  );
  const K = x[i].length;
  return (
    <CalcPanel target={<Tex>{String.raw`${names.y}_{${i},${j}}`}</Tex>}>
      <CalcLine
        label="definition"
        tex={String.raw`${names.y}_{${i},${j}} = \sum_{k=0}^{${K - 1}} ${colored(ROW_COLOR, `${names.x}_{${i},k}`)}\,${colored(COL_COLOR, `(${names.w})_{k,${j}}`)} + (${names.b})_{${j}}`}
      />
      <CalcLine label="substitute" tex={String.raw`= ${e.factors} + ${ta(b[j])}`} />
      <CalcLine label="multiply" tex={String.raw`= ${e.products} ${tsigned(b[j])}`} />
      <CalcLine label="sum" tex={String.raw`= ${boxed(y[i][j])}`} />
      {extra}
    </CalcPanel>
  );
}

function LinearViz({
  x,
  w,
  b,
  y,
  names,
  cell,
  setCell,
  xSize = "normal",
  size = "normal",
  xColGroups,
}: {
  x: Matrix;
  w: Matrix;
  b: Vector;
  y: Matrix;
  names: LinearNames;
  cell: Cell;
  setCell: (f: (c: Cell) => Cell) => void;
  xSize?: CellSize;
  size?: CellSize;
  xColGroups?: ColGroup[];
}) {
  const { trace } = useStep();
  const T = trace.ids.length;
  const { i, j, k } = cell;
  return (
    <MatMulView
      corner={
        <div className="mm-hint">
          <Tex>{String.raw`${names.y} = ${names.x}\,${names.w} + ${names.b}`}</Tex>
          <div>
            <span style={{ color: ROW_COLOR }}>row i</span> · <span style={{ color: COL_COLOR }}>column j</span>,
            plus bias j
          </div>
        </div>
      }
      b={
        <MatrixView
          data={w}
          label={names.w}
          rowLabels={indexLabels(w.length)}
          size={size}
          highlight={{ cols: [j], focus: k !== undefined ? [k, j] : null }}
          onHover={(k, j) => setCell((c) => ({ i: c.i, j, k }))}
        />
      }
      bias={
        <MatrixView
          data={[b]}
          label={names.b}
          colLabels="none"
          size={size}
          highlight={{ cols: [j] }}
          onHover={(_, j) => setCell((c) => ({ i: c.i, j }))}
        />
      }
      a={
        <MatrixView
          data={x}
          label={names.x}
          rowLabels={tokenRowLabels(trace.tokens)}
          markRow={T - 1}
          size={xSize}
          colGroups={xColGroups}
          highlight={{ rows: [i], focus: k !== undefined ? [i, k] : null }}
          onHover={(i, k) => setCell((c) => ({ i, j: c.j, k }))}
        />
      }
      c={
        <MatrixView
          data={y}
          label={names.y}
          size={size}
          highlight={{ focus: [i, j] }}
          onHover={(i, j) => setCell(() => ({ i, j }))}
        />
      }
    />
  );
}

// ---------------------------------------------------------------------------
// LayerNorms
// ---------------------------------------------------------------------------

export function Ln1Step() {
  const { block, layer } = useBlock();
  return (
    <LayerNormStep
      ln={block.ln1}
      names={{ input: "X", output: "Y", gamma: String.raw`\gamma_1`, beta: String.raw`\beta_1` }}
      intro={
        layer === 0 ? (
          <p>
            Block 1 begins. Every block has two sub-layers, multi-head attention and then an MLP, and each sub-layer
            starts by normalising its input. The input <Tex>X</Tex> here is the residual stream{" "}
            <Tex>{String.raw`X^{(0)}`}</Tex> from the embeddings.
          </p>
        ) : (
          <p>
            Block {layer + 1} has exactly the same structure as block {layer} but its own weights. Its input{" "}
            <Tex>{String.raw`X = X^{(${layer})}`}</Tex> is the output of block {layer}: the embeddings plus everything
            the earlier blocks added to them.
          </p>
        )
      }
    />
  );
}

export function Ln2Step() {
  const { block } = useBlock();
  return (
    <LayerNormStep
      ln={block.ln2}
      names={{ input: "R", output: "N", gamma: String.raw`\gamma_2`, beta: String.raw`\beta_2` }}
      intro={
        <p>
          Second sub-layer: the MLP. Like attention, it reads a normalised copy of the residual stream{" "}
          <Tex>R</Tex>, using its own LayerNorm parameters <Tex>{String.raw`\gamma_2, \beta_2`}</Tex>.
        </p>
      }
    />
  );
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

export function QkvStep() {
  const { block, params, T, layer } = useBlock();
  const [which, setWhich] = useState<"q" | "k" | "v">("q");
  const [cell, setCell] = useCell({ i: T - 1, j: 0 });
  const U = which.toUpperCase();
  const w = { q: params.attn.wq, k: params.attn.wk, v: params.attn.wv }[which];
  const b = { q: params.attn.bq, k: params.attn.bk, v: params.attn.bv }[which];
  const names = { x: "Y", w: `W_${U}`, b: `b_${U}`, y: U };
  const allZero = b.every((v) => v === 0);

  return (
    <StepLayout
      explain={
        <>
          <p>
            Attention starts by projecting each token's normalised vector <Tex>{String.raw`Y_{i,:}`}</Tex> three
            ways, each with its own learned {w.length}×{w.length} matrix and bias:
          </p>
          <ul>
            <li>
              <strong>query</strong> <Tex>{String.raw`Q_{i,:}`}</Tex>: what token i is looking for;
            </li>
            <li>
              <strong>key</strong> <Tex>{String.raw`K_{j,:}`}</Tex>: what token j offers, to be matched against
              queries;
            </li>
            <li>
              <strong>value</strong> <Tex>{String.raw`V_{j,:}`}</Tex>: the information token j passes on when it is
              attended to.
            </li>
          </ul>
          <p>The same weights are applied at every position: one matrix product handles all T rows at once.</p>
        </>
      }
      formula={
        <>
          <Tex display>{String.raw`Q = Y W_Q + b_Q`}</Tex>
          <Tex display>{String.raw`K = Y W_K + b_K`}</Tex>
          <Tex display>{String.raw`V = Y W_V + b_V`}</Tex>
          <div className="formula-caption">
            <Tex>{String.raw`W_{Q,K,V} \in \mathbb{R}^{${w.length}\times${w.length}}`}</Tex>; bias added to every row
          </div>
        </>
      }
      deeper={
        <>
          <p>
            <strong>KV cache.</strong> Thanks to the causal mask (two steps ahead), nothing at position j ever
            depends on tokens after j. So when a new token is appended, the K and V rows of all earlier positions are
            unchanged. Real inference engines cache them and compute only the new token's row of Q, K and V. We
            recompute every row here for clarity; the numbers are identical.
          </p>
          <p>
            <strong>
              Why is <Tex>b_K</Tex> exactly zero{layer === 0 ? "" : " again"}?
            </strong>{" "}
            Adding the same vector <Tex>b_K</Tex> to every key adds the constant{" "}
            <Tex>{String.raw`Q_{i,:}\cdot b_K`}</Tex> to every score in row i, and softmax ignores constants added to
            a whole row. So <Tex>b_K</Tex> has no effect on the output, receives zero gradient, and stays at its
            initial value of 0.
          </p>
        </>
      }
      calc={
        <LinearCalc
          x={block.ln1.output}
          w={w}
          b={b}
          y={block[which]}
          names={names}
          cell={cell}
          extra={
            which === "k" && allZero ? (
              <CalcNote>
                <Tex>b_K = 0</Tex> exactly: see “Go deeper” above.
              </CalcNote>
            ) : undefined
          }
        />
      }
    >
      <Tabs
        value={which}
        onChange={setWhich}
        options={[
          { value: "q", label: "Queries Q" },
          { value: "k", label: "Keys K" },
          { value: "v", label: "Values V" },
        ]}
      />
      <LinearViz x={block.ln1.output} w={w} b={b} y={block[which]} names={names} cell={cell} setCell={setCell} />
    </StepLayout>
  );
}

export function SplitHeadsStep() {
  const { block, cfg, T, trace } = useBlock();
  const [sel, setSel] = useState({ m: "q" as "q" | "k" | "v", h: 0, i: T - 1, c: 0 });
  const dh = cfg.dHead;
  const groups = headColGroups(cfg.nHeads, dh);
  const col = sel.h * dh + sel.c;
  const M = sel.m.toUpperCase();

  return (
    <StepLayout
      explain={
        <>
          <p>
            Multi-head attention runs H = {cfg.nHeads} independent attention computations in parallel. Head h uses
            its own slice of <Tex>{`d_h = d/H = ${cfg.dModel}/${cfg.nHeads} = ${dh}`}</Tex> columns of Q, K and V.
          </p>
          <p>
            No arithmetic happens here, only a reshape from T×{cfg.dModel} to H×T×{dh}. Because each head computes
            its own attention pattern, one head can, say, track the subject of the sentence while another looks at
            the previous word, at the same total cost as a single {cfg.dModel}-wide head.
          </p>
        </>
      }
      formula={
        <>
          <Tex display>{String.raw`Q^{(h)} = Q_{:,\;(h-1)d_h\,:\,h\,d_h}`}</Tex>
          <div className="formula-caption">and the same for K and V, h = 1, …, {cfg.nHeads}</div>
          <Tex display>{String.raw`\texttt{q.view(T, H, d\_h).transpose(0, 1)}`}</Tex>
        </>
      }
      calc={
        <CalcPanel target={<Tex>{String.raw`${M}^{(${sel.h + 1})}_{${sel.i},${sel.c}}`}</Tex>}>
          <CalcLine
            label="slice"
            tex={String.raw`${M}^{(${sel.h + 1})}_{${sel.i},${sel.c}} = ${M}_{${sel.i},\,(${sel.h + 1}-1)\cdot ${dh} + ${sel.c}} = ${M}_{${sel.i},${col}} = ${boxed(block[sel.m][sel.i][col])}`}
          />
        </CalcPanel>
      }
    >
      <div className="split-grid">
        {(["q", "k", "v"] as const).map((m) => (
          <EqRow key={m}>
            <MatrixView
              data={block[m]}
              label={m.toUpperCase()}
              rowLabels={tokenRowLabels(trace.tokens)}
              markRow={T - 1}
              colGroups={groups}
              highlight={{ focus: sel.m === m ? [sel.i, col] : null }}
              onHover={(i, j) => setSel({ m, h: Math.floor(j / dh), i, c: j % dh })}
            />
            <Op>
              <span className="op-text">split →</span>
            </Op>
            {block.heads.map((hd, h) => (
              <MatrixView
                key={h}
                data={hd[m]}
                label={`${m.toUpperCase()}^{(${h + 1})}`}
                colGroups={[{ start: 0, end: dh, label: `head ${h + 1}`, color: HEAD_COLORS[h] }]}
                highlight={{ focus: sel.m === m && sel.h === h ? [sel.i, sel.c] : null }}
                onHover={(i, c) => setSel({ m, h, i, c })}
              />
            ))}
          </EqRow>
        ))}
      </div>
    </StepLayout>
  );
}

function defaultAttendedKey(weights: Matrix): number {
  const row = weights[weights.length - 1];
  return row.indexOf(Math.max(...row));
}

export function ScoresStep() {
  const { block, cfg, head, T, trace } = useBlock();
  const hd = block.heads[head];
  const [cell, setCell] = useCell({ i: T - 1, j: defaultAttendedKey(hd.weights) });
  const { i, j, k } = cell;
  const kT = transpose(hd.k);
  const e = dotExpansion(hd.q[i], hd.k[j], { emphasize: k });
  const h = head + 1;
  const scale = Math.sqrt(cfg.dHead);

  return (
    <StepLayout
      explain={
        <>
          <p>
            Each head compares every query with every key using a dot product. <Tex>{String.raw`S_{i,j}`}</Tex> is
            large when query i and key j point in similar directions, i.e. when token j has what token i is looking
            for. All T×T pairs come out of a single matrix product{" "}
            <Tex>{String.raw`Q^{(h)}K^{(h)\top}`}</Tex>, which is then divided by{" "}
            <Tex>{String.raw`\sqrt{d_h} = \sqrt{${cfg.dHead}} = ${scale}`}</Tex>.
          </p>
          <p>
            Row <Tex>{String.raw`i = ${T - 1}`}</Tex> (the last token, <Tok>{trace.tokens[T - 1]}</Tok>) is the one
            that matters for predicting the next token.
          </p>
        </>
      }
      formula={
        <>
          <Tex display>{String.raw`S^{(h)} = \frac{Q^{(h)}K^{(h)\top}}{\sqrt{d_h}}`}</Tex>
          <Tex display>{String.raw`S^{(h)}_{i,j} = \frac{1}{\sqrt{d_h}}\sum_{c=0}^{d_h-1} Q^{(h)}_{i,c}\,K^{(h)}_{j,c}`}</Tex>
        </>
      }
      deeper={
        <p>
          <strong>
            Why divide by <Tex>{String.raw`\sqrt{d_h}`}</Tex>?
          </strong>{" "}
          If the components of q and k were independent with mean 0 and variance 1, their dot product would be a sum
          of <Tex>d_h</Tex> such products and have variance <Tex>d_h</Tex>. Dividing by{" "}
          <Tex>{String.raw`\sqrt{d_h}`}</Tex> brings the variance back to 1, so the softmax does not saturate (putting
          nearly all weight on one token, with vanishing gradients) as heads get wider. GPT-2 has{" "}
          <Tex>d_h = 64</Tex>, a factor of 8.
        </p>
      }
      calc={
        <CalcPanel
          target={
            <>
              head {h}: <Tex>{String.raw`S^{(${h})}_{${i},${j}}`}</Tex> (query <Tok>{trace.tokens[i]}</Tok>, key{" "}
              <Tok>{trace.tokens[j]}</Tok>)
            </>
          }
        >
          <CalcLine
            label="definition"
            tex={String.raw`S^{(${h})}_{${i},${j}} = \tfrac{1}{\sqrt{${cfg.dHead}}}\sum_{c=0}^{${cfg.dHead - 1}} ${colored(ROW_COLOR, `Q^{(${h})}_{${i},c}`)}\,${colored(COL_COLOR, `K^{(${h})}_{${j},c}`)}`}
          />
          <CalcLine label="substitute" tex={String.raw`= \tfrac{1}{${scale}}\big(${e.factors}\big)`} />
          <CalcLine label="multiply" tex={String.raw`= \tfrac{1}{${scale}}\big(${e.products}\big)`} />
          <CalcLine label="scale" tex={String.raw`= \tfrac{${tn(hd.dots[i][j])}}{${scale}} = ${boxed(hd.scores[i][j])}`} />
          {j > i && (
            <CalcNote>
              Key position {j} comes after query position {i}. This score is computed, but the causal mask in the
              next step will remove it.
            </CalcNote>
          )}
        </CalcPanel>
      }
    >
      <HeadTabs />
      <MatMulView
        corner={
          <div className="mm-hint">
            <Tex>{String.raw`Q^{(${h})}K^{(${h})\top}`}</Tex>
            <div>
              <span style={{ color: ROW_COLOR }}>query row i</span> ·{" "}
              <span style={{ color: COL_COLOR }}>key column j</span>
            </div>
          </div>
        }
        b={
          <MatrixView
            data={kT}
            label={`K^{(${h})\\top}`}
            rowLabels={indexLabels(cfg.dHead)}
            colLabels={tokenColLabels(trace.tokens)}
            highlight={{ cols: [j], focus: k !== undefined ? [k, j] : null }}
            onHover={(k, j) => setCell((c) => ({ i: c.i, j, k }))}
          />
        }
        a={
          <MatrixView
            data={hd.q}
            label={`Q^{(${h})}`}
            rowLabels={tokenRowLabels(trace.tokens)}
            markRow={T - 1}
            highlight={{ rows: [i], focus: k !== undefined ? [i, k] : null }}
            onHover={(i, k) => setCell((c) => ({ i, j: c.j, k }))}
          />
        }
        c={
          <MatrixView
            data={hd.dots}
            label={`Q^{(${h})}K^{(${h})\\top}`}
            colLabels={tokenColLabels(trace.tokens)}
            highlight={{ focus: [i, j] }}
            dimCell={(i, j) => j > i}
            onHover={(i, j) => setCell(() => ({ i, j }))}
          />
        }
        after={
          <EqRow>
            <Op tex={String.raw`\div ${scale} =`} />
            <MatrixView
              data={hd.scores}
              label={`S^{(${h})}`}
              colLabels={tokenColLabels(trace.tokens)}
              highlight={{ focus: [i, j] }}
              dimCell={(i, j) => j > i}
              onHover={(i, j) => setCell(() => ({ i, j }))}
            />
          </EqRow>
        }
      />
    </StepLayout>
  );
}

export function MaskStep() {
  const { block, head, T, trace } = useBlock();
  const hd = block.heads[head];
  const [cell, setCell] = useCell({ i: T - 2 >= 0 ? T - 2 : 0, j: T - 1 });
  const { i, j } = cell;
  const mask = trace.ids.map((_, r) => trace.ids.map((_, c) => (c <= r ? 0 : -Infinity)));
  const h = head + 1;
  const hl = { focus: [i, j] as [number, number] };
  const hover = (i: number, j: number) => setCell({ i, j });
  const cols = tokenColLabels(trace.tokens);

  return (
    <StepLayout
      explain={
        <>
          <p>
            A language model predicts each token from the ones <em>before</em> it, so position i may only attend to
            positions <Tex>{String.raw`j \le i`}</Tex>. Adding <Tex>{String.raw`-\infty`}</Tex> above the diagonal
            guarantees those entries get exactly zero weight after the softmax, because{" "}
            <Tex>{String.raw`e^{-\infty} = 0`}</Tex>.
          </p>
          <p>
            For the token we are generating from (the last row) nothing is masked: every other token is in its past.
            The mask matters for the other rows, and it is essential in training, where one forward pass predicts
            the next token at every position at once. Without it, position i could simply copy the answer from
            position i + 1.
          </p>
        </>
      }
      formula={
        <>
          <Tex display>{String.raw`\tilde S^{(h)} = S^{(h)} + M`}</Tex>
          <Tex display>{String.raw`M_{i,j} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}`}</Tex>
        </>
      }
      calc={
        <CalcPanel
          target={
            <>
              head {h}: <Tex>{String.raw`\tilde S^{(${h})}_{${i},${j}}`}</Tex>
            </>
          }
        >
          <CalcLine
            label="add mask"
            tex={String.raw`\tilde S^{(${h})}_{${i},${j}} = S^{(${h})}_{${i},${j}} + M_{${i},${j}} = ${tn(hd.scores[i][j])} + ${j <= i ? "0" : String.raw`(-\infty)`} = ${j <= i ? boxed(hd.masked[i][j]) : String.raw`\boxed{-\infty}`}`}
          />
          <CalcNote>
            {j <= i ? (
              <>
                <Tex>{String.raw`j = ${j} \le i = ${i}`}</Tex>: <Tok>{trace.tokens[i]}</Tok> may look at{" "}
                <Tok>{trace.tokens[j]}</Tok> (position {j} is {j === i ? "itself" : "in its past"}), so the score
                passes through unchanged.
              </>
            ) : (
              <>
                <Tex>{String.raw`j = ${j} > i = ${i}`}</Tex>: <Tok>{trace.tokens[j]}</Tok> is in the future of{" "}
                <Tok>{trace.tokens[i]}</Tok>, so it is blocked.
              </>
            )}
          </CalcNote>
        </CalcPanel>
      }
    >
      <HeadTabs />
      <EqRow>
        <MatrixView data={hd.scores} label={`S^{(${h})}`} rowLabels={tokenRowLabels(trace.tokens)} markRow={T - 1} colLabels={cols} highlight={hl} onHover={hover} />
        <Op tex="+" />
        <MatrixView data={mask} label="M" colLabels={cols} dp={0} maxAbs={1} highlight={hl} onHover={hover} />
        <Op tex="=" />
        <MatrixView data={hd.masked} label={`\\tilde S^{(${h})}`} colLabels={cols} highlight={hl} onHover={hover} />
      </EqRow>
    </StepLayout>
  );
}

export function SoftmaxStep() {
  const { block, head, T, trace } = useBlock();
  const hd = block.heads[head];
  const [cell, setCell] = useCell({ i: T - 1, j: defaultAttendedKey(hd.weights) });
  const { i, j } = cell;
  const h = head + 1;
  const hl = { rows: [i], focus: [i, j] as [number, number] };
  const hover = (i: number, j: number) => setCell({ i, j });
  const cols = tokenColLabels(trace.tokens);
  const row = hd.masked[i];
  const shifted = row.map((s) => s - hd.rowMax[i]);
  const expTerm = (a: number, jj: number) => {
    const t = a === -Infinity ? String.raw`e^{-\infty}` : `e^{${tn(a)}}`;
    return jj === j ? emphasized(t) : t;
  };
  const expVal = (v: number, jj: number) => (jj === j ? emphasized(tn(v)) : tn(v));
  const top = hd.weights[T - 1].indexOf(Math.max(...hd.weights[T - 1]));

  return (
    <StepLayout
      explain={
        <>
          <p>
            Softmax turns each row of scores into weights that are positive and sum to 1.{" "}
            <Tex>{String.raw`A_{i,j}`}</Tex> is the fraction of its attention that token i pays to token j. Larger
            scores win exponentially, and masked entries get exactly 0.
          </p>
          <p>
            In head {h}, the last token <Tok>{trace.tokens[T - 1]}</Tok> puts the most weight (
            {(hd.weights[T - 1][top] * 100).toFixed(1)}%) on <Tok>{trace.tokens[top]}</Tok> at position {top}. The
            diagram on the right shows where the selected query token looks; hover over a query token to switch
            rows.
          </p>
        </>
      }
      formula={
        <>
          <Tex display>{String.raw`A^{(h)}_{i,j} = \frac{e^{\tilde S_{i,j} - m_i}}{\sum_{j'} e^{\tilde S_{i,j'} - m_i}}`}</Tex>
          <Tex display>{String.raw`m_i = \max_{j} \tilde S_{i,j}`}</Tex>
        </>
      }
      deeper={
        <p>
          Subtracting the row maximum <Tex>m_i</Tex> leaves the result unchanged: numerator and denominator are both
          multiplied by <Tex>{String.raw`e^{-m_i}`}</Tex>. It is done purely for numerical safety, because{" "}
          <Tex>{String.raw`e^{x}`}</Tex> overflows for large x, while after the shift every exponent is ≤ 0. This is
          also why the key bias <Tex>b_K</Tex> is useless: it shifts a whole row by a constant.
        </p>
      }
      calc={
        <CalcPanel
          target={
            <>
              head {h}: <Tex>{String.raw`A^{(${h})}_{${i},${j}}`}</Tex> (how much <Tok>{trace.tokens[i]}</Tok>{" "}
              attends to <Tok>{trace.tokens[j]}</Tok>)
            </>
          }
        >
          <CalcLine label="row max" tex={String.raw`m_{${i}} = \max(${row.map((s) => tn(s)).join(",\\ ")}) = ${tn(hd.rowMax[i])}`} />
          <CalcLine
            label="denominator"
            tex={String.raw`\textstyle\sum_{j'} e^{\tilde S_{${i},j'} - m_{${i}}} = ${shifted.map(expTerm).join(" + ")} = ${hd.exps[i].map(expVal).join(" + ")} = ${tn(hd.rowSum[i])}`}
          />
          <CalcLine
            label="weight"
            tex={String.raw`A^{(${h})}_{${i},${j}} = \dfrac{${expTerm(shifted[j], -1)}}{${tn(hd.rowSum[i])}} = \dfrac{${tn(hd.exps[i][j])}}{${tn(hd.rowSum[i])}} = ${boxed(hd.weights[i][j])}`}
          />
          <CalcLine
            label="check"
            tex={String.raw`\textstyle\sum_j A^{(${h})}_{${i},j} = ${hd.weights[i].map((a) => tn(a)).join(" + ")} = ${tn(hd.weights[i].reduce((s, a) => s + a, 0))}`}
          />
        </CalcPanel>
      }
    >
      <HeadTabs />
      <div className="softmax-layout">
        <EqRow>
          <MatrixView data={hd.masked} label={`\\tilde S^{(${h})}`} rowLabels={tokenRowLabels(trace.tokens)} markRow={T - 1} colLabels={cols} highlight={hl} onHover={hover} />
          <Op>
            <span className="op-text">softmax each row →</span>
          </Op>
          <MatrixView data={hd.weights} label={`A^{(${h})}`} colLabels={cols} scale="sequential" maxAbs={1} dp={2} highlight={hl} onHover={hover} />
        </EqRow>
        <div className="arcs-box">
          <div className="arcs-title">Where query {i} looks (head {h})</div>
          <AttentionArcs
            tokens={trace.tokens}
            weights={hd.weights}
            row={i}
            onRow={(r) => setCell((c) => ({ i: r, j: Math.min(c.j, r) }))}
            color={HEAD_COLORS[head]}
          />
        </div>
      </div>
    </StepLayout>
  );
}

export function WeightedSumStep() {
  const { block, head, T, trace, cfg } = useBlock();
  const hd = block.heads[head];
  const [cell, setCell] = useCell({ i: T - 1, j: 0 });
  const { i, j, k } = cell;
  const h = head + 1;
  const e = dotExpansion(
    hd.weights[i],
    hd.v.map((r) => r[j]),
    { emphasize: k },
  );

  return (
    <StepLayout
      explain={
        <>
          <p>
            Each token's output from this head is the weighted average of the value vectors, using its row of
            attention weights. The last token's output row is mostly the value vector of whichever tokens it attends
            to.
          </p>
          <p>
            <strong>This is the only place in the entire network where information moves between positions.</strong>{" "}
            Everything else (LayerNorm, the Q/K/V and output projections, the MLP) processes each position on its
            own.
          </p>
        </>
      }
      formula={
        <>
          <Tex display>{String.raw`Z^{(h)} = A^{(h)} V^{(h)}`}</Tex>
          <Tex display>{String.raw`Z^{(h)}_{i,c} = \sum_{j=0}^{T-1} A^{(h)}_{i,j}\,V^{(h)}_{j,c}`}</Tex>
          <div className="formula-caption">
            <Tex>{String.raw`Z^{(h)} \in \mathbb{R}^{T\times ${cfg.dHead}}`}</Tex>
          </div>
        </>
      }
      calc={
        <CalcPanel
          target={
            <>
              head {h}: <Tex>{String.raw`Z^{(${h})}_{${i},${j}}`}</Tex>
            </>
          }
        >
          <CalcLine
            label="definition"
            tex={String.raw`Z^{(${h})}_{${i},${j}} = \sum_{j'=0}^{${T - 1}} ${colored(ROW_COLOR, `A^{(${h})}_{${i},j'}`)}\,${colored(COL_COLOR, `V^{(${h})}_{j',${j}}`)}`}
          />
          <CalcLine label="substitute" tex={String.raw`= ${e.factors}`} />
          <CalcLine label="multiply" tex={String.raw`= ${e.products}`} />
          <CalcLine label="sum" tex={String.raw`= ${boxed(hd.output[i][j])}`} />
          {i < T - 1 && (
            <CalcNote>
              Terms for <Tex>{String.raw`j' > ${i}`}</Tex> are zero because of the causal mask.
            </CalcNote>
          )}
        </CalcPanel>
      }
    >
      <HeadTabs />
      <MatMulView
        corner={
          <div className="mm-hint">
            <Tex>{String.raw`A^{(${h})}V^{(${h})}`}</Tex>
            <div>
              <span style={{ color: ROW_COLOR }}>weights row i</span> ·{" "}
              <span style={{ color: COL_COLOR }}>value column c</span>
            </div>
          </div>
        }
        b={
          <MatrixView
            data={hd.v}
            label={`V^{(${h})}`}
            rowLabels={tokenRowLabels(trace.tokens)}
            highlight={{ cols: [j], focus: k !== undefined ? [k, j] : null }}
            onHover={(k, j) => setCell((c) => ({ i: c.i, j, k }))}
          />
        }
        a={
          <MatrixView
            data={hd.weights}
            label={`A^{(${h})}`}
            rowLabels={tokenRowLabels(trace.tokens)}
            markRow={T - 1}
            colLabels={tokenColLabels(trace.tokens)}
            scale="sequential"
            maxAbs={1}
            highlight={{ rows: [i], focus: k !== undefined ? [i, k] : null }}
            onHover={(i, k) => setCell((c) => ({ i, j: c.j, k }))}
          />
        }
        c={
          <MatrixView
            data={hd.output}
            label={`Z^{(${h})}`}
            highlight={{ focus: [i, j] }}
            onHover={(i, j) => setCell(() => ({ i, j }))}
          />
        }
      />
    </StepLayout>
  );
}

export function AttnProjStep() {
  const { block, params, T, cfg } = useBlock();
  const [cell, setCell] = useCell({ i: T - 1, j: 0 });
  const names = { x: "Z", w: "W_O", b: "b_O", y: "O" };
  const dh = cfg.dHead;
  const { i, j } = cell;
  const parts = block.heads.map((_, h) => {
    let s = 0;
    for (let c = h * dh; c < (h + 1) * dh; c++) s += block.concat[i][c] * params.attn.wo[c][j];
    return s;
  });

  return (
    <StepLayout
      explain={
        <>
          <p>
            The head outputs are concatenated side by side back into a T×{cfg.dModel} matrix <Tex>Z</Tex> and
            multiplied by the output projection <Tex>W_O</Tex>. This mixes information across heads and maps the
            result back into the residual stream's coordinates.
          </p>
          <p>
            Splitting <Tex>W_O</Tex> by rows shows this is the same as every head projecting its own output and the
            results being summed. The calculation below shows the contribution of each head separately.
          </p>
        </>
      }
      formula={
        <>
          <Tex display>{String.raw`Z = \big[\,Z^{(1)}\;\; Z^{(2)}\,\big]`}</Tex>
          <Tex display>{String.raw`O = Z\,W_O + b_O`}</Tex>
          <Tex display>{String.raw`\phantom{O} = \sum_{h} Z^{(h)}\,W_O^{[h]} + b_O`}</Tex>
          <div className="formula-caption">
            <Tex>{String.raw`W_O^{[h]}`}</Tex> = the {dh} rows of <Tex>W_O</Tex> belonging to head h
          </div>
        </>
      }
      calc={
        <LinearCalc
          x={block.concat}
          w={params.attn.wo}
          b={params.attn.bo}
          y={block.attnOut}
          names={names}
          cell={cell}
          extra={
            <CalcLine
              label="by head"
              tex={String.raw`= ${parts.map((p, h) => String.raw`\underbrace{${tn(p)}}_{\text{head ${h + 1}}}`).join(" + ")} + \underbrace{${ta(params.attn.bo[j])}}_{\text{bias}}`}
            />
          }
        />
      }
    >
      <LinearViz
        x={block.concat}
        w={params.attn.wo}
        b={params.attn.bo}
        y={block.attnOut}
        names={names}
        cell={cell}
        setCell={setCell}
        xColGroups={headColGroups(cfg.nHeads, dh)}
      />
    </StepLayout>
  );
}

// ---------------------------------------------------------------------------
// Residual connections
// ---------------------------------------------------------------------------

function norm(v: number[]): number {
  return Math.sqrt(v.reduce((s, x) => s + x * x, 0));
}

function ResidualStep({
  a,
  b,
  c,
  names,
  explain,
}: {
  a: Matrix;
  b: Matrix;
  c: Matrix;
  names: [string, string, string];
  explain: ReactNode;
}) {
  const { trace } = useStep();
  const T = trace.ids.length;
  const [cell, setCell] = useCell({ i: T - 1, j: 0 });
  const { i, j } = cell;
  const hl = { focus: [i, j] as [number, number] };
  const hover = (i: number, j: number) => setCell({ i, j });
  const [A, B, C] = names;

  return (
    <StepLayout
      explain={explain}
      formula={<Tex display>{String.raw`${C} = ${A} + ${B}`}</Tex>}
      calc={
        <CalcPanel target={<Tex>{String.raw`${C}_{${i},${j}}`}</Tex>}>
          <CalcLine
            label="add"
            tex={String.raw`${C}_{${i},${j}} = ${A}_{${i},${j}} + ${B}_{${i},${j}} = ${tn(a[i][j])} + ${ta(b[i][j])} = ${boxed(c[i][j])}`}
          />
          <CalcLine
            label="row sizes"
            tex={String.raw`\lVert ${A}_{${i},:}\rVert = ${tn(norm(a[i]), 3)},\quad \lVert ${B}_{${i},:}\rVert = ${tn(norm(b[i]), 3)},\quad \lVert ${C}_{${i},:}\rVert = ${tn(norm(c[i]), 3)}`}
          />
        </CalcPanel>
      }
    >
      <EqRow>
        <MatrixView data={a} label={A} rowLabels={tokenRowLabels(trace.tokens)} markRow={T - 1} highlight={hl} onHover={hover} />
        <Op tex="+" />
        <MatrixView data={b} label={B} highlight={hl} onHover={hover} />
        <Op tex="=" />
        <MatrixView data={c} label={C} highlight={hl} onHover={hover} />
      </EqRow>
    </StepLayout>
  );
}

export function Resid1Step() {
  const { block } = useBlock();
  return (
    <ResidualStep
      a={block.input}
      b={block.attnOut}
      c={block.resid1}
      names={["X", "O", "R"]}
      explain={
        <>
          <p>
            The attention output is <em>added</em> to the block's input instead of replacing it. Think of the
            residual stream as shared memory: each sub-layer reads it (through a LayerNorm), computes an update, and
            writes the update back by addition.
          </p>
          <p>
            Information from the embeddings survives unless a layer actively changes it, and in training, gradients
            flow straight through the additions. That is what makes deep stacks of blocks trainable.
          </p>
        </>
      }
    />
  );
}

export function Resid2Step() {
  const { block, layer, cfg } = useBlock();
  const out = `X^{(${layer + 1})}`;
  const last = layer === cfg.nLayers - 1;
  return (
    <ResidualStep
      a={block.resid1}
      b={block.down}
      c={block.resid2}
      names={["R", "D", out]}
      explain={
        <p>
          The same pattern for the MLP: its output <Tex>D</Tex> is added to the stream. The result{" "}
          <Tex>{out}</Tex> is the output of block {layer + 1} and{" "}
          {last ? "goes to the final LayerNorm." : `becomes the input of block ${layer + 2}.`}
        </p>
      }
    />
  );
}

// ---------------------------------------------------------------------------
// MLP
// ---------------------------------------------------------------------------

export function MlpUpStep() {
  const { block, params, T, cfg } = useBlock();
  const [cell, setCell] = useCell({ i: T - 1, j: 0 });
  const names = { x: "N", w: "W_1", b: "b_1", y: "U" };
  return (
    <StepLayout
      explain={
        <>
          <p>
            The MLP (feed-forward network) is applied to each position independently, with the same weights
            everywhere. First it expands each {cfg.dModel}-dimensional vector to <Tex>d_ff</Tex> = {cfg.dFF} hidden
            units, 4× wider as in GPT-2.
          </p>
          <p>
            Hidden unit j computes the dot product of the token's vector with column j of <Tex>W_1</Tex>, plus a
            bias. You can think of each column as a learned pattern detector.
          </p>
        </>
      }
      formula={
        <>
          <Tex display>{String.raw`U = N\,W_1 + b_1`}</Tex>
          <div className="formula-caption">
            <Tex>{String.raw`W_1 \in \mathbb{R}^{${cfg.dModel}\times${cfg.dFF}},\; U \in \mathbb{R}^{T\times${cfg.dFF}}`}</Tex>
          </div>
        </>
      }
      deeper={
        <p>
          Interpretability research suggests MLP layers behave like key–value memories: columns of <Tex>W_1</Tex>{" "}
          detect patterns in the residual stream, and the matching rows of <Tex>W_2</Tex> write associated
          information back. About two-thirds of GPT-2's parameters are in its MLPs.
        </p>
      }
      calc={<LinearCalc x={block.ln2.output} w={params.mlp.w1} b={params.mlp.b1} y={block.up} names={names} cell={cell} />}
    >
      <LinearViz x={block.ln2.output} w={params.mlp.w1} b={params.mlp.b1} y={block.up} names={names} cell={cell} setCell={setCell} size="compact" />
    </StepLayout>
  );
}

export function GeluStep() {
  const { block, T, trace, cfg } = useBlock();
  const [cell, setCell] = useCell({ i: T - 1, j: 0 });
  const { i, j } = cell;
  const u = block.up[i][j];
  const inner = GELU_C * (u + 0.044715 * u ** 3);
  const t = Math.tanh(inner);
  const hl = { rows: [i], focus: [i, j] as [number, number] };
  const hover = (i: number, j: number) => setCell({ i, j });
  const active = block.act[i].filter((g) => g > 0.1).length;

  return (
    <StepLayout
      explain={
        <>
          <p>
            GELU is applied to each of the {cfg.dFF} hidden values separately. It is a smooth version of ReLU: large
            positive inputs pass through almost unchanged, large negative inputs are squashed to about 0, and there
            is a small negative dip (minimum ≈ −0.17 at u ≈ −0.75).
          </p>
          <p>
            This non-linearity is essential. Without it, <Tex>W_1</Tex> and <Tex>W_2</Tex> would multiply into a
            single linear map and the MLP could learn nothing a single matrix couldn't. For the selected token{" "}
            <Tok>{trace.tokens[i]}</Tok>, {active} of {cfg.dFF} hidden units are clearly active (&gt; 0.1).
          </p>
        </>
      }
      formula={
        <>
          <Tex display>{String.raw`G = \mathrm{GELU}(U)\quad\text{(elementwise)}`}</Tex>
          <Tex display>{String.raw`\mathrm{GELU}(u) = u\,\Phi(u)`}</Tex>
          <Tex display>{String.raw`\approx \tfrac{1}{2}u\Big(1 + \tanh\big(\sqrt{2/\pi}\,(u + 0.044715\,u^3)\big)\Big)`}</Tex>
          <div className="formula-caption">GPT-2 uses the tanh approximation, and so does this model.</div>
        </>
      }
      calc={
        <CalcPanel target={<Tex>{String.raw`G_{${i},${j}} = \mathrm{GELU}(U_{${i},${j}})`}</Tex>}>
          <CalcLine label="input" tex={String.raw`u = U_{${i},${j}} = ${tn(u)}`} />
          <CalcLine
            label="inner"
            tex={String.raw`\sqrt{2/\pi}\,(u + 0.044715\,u^3) = 0.7979 \times \big(${tn(u)} + 0.044715 \times ${tp(u)}^3\big) = 0.7979 \times ${tp(u + 0.044715 * u ** 3)} = ${tn(inner)}`}
          />
          <CalcLine label="tanh" tex={String.raw`\tanh(${tn(inner)}) = ${tn(t)}`} />
          <CalcLine
            label="result"
            tex={String.raw`G_{${i},${j}} = \tfrac{1}{2} \times ${tp(u)} \times (1 + ${ta(t)}) = ${boxed(gelu(u))}`}
          />
        </CalcPanel>
      }
    >
      <div className="gelu-layout">
        <div className="gelu-mats">
          <MatrixView data={block.up} label="U" rowLabels={tokenRowLabels(trace.tokens)} markRow={T - 1} size="compact" highlight={hl} onHover={hover} />
          <div className="arch-down">↓ GELU, elementwise</div>
          <MatrixView data={block.act} label="G" rowLabels={tokenRowLabels(trace.tokens)} markRow={T - 1} size="compact" highlight={hl} onHover={hover} />
        </div>
        <div className="gelu-side">
          <div className="arcs-title">
            GELU with the {cfg.dFF} values of row {i} (<Tok>{trace.tokens[i]}</Tok>)
          </div>
          <GeluPlot values={block.up[i]} focus={j} />
        </div>
      </div>
    </StepLayout>
  );
}

export function MlpDownStep() {
  const { block, params, T, cfg } = useBlock();
  const [cell, setCell] = useCell({ i: T - 1, j: 0 });
  const names = { x: "G", w: "W_2", b: "b_2", y: "D" };
  return (
    <StepLayout
      explain={
        <p>
          The {cfg.dFF} activations are projected back down to d = {cfg.dModel} so the result can be added to the
          residual stream. Each output feature is a weighted sum of all {cfg.dFF} hidden units. Hidden units that
          GELU switched off (≈ 0) contribute almost nothing.
        </p>
      }
      formula={
        <>
          <Tex display>{String.raw`D = G\,W_2 + b_2`}</Tex>
          <Tex display>{String.raw`\mathrm{MLP}(N) = \mathrm{GELU}(N W_1 + b_1)\,W_2 + b_2`}</Tex>
          <div className="formula-caption">
            <Tex>{String.raw`W_2 \in \mathbb{R}^{${cfg.dFF}\times${cfg.dModel}}`}</Tex>
          </div>
        </>
      }
      calc={<LinearCalc x={block.act} w={params.mlp.w2} b={params.mlp.b2} y={block.down} names={names} cell={cell} />}
    >
      <LinearViz
        x={block.act}
        w={params.mlp.w2}
        b={params.mlp.b2}
        y={block.down}
        names={names}
        cell={cell}
        setCell={setCell}
        xSize="compact"
        size="compact"
      />
    </StepLayout>
  );
}
