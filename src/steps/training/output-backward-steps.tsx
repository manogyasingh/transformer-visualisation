import { useState } from "react";
import { CalcLine, CalcNote, CalcPanel } from "../../components/calc-panel";
import { EqRow, MatMulView, Op, Tabs } from "../../components/layout";
import { MatrixView } from "../../components/matrix-view";
import { StepLayout } from "../../components/step-layout";
import { Tex } from "../../components/tex";
import { colored, COL_COLOR, dotExpansion, ROW_COLOR, tg, tga, tgSum, tn } from "../../lib/format";
import { transpose } from "../../model/linalg";
import { useTraining } from "../../training/training-context";
import { useStep } from "../step-context";
import { Tok } from "../shared";
import { boxedG, d, LayerNormBackwardView, useTokens } from "./backward-views";

function useVocabLabels() {
  const { cfg } = useStep();
  return {
    vocabCols: cfg.vocab.map((w) => <span key={w}>{w}</span>),
    vocabRows: cfg.vocab.map((w, v) => (
      <span className="rl" key={v}>
        <span className="rl-pos">{v}</span>
        <span className="rl-tok">{w}</span>
      </span>
    )),
  };
}

export function TDLogitsStep() {
  const { trace } = useTraining();
  const { cfg } = useStep();
  const { T, rowLabels, tokens } = useTokens();
  const { vocabCols } = useVocabLabels();
  const [cell, setCell] = useState({ i: T - 1, v: trace.targets[T - 1] });
  const { i, v } = cell;
  const isTarget = v === trace.targets[i];
  const targets = trace.targets.map((t, r) => [r, t] as [number, number]);
  const hover = (i: number, v: number) => setCell({ i, v });
  return (
    <StepLayout
      explain={
        <>
          <p>
            Backpropagation runs the chain rule from the loss back to every weight, visiting the operations in
            reverse order. It starts with one of the most useful results in deep learning: for softmax followed by
            cross-entropy, the gradient of the loss with respect to the logits is simply{" "}
            <strong>predicted probability minus actual</strong>.
          </p>
          <p>
            At each position the correct token's logit gets a negative gradient (<Tex>{"P - 1 < 0"}</Tex>), so the
            update will push it <em>up</em>. Every other logit gets a positive gradient equal to its probability,
            so it is pushed down, harder the more probability it wrongly took. The <Tex>{`1/${T}`}</Tex> comes from
            averaging over the positions.
          </p>
        </>
      }
      formula={
        <>
          <Tex display>{`${d("Z")} = \\frac{1}{${T}}\\big(P - Y\\big)`}</Tex>
          <Tex display>{"Y_{i,v} = [v = y_i]\\quad\\text{(one-hot targets)}"}</Tex>
        </>
      }
      deeper={
        <p>
          Derivation for one row: <Tex>{"\\ell = -z_y + \\log\\sum_u e^{z_u}"}</Tex>, so{" "}
          <Tex>{"\\partial\\ell/\\partial z_v = -[v=y] + e^{z_v}/\\sum_u e^{z_u} = P_v - [v=y]"}</Tex>. Each row of{" "}
          <Tex>{d("Z")}</Tex> sums to zero, because the probabilities sum to 1.
        </p>
      }
      calc={
        <CalcPanel
          target={
            <>
              <Tex>{`${d("Z")}_{${i},${v}}`}</Tex>: logit of <Tok>{cfg.vocab[v]}</Tok> after <Tok>{tokens.slice(0, i + 1).join(" ")}</Tok>
            </>
          }
        >
          <CalcLine
            label="formula"
            tex={`${d("Z")}_{${i},${v}} = \\tfrac{1}{${T}}\\big(P_{${i},${v}} - [${v} = y_{${i}}]\\big) = \\tfrac{1}{${T}}(${tn(trace.probs[i][v])} - ${isTarget ? 1 : 0}) = ${boxedG(trace.dLogits[i][v])}`}
          />
          <CalcNote>
            {isTarget
              ? `This is the correct token (y_${i} = ${cfg.vocab[v]}): the negative gradient means gradient descent will raise this logit.`
              : `The correct token here is “${cfg.vocab[trace.targets[i]]}”, so this logit will be lowered in proportion to its probability.`}
          </CalcNote>
        </CalcPanel>
      }
    >
      <div className="stack-viz">
        <MatrixView data={trace.probs} label="P" rowLabels={rowLabels} markRow={T - 1} colLabels={vocabCols} verticalColLabels size="medium" scale="sequential" maxAbs={1} highlight={{ cells: targets, focus: [i, v] }} onHover={hover} />
        <div className="arch-down">↓ subtract 1 at each target (outlined), divide by T = {T}</div>
        <MatrixView data={trace.dLogits} label={d("Z")} rowLabels={rowLabels} markRow={T - 1} size="medium" autoScale highlight={{ cells: targets, focus: [i, v] }} onHover={hover} />
      </div>
    </StepLayout>
  );
}

export function TUnembedStep() {
  const { trace, model } = useTraining();
  const { cfg } = useStep();
  const { T, rowLabels, colLabels } = useTokens();
  const { vocabCols, vocabRows } = useVocabLabels();
  const [tab, setTab] = useState<"dF" | "dW">("dF");
  const [fCell, setFCell] = useState({ i: T - 1, j: 0 });
  const [wCell, setWCell] = useState({ v: trace.targets[T - 1], j: 0 });
  const F = trace.forward.lnf.output;
  const wte = model.params.wte;

  const calc =
    tab === "dF" ? (
      <CalcPanel target={<Tex>{`${d("F")}_{${fCell.i},${fCell.j}}`}</Tex>}>
        <CalcLine
          label="chain rule"
          tex={`${d("F")}_{${fCell.i},${fCell.j}} = \\sum_{v=0}^{${cfg.vocab.length - 1}} ${colored(ROW_COLOR, `${d("Z")}_{${fCell.i},v}`)}\\,${colored(COL_COLOR, `(W_E)_{v,${fCell.j}}`)}`}
        />
        <CalcLine label="substitute" tex={`= ${dotExpansion(trace.dLogits[fCell.i], wte.map((r) => r[fCell.j]), { sig: true }).factors}`} />
        <CalcLine label="sum" tex={`= ${boxedG(trace.dF[fCell.i][fCell.j])}`} />
      </CalcPanel>
    ) : (
      <CalcPanel target={<Tex>{`${d("W_E")}^{\\text{out}}_{${wCell.v},${wCell.j}}`}</Tex>}>
        <CalcLine
          label="chain rule"
          tex={`${d("W_E")}^{\\text{out}}_{${wCell.v},${wCell.j}} = \\sum_{i=0}^{${T - 1}} ${colored(ROW_COLOR, `${d("Z")}_{i,${wCell.v}}`)}\\,${colored(COL_COLOR, `F_{i,${wCell.j}}`)}`}
        />
        <CalcLine label="substitute" tex={`= ${dotExpansion(trace.dLogits.map((r) => r[wCell.v]), F.map((r) => r[wCell.j]), { sig: true }).factors}`} />
        <CalcLine label="sum" tex={`= ${boxedG(trace.dWteOut[wCell.v][wCell.j])}`} />
      </CalcPanel>
    );

  return (
    <StepLayout
      explain={
        <>
          <p>
            Forward, the logits were <Tex>{"Z = F W_E^\\top"}</Tex>. Backward this gives two gradients: one for{" "}
            <Tex>F</Tex>, the final hidden vectors, which continues down the network, and one for the embedding matrix{" "}
            <Tex>W_E</Tex>.
          </p>
          <p>
            Because GPT-2 <strong>ties</strong> the input and output embeddings, <Tex>W_E</Tex> is used twice: here as
            the unembedding and at the very bottom as the token lookup table. This is its first gradient contribution,{" "}
            <Tex>{`${d("W_E")}^{\\text{out}}`}</Tex>. The second arrives at the end of the backward pass and the two are
            added.
          </p>
        </>
      }
      formula={
        <>
          <Tex display>{`${d("F")} = ${d("Z")}\\,W_E`}</Tex>
          <Tex display>{`${d("W_E")}^{\\text{out}} = ${d("Z")}^{\\top} F`}</Tex>
        </>
      }
      calc={calc}
    >
      <Tabs
        value={tab}
        onChange={setTab}
        options={[
          { value: "dF", label: <Tex>{`\\text{hidden-state gradient } ${d("F")}`}</Tex> },
          { value: "dW", label: <Tex>{`\\text{embedding gradient } ${d("W_E")}^{\\text{out}}`}</Tex> },
        ]}
      />
      {tab === "dF" ? (
        <MatMulView
          b={<MatrixView data={wte} label="W_E" rowLabels={vocabRows} highlight={{ cols: [fCell.j] }} onHover={(_, j) => setFCell((c) => ({ ...c, j }))} />}
          a={<MatrixView data={trace.dLogits} label={d("Z")} rowLabels={rowLabels} markRow={T - 1} colLabels={vocabCols} verticalColLabels size="compact" autoScale highlight={{ rows: [fCell.i] }} onHover={(i) => setFCell((c) => ({ ...c, i }))} />}
          c={<MatrixView data={trace.dF} label={d("F")} autoScale highlight={{ focus: [fCell.i, fCell.j] }} onHover={(i, j) => setFCell({ i, j })} />}
        />
      ) : (
        <MatMulView
          b={<MatrixView data={F} label="F" rowLabels={rowLabels} highlight={{ cols: [wCell.j] }} onHover={(_, j) => setWCell((c) => ({ ...c, j }))} />}
          a={<MatrixView data={transpose(trace.dLogits)} label={`${d("Z")}^\\top`} rowLabels={vocabRows} colLabels={colLabels} autoScale highlight={{ rows: [wCell.v] }} onHover={(v) => setWCell((c) => ({ ...c, v }))} />}
          c={<MatrixView data={trace.dWteOut} label={`${d("W_E")}^{\\text{out}}`} autoScale highlight={{ focus: [wCell.v, wCell.j] }} onHover={(v, j) => setWCell({ v, j })} />}
        />
      )}
    </StepLayout>
  );
}

export function TLnfStep() {
  const { trace } = useTraining();
  const { cfg } = useStep();
  return (
    <LayerNormBackwardView
      ln={trace.forward.lnf}
      grad={trace.lnf}
      names={{ x: "X", y: "F", gamma: "\\gamma_f", beta: "\\beta_f" }}
      intro={
        <p>
          Next in reverse order is the final LayerNorm, <Tex>{`F = \\mathrm{LN}_f(X)`}</Tex> with{" "}
          <Tex>{`X = X^{(${cfg.nLayers})}`}</Tex>, the output of the last block. Its input gradient is the gradient that
          enters block {cfg.nLayers}.
        </p>
      }
    />
  );
}

export function TEmbedStep() {
  const { trace, example } = useTraining();
  const { cfg } = useStep();
  const { T, rowLabels } = useTokens();
  const { vocabRows } = useVocabLabels();
  const [cell, setCell] = useState({ v: example.ids[T - 1], j: 0 });
  const { v, j } = cell;
  const positions = example.ids.map((id, i) => (id === v ? i : -1)).filter((i) => i >= 0);
  const used = new Set(example.ids);
  const hoverW = (v: number, j: number) => setCell({ v, j });
  const total = trace.grads.wte;
  return (
    <StepLayout
      explain={
        <>
          <p>
            The last operation to undo is the very first one: <Tex>{"X^{(0)} = E + P"}</Tex>, where row i of{" "}
            <Tex>E</Tex> is row <Tex>{"t_i"}</Tex> of <Tex>W_E</Tex> and row i of <Tex>P</Tex> is row i of{" "}
            <Tex>W_P</Tex>. A lookup has a simple gradient: each position's gradient is added into the table row it
            came from.
          </p>
          <p>
            Only tokens that appear in the batch get an input-side gradient. A token that appears twice gets the sum of
            both positions' gradients. Adding the output-side gradient from the unembedding step gives the full
            gradient of the tied matrix <Tex>W_E</Tex>.
          </p>
        </>
      }
      formula={
        <>
          <Tex display>{`${d("W_P")}_{i,:} = ${d("X^{(0)}")}_{i,:}\\quad (i < T)`}</Tex>
          <Tex display>{`${d("W_E")}^{\\text{in}}_{v,:} = \\sum_{i\\,:\\,t_i = v} ${d("X^{(0)}")}_{i,:}`}</Tex>
          <Tex display>{`${d("W_E")} = ${d("W_E")}^{\\text{out}} + ${d("W_E")}^{\\text{in}}`}</Tex>
        </>
      }
      calc={
        <CalcPanel
          target={
            <>
              <Tex>{`${d("W_E")}_{${v},${j}}`}</Tex>, the row of <Tok>{cfg.vocab[v]}</Tok>
            </>
          }
        >
          <CalcLine
            label="input side"
            tex={
              positions.length
                ? `${d("W_E")}^{\\text{in}}_{${v},${j}} = ${positions.map((i) => `${d("X^{(0)}")}_{${i},${j}}`).join(" + ")} = ${tgSum(positions.map((i) => trace.dX0[i][j]))} = ${tg(trace.dWteIn[v][j])}`
                : `${d("W_E")}^{\\text{in}}_{${v},${j}} = 0\\quad(\\texttt{${cfg.vocab[v]}}\\text{ is not in the input})`
            }
          />
          <CalcLine label="output side" tex={`${d("W_E")}^{\\text{out}}_{${v},${j}} = ${tg(trace.dWteOut[v][j])}\\quad\\text{(unembedding step)}`} />
          <CalcLine
            label="total"
            tex={`${d("W_E")}_{${v},${j}} = ${tg(trace.dWteOut[v][j])} + ${tga(trace.dWteIn[v][j])} = ${boxedG(total[v][j])}`}
          />
        </CalcPanel>
      }
    >
      <EqRow>
        <MatrixView data={trace.dX0} label={d("X^{(0)}")} rowLabels={rowLabels} markRow={T - 1} autoScale highlight={{ rows: positions }} />
        <Op>
          <span className="op-text">row i goes to row i of W_P →</span>
        </Op>
        <MatrixView data={trace.dWpe} label={d("W_P")} autoScale dimRow={(r) => r >= T} highlight={{ rows: positions }} />
      </EqRow>
      <h3 className="viz-title">
        Row i of <Tex>{d("X^{(0)}")}</Tex> is also added into row <Tex>t_i</Tex> of the embedding gradient, which is summed
        with the unembedding contribution
      </h3>
      <EqRow>
        <MatrixView data={trace.dWteOut} label={`${d("W_E")}^{\\text{out}}`} rowLabels={vocabRows} size="medium" autoScale highlight={{ focus: [v, j] }} onHover={hoverW} />
        <Op tex="+" />
        <MatrixView data={trace.dWteIn} label={`${d("W_E")}^{\\text{in}}`} size="medium" autoScale dimRow={(r) => !used.has(r)} highlight={{ focus: [v, j] }} onHover={hoverW} />
        <Op tex="=" />
        <MatrixView data={total} label={d("W_E")} size="medium" autoScale highlight={{ focus: [v, j] }} onHover={hoverW} />
      </EqRow>
    </StepLayout>
  );
}
