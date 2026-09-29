import { Fragment, useMemo, useState } from "react";
import { CalcLine, CalcNote, CalcPanel } from "../../components/calc-panel";
import { BarList } from "../../components/charts";
import { MatrixView } from "../../components/matrix-view";
import { StepLayout } from "../../components/step-layout";
import { Tex } from "../../components/tex";
import { boxed, colored, COL_COLOR, dotExpansion, ROW_COLOR, tn } from "../../lib/format";
import { corpusNextCounts } from "../../model/corpus";
import { useTraining } from "../../training/training-context";
import { INIT_STD } from "../../training/trainer";
import { useStep } from "../step-context";
import { indexLabels, Tok } from "../shared";
import { d, useTokens } from "./backward-views";
import { TrainerControls } from "./trainer-controls";

function TrainingDiagram() {
  const { steps, goTo, step: current } = useStep();
  const sections: { title: string; ids: string[] }[] = [];
  for (const s of steps) {
    const last = sections[sections.length - 1];
    if (last && last.title === s.section) last.ids.push(s.id);
    else sections.push({ title: s.section, ids: [s.id] });
  }
  return (
    <div className="arch">
      {sections.slice(1).map((sec, n) => (
        <Fragment key={sec.title}>
          {n > 0 && <div className="arch-down">↓</div>}
          <div className={sec.title.startsWith("Backward") ? "arch-block backward" : "arch-block"}>
            <div className="arch-block-title">{sec.title}</div>
            <div className="arch-flow">
              {sec.ids.map((id, k) => {
                const s = steps.find((x) => x.id === id)!;
                return (
                  <Fragment key={id}>
                    {k > 0 && <span className="arch-arrow">→</span>}
                    <button className={`arch-chip${current.id === id ? " current" : ""}`} onClick={() => goTo(id)}>
                      {s.navLabel}
                    </button>
                  </Fragment>
                );
              })}
            </div>
          </div>
        </Fragment>
      ))}
      <div className="arch-down">↺ repeat with the next batch, thousands of times</div>
    </div>
  );
}

export function TOverviewStep() {
  const { example, trainer } = useTraining();
  return (
    <StepLayout
      explain={
        <>
          <p>
            <strong>Pretraining</strong> is how the weights you saw in the inference walkthrough were found. The model
            starts with random weights and repeatedly performs one <em>training step</em>:
          </p>
          <ol>
            <li>take a batch of real text;</li>
            <li>run the forward pass, which predicts the next token at every position;</li>
            <li>measure how wrong those predictions were with the cross-entropy loss;</li>
            <li>
              <strong>backpropagate</strong>: apply the chain rule backwards through every layer to get the gradient
              of the loss with respect to every one of the model's weights;
            </li>
            <li>nudge every weight against its gradient (here with the Adam optimiser).</li>
          </ol>
          <p>
            This walkthrough performs one such step on the sentence <Tok>{example.text}</Tok> (change it at the top),
            with every number exact. It uses the weights of a model being trained live in your browser, currently at
            step {trainer.state.step}. Train it for a while (buttons below, or the final step) and come back to see a
            training step from later in training.
          </p>
          <p>
            Notation: <Tex>{"\\delta A"}</Tex> means <Tex>{"\\partial L / \\partial A"}</Tex>, the gradient of the
            loss with respect to <Tex>A</Tex>. It always has the same shape as <Tex>A</Tex>.
          </p>
        </>
      }
      formula={
        <>
          <Tex display>{"L(\\theta) = -\\frac{1}{N}\\sum_{i} \\log p_\\theta(t_{i+1} \\mid t_0,\\dots,t_i)"}</Tex>
          <Tex display>{"g = \\nabla_\\theta L(\\theta)\\quad\\text{(backpropagation)}"}</Tex>
          <Tex display>{"\\theta \\leftarrow \\theta - \\eta\\,\\mathrm{Adam}(g)"}</Tex>
          <div className="formula-caption">θ = all 1,952 weights; N = number of predicted tokens in the batch</div>
        </>
      }
    >
      <TrainerControls />
      <h3 className="viz-title">One training step (click any stage to jump to it)</h3>
      <TrainingDiagram />
    </StepLayout>
  );
}

export function TDataStep() {
  const { examples, exampleIndex, setExampleIndex, example, trainer } = useTraining();
  const { cfg } = useStep();
  const T = example.ids.length;
  const [pos, setPos] = useState(T - 1);
  const total = examples.reduce((s, e) => s + e.weight, 0);
  return (
    <StepLayout
      explain={
        <>
          <p>
            Pretraining needs no labels: the text supervises itself. Shifting a sentence by one token gives the inputs
            and the targets. A sentence of {example.tokens.length} tokens yields {T} prediction tasks, one per
            position, and thanks to the causal mask a single forward pass solves all of them at once. Position i sees
            only tokens 0…i and must predict token i + 1.
          </p>
          <p>
            In the live trainer each step uses a <strong>minibatch</strong> of {trainer.settings.batchSize} sentences
            drawn at random from the corpus (in proportion to their counts), and the loss is averaged over every
            target token in the batch. In this walkthrough the batch is the one sentence selected below.
          </p>
        </>
      }
      formula={
        <>
          <Tex display>{`x = (t_0, \\dots, t_{${T - 1}}),\\quad y_i = t_{i+1}`}</Tex>
          <Tex display>{`L = \\frac{1}{${T}}\\sum_{i=0}^{${T - 1}} -\\log p(y_i \\mid x_0,\\dots,x_i)`}</Tex>
        </>
      }
      calc={
        <CalcPanel target={`position ${pos}`}>
          <CalcLine
            label="task"
            tex={`\\text{given } (${example.tokens
              .slice(0, pos + 1)
              .map((t) => `\\texttt{${t}}`)
              .join(", ")}) \\;\\Rightarrow\\; \\text{predict } y_{${pos}} = \\texttt{${example.tokens[pos + 1]}} \\;(\\text{id } ${example.targets[pos]})`}
          />
        </CalcPanel>
      }
    >
      <div className="shift-viz">
        <div className="shift-row">
          <span className="shift-label">sentence</span>
          {example.tokens.map((t, i) => (
            <span key={i} className="token-chip">
              {t}
            </span>
          ))}
        </div>
        <div className="shift-row">
          <span className="shift-label">input x</span>
          {example.ids.map((id, i) => (
            <span key={i} className={`token-card small${i === pos ? " focus" : ""}${i < pos ? " related" : ""}`} onMouseEnter={() => setPos(i)}>
              <span className="token-card-pos">{i}</span>
              <span className="token-card-word">{cfg.vocab[id]}</span>
            </span>
          ))}
        </div>
        <div className="shift-row">
          <span className="shift-label">target y</span>
          {example.targets.map((id, i) => (
            <span key={i} className={`token-card small target${i === pos ? " focus" : ""}`} onMouseEnter={() => setPos(i)}>
              <span className="token-card-pos">↑ {i}</span>
              <span className="token-card-word">{cfg.vocab[id]}</span>
            </span>
          ))}
        </div>
      </div>
      <h3 className="viz-title">The corpus ({total} sentences; click one to use it)</h3>
      <table className="corpus-table clickable">
        <thead>
          <tr>
            <th>sentence</th>
            <th>copies</th>
            <th>sampling probability</th>
          </tr>
        </thead>
        <tbody>
          {examples.map((e, i) => (
            <tr key={e.text} className={i === exampleIndex ? "current" : ""} onClick={() => setExampleIndex(i)}>
              <td>
                <code>{e.text}</code>
              </td>
              <td>{e.weight}</td>
              <td>
                {e.weight}/{total} = {(e.weight / total).toFixed(3)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </StepLayout>
  );
}

export function TInitStep() {
  const { model, trace, trainer } = useTraining();
  const { cfg } = useStep();
  const { T, tokens } = useTokens();
  const V = cfg.vocab.length;
  const last = trace.probs[T - 1];
  const order = last.map((_, v) => v).sort((a, b) => last[b] - last[a]);
  const vocabLabels = cfg.vocab.map((w, v) => (
    <span className="rl" key={v}>
      <span className="rl-pos">{v}</span>
      <span className="rl-tok">{w}</span>
    </span>
  ));
  const atInit = trainer.state.step === 0 && trainer.origin === "init";
  return (
    <StepLayout
      explain={
        <>
          <p>
            Before training, every weight is random. Embeddings are drawn from{" "}
            <Tex>{`\\mathcal{N}(0, ${INIT_STD.embedding}^2)`}</Tex>, weight matrices from{" "}
            <Tex>{"\\mathcal{N}(0, 1/n_{\\text{in}})"}</Tex> (so a layer's outputs have roughly the same scale as its
            inputs), biases start at 0, and LayerNorm starts as the identity (<Tex>{"\\gamma = 1, \\beta = 0"}</Tex>).
          </p>
          <p>
            A random model knows nothing, so its predictions are close to uniform over the {V} words, and its loss is
            close to <Tex>{`\\ln ${V} = ${Math.log(V).toFixed(4)}`}</Tex>, the loss of guessing uniformly.{" "}
            {atInit
              ? "The trainer is at step 0, so these really are the starting weights."
              : `The trainer is at step ${trainer.state.step}, so these weights have already been trained; press "Re-initialise" to see fresh random weights.`}
          </p>
        </>
      }
      formula={
        <>
          <Tex display>{`W_E, W_P \\sim \\mathcal{N}(0, ${INIT_STD.embedding}^2)`}</Tex>
          <Tex display>{"W \\sim \\mathcal{N}(0, 1/n_{\\text{in}}),\\quad b = 0"}</Tex>
          <Tex display>{"\\gamma = \\mathbf{1},\\quad \\beta = \\mathbf{0}"}</Tex>
        </>
      }
      calc={
        <CalcPanel target="loss of this sentence at the current weights">
          <CalcLine label="uniform guess" tex={`-\\ln\\tfrac{1}{${V}} = \\ln ${V} = ${tn(Math.log(V))}`} />
          <CalcLine label="this model" tex={`L = ${boxed(trace.loss)}`} />
        </CalcPanel>
      }
    >
      <TrainerControls compact />
      <div className="init-layout">
        <MatrixView data={model.params.wte} label="W_E" rowLabels={vocabLabels} size="medium" />
        <MatrixView data={model.params.layers[0].attn.wq} label="W_Q^{(1)}" rowLabels={indexLabels(cfg.dModel)} size="medium" />
        <div>
          <div className="arcs-title">
            Predicted next token after <Tok>{tokens.join(" ")}</Tok>
          </div>
          <BarList
            items={order.slice(0, 10).map((v) => ({ key: v, label: cfg.vocab[v], value: last[v] }))}
            min={0}
            max={1}
            dp={3}
            color="#7c3aed"
            width={200}
          />
        </div>
      </div>
    </StepLayout>
  );
}

export function TForwardStep() {
  const { trace, model } = useTraining();
  const { cfg, goTo } = useStep();
  const { T, rowLabels, tokens } = useTokens();
  const z = trace.forward.logitsAll;
  const [cell, setCell] = useState({ i: T - 1, v: trace.targets[T - 1] });
  const { i, v } = cell;
  const F = trace.forward.lnf.output;
  const e = dotExpansion(F[i], model.params.wte[v]);
  const vocabLabels = cfg.vocab.map((w) => <span key={w}>{w}</span>);
  return (
    <StepLayout
      explain={
        <>
          <p>
            The forward pass is exactly the one in the inference walkthrough (embeddings, {cfg.nLayers} blocks, final
            LayerNorm, unembedding) run on the input tokens. The difference is what we keep: for generation only the
            last row of logits mattered; for training <strong>every row</strong> is a prediction to be scored.
          </p>
          <p>
            The input is the real text, not the model's own samples. This is called <em>teacher forcing</em>. The
            outlined cells are the logits of the correct next tokens, which training will push up.
          </p>
        </>
      }
      formula={
        <>
          <Tex display>{"Z = \\mathrm{LN}_f\\big(X^{(2)}\\big)\\,W_E^\\top = F\\,W_E^\\top"}</Tex>
          <Tex display>{`Z \\in \\mathbb{R}^{${T}\\times${cfg.vocab.length}}`}</Tex>
        </>
      }
      deeper={
        <p>
          Every intermediate matrix of this forward pass (Q, K, V, attention weights, MLP activations…) is needed again
          during backpropagation, so frameworks keep them in memory. That stored activation memory is a large part of
          why training needs far more memory than inference.{" "}
          <button className="link-btn" onClick={() => goTo("overview")}>
            See the forward pass step by step in the inference walkthrough
          </button>{" "}
          (it uses the fully trained weights).
        </p>
      }
      calc={
        <CalcPanel
          target={
            <>
              <Tex>{`Z_{${i},${v}}`}</Tex>: after <Tok>{tokens.slice(0, i + 1).join(" ")}</Tok>, the logit of{" "}
              <Tok>{cfg.vocab[v]}</Tok>
            </>
          }
        >
          <CalcLine
            label="dot product"
            tex={`Z_{${i},${v}} = \\sum_j ${colored(ROW_COLOR, `F_{${i},j}`)}\\,${colored(COL_COLOR, `(W_E)_{${v},j}`)} = ${e.factors} = ${boxed(z[i][v])}`}
          />
          {v === trace.targets[i] && <CalcNote>This is the correct next token at position {i}.</CalcNote>}
        </CalcPanel>
      }
    >
      <MatrixView
        data={z}
        label="Z"
        rowLabels={rowLabels}
        markRow={T - 1}
        colLabels={vocabLabels}
        verticalColLabels
        size="medium"
        highlight={{ cells: trace.targets.map((t, r) => [r, t] as [number, number]), focus: [i, v] }}
        onHover={(i, v) => setCell({ i, v })}
      />
    </StepLayout>
  );
}

export function TProbsStep() {
  const { trace } = useTraining();
  const { cfg } = useStep();
  const { T, rowLabels, tokens } = useTokens();
  const [cell, setCell] = useState({ i: T - 1, v: trace.targets[T - 1] });
  const { i, v } = cell;
  const z = trace.forward.logitsAll[i];
  const m = Math.max(...z);
  const S = z.reduce((s, x) => s + Math.exp(x - m), 0);
  const vocabLabels = cfg.vocab.map((w) => <span key={w}>{w}</span>);
  return (
    <StepLayout
      explain={
        <>
          <p>
            Softmax turns each row of logits into a probability distribution over the next token, exactly as at the
            end of inference but for every position. The outlined cells are the probabilities the model assigned to
            what actually came next: those are the only numbers the loss looks at.
          </p>
        </>
      }
      formula={<Tex display>{"P_{i,v} = \\frac{e^{Z_{i,v} - m_i}}{\\sum_u e^{Z_{i,u} - m_i}},\\quad m_i = \\max_u Z_{i,u}"}</Tex>}
      calc={
        <CalcPanel
          target={
            <>
              <Tex>{`P_{${i},${v}}`}</Tex>: probability of <Tok>{cfg.vocab[v]}</Tok> after <Tok>{tokens.slice(0, i + 1).join(" ")}</Tok>
            </>
          }
        >
          <CalcLine label="row max" tex={`m_{${i}} = ${tn(m)}`} />
          <CalcLine label="denominator" tex={`\\textstyle\\sum_u e^{Z_{${i},u} - m_{${i}}} = ${tn(S)}`} />
          <CalcLine label="probability" tex={`P_{${i},${v}} = \\dfrac{e^{${tn(z[v])} - ${tn(m)}}}{${tn(S)}} = ${boxed(trace.probs[i][v])}`} />
        </CalcPanel>
      }
    >
      <MatrixView
        data={trace.probs}
        label="P"
        rowLabels={rowLabels}
        markRow={T - 1}
        colLabels={vocabLabels}
        verticalColLabels
        size="medium"
        scale="sequential"
        maxAbs={1}
        highlight={{ cells: trace.targets.map((t, r) => [r, t] as [number, number]), focus: [i, v] }}
        onHover={(i, v) => setCell({ i, v })}
      />
    </StepLayout>
  );
}

export function TLossStep() {
  const { trace, example } = useTraining();
  const { cfg, model } = useStep();
  const { T, tokens } = useTokens();
  const [pos, setPos] = useState(T - 1);
  const best = useMemo(
    () =>
      example.targets.map((t, i) => {
        const { counts, total } = corpusNextCounts(model, example.ids.slice(0, i + 1));
        return total ? (counts.get(t) ?? 0) / total : NaN;
      }),
    [model, example],
  );
  const bestLoss = best.reduce((s, p) => s - Math.log(p), 0) / T;
  const nll = trace.nll;
  return (
    <StepLayout
      explain={
        <>
          <p>
            The loss for one position is <Tex>{"-\\ln p"}</Tex>, where p is the probability the model gave the
            correct next token: 0 if it was certain and right, <Tex>{`\\ln ${cfg.vocab.length} \\approx ${Math.log(cfg.vocab.length).toFixed(2)}`}</Tex> for a
            uniform guess, and unbounded as p → 0. The sentence's loss is the mean over its {T} positions.
          </p>
          <p>
            It usually cannot reach 0, because the same prefix is followed by different words in the corpus (for
            example, <Tok>the cat sat on the</Tok> continues with “mat” 3 times out of 4). The last column shows the
            best possible probability, the corpus frequency. The loss is minimised exactly when the model's
            probabilities match those frequencies.
          </p>
        </>
      }
      formula={
        <>
          <Tex display>{"\\ell_i = -\\ln P_{i,\\,y_i}"}</Tex>
          <Tex display>{`L = \\frac{1}{${T}}\\sum_{i=0}^{${T - 1}} \\ell_i`}</Tex>
        </>
      }
      calc={
        <CalcPanel target={`the loss of this sentence (position ${pos} highlighted)`}>
          <CalcLine
            label={`position ${pos}`}
            tex={`\\ell_{${pos}} = -\\ln P_{${pos},\\texttt{${cfg.vocab[example.targets[pos]]}}} = -\\ln ${tn(trace.probs[pos][example.targets[pos]])} = ${tn(nll[pos])}`}
          />
          <CalcLine
            label="mean"
            tex={`L = \\tfrac{1}{${T}}(${nll.map((x, i) => (i === pos ? `\\colorbox{#fde68a}{$${tn(x)}$}` : tn(x))).join(" + ")}) = ${boxed(trace.loss)}`}
          />
          <CalcNote>
            Best achievable for this sentence (predicting the corpus frequencies): {bestLoss.toFixed(4)}.
          </CalcNote>
        </CalcPanel>
      }
    >
      <table className="softmax-table loss-table">
        <thead>
          <tr>
            <th>i</th>
            <th>given</th>
            <th>target</th>
            <th>
              <Tex>{"P_{i,y_i}"}</Tex>
            </th>
            <th>
              <Tex>{"\\ell_i = -\\ln P"}</Tex>
            </th>
            <th />
            <th>best possible P</th>
          </tr>
        </thead>
        <tbody>
          {example.targets.map((t, i) => (
            <tr key={i} className={i === pos ? "focus" : ""} onMouseEnter={() => setPos(i)}>
              <td>{i}</td>
              <td className="tok-cell">{tokens.slice(0, i + 1).join(" ")}</td>
              <td className="tok-cell">
                <strong>{cfg.vocab[t]}</strong>
              </td>
              <td>{trace.probs[i][t].toFixed(4)}</td>
              <td>{nll[i].toFixed(4)}</td>
              <td className="prob-bar-cell">
                <div className="prob-bar loss-bar" style={{ width: `${Math.min(1, nll[i] / Math.log(cfg.vocab.length)) * 100}%` }} />
              </td>
              <td className="corpus-cell">{Number.isNaN(best[i]) ? "" : best[i].toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="viz-caption">
        Mean loss <strong>{trace.loss.toFixed(4)}</strong>; next, backpropagation computes <Tex>{d("\\theta")}</Tex> for
        every weight.
      </p>
    </StepLayout>
  );
}
