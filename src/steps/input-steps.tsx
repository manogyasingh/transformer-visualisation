import { useState } from "react";
import { ArchitectureDiagram } from "../components/architecture-diagram";
import { CalcLine, CalcNote, CalcPanel } from "../components/calc-panel";
import { EqRow, Op } from "../components/layout";
import { MatrixView } from "../components/matrix-view";
import { StepLayout } from "../components/step-layout";
import { Tex } from "../components/tex";
import { boxed, ta, tn } from "../lib/format";
import { useStep } from "./step-context";
import { Tok, tokenRowLabels, useCell } from "./shared";

function parameterCount(cfg: ReturnType<typeof useStep>["cfg"]): number {
  const { dModel: d, dFF: f, vocab, nCtx, nLayers } = cfg;
  const perBlock = 2 * d + 4 * (d * d + d) + 2 * d + (d * f + f) + (f * d + d);
  return vocab.length * d + nCtx * d + nLayers * perBlock + 2 * d;
}

export function OverviewStep() {
  const { cfg, trace, model } = useStep();
  const t = model.training;
  const nSentences = t.corpus.reduce((s, c) => s + c.count, 0);
  return (
    <StepLayout
      explain={
        <>
          <p>
            This is a complete, working GPT: the same architecture as GPT-2, shrunk until every number fits on
            screen. It was trained on {nSentences} toy sentences, and everything you will see is{" "}
            <strong>computed live in your browser from its trained weights</strong>, not illustrated.
          </p>
          <p>
            Your prompt is <Tok>{trace.tokens.join(" ")}</Tok>. The following {"steps"} walk through every
            lookup, matrix multiplication, normalisation and softmax that turns it into a probability distribution
            over the next token, and then pick one. Change the prompt at the top at any time; every step
            recomputes.
          </p>
          <table className="hp-table">
            <thead>
              <tr>
                <th />
                <th>this model</th>
                <th>GPT-2 small</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>vocabulary size V</td>
                <td>{cfg.vocab.length} words</td>
                <td>50,257 BPE tokens</td>
              </tr>
              <tr>
                <td>context length n_ctx</td>
                <td>{cfg.nCtx}</td>
                <td>1,024</td>
              </tr>
              <tr>
                <td>model width d</td>
                <td>{cfg.dModel}</td>
                <td>768</td>
              </tr>
              <tr>
                <td>heads H × head width d_h</td>
                <td>
                  {cfg.nHeads} × {cfg.dHead}
                </td>
                <td>12 × 64</td>
              </tr>
              <tr>
                <td>MLP hidden width d_ff</td>
                <td>{cfg.dFF}</td>
                <td>3,072</td>
              </tr>
              <tr>
                <td>blocks L</td>
                <td>{cfg.nLayers}</td>
                <td>12</td>
              </tr>
              <tr>
                <td>parameters</td>
                <td>{parameterCount(cfg).toLocaleString()}</td>
                <td>124 million</td>
              </tr>
            </tbody>
          </table>
        </>
      }
      formula={
        <>
          <div className="formula-caption">The whole computation</div>
          <Tex display>{String.raw`X^{(0)} = E + P`}</Tex>
          <Tex display>{String.raw`X^{(\ell)} = \mathrm{Block}_\ell\big(X^{(\ell-1)}\big),\quad \ell = 1,\dots,${cfg.nLayers}`}</Tex>
          <Tex display>{String.raw`z = \mathrm{LN}_f\big(X^{(${cfg.nLayers})}\big)_{T-1}\,W_E^{\top} \in \mathbb{R}^{${cfg.vocab.length}}`}</Tex>
          <Tex display>{String.raw`p = \mathrm{softmax}(z/\tau),\qquad \hat t \sim p`}</Tex>
          <div className="formula-caption">Each block</div>
          <Tex display>{String.raw`R = X + \mathrm{Attn}\big(\mathrm{LN}_1(X)\big)`}</Tex>
          <Tex display>{String.raw`X' = R + \mathrm{MLP}\big(\mathrm{LN}_2(R)\big)`}</Tex>
        </>
      }
      deeper={
        <>
          <p>
            <strong>How to read the visualisations.</strong> Rows are token positions (0-indexed, like in code) and
            columns are feature dimensions. We use the row-vector convention of GPT-2's code, so a linear layer is{" "}
            <Tex>{String.raw`y = xW + b`}</Tex> with <Tex>{String.raw`W`}</Tex> stored as (in × out). Orange cells are
            positive, blue negative, and the intensity is proportional to magnitude (scaled per matrix); purple
            cells are probabilities. The <span className="marked-inline">▸</span> marks the last position, whose
            final vector predicts the next token.
          </p>
          <p>
            <strong>Precision.</strong> Cells show 2 decimals, calculations show 4. All arithmetic uses 64-bit
            floats. The weights were rounded to 4 decimals after training, so every weight you see in a
            calculation is the exact value the model uses.
          </p>
          <p>
            <strong>Training.</strong> The model was trained with Adam for {t.steps.toLocaleString()} full-batch
            steps by <code>training/train-tiny-gpt.py</code> (plain NumPy with a hand-written, gradient-checked
            backward pass). Its loss is {t.roundedLoss.toFixed(4)} nats per token; the best achievable on this
            corpus is {t.optimalLoss.toFixed(4)}, so it has learned the corpus statistics essentially perfectly.
          </p>
          <table className="corpus-table">
            <thead>
              <tr>
                <th>training sentence</th>
                <th>copies</th>
              </tr>
            </thead>
            <tbody>
              {t.corpus.map((c) => (
                <tr key={c.text}>
                  <td>
                    <code>{c.text}</code>
                  </td>
                  <td>{c.count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      }
    >
      <h3 className="viz-title">Architecture (click any stage to jump to it)</h3>
      <ArchitectureDiagram />
      <p className="viz-caption">
        Use <kbd>←</kbd> <kbd>→</kbd> or the buttons below to step through. Hover over any number to see exactly
        how it was computed.
      </p>
    </StepLayout>
  );
}

export function TokenizeStep() {
  const { cfg, trace } = useStep();
  const T = trace.ids.length;
  const [pos, setPos] = useState(T - 1);
  const [hoverVocab, setHoverVocab] = useState<number | null>(null);
  const id = trace.ids[pos];
  const others = trace.ids.map((x, i) => (x === id && i !== pos ? i : -1)).filter((i) => i >= 0);

  return (
    <StepLayout
      explain={
        <>
          <p>
            A GPT never sees characters. The text is cut into <strong>tokens</strong> from a fixed vocabulary, and
            each token is replaced by its integer index. Our toy vocabulary has V = {cfg.vocab.length} whole words,
            so tokenizing is just splitting on spaces.
          </p>
          <p>
            GPT-2 uses byte-pair encoding (BPE) with 50,257 sub-word tokens (“unbelievable” → “un”, “believ”,
            “able”), but the output has the same form: a list of T integers. Here T = {T}, and the model's job is to
            produce a distribution over the token at position {T}.
          </p>
        </>
      }
      formula={
        <>
          <Tex display>{String.raw`t = (t_0, \dots, t_{T-1}),\quad t_i \in \{0, \dots, V-1\}`}</Tex>
          <Tex display>{String.raw`t = (${trace.ids.join(", ")}),\quad T = ${T}`}</Tex>
        </>
      }
      calc={
        <CalcPanel target={`position ${pos}`}>
          <CalcLine
            label="lookup"
            tex={String.raw`t_{${pos}} = \mathrm{vocab}[\texttt{"${trace.tokens[pos]}"}] = \boxed{${id}}`}
          />
          {others.length > 0 && (
            <CalcNote>
              The same word also appears at position {others.join(", ")} and gets the same id {id}. Until position
              information is added (two steps from now) the model cannot tell these copies apart.
            </CalcNote>
          )}
        </CalcPanel>
      }
    >
      <div className="tokenize-viz">
        <div className="prompt-string">“{trace.tokens.join(" ")}”</div>
        <div className="arch-down">↓ split on spaces, look up each word</div>
        <div className="token-row">
          {trace.tokens.map((tok, i) => (
            <div
              key={i}
              className={`token-card${i === pos ? " focus" : ""}${hoverVocab === trace.ids[i] ? " related" : ""}`}
              onMouseEnter={() => setPos(i)}
            >
              <div className="token-card-pos">position {i}</div>
              <div className="token-card-word">{tok}</div>
              <div className="token-card-id">id {trace.ids[i]}</div>
            </div>
          ))}
          <div className="token-card ghost">
            <div className="token-card-pos">position {T}</div>
            <div className="token-card-word">?</div>
            <div className="token-card-id">to predict</div>
          </div>
        </div>
        <h3 className="viz-title">Vocabulary (V = {cfg.vocab.length})</h3>
        <div className="vocab-grid">
          {cfg.vocab.map((w, v) => (
            <div
              key={v}
              className={`vocab-entry${trace.ids.includes(v) ? " used" : ""}${v === id ? " focus" : ""}`}
              onMouseEnter={() => {
                setHoverVocab(v);
                const first = trace.ids.indexOf(v);
                if (first >= 0) setPos(first);
              }}
              onMouseLeave={() => setHoverVocab(null)}
            >
              <span className="vocab-id">{v}</span>
              <span className="vocab-word">{w}</span>
            </div>
          ))}
        </div>
      </div>
    </StepLayout>
  );
}

export function TokenEmbeddingStep() {
  const { cfg, trace, model } = useStep();
  const T = trace.ids.length;
  const [cell, setCell] = useCell({ i: T - 1, j: 0 });
  const id = trace.ids[cell.i];
  const wte = model.params.wte;
  const vocabLabels = cfg.vocab.map((w, v) => (
    <span className="rl" key={v}>
      <span className="rl-pos">{v}</span>
      <span className="rl-tok">{w}</span>
    </span>
  ));

  return (
    <StepLayout
      explain={
        <>
          <p>
            Each token id selects one row of the <strong>token embedding matrix</strong>{" "}
            <Tex>{String.raw`W_E \in \mathbb{R}^{${cfg.vocab.length}\times ${cfg.dModel}}`}</Tex>. That row, d ={" "}
            {cfg.dModel} learned numbers, is the model's representation of the token. Stacking the T = {T} selected
            rows gives <Tex>E</Tex>, one row per position.
          </p>
          <p>
            Formally this is a product of one-hot vectors with <Tex>W_E</Tex>, but no multiplication actually
            happens: it is an array index. Repeated words get identical rows (look at every copy of{" "}
            <Tok>{trace.tokens[0]}</Tok>).
          </p>
        </>
      }
      formula={
        <>
          <Tex display>{String.raw`E_{i,:} = (W_E)_{t_i,:}`}</Tex>
          <Tex display>{String.raw`E = \mathrm{onehot}(t)\,W_E,\quad \mathrm{onehot}(t)\in\{0,1\}^{T\times V}`}</Tex>
        </>
      }
      deeper={
        <p>
          <Tex>W_E</Tex> does double duty. GPT-2 <em>ties</em> the input and output embeddings: the very same
          matrix, transposed, turns the final hidden vector back into a score per vocabulary word at the end of the
          network. Tokens that can follow the same contexts are pushed towards similar rows.
        </p>
      }
      calc={
        <CalcPanel target={<Tex>{String.raw`E_{${cell.i},${cell.j}}`}</Tex>}>
          <CalcLine
            label="one-hot product"
            tex={String.raw`E_{${cell.i},${cell.j}} = \sum_{v=0}^{${cfg.vocab.length - 1}} \mathrm{onehot}(t_{${cell.i}})_v\,(W_E)_{v,${cell.j}} = (W_E)_{t_{${cell.i}},${cell.j}}`}
          />
          <CalcLine
            label="lookup"
            tex={String.raw`= (W_E)_{${id},${cell.j}} = ${boxed(wte[id][cell.j])}\qquad(t_{${cell.i}} = ${id} = \texttt{"${cfg.vocab[id]}"})`}
          />
        </CalcPanel>
      }
    >
      <EqRow>
        <MatrixView
          data={wte}
          label="W_E"
          rowLabels={vocabLabels}
          highlight={{ rows: [...new Set(trace.ids)], focus: [id, cell.j] }}
          dimRow={(v) => !trace.ids.includes(v)}
          onHover={(v, j) => {
            const i = trace.ids.indexOf(v);
            if (i >= 0) setCell({ i, j });
          }}
        />
        <Op>
          <span className="op-text">select rows t_i →</span>
        </Op>
        <MatrixView
          data={trace.tokEmb}
          label="E"
          rowLabels={tokenRowLabels(trace.tokens)}
          markRow={T - 1}
          highlight={{ focus: [cell.i, cell.j] }}
          onHover={(i, j) => setCell({ i, j })}
        />
      </EqRow>
    </StepLayout>
  );
}

export function PositionEmbeddingStep() {
  const { cfg, trace } = useStep();
  const T = trace.ids.length;
  const [cell, setCell] = useCell({ i: T - 1, j: 0 });
  const { i, j } = cell;
  const hl = { focus: [i, j] as [number, number] };
  const hover = (i: number, j: number) => setCell({ i, j });

  return (
    <StepLayout
      explain={
        <>
          <p>
            Attention, coming up next, treats its input as a <em>set</em>: shuffle the input rows and the output
            rows are shuffled the same way but otherwise unchanged. Word order must therefore be injected
            explicitly. GPT-2 learns a second table{" "}
            <Tex>{String.raw`W_P \in \mathbb{R}^{${cfg.nCtx}\times ${cfg.dModel}}`}</Tex>, one row per position, and
            adds row <Tex>i</Tex> to the token embedding at position <Tex>i</Tex>.
          </p>
          <p>
            The sum <Tex>{String.raw`X^{(0)}`}</Tex> is the start of the <strong>residual stream</strong>: one{" "}
            {cfg.dModel}-number vector per position that every later layer reads from and adds to. Repeated words
            now have different vectors.
          </p>
        </>
      }
      formula={
        <>
          <Tex display>{String.raw`P_{i,:} = (W_P)_{i,:},\quad i = 0,\dots,T-1`}</Tex>
          <Tex display>{String.raw`X^{(0)} = E + P`}</Tex>
        </>
      }
      deeper={
        <p>
          Newer models (LLaMA, Mistral, Qwen…) use rotary position embeddings (RoPE) instead: nothing is added here,
          and queries and keys are rotated by position-dependent angles inside attention. Learned absolute
          embeddings cannot extrapolate: this model has no row for position {cfg.nCtx}, which is why the context is
          limited to {cfg.nCtx} tokens.
        </p>
      }
      calc={
        <CalcPanel target={<Tex>{String.raw`X^{(0)}_{${i},${j}}`}</Tex>}>
          <CalcLine
            label="add"
            tex={String.raw`X^{(0)}_{${i},${j}} = E_{${i},${j}} + P_{${i},${j}} = ${tn(trace.tokEmb[i][j])} + ${ta(trace.posEmb[i][j])} = ${boxed(trace.h0[i][j])}`}
          />
          <CalcNote>
            <Tex>{String.raw`E_{${i},:}`}</Tex> is the embedding of <Tok>{trace.tokens[i]}</Tok>;{" "}
            <Tex>{String.raw`P_{${i},:}`}</Tex> is the embedding of position {i}.
          </CalcNote>
        </CalcPanel>
      }
    >
      <EqRow>
        <MatrixView data={trace.tokEmb} label="E" rowLabels={tokenRowLabels(trace.tokens)} markRow={T - 1} highlight={hl} onHover={hover} />
        <Op tex="+" />
        <MatrixView data={trace.posEmb} label={String.raw`P = W_P[0{:}${T}]`} highlight={hl} onHover={hover} />
        <Op tex="=" />
        <MatrixView data={trace.h0} label="X^{(0)}" highlight={hl} onHover={hover} />
      </EqRow>
    </StepLayout>
  );
}
