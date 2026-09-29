import { useMemo } from "react";
import { CalcLine, CalcNote, CalcPanel } from "../components/calc-panel";
import { BarList, NumberLine, TemperatureCurve, type Segment } from "../components/charts";
import { MatMulView } from "../components/layout";
import { MatrixView } from "../components/matrix-view";
import { StepLayout } from "../components/step-layout";
import { Tex } from "../components/tex";
import { boxed, colored, COL_COLOR, dotExpansion, fmt, ROW_COLOR, tn, tsci } from "../lib/format";
import { corpusNextCounts } from "../model/corpus";
import { runForward } from "../model/forward";
import { argmax, transpose } from "../model/linalg";
import { decode, type Strategy } from "../model/sampling";
import type { TinyGpt } from "../model/types";
import { useStep } from "./step-context";
import { LayerNormStep } from "./layer-norm-step";
import { indexLabels, Tok, tokenRowLabels, useCell } from "./shared";

function softmax(z: number[]): number[] {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

export function FinalLnStep() {
  const { trace, cfg } = useStep();
  return (
    <LayerNormStep
      ln={trace.lnf}
      names={{ input: "X", output: "F", gamma: String.raw`\gamma_f`, beta: String.raw`\beta_f` }}
      onlyLastRowMatters
      intro={
        <p>
          All {cfg.nLayers} blocks are done. The residual stream <Tex>{String.raw`X = X^{(${cfg.nLayers})}`}</Tex>{" "}
          gets one last LayerNorm (GPT-2's <code>ln_f</code>) before it is read out, because the un-normalised
          stream can have any scale.
        </p>
      }
    />
  );
}

export function LogitsStep() {
  const { trace, cfg, model } = useStep();
  const T = trace.ids.length;
  const V = cfg.vocab.length;
  const [cell, setCell] = useCell({ i: 0, j: argmax(trace.logits) });
  const { j: v, k } = cell;
  const f = trace.lnf.output[T - 1];
  const wteT = useMemo(() => transpose(model.params.wte), [model]);
  const e = dotExpansion(f, model.params.wte[v], { emphasize: k });
  const order = trace.logits.map((_, i) => i).sort((a, b) => trace.logits[b] - trace.logits[a]);
  const rank = order.indexOf(v) + 1;
  const lo = Math.min(0, ...trace.logits);
  const hi = Math.max(0, ...trace.logits);
  const vocabLabels = cfg.vocab.map((w) => <span key={w}>{w}</span>);

  return (
    <StepLayout
      explain={
        <>
          <p>
            To predict the next token we only need the <strong>last row</strong> of <Tex>F</Tex>. It is compared
            with every vocabulary word's embedding by a dot product, giving one score, a <strong>logit</strong>,
            per word: {V} numbers.
          </p>
          <p>
            This is <strong>weight tying</strong>: the unembedding matrix is <Tex>{String.raw`W_E^\top`}</Tex>, the
            same matrix that embedded the input tokens. A high logit means the final vector points in the same
            direction as that word's embedding. Logits are unnormalised log-probabilities: only their differences
            matter.
          </p>
        </>
      }
      formula={
        <>
          <Tex display>{String.raw`z = F_{T-1,:}\;W_E^{\top} \in \mathbb{R}^{${V}}`}</Tex>
          <Tex display>{String.raw`z_v = \sum_{j=0}^{d-1} F_{T-1,j}\,(W_E)_{v,j}`}</Tex>
        </>
      }
      deeper={<PerPositionPredictions />}
      calc={
        <CalcPanel
          target={
            <>
              <Tex>{`z_{${v}}`}</Tex>, the logit for <Tok>{cfg.vocab[v]}</Tok>
            </>
          }
        >
          <CalcLine
            label="definition"
            tex={String.raw`z_{${v}} = \sum_{j=0}^{${cfg.dModel - 1}} ${colored(ROW_COLOR, `F_{${T - 1},j}`)}\,${colored(COL_COLOR, `(W_E)_{${v},j}`)}`}
          />
          <CalcLine label="substitute" tex={String.raw`= ${e.factors}`} />
          <CalcLine label="multiply" tex={String.raw`= ${e.products}`} />
          <CalcLine label="sum" tex={String.raw`= ${boxed(trace.logits[v])}`} />
          <CalcNote>
            Rank {rank} of {V}.
          </CalcNote>
        </CalcPanel>
      }
    >
      <div className="logits-layout">
        <MatMulView
          corner={
            <div className="mm-hint">
              <Tex>{String.raw`F_{${T - 1},:}\,W_E^\top`}</Tex>
              <div>
                <span style={{ color: ROW_COLOR }}>last hidden vector</span> ·{" "}
                <span style={{ color: COL_COLOR }}>embedding of word v</span>
              </div>
            </div>
          }
          b={
            <MatrixView
              data={wteT}
              label={String.raw`W_E^\top`}
              rowLabels={indexLabels(cfg.dModel)}
              colLabels={vocabLabels}
              verticalColLabels
              size="medium"
              highlight={{ cols: [v], focus: k !== undefined ? [k, v] : null }}
              onHover={(k, v) => setCell({ i: 0, j: v, k })}
            />
          }
          a={
            <MatrixView
              data={[f]}
              label={`F_{${T - 1},:}`}
              rowLabels={tokenRowLabels(trace.tokens).slice(T - 1)}
              markRow={0}
              highlight={{ rows: [0], focus: k !== undefined ? [0, k] : null }}
              onHover={(_, k) => setCell((c) => ({ ...c, k }))}
            />
          }
          c={
            <MatrixView
              data={[trace.logits]}
              label="z"
              colLabels="none"
              size="medium"
              highlight={{ focus: [0, v] }}
              onHover={(_, v) => setCell({ i: 0, j: v })}
            />
          }
        />
        <div className="logits-bars">
          <div className="arcs-title">Logits, sorted</div>
          <BarList
            items={order.map((id) => ({ key: id, label: cfg.vocab[id], value: trace.logits[id], focus: id === v }))}
            min={lo}
            max={hi}
            onHover={(id) => setCell({ i: 0, j: id })}
          />
        </div>
      </div>
    </StepLayout>
  );
}

function PerPositionPredictions() {
  const { trace, cfg } = useStep();
  return (
    <>
      <p>
        The model actually computed logits for <em>every</em> position, a T×{cfg.vocab.length} matrix. In training
        all of them are used at once (position i is scored on predicting token i + 1), so one forward pass yields T
        training examples. For generation only the last row matters. Here is what every row predicts (at τ = 1):
      </p>
      <table className="pred-table">
        <thead>
          <tr>
            <th>after the prefix</th>
            <th>top predictions</th>
          </tr>
        </thead>
        <tbody>
          {trace.logitsAll.map((row, i) => {
            const p = softmax(row);
            const top = p
              .map((_, id) => id)
              .sort((a, b) => p[b] - p[a])
              .slice(0, 3);
            return (
              <tr key={i} className={i === trace.ids.length - 1 ? "current" : ""}>
                <td>
                  <code>{trace.tokens.slice(0, i + 1).join(" ")}</code>
                </td>
                <td>
                  {top.map((id) => (
                    <span key={id} className="pred-chip">
                      {cfg.vocab[id]} <small>{p[id].toFixed(2)}</small>
                    </span>
                  ))}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </>
  );
}

function TemperatureControl() {
  const { decoding, setDecoding } = useStep();
  return (
    <div className="control-row">
      <label>
        temperature τ = <strong>{decoding.temperature.toFixed(2)}</strong>
        <input
          type="range"
          min={0.1}
          max={2.5}
          step={0.05}
          value={decoding.temperature}
          onChange={(e) => setDecoding({ ...decoding, temperature: Number(e.target.value) })}
        />
      </label>
      {[0.5, 1, 2].map((t) => (
        <button key={t} className="small-btn" onClick={() => setDecoding({ ...decoding, temperature: t })}>
          τ = {t}
        </button>
      ))}
    </div>
  );
}

const CURVE_COLORS = ["#ea580c", "#7c3aed", "#0891b2"];

export function TemperatureStep() {
  const { trace, cfg, sampling } = useStep();
  const [cell, setCell] = useCell({ i: 0, j: argmax(trace.logits) });
  const v = cell.j;
  const tau = sampling.temperature;
  const order = sampling.order;
  const top = order[0];
  const byLogit = trace.logits.map((_, i) => i).sort((a, b) => trace.logits[b] - trace.logits[a]);
  const lo = Math.min(0, ...trace.logits, ...sampling.scaled);
  const hi = Math.max(0, ...trace.logits, ...sampling.scaled);
  const curveIds = byLogit.slice(0, 3);

  return (
    <StepLayout
      explain={
        <>
          <p>
            Before the softmax, the logits are divided by a <strong>temperature</strong> <Tex>\tau</Tex>. Softmax
            depends only on differences between logits, and dividing by <Tex>\tau</Tex> scales every difference by{" "}
            <Tex>{String.raw`1/\tau`}</Tex>:
          </p>
          <ul>
            <li>
              <Tex>{String.raw`\tau < 1`}</Tex> stretches the gaps: sharper, more deterministic;
            </li>
            <li>
              <Tex>{String.raw`\tau > 1`}</Tex> shrinks them: flatter, more random;
            </li>
            <li>
              <Tex>{String.raw`\tau \to 0`}</Tex> approaches greedy argmax; <Tex>{String.raw`\tau = 1`}</Tex> leaves the
              model's learned distribution unchanged.
            </li>
          </ul>
        </>
      }
      formula={
        <>
          <Tex display>{String.raw`z'_v = z_v / \tau`}</Tex>
          <Tex display>{String.raw`z'_a - z'_b = (z_a - z_b)/\tau`}</Tex>
        </>
      }
      calc={
        <CalcPanel
          target={
            <>
              <Tex>{`z'_{${v}}`}</Tex> for <Tok>{cfg.vocab[v]}</Tok>
            </>
          }
        >
          <CalcLine
            label="divide"
            tex={String.raw`z'_{${v}} = \frac{z_{${v}}}{\tau} = \frac{${tn(trace.logits[v])}}{${tn(tau, 2)}} = ${boxed(sampling.scaled[v])}`}
          />
          {v !== top && (
            <CalcLine
              label="gap to top"
              tex={String.raw`z'_{${top}} - z'_{${v}} = \frac{${tn(trace.logits[top])} - (${tn(trace.logits[v])})}{${tn(tau, 2)}} = ${tn(sampling.scaled[top] - sampling.scaled[v])}`}
            />
          )}
        </CalcPanel>
      }
    >
      <TemperatureControl />
      <div className="temp-layout">
        <div>
          <div className="arcs-title">Logits z</div>
          <BarList
            items={byLogit.map((id) => ({ key: id, label: cfg.vocab[id], value: trace.logits[id], focus: id === v }))}
            min={lo}
            max={hi}
            width={200}
            onHover={(id) => setCell({ i: 0, j: id })}
          />
        </div>
        <div>
          <div className="arcs-title">
            Scaled logits <Tex>{`z' = z/${tau.toFixed(2)}`}</Tex>
          </div>
          <BarList
            items={byLogit.map((id) => ({ key: id, label: cfg.vocab[id], value: sampling.scaled[id], focus: id === v }))}
            min={lo}
            max={hi}
            width={200}
            onHover={(id) => setCell({ i: 0, j: id })}
          />
        </div>
        <div>
          <div className="arcs-title">Probability of the top 3 words as τ varies</div>
          <TemperatureCurve
            logits={trace.logits}
            labels={curveIds.map((id) => cfg.vocab[id])}
            ids={curveIds}
            tau={tau}
            colors={CURVE_COLORS}
          />
        </div>
      </div>
    </StepLayout>
  );
}

export function SoftmaxOutStep() {
  const { trace, cfg, sampling, model } = useStep();
  const [cell, setCell] = useCell({ i: 0, j: sampling.order[0] });
  const v = cell.j;
  const { counts, total } = useMemo(() => corpusNextCounts(model, trace.ids), [model, trace.ids]);
  const m = sampling.maxScaled;
  const argTop = sampling.order[0];
  const expTerms = sampling.scaled.map((z) => `e^{${tn(z - m)}}`).join(" + ");
  const expVals = sampling.exps.map((x) => tsci(x)).join(" + ");

  return (
    <StepLayout
      explain={
        <>
          <p>
            Softmax turns the scaled logits into a probability distribution over the vocabulary: exponentiate (so
            every value is positive), then divide by the total (so they sum to 1). This <Tex>p</Tex> is the model's
            estimate of <Tex>{String.raw`P(t_T \mid t_0,\dots,t_{T-1})`}</Tex>.
          </p>
          {total > 0 ? (
            <p>
              The last column shows how this prompt actually continued in the training data ({total} sentence
              {total === 1 ? "" : "s"}). At τ = 1 the model reproduces those frequencies almost exactly, which is
              exactly what minimising cross-entropy asks for.
            </p>
          ) : (
            <p>
              This prompt never occurs in the training data, so the model is extrapolating from similar sentences.
            </p>
          )}
        </>
      }
      formula={
        <>
          <Tex display>{String.raw`p_v = \frac{e^{z'_v - m}}{\sum_{u=0}^{V-1} e^{z'_u - m}}`}</Tex>
          <Tex display>{String.raw`m = \max_u z'_u`}</Tex>
        </>
      }
      calc={
        <CalcPanel
          target={
            <>
              <Tex>{`p_{${v}}`}</Tex>, the probability of <Tok>{cfg.vocab[v]}</Tok>
            </>
          }
        >
          <CalcLine label="max" tex={String.raw`m = z'_{${argTop}} = ${tn(m)}\quad(\texttt{${cfg.vocab[argTop]}})`} />
          <CalcLine label="denominator" tex={String.raw`\textstyle\sum_u e^{z'_u - m} = ${expTerms}`} />
          <CalcLine tex={String.raw`= ${expVals} = ${tn(sampling.sumExp)}`} />
          <CalcLine
            label="probability"
            tex={String.raw`p_{${v}} = \dfrac{e^{${tn(sampling.scaled[v])} - ${tn(m)}}}{${tn(sampling.sumExp)}} = \dfrac{${tsci(sampling.exps[v])}}{${tn(sampling.sumExp)}} = \boxed{${tsci(sampling.probs[v])}}`}
          />
        </CalcPanel>
      }
    >
      <table className="softmax-table">
        <thead>
          <tr>
            <th>token</th>
            <th>
              <Tex>z'</Tex>
            </th>
            <th>
              <Tex>z' - m</Tex>
            </th>
            <th>
              <Tex>{String.raw`e^{z'-m}`}</Tex>
            </th>
            <th>
              <Tex>p</Tex>
            </th>
            <th />
            {total > 0 && <th>training data</th>}
          </tr>
        </thead>
        <tbody>
          {sampling.order.map((id) => (
            <tr key={id} className={id === v ? "focus" : ""} onMouseEnter={() => setCell({ i: 0, j: id })}>
              <td className="tok-cell">{cfg.vocab[id]}</td>
              <td>{fmt(sampling.scaled[id], 3)}</td>
              <td>{fmt(sampling.scaled[id] - m, 3)}</td>
              <td>{sampling.exps[id] < 1e-4 ? sampling.exps[id].toExponential(2) : fmt(sampling.exps[id], 4)}</td>
              <td>{sampling.probs[id] < 1e-4 ? sampling.probs[id].toExponential(2) : fmt(sampling.probs[id], 4)}</td>
              <td className="prob-bar-cell">
                <div className="prob-bar" style={{ width: `${sampling.probs[id] * 100}%` }} />
              </td>
              {total > 0 && (
                <td className="corpus-cell">
                  {counts.get(id) ? `${counts.get(id)}/${total} = ${((counts.get(id) ?? 0) / total).toFixed(2)}` : ""}
                </td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </StepLayout>
  );
}

const STRATEGIES: { value: Strategy; label: string }[] = [
  { value: "greedy", label: "Greedy" },
  { value: "sample", label: "Sample" },
  { value: "top-k", label: "Top-k" },
  { value: "top-p", label: "Top-p (nucleus)" },
];

function DecodingControls() {
  const { decoding, setDecoding, cfg, goTo } = useStep();
  return (
    <div className="decoding-controls">
      <div className="tabs">
        {STRATEGIES.map((s) => (
          <button
            key={s.value}
            className={decoding.strategy === s.value ? "tab active" : "tab"}
            onClick={() => setDecoding({ ...decoding, strategy: s.value })}
          >
            {s.label}
          </button>
        ))}
      </div>
      {decoding.strategy === "top-k" && (
        <label>
          k = <strong>{decoding.topK}</strong>
          <input
            type="range"
            min={1}
            max={cfg.vocab.length}
            value={decoding.topK}
            onChange={(e) => setDecoding({ ...decoding, topK: Number(e.target.value) })}
          />
        </label>
      )}
      {decoding.strategy === "top-p" && (
        <label>
          p = <strong>{decoding.topP.toFixed(2)}</strong>
          <input
            type="range"
            min={0.05}
            max={1}
            step={0.05}
            value={decoding.topP}
            onChange={(e) => setDecoding({ ...decoding, topP: Number(e.target.value) })}
          />
        </label>
      )}
      {decoding.strategy !== "greedy" && (
        <span className="seed-control">
          random seed
          <input
            type="number"
            value={decoding.seed}
            onChange={(e) => setDecoding({ ...decoding, seed: Math.trunc(Number(e.target.value)) || 0 })}
          />
          <button className="small-btn" onClick={() => setDecoding({ ...decoding, seed: decoding.seed + 1 })}>
            New random draw
          </button>
        </span>
      )}
      <button className="link-btn" onClick={() => goTo("temperature")}>
        τ = {decoding.temperature.toFixed(2)} (change)
      </button>
    </div>
  );
}

export function SampleStep() {
  const { trace, cfg, sampling, decoding } = useStep();
  const s = sampling;
  const T = trace.ids.length;
  const keptIds = s.order.filter((id) => s.kept[id]);
  const chosenRank = s.order.indexOf(s.chosen);
  const segments: Segment[] = [];
  s.order.forEach((id, r) => {
    if (!s.kept[id] || s.filtered[id] === 0) return;
    segments.push({
      key: id,
      label: cfg.vocab[id],
      from: s.cumulative[r] - s.filtered[id],
      to: s.cumulative[r],
      chosen: id === s.chosen,
    });
  });
  const lowerF = chosenRank === 0 ? 0 : s.cumulative[chosenRank - 1];
  const filtering = decoding.strategy === "top-k" || decoding.strategy === "top-p";
  const shown = s.order.filter((id) => s.probs[id] >= 1e-4 || s.kept[id]).slice(0, 10);

  return (
    <StepLayout
      explain={
        <>
          <p>
            The model's output is a distribution. A <strong>decoding strategy</strong> turns it into one token:
          </p>
          <ul>
            <li>
              <strong>Greedy</strong>: always take the most likely token. Deterministic, but tends to be repetitive.
            </li>
            <li>
              <strong>Sample</strong>: pick token v with probability <Tex>p_v</Tex>. Concretely, draw{" "}
              <Tex>{String.raw`u \sim \mathrm{Uniform}[0,1)`}</Tex>, lay the probabilities end to end on [0, 1), and
              take the token whose segment contains u.
            </li>
            <li>
              <strong>Top-k</strong>: keep only the k most likely tokens, renormalise, then sample.
            </li>
            <li>
              <strong>Top-p</strong>: keep the smallest set of most likely tokens whose total probability reaches p,
              renormalise, then sample.
            </li>
          </ul>
        </>
      }
      formula={
        <>
          {filtering && (
            <Tex display>{String.raw`\tilde p_v = \frac{p_v\,[v \in \mathcal{K}]}{\sum_{u\in\mathcal{K}} p_u}`}</Tex>
          )}
          {decoding.strategy === "greedy" ? (
            <Tex display>{String.raw`\hat t = \arg\max_v p_v`}</Tex>
          ) : (
            <>
              <Tex display>{String.raw`F_r = \sum_{r' \le r} \tilde p_{(r')}`}</Tex>
              <Tex display>{String.raw`\hat t = (r)\ \text{ such that }\ F_{r-1} \le u < F_r`}</Tex>
              <div className="formula-caption">(r) = the r-th most likely token</div>
            </>
          )}
        </>
      }
      calc={
        <CalcPanel
          target={
            <>
              the next token: <Tok>{cfg.vocab[s.chosen]}</Tok>
            </>
          }
        >
          {decoding.strategy === "greedy" ? (
            <CalcLine
              label="argmax"
              tex={String.raw`\hat t = \arg\max_v p_v = ${s.chosen}\ (\texttt{${cfg.vocab[s.chosen]}}),\quad p_{${s.chosen}} = ${tn(s.probs[s.chosen])}`}
            />
          ) : (
            <>
              {filtering && (
                <>
                  <CalcLine
                    label="keep"
                    tex={String.raw`\mathcal{K} = \{${keptIds.map((id) => `\\texttt{${cfg.vocab[id]}}`).join(", ")}\},\quad \textstyle\sum_{u\in\mathcal{K}} p_u = ${keptIds.map((id) => tn(s.probs[id])).join(" + ")} = ${tn(s.keptMass)}`}
                  />
                  <CalcLine
                    label="renormalise"
                    tex={String.raw`\tilde p_{${s.chosen}} = \frac{${tn(s.probs[s.chosen])}}{${tn(s.keptMass)}} = ${tn(s.filtered[s.chosen])}`}
                  />
                </>
              )}
              <CalcLine
                label="random draw"
                tex={String.raw`u = ${tn(s.u ?? 0)}\quad\text{(seeded PRNG: seed ${decoding.seed}, position ${T})}`}
              />
              <CalcLine
                label="locate u"
                tex={String.raw`F_{${chosenRank}} = ${tn(lowerF)} \le u = ${tn(s.u ?? 0)} < F_{${chosenRank + 1}} = ${tn(s.cumulative[chosenRank])}\;\Rightarrow\; \hat t = \boxed{\texttt{${cfg.vocab[s.chosen]}}}`}
              />
            </>
          )}
        </CalcPanel>
      }
    >
      <DecodingControls />
      <div className="sample-layout">
        <table className="softmax-table">
          <thead>
            <tr>
              <th>rank r</th>
              <th>token</th>
              <th>
                <Tex>p</Tex>
              </th>
              {filtering && <th>kept?</th>}
              {filtering && (
                <th>
                  <Tex>{String.raw`\tilde p`}</Tex>
                </th>
              )}
              {decoding.strategy !== "greedy" && (
                <th>
                  <Tex>F_r</Tex>
                </th>
              )}
            </tr>
          </thead>
          <tbody>
            {shown.map((id) => {
              const r = s.order.indexOf(id);
              return (
                <tr key={id} className={`${id === s.chosen ? "chosen" : ""}${s.kept[id] ? "" : " removed"}`}>
                  <td>{r + 1}</td>
                  <td className="tok-cell">{cfg.vocab[id]}</td>
                  <td>{fmt(s.probs[id], 4)}</td>
                  {filtering && <td>{s.kept[id] ? "yes" : "no"}</td>}
                  {filtering && <td>{fmt(s.filtered[id], 4)}</td>}
                  {decoding.strategy !== "greedy" && <td>{fmt(s.cumulative[r], 4)}</td>}
                </tr>
              );
            })}
          </tbody>
        </table>
        <div className="sample-side">
          {decoding.strategy !== "greedy" && (
            <>
              <div className="arcs-title">The [0, 1) number line, one segment per kept token</div>
              <NumberLine segments={segments} u={s.u} />
            </>
          )}
          <div className="chosen-token">
            next token <span className="chosen-chip">{cfg.vocab[s.chosen]}</span>
            <small>
              with probability {fmt(s.probs[s.chosen], 4)}
              {filtering ? ` (${fmt(s.filtered[s.chosen], 4)} after filtering)` : ""}
            </small>
          </div>
        </div>
      </div>
    </StepLayout>
  );
}

/** Multiply-adds in all matrix products of one forward pass (only the last row is unembedded). */
function multiplyAdds(cfg: TinyGpt["config"], T: number): number {
  const { dModel: d, dHead, nHeads, dFF, nLayers, vocab } = cfg;
  const perBlock = 3 * T * d * d + 2 * nHeads * T * T * dHead + T * d * d + 2 * T * d * dFF;
  return nLayers * perBlock + d * vocab.length;
}

export function ResultStep() {
  const { trace, cfg, sampling, decoding, model, setPrompt, goTo } = useStep();
  const chosen = sampling.chosen;
  const next = [...trace.ids, chosen];
  const period = cfg.vocab.indexOf(".");
  const canContinue = chosen !== period && next.length <= cfg.nCtx;
  const p = sampling.probs[chosen];

  const rollout = useMemo(() => {
    const rows: { context: string; token: string; p: number }[] = [];
    let ids = next;
    while (ids[ids.length - 1] !== period && ids.length <= cfg.nCtx) {
      const tr = runForward(model, ids);
      const r = decode(tr.logits, decoding, ids.length);
      rows.push({ context: tr.tokens.join(" "), token: cfg.vocab[r.chosen], p: r.probs[r.chosen] });
      ids = [...ids, r.chosen];
    }
    return {
      rows,
      final: ids.map((id) => cfg.vocab[id]).join(" "),
      full: ids.length > cfg.nCtx && ids[ids.length - 1] !== period,
    };
  }, [trace.ids, chosen, decoding, model, cfg]);

  return (
    <StepLayout
      explain={
        <>
          <p>
            That is one token: {cfg.nLayers} blocks, {cfg.nLayers * cfg.nHeads} attention heads and{" "}
            {multiplyAdds(cfg, trace.ids.length).toLocaleString()} multiply-adds in matrix products. To generate
            text, the model appends the new token and runs the entire forward pass again on the longer sequence.
            That loop is what <em>autoregressive</em> means.
          </p>
          <p>
            In practice each new token is much cheaper than the first. With a <strong>KV cache</strong>, the keys and
            values of earlier positions are reused (they cannot change, because of the causal mask), so every layer
            only computes the new token's row: its query, key and value, one row of attention over all cached keys,
            and one row through the MLP.
          </p>
        </>
      }
      formula={
        <>
          <Tex display>{String.raw`t_T = \hat t`}</Tex>
          <Tex display>{String.raw`t_{T+1} \sim p(\,\cdot \mid t_0, \dots, t_T)`}</Tex>
          <Tex display>{String.raw`\log p(\hat t) = \ln ${tn(p)} = ${tn(Math.log(p))}`}</Tex>
        </>
      }
      calc={
        <CalcPanel target="training view">
          <CalcLine
            label="cross-entropy"
            tex={String.raw`-\ln p_{\hat t} = -\ln ${tn(p)} = ${boxed(-Math.log(p))}\ \text{nats}`}
          />
          <CalcNote>
            If <Tok>{cfg.vocab[chosen]}</Tok> were the true next token in a training sentence, this would be the
            loss at this position. Training adjusts every weight you have seen to make this number smaller on
            average.
          </CalcNote>
        </CalcPanel>
      }
    >
      <div className="result-sentence">
        {trace.tokens.map((t, i) => (
          <span key={i} className="token-chip">
            {t}
          </span>
        ))}
        <span className="token-chip new">{cfg.vocab[chosen]}</span>
      </div>
      <div className="result-actions">
        <button
          className="primary-btn"
          disabled={!canContinue}
          onClick={() => {
            setPrompt(next);
            goTo("tokenize");
          }}
        >
          Append “{cfg.vocab[chosen]}” and generate the next token →
        </button>
        {!canContinue && (
          <span className="muted">
            {chosen === period
              ? "The sentence is finished (the model produced “.”)."
              : `The context window (n_ctx = ${cfg.nCtx}) is full.`}
          </span>
        )}
        <button className="small-btn" onClick={() => goTo("sample")}>
          ← Try another decoding strategy
        </button>
      </div>
      {rollout.rows.length > 0 && (
        <div className="rollout">
          <h3 className="viz-title">If we keep going with the same settings</h3>
          <table className="pred-table">
            <thead>
              <tr>
                <th>context</th>
                <th>next token</th>
                <th>p</th>
              </tr>
            </thead>
            <tbody>
              {rollout.rows.map((r) => (
                <tr key={r.context}>
                  <td>
                    <code>{r.context}</code>
                  </td>
                  <td>
                    <span className="pred-chip">{r.token}</span>
                  </td>
                  <td>{r.p.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="viz-caption">
            Final text: <code>{rollout.final}</code>
            {rollout.full ? ` (stopped: context window of ${cfg.nCtx} is full)` : ""}
          </p>
        </div>
      )}
    </StepLayout>
  );
}
