import { useMemo, useState } from "react";
import { CalcLine, CalcNote, CalcPanel } from "../../components/calc-panel";
import { EqRow, Op } from "../../components/layout";
import { MatrixView } from "../../components/matrix-view";
import { StepLayout } from "../../components/step-layout";
import { Tex } from "../../components/tex";
import { boxed, tg, tgp, tn, tp } from "../../lib/format";
import { batchLoss } from "../../model/backward";
import { runForward } from "../../model/forward";
import { asMatrix, cloneParams, getElement, leafIsMatrix, paramList, setElement, type ParamRef } from "../../model/params";
import type { ModelConfig, ModelParams } from "../../model/types";
import { useTraining } from "../../training/training-context";
import { ADAM, adamElement, adamUpdate, corpusExamples, lrAt } from "../../training/trainer";
import { useStep } from "../step-context";
import { Tok } from "../shared";
import { d, sizeFor, useTokens } from "./backward-views";

function ParamSelect({ refs, value, onChange }: { refs: ParamRef[]; value: string; onChange: (id: string) => void }) {
  return (
    <label className="param-select">
      parameter
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        {refs.map((r) => (
          <option key={r.id} value={r.id}>
            {r.id}
          </option>
        ))}
      </select>
    </label>
  );
}

function perturbedLoss(
  cfg: ModelConfig,
  params: ModelParams,
  ref: ParamRef,
  i: number,
  j: number,
  delta: number,
  ids: number[],
  targets: number[],
): number {
  const p = cloneParams(params);
  const leaf = ref.get(p);
  setElement(leaf, i, j, getElement(leaf, i, j) + delta);
  return batchLoss({ config: cfg, params: p }, [{ ids, targets, weight: 1 }]);
}

const H = 1e-5;

export function TGradCheckStep() {
  const { trace, model, example } = useTraining();
  const { cfg } = useStep();
  const refs = useMemo(() => paramList(cfg), [cfg]);
  const [pid, setPid] = useState("b1.wq");
  const ref = refs.find((r) => r.id === pid)!;
  const grad = asMatrix(ref.get(trace.grads));
  const theta = asMatrix(ref.get(model.params));
  const [cell, setCell] = useState({ i: 0, j: 0 });
  const i = Math.min(cell.i, grad.length - 1);
  const j = Math.min(cell.j, grad[0].length - 1);

  const check = useMemo(() => {
    const plus = perturbedLoss(cfg, model.params, ref, i, j, H, example.ids, example.targets);
    const minus = perturbedLoss(cfg, model.params, ref, i, j, -H, example.ids, example.targets);
    const numeric = (plus - minus) / (2 * H);
    const analytic = grad[i][j];
    return { plus, minus, numeric, analytic, rel: Math.abs(numeric - analytic) / Math.max(1e-12, Math.abs(numeric) + Math.abs(analytic)) };
  }, [cfg, model.params, ref, i, j, example, grad]);

  const table = useMemo(
    () =>
      refs.map((r) => {
        const g = asMatrix(r.get(trace.grads));
        let bi = 0;
        let bj = 0;
        g.forEach((row, a) => row.forEach((x, b) => Math.abs(x) > Math.abs(g[bi][bj]) && ((bi = a), (bj = b))));
        const numeric =
          (perturbedLoss(cfg, model.params, r, bi, bj, H, example.ids, example.targets) -
            perturbedLoss(cfg, model.params, r, bi, bj, -H, example.ids, example.targets)) /
          (2 * H);
        const analytic = g[bi][bj];
        return { r, bi, bj, analytic, numeric, rel: Math.abs(numeric - analytic) / Math.max(1e-12, Math.abs(numeric) + Math.abs(analytic)) };
      }),
    [refs, trace, cfg, model.params, example],
  );
  const worst = Math.max(...table.filter((row) => Math.abs(row.analytic) > 1e-9).map((row) => row.rel));

  return (
    <StepLayout
      explain={
        <>
          <p>
            Backpropagation is intricate, so it is worth checking. A derivative is a limit of a difference quotient,
            so we can estimate any single gradient entry by nudging that one weight by{" "}
            <Tex>{"\\pm h"}</Tex>, rerunning the forward pass twice, and measuring how much the loss changed. This
            needs two forward passes <em>per weight</em>, which is hopeless for training (GPT-2 has 124 million weights)
            but perfect for testing.
          </p>
          <p>
            Pick any parameter and hover over any entry. Across one entry of every parameter tensor, the worst
            relative disagreement is <strong>{worst.toExponential(1)}</strong>, which is floating-point noise.
          </p>
        </>
      }
      formula={
        <>
          <Tex display>{"\\frac{\\partial L}{\\partial \\theta_k} \\approx \\frac{L(\\theta + h e_k) - L(\\theta - h e_k)}{2h}"}</Tex>
          <div className="formula-caption">
            central difference, error <Tex>{"O(h^2)"}</Tex>; here <Tex>{"h = 10^{-5}"}</Tex>
          </div>
        </>
      }
      calc={
        <CalcPanel
          target={
            <>
              <Tex>{`${d(ref.label)}`}</Tex> entry ({i}, {j})
            </>
          }
        >
          <CalcLine label="nudge up" tex={`L(\\theta + h e_k) = ${check.plus.toFixed(12)}`} />
          <CalcLine label="nudge down" tex={`L(\\theta - h e_k) = ${check.minus.toFixed(12)}`} />
          <CalcLine label="numeric" tex={`\\frac{${check.plus.toFixed(12)} - ${check.minus.toFixed(12)}}{2\\times10^{-5}} = ${tg(check.numeric, 8)}`} />
          <CalcLine label="backprop" tex={`${tg(check.analytic, 8)}`} />
          <CalcLine label="rel. error" tex={`\\frac{|a - n|}{|a| + |n|} = \\boxed{${tg(check.rel, 3)}}`} />
        </CalcPanel>
      }
    >
      <ParamSelect refs={refs} value={pid} onChange={(id) => {
        setPid(id);
        setCell({ i: 0, j: 0 });
      }} />
      <div className="gradcheck-layout">
        <MatrixView
          data={grad}
          label={d(ref.label)}
          size={sizeFor(grad[0].length)}
          autoScale
          highlight={{ focus: [i, j] }}
          onHover={(i, j) => setCell({ i, j })}
        />
        <MatrixView data={theta} label={ref.label} size={sizeFor(grad[0].length)} highlight={{ focus: [i, j] }} onHover={(i, j) => setCell({ i, j })} />
      </div>
      <h3 className="viz-title">One entry (the largest gradient) of every parameter tensor</h3>
      <table className="softmax-table gradcheck-table">
        <thead>
          <tr>
            <th>parameter</th>
            <th>entry</th>
            <th>backprop</th>
            <th>finite difference</th>
            <th>relative error</th>
          </tr>
        </thead>
        <tbody>
          {table.map((row) => (
            <tr key={row.r.id} className={row.r.id === pid ? "focus" : ""} onClick={() => {
              setPid(row.r.id);
              setCell({ i: row.bi, j: row.bj });
            }}>
              <td className="tok-cell">
                <Tex>{row.r.label}</Tex>
              </td>
              <td>
                ({row.bi}, {row.bj})
              </td>
              <td>{row.analytic.toExponential(6)}</td>
              <td>{row.numeric.toExponential(6)}</td>
              <td>{Math.abs(row.analytic) < 1e-9 ? "gradient ≈ 0" : row.rel.toExponential(1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </StepLayout>
  );
}

function useAdamPreview() {
  const { trainer, trace } = useTraining();
  return useMemo(() => adamUpdate(trainer.state, trace.grads, trainer.settings), [trainer.state, trace, trainer.settings]);
}

export function TAdamStep() {
  const { trainer, trace } = useTraining();
  const { cfg } = useStep();
  const refs = useMemo(() => paramList(cfg), [cfg]);
  const [pid, setPid] = useState("b1.wq");
  const ref = refs.find((r) => r.id === pid)!;
  const preview = useAdamPreview();
  const state = trainer.state;
  const theta = asMatrix(ref.get(state.params));
  const g = asMatrix(ref.get(trace.grads));
  const mPrev = asMatrix(ref.get(state.m));
  const vPrev = asMatrix(ref.get(state.v));
  const next = asMatrix(ref.get(preview.params));
  const delta = theta.map((r, i) => r.map((x, j) => x - next[i][j]));
  const [cell, setCell] = useState({ i: 0, j: 0 });
  const i = Math.min(cell.i, theta.length - 1);
  const j = Math.min(cell.j, theta[0].length - 1);
  const t = state.step + 1;
  const lr = lrAt(t, trainer.settings);
  const decays = leafIsMatrix(ref.get(state.params));
  const el = adamElement(theta[i][j], g[i][j], mPrev[i][j], vPrev[i][j], t, lr, decays);
  const size = sizeFor(theta[0].length);
  const hl = { focus: [i, j] as [number, number] };
  const hover = (i: number, j: number) => setCell({ i, j });
  const { beta1: b1, beta2: b2, weightDecay: wd } = ADAM;

  return (
    <StepLayout
      explain={
        <>
          <p>
            Plain gradient descent would set <Tex>{"\\theta \\leftarrow \\theta - \\eta\\,g"}</Tex>. Adam, used for
            essentially every large language model, adds two running averages per weight: <Tex>m</Tex>, an average of
            recent gradients (momentum, which smooths out noise between batches), and <Tex>v</Tex>, an average of
            recent <em>squared</em> gradients (which sets a per-weight scale). Dividing <Tex>m</Tex> by{" "}
            <Tex>{"\\sqrt v"}</Tex> makes every weight move by roughly the learning rate, whether its gradients are
            large or tiny.
          </p>
          <p>
            The hats undo the bias of starting <Tex>m, v</Tex> at zero. The <Tex>{"\\lambda\\theta"}</Tex> term is
            decoupled weight decay (AdamW), applied to weight matrices only, which gently pulls weights towards zero.
            The learning rate <Tex>{"\\eta_t"}</Tex> follows a cosine schedule from {trainer.settings.lr} down to 5% of
            that.
          </p>
        </>
      }
      formula={
        <>
          <Tex display>{"m_t = \\beta_1 m_{t-1} + (1-\\beta_1)\\,g"}</Tex>
          <Tex display>{"v_t = \\beta_2 v_{t-1} + (1-\\beta_2)\\,g^2"}</Tex>
          <Tex display>{"\\hat m_t = \\frac{m_t}{1-\\beta_1^t},\\quad \\hat v_t = \\frac{v_t}{1-\\beta_2^t}"}</Tex>
          <Tex display>{"\\theta \\leftarrow \\theta - \\eta_t\\Big(\\frac{\\hat m_t}{\\sqrt{\\hat v_t}+\\epsilon} + \\lambda\\theta\\Big)"}</Tex>
          <div className="formula-caption">
            <Tex>{`\\beta_1 = ${b1},\\ \\beta_2 = ${b2},\\ \\epsilon = 10^{-8},\\ \\lambda = ${wd}`}</Tex>
          </div>
        </>
      }
      calc={
        <CalcPanel
          target={
            <>
              update of <Tex>{ref.label}</Tex> entry ({i}, {j}) at step t = {t}
            </>
          }
        >
          <CalcLine label="learning rate" tex={`\\eta_{${t}} = ${trainer.settings.lr}\\big(0.05 + 0.95\\cdot\\tfrac12(1 + \\cos(\\pi\\cdot\\tfrac{${Math.min(t, trainer.settings.totalSteps)}}{${trainer.settings.totalSteps}}))\\big) = ${tg(lr)}`} />
          <CalcLine label="momentum" tex={`m_{${t}} = ${b1}\\cdot${tgp(mPrev[i][j])} + ${(1 - b1).toFixed(1)}\\cdot${tgp(g[i][j])} = ${tg(el.m)}`} />
          <CalcLine label="2nd moment" tex={`v_{${t}} = ${b2}\\cdot${tgp(vPrev[i][j])} + ${(1 - b2).toFixed(2)}\\cdot${tgp(g[i][j])}^2 = ${tg(el.v)}`} />
          <CalcLine label="bias-corrected" tex={`\\hat m = \\frac{${tg(el.m)}}{1 - ${b1}^{${t}}} = ${tg(el.mHat)},\\quad \\hat v = \\frac{${tg(el.v)}}{1 - ${b2}^{${t}}} = ${tg(el.vHat)}`} />
          <CalcLine
            label="step"
            tex={`\\Delta\\theta = ${tg(lr)}\\Big(\\frac{${tg(el.mHat)}}{\\sqrt{${tg(el.vHat)}} + 10^{-8}}${decays ? ` + ${wd}\\cdot${tp(theta[i][j])}` : ""}\\Big) = ${tg(el.update)}`}
          />
          <CalcLine label="new weight" tex={`\\theta' = ${tn(theta[i][j], 6)} - ${tgp(el.update)} = ${boxed(el.pNew, 6)}`} />
          {state.step === 0 && (
            <CalcNote>
              At the very first step m and v are 0, so <Tex>{"\\hat m/\\sqrt{\\hat v} = g/|g| = \\pm 1"}</Tex>: every
              weight moves by exactly the learning rate.
            </CalcNote>
          )}
        </CalcPanel>
      }
    >
      <div className="control-row">
        <ParamSelect refs={refs} value={pid} onChange={(id) => {
          setPid(id);
          setCell({ i: 0, j: 0 });
        }} />
        <button className="primary-btn" onClick={() => trainer.applyGradients(trace.grads, trace.loss)} disabled={trainer.running}>
          Apply this update (step {state.step} → {state.step + 1})
        </button>
      </div>
      <EqRow>
        <MatrixView data={theta} label={ref.label} size={size} highlight={hl} onHover={hover} />
        <Op tex="-" />
        <MatrixView data={delta} label="\Delta\theta" size={size} autoScale highlight={hl} onHover={hover} />
        <Op tex="=" />
        <MatrixView data={next} label={`${ref.label}{}'`} size={size} highlight={hl} onHover={hover} />
      </EqRow>
      <h3 className="viz-title">The gradient that drove it</h3>
      <MatrixView data={g} label={d(ref.label)} size={size} autoScale highlight={hl} onHover={hover} />
    </StepLayout>
  );
}

function probsOf(model: { config: ModelConfig; params: ModelParams }, ids: number[]): number[][] {
  return runForward(model, ids).logitsAll.map((z) => {
    const m = Math.max(...z);
    const e = z.map((v) => Math.exp(v - m));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
  });
}

export function TAfterStep() {
  const { trainer, trace, example, model } = useTraining();
  const { cfg, goTo, model: shipped } = useStep();
  const { tokens } = useTokens();
  const preview = useAdamPreview();
  const after = useMemo(() => ({ config: cfg, params: preview.params }), [cfg, preview]);
  const probsAfter = useMemo(() => probsOf(after, example.ids), [after, example]);
  const lossAfter = useMemo(() => batchLoss(after, [{ ids: example.ids, targets: example.targets, weight: 1 }]), [after, example]);
  const corpus = useMemo(() => corpusExamples(shipped), [shipped]);
  const corpusBefore = useMemo(() => batchLoss(model, corpus), [model, corpus]);
  const corpusAfter = useMemo(() => batchLoss(after, corpus), [after, corpus]);
  const apply = () => trainer.applyGradients(trace.grads, trace.loss);

  return (
    <StepLayout
      explain={
        <>
          <p>
            Here is what one update would do, computed by running the forward pass again with the updated weights{" "}
            <Tex>{"\\theta'"}</Tex>. On this sentence the loss should drop, and the probability of most correct next
            tokens should rise.
          </p>
          <p>
            The effect on the <em>whole corpus</em> can be smaller, or even negative: a step fitted to one sentence
            can hurt others. That is why real training averages the gradient over a batch of many sentences and takes
            many small steps. Apply the update and walk through the next step, or jump to the last page to watch
            thousands of steps.
          </p>
        </>
      }
      formula={
        <>
          <Tex display>{`L_{\\text{sentence}}:\\ ${tn(trace.loss)} \\to ${tn(lossAfter)}`}</Tex>
          <Tex display>{`L_{\\text{corpus}}:\\ ${tn(corpusBefore)} \\to ${tn(corpusAfter)}`}</Tex>
        </>
      }
      calc={
        <CalcPanel target="effect of one update">
          <CalcLine label="this sentence" tex={`\\Delta L = ${tn(lossAfter)} - ${tn(trace.loss)} = ${boxed(lossAfter - trace.loss)}`} />
          <CalcLine label="whole corpus" tex={`\\Delta L = ${tn(corpusAfter)} - ${tn(corpusBefore)} = ${boxed(corpusAfter - corpusBefore)}`} />
        </CalcPanel>
      }
    >
      <div className="control-row">
        <button className="primary-btn" onClick={apply} disabled={trainer.running}>
          Apply this update (step {trainer.state.step} → {trainer.state.step + 1})
        </button>
        <button
          className="small-btn"
          disabled={trainer.running}
          onClick={() => {
            apply();
            goTo("t-forward");
          }}
        >
          Apply and walk through the next training step
        </button>
        <button className="small-btn" onClick={() => goTo("t-dynamics")}>
          Watch thousands of steps →
        </button>
      </div>
      <table className="softmax-table">
        <thead>
          <tr>
            <th>i</th>
            <th>given</th>
            <th>target</th>
            <th>p before</th>
            <th>p after</th>
            <th>change</th>
          </tr>
        </thead>
        <tbody>
          {example.targets.map((t, i) => {
            const before = trace.probs[i][t];
            const aft = probsAfter[i][t];
            return (
              <tr key={i}>
                <td>{i}</td>
                <td className="tok-cell">{tokens.slice(0, i + 1).join(" ")}</td>
                <td className="tok-cell">
                  <strong>{cfg.vocab[t]}</strong>
                </td>
                <td>{before.toFixed(4)}</td>
                <td>{aft.toFixed(4)}</td>
                <td className={aft >= before ? "delta-up" : "delta-down"}>
                  {aft >= before ? "+" : "−"}
                  {Math.abs(aft - before).toFixed(4)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="viz-caption">
        Sentence: <Tok>{example.text}</Tok>
      </p>
    </StepLayout>
  );
}
