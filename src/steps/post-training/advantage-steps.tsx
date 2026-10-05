import { useMemo } from "react";
import { CalcLine, CalcNote, CalcPanel } from "../../components/calc-panel";
import { StepLayout } from "../../components/step-layout";
import { Tex } from "../../components/tex";
import { boxed, dotExpansion, ta, tn, tp } from "../../lib/format";
import { scalarOutputs } from "../../model/reward-model";
import { usePpo } from "../../post-training/ppo-context";
import { inputsOf, positionOf, WHITEN_EPS } from "../../post-training/ppo";
import { Tok } from "../shared";
import { TokenGrid, useTokenFocus, useWords } from "./shared";

export function PValuesStep() {
  const { trace, models, trainer } = usePpo();
  const words = useWords();
  const [focus, setFocus] = useTokenFocus(() => 0);
  const r = trace.rollouts[focus.i];
  const { t } = focus;
  const value = trace.before.value.params;
  const inputs = inputsOf(r);
  const pos = positionOf(r, t);
  const h = useMemo(
    () => scalarOutputs(models.scalarCfg, value, inputs).forward.lnf.output[pos],
    [models, value, inputs.join(","), pos],
  );
  const e = dotExpansion(h, value.head.w);
  const b = value.head.b[0];
  const state = words([...r.prompt, ...r.response.slice(0, t)]);
  const atStart = trainer.state.iteration === 0;
  return (
    <StepLayout
      explain={
        <>
          <p>
            Was a token a good choice? That depends on what was expected at that point. The value model estimates{" "}
            <Tex>{"V(s_t)"}</Tex>, the sum of the rewards still to come from state <Tex>{"s_t"}</Tex>. It is a separate
            transformer, initialised from the reward model as in Stiennon et al. and InstructGPT, with its scalar head
            read at every position. <Tex>{"V(s_t)"}</Tex> is the output at the last token of <Tex>{"s_t"}</Tex>, the
            position just before <Tex>{"a_t"}</Tex> is chosen. Each number below is shown under the token about to
            be chosen.
          </p>
          <p>
            {atStart
              ? "Right now it is badly calibrated. The reward model only ever scored finished sentences, so an unfinished one such as “the” looks like nonsense to it, and its values for early states sit far below the rewards that actually follow. Huang et al. (2024) saw the same thing at scale. The value loss fixes this over the iterations."
              : `It has been trained for ${trainer.state.iteration} iterations of the value loss and now tracks the returns much better than at the start, when it was just the reward model.`}
          </p>
        </>
      }
      formula={
        <>
          <Tex display>{"V_\\psi(s_t) = h_t\\cdot w_V + b_V"}</Tex>
          <Tex display>{"V_\\psi(s_t) \\approx \\mathbb{E}\\Big[\\textstyle\\sum_{l \\ge t} r_l \\;\\Big|\\; s_t\\Big]"}</Tex>
        </>
      }
      calc={
        <CalcPanel target={<>value of the state <Tok>{state}</Tok> (response {focus.i + 1}, before token {t})</>}>
          <CalcLine label="value" tex={`V(s_{${t}}) = \\sum_j h_j\\,(w_V)_j + b_V = ${e.factors} ${b < 0 ? "-" : "+"} ${tn(Math.abs(b))}`} />
          <CalcLine label="" tex={`= ${boxed(r.values[t])}`} />
          <CalcNote>
            For comparison, this response's rewards from here on sum to{" "}
            {r.rewards.slice(t).reduce((s, x) => s + x, 0).toFixed(4)}.
          </CalcNote>
        </CalcPanel>
      }
    >
      <TokenGrid
        value={(ro, _, k) => ro.values[k]}
        focus={focus}
        onFocus={setFocus}
        trailing={{ header: "score", cell: (ro) => ro.score.toFixed(3) }}
      />
    </StepLayout>
  );
}

export function PGaeStep() {
  const { trace, trainer } = usePpo();
  const words = useWords();
  const [focus, setFocus] = useTokenFocus(() => 0);
  const r = trace.rollouts[focus.i];
  const { t } = focus;
  const T = r.response.length;
  const { gamma, lambda } = trainer.settings;
  const vNext = t + 1 < T ? r.values[t + 1] : 0;
  const aNext = t + 1 < T ? r.advantages[t + 1] : 0;
  return (
    <StepLayout
      explain={
        <>
          <p>
            The TD error <Tex>{"\\delta_t = r_t + \\gamma V(s_{t+1}) - V(s_t)"}</Tex> compares what happened after
            choosing <Tex>{"a_t"}</Tex> (the reward, plus the value of the state it led to) with what the value
            model expected beforehand. Generalized advantage estimation (Schulman et al., 2016) blends these into{" "}
            <Tex>{"\\hat A_t = \\sum_l (\\gamma\\lambda)^l\\,\\delta_{t+l}"}</Tex>, computed in one backward sweep.
            With <Tex>{"\\lambda = 0"}</Tex> it trusts the value model completely: low variance, but biased when the
            value model is wrong. With <Tex>{"\\lambda = 1"}</Tex> it is the actual sum of rewards minus{" "}
            <Tex>{"V(s_t)"}</Tex>: unbiased, but noisy.
          </p>
          <p>
            RLHF uses <Tex>{`\\gamma = ${gamma}`}</Tex> (no discounting; InstructGPT) and{" "}
            <Tex>{`\\lambda = ${lambda}`}</Tex>. After the last token the episode is over, so{" "}
            <Tex>{"V(s_T) = 0"}</Tex>. The returns <Tex>{"R_t = \\hat A_t + V(s_t)"}</Tex> become the value model's
            targets.
          </p>
        </>
      }
      formula={
        <>
          <Tex display>{"\\delta_t = r_t + \\gamma V(s_{t+1}) - V(s_t)"}</Tex>
          <Tex display>{"\\hat A_t = \\delta_t + \\gamma\\lambda\\,\\hat A_{t+1},\\quad \\hat A_T = 0"}</Tex>
          <Tex display>{"R_t = \\hat A_t + V(s_t)"}</Tex>
        </>
      }
      calc={
        <CalcPanel target={<>response {focus.i + 1}, token {t} (<Tok>{words([r.response[t]])}</Tok>)</>}>
          <CalcLine
            label="TD error"
            tex={`\\delta_{${t}} = ${tn(r.rewards[t])} + ${gamma}\\cdot${tp(vNext)} - ${tp(r.values[t])} = ${tn(r.deltas[t])}`}
          />
          <CalcLine
            label="advantage"
            tex={`\\hat A_{${t}} = ${tn(r.deltas[t])} + ${(gamma * lambda).toFixed(2)}\\cdot${tp(aNext)} = ${boxed(r.advantages[t])}`}
          />
          <CalcLine label="return" tex={`R_{${t}} = ${tn(r.advantages[t])} + ${ta(r.values[t])} = ${tn(r.returns[t])}`} />
          {t === T - 1 && <CalcNote>The last token: the next state is terminal, so V(s_{t+1}) = 0 and there is no later advantage.</CalcNote>}
        </CalcPanel>
      }
    >
      <h3 className="viz-title">
        Response {focus.i + 1}: <Tok>{words(r.prompt)}</Tok> → <Tok>{words(r.response)}</Tok> (computed bottom-up)
      </h3>
      <table className="softmax-table gae-table">
        <thead>
          <tr>
            <th>t</th>
            <th>token</th>
            <th>
              <Tex>{"r_t"}</Tex>
            </th>
            <th>
              <Tex>{"V(s_t)"}</Tex>
            </th>
            <th>
              <Tex>{"V(s_{t+1})"}</Tex>
            </th>
            <th>
              <Tex>{"\\delta_t"}</Tex>
            </th>
            <th>
              <Tex>{"\\hat A_t"}</Tex>
            </th>
            <th>
              <Tex>{"R_t"}</Tex>
            </th>
          </tr>
        </thead>
        <tbody>
          {r.response.map((a, k) => (
            <tr key={k} className={k === t ? "focus" : ""} onMouseEnter={() => setFocus({ i: focus.i, t: k })}>
              <td>{k}</td>
              <td className="tok-cell">{words([a])}</td>
              <td>{r.rewards[k].toFixed(4)}</td>
              <td>{r.values[k].toFixed(4)}</td>
              <td>{(k + 1 < T ? r.values[k + 1] : 0).toFixed(4)}</td>
              <td>{r.deltas[k].toFixed(4)}</td>
              <td>
                <strong>{r.advantages[k].toFixed(4)}</strong>
              </td>
              <td>{r.returns[k].toFixed(4)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <h3 className="viz-title">
        Advantages <Tex>{"\\hat A_t"}</Tex> of the whole batch
      </h3>
      <TokenGrid value={(ro, _, k) => ro.advantages[k]} focus={focus} onFocus={setFocus} />
    </StepLayout>
  );
}

export function PWhitenStep() {
  const { trace } = usePpo();
  const words = useWords();
  const [focus, setFocus] = useTokenFocus(() => 0);
  const r = trace.rollouts[focus.i];
  const { t } = focus;
  const N = trace.rollouts.reduce((s, ro) => s + ro.response.length, 0);
  const { advMean: mu, advStd: sd } = trace;
  return (
    <StepLayout
      explain={
        <>
          <p>
            Before they are used, the advantages are standardised over all {N} tokens of the batch: subtract the
            mean, divide by the standard deviation (Huang et al., 2024, detail 25, following Ziegler et al.'s code).
            The size of the update then no longer depends on the scale of the rewards. And “better than expected”
            becomes “better than the rest of the batch”, so roughly half the tokens will be pushed up and half down.
          </p>
          <p>
            The mean here is <Tex>{tn(mu, 3)}</Tex>.{" "}
            {mu > 0.3
              ? "It is well above 0 because the value model still underestimates the returns. Whitening removes that bias from the policy update."
              : "It is close to 0 because the value model has learned to predict the returns."}
          </p>
        </>
      }
      formula={
        <>
          <Tex display>{"\\mu = \\frac{1}{N}\\sum \\hat A_t,\\quad \\sigma^2 = \\frac{1}{N}\\sum(\\hat A_t - \\mu)^2"}</Tex>
          <Tex display>{"\\hat A_t \\leftarrow \\frac{\\hat A_t - \\mu}{\\sqrt{\\sigma^2 + 10^{-8}}}"}</Tex>
          <div className="formula-caption">N = {N} response tokens in the batch</div>
        </>
      }
      calc={
        <CalcPanel target={<>response {focus.i + 1}, token {t} (<Tok>{words([r.response[t]])}</Tok>)</>}>
          <CalcLine label="batch statistics" tex={`\\mu = ${tn(mu)},\\quad \\sqrt{\\sigma^2 + 10^{-8}} = ${tn(sd)}`} />
          <CalcLine label="whitened" tex={`\\frac{${tn(r.advantages[t])} - ${tp(mu)}}{${tn(sd)}} = ${boxed(r.adv[t])}`} />
          <CalcNote>ε = {WHITEN_EPS} only guards against a batch where every advantage is equal.</CalcNote>
        </CalcPanel>
      }
    >
      <TokenGrid value={(ro, _, k) => ro.adv[k]} focus={focus} onFocus={setFocus} />
      <p className="viz-caption">
        Orange tokens will be made more likely, blue ones less likely. The strength depends on the ratio and the
        clip, next.
      </p>
    </StepLayout>
  );
}
