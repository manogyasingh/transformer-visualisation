import { useMemo, useState } from "react";
import { CalcLine, CalcNote, CalcPanel } from "../../components/calc-panel";
import { BarList, NumberLine } from "../../components/charts";
import { Tabs } from "../../components/layout";
import { StepLayout } from "../../components/step-layout";
import { Tex } from "../../components/tex";
import { boxed, dotExpansion, tn, tp } from "../../lib/format";
import { labelOf, scalarOutputs } from "../../model/reward-model";
import { usePpo } from "../../post-training/ppo-context";
import { Tok } from "../shared";
import { samplingSegments, TokenGrid, useTokenFocus, useWords } from "./shared";

const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);

export function PRolloutStep() {
  const { trace, models, trainer } = usePpo();
  const words = useWords();
  const [focus, setFocus] = useTokenFocus(() => 0);
  const r = trace.rollouts[focus.i];
  const s = r.sampling[focus.t];
  const a = r.response[focus.t];
  const vocab = models.cfg.vocab;
  const state = words([...r.prompt, ...r.response.slice(0, focus.t)]);
  const lower = s.cumulative[s.order.indexOf(a)] - s.filtered[a];
  const shown = s.order.filter((id) => s.probs[id] >= 1e-3).slice(0, 6);
  return (
    <StepLayout
      explain={
        <>
          <p>
            Every iteration collects fresh experience. For each of the {models.prompts.length} prompts the policy writes{" "}
            {trainer.settings.samplesPerPrompt} responses by plain sampling at temperature 1 (InstructGPT, appendix
            C.4), exactly as in the inference chapter. A response ends at the full stop (our end-of-text token) or
            when the sentence reaches {models.cfg.nCtx + 1} tokens, the most the policy can read and extend. The
            weights that write the batch are called <Tex>{"\\theta_{\\text{old}}"}</Tex>. They stay fixed for the
            whole iteration while a copy, <Tex>{"\\theta"}</Tex>, is updated.
          </p>
          <p>
            Each number below is the probability the policy gave the token it chose. Click a row to follow that
            response through the next pages (or pick it at the top), and hover a token to see how it was drawn.
          </p>
        </>
      }
      formula={
        <>
          <Tex display>{"y \\sim \\pi_{\\theta_{\\text{old}}}(\\cdot \\mid x)"}</Tex>
          <Tex display>{"a_t \\sim \\pi_{\\theta_{\\text{old}}}(\\cdot \\mid s_t),\\quad s_t = (x, a_0, \\dots, a_{t-1})"}</Tex>
        </>
      }
      calc={
        <CalcPanel
          target={
            <>
              response {focus.i + 1}, token {focus.t}: after <Tok>{state}</Tok>
            </>
          }
        >
          <CalcLine label="uniform draw" tex={`u = ${tn(s.u ?? 0)}`} />
          <CalcLine
            label="interval"
            tex={`u \\in [${tn(lower)},\\ ${tn(s.cumulative[s.order.indexOf(a)])})\\;\\Rightarrow\\; a_{${focus.t}} = \\texttt{${vocab[a]}},\\quad \\pi_{\\theta_{\\text{old}}}(a_{${focus.t}} \\mid s_{${focus.t}}) = ${boxed(s.probs[a])}`}
          />
          <CalcNote>
            Tokens are laid out on [0, 1) from most to least likely, each taking a length equal to its probability;
            the token whose segment contains u is the sample.
          </CalcNote>
        </CalcPanel>
      }
    >
      <TokenGrid
        value={(ro, _, t) => ro.sampling[t].probs[ro.response[t]]}
        scale="sequential"
        maxAbs={1}
        focus={focus}
        onFocus={setFocus}
        trailing={{ header: "finished?", cell: (ro) => (ro.ended ? "yes" : "no: hit the limit") }}
      />
      <div className="sample-detail">
        <div>
          <div className="arcs-title">
            Next-token distribution after <Tok>{state}</Tok>
          </div>
          <BarList
            items={shown.map((id) => ({ key: id, label: vocab[id], value: s.probs[id], focus: id === a }))}
            min={0}
            max={1}
            dp={3}
            color="#7c3aed"
            width={180}
          />
        </div>
        <div className="sample-line">
          <div className="arcs-title">The sample: where u falls on [0, 1)</div>
          <NumberLine segments={samplingSegments(s, vocab)} u={s.u} />
        </div>
      </div>
    </StepLayout>
  );
}

type LogpView = "kl" | "old" | "ref";

export function PLogprobsStep() {
  const { trace, trainer } = usePpo();
  const words = useWords();
  const [view, setView] = useState<LogpView>("kl");
  const [focus, setFocus] = useTokenFocus();
  const r = trace.rollouts[focus.i];
  const { t } = focus;
  const state = words([...r.prompt, ...r.response.slice(0, t)]);
  const atStart = trainer.state.iteration === 0;
  const value = {
    kl: (ro: typeof r, _: number, k: number) => ro.kl[k],
    old: (ro: typeof r, _: number, k: number) => ro.logpOld[k],
    ref: (ro: typeof r, _: number, k: number) => ro.logpRef[k],
  }[view];
  return (
    <StepLayout
      explain={
        <>
          <p>
            PPO needs two log-probabilities for every response token. The first is under the policy that wrote it,{" "}
            <Tex>{"\\log\\pi_{\\theta_{\\text{old}}}(a_t \\mid s_t)"}</Tex>. The second is under the frozen reference,{" "}
            <Tex>{"\\log\\pi_{\\text{ref}}(a_t \\mid s_t)"}</Tex>. Each takes one forward pass over the whole sentence
            (teacher forcing, as in pretraining), reading the log-softmax at the position before each token.
          </p>
          <p>
            Their difference <Tex>{"k_t"}</Tex> is a one-sample estimate of the KL divergence: averaged over
            responses, <Tex>{"\\sum_t k_t"}</Tex> equals{" "}
            <Tex>{"\\mathrm{KL}(\\pi_{\\theta_{\\text{old}}}(\\cdot\\mid x)\\,\\|\\,\\pi_{\\text{ref}}(\\cdot\\mid x))"}</Tex>{" "}
            (Ziegler et al., 2019). A single token's <Tex>{"k_t"}</Tex> can be negative, where the policy now finds
            the token less likely than the reference did.{" "}
            {atStart
              ? "At iteration 0 the policy is the reference, so every difference is exactly 0. Run a few iterations to watch them grow."
              : `After ${trainer.state.iteration} iterations the policy has drifted from the reference.`}
          </p>
        </>
      }
      formula={
        <>
          <Tex display>{"k_t = \\log\\pi_{\\theta_{\\text{old}}}(a_t\\mid s_t) - \\log\\pi_{\\text{ref}}(a_t\\mid s_t)"}</Tex>
          <Tex display>{"\\mathbb{E}_{y}\\Big[\\textstyle\\sum_t k_t\\Big] = \\mathrm{KL}\\big(\\pi_{\\theta_{\\text{old}}}\\,\\|\\,\\pi_{\\text{ref}}\\big)"}</Tex>
        </>
      }
      calc={
        <CalcPanel
          target={
            <>
              response {focus.i + 1}: <Tok>{words([r.response[t]])}</Tok> after <Tok>{state}</Tok>
            </>
          }
        >
          <CalcLine label="policy" tex={`\\log\\pi_{\\theta_{\\text{old}}} = \\ln ${tn(Math.exp(r.logpOld[t]))} = ${tn(r.logpOld[t])}`} />
          <CalcLine label="reference" tex={`\\log\\pi_{\\text{ref}} = \\ln ${tn(Math.exp(r.logpRef[t]))} = ${tn(r.logpRef[t])}`} />
          <CalcLine label="difference" tex={`k_{${t}} = ${tn(r.logpOld[t])} - ${tp(r.logpRef[t])} = ${boxed(r.kl[t])}`} />
        </CalcPanel>
      }
    >
      <Tabs
        value={view}
        onChange={setView}
        options={[
          { value: "kl", label: <Tex>{"k_t"}</Tex> },
          { value: "old", label: <Tex>{"\\log\\pi_{\\theta_{\\text{old}}}"}</Tex> },
          { value: "ref", label: <Tex>{"\\log\\pi_{\\text{ref}}"}</Tex> },
        ]}
      />
      <TokenGrid
        value={value}
        dp={view === "kl" ? 4 : 3}
        scale={view === "kl" ? "diverging" : "sequential"}
        maxAbs={view === "kl" && atStart ? 1 : undefined}
        focus={focus}
        onFocus={setFocus}
        trailing={{ header: <Tex>{"\\sum_t k_t"}</Tex>, cell: (ro) => sum(ro.kl).toFixed(4) }}
      />
    </StepLayout>
  );
}

export function PScoreStep() {
  const { trace, models, trainer } = usePpo();
  const words = useWords();
  const [i, setI] = useState<number | null>(null);
  const idx = Math.max(0, Math.min(i ?? trace.rollouts.findIndex((r) => r.ended), trace.rollouts.length - 1));
  const r = trace.rollouts[idx];
  const ids = [...r.prompt, ...r.response];
  const h = useMemo(() => {
    const out = scalarOutputs(models.scalarCfg, models.reward, ids);
    return out.forward.lnf.output[ids.length - 1];
  }, [models, ids.join(",")]);
  const { w, b } = models.reward.head;
  const e = dotExpansion(h, w);
  const mean = trace.rollouts.reduce((s, ro) => s + ro.score, 0) / trace.rollouts.length;
  return (
    <StepLayout
      explain={
        <>
          <p>
            Once a response is finished, the reward model reads the whole sentence, and its scalar head turns the
            final hidden state of the last token into one number. A response that hit the length limit without a full
            stop is not scored, because the reward model only learned to score finished sentences. It gets a fixed{" "}
            {trainer.settings.truncatedScore} instead (Ziegler et al., 2019; the “EOS trick” in Huang et al., 2024).
          </p>
          <p>
            The reward model stays frozen throughout RL. This score is the only way human preferences reach PPO, so
            the policy will learn whatever the reward model rewards, including its mistakes. The hidden labeler's
            verdict is shown for comparison. PPO never sees it.
          </p>
        </>
      }
      formula={
        <>
          <Tex display>{"r_\\phi(x, y) = h_{\\text{last}}\\cdot w_r + b_r"}</Tex>
          <div className="formula-caption">
            h = final LayerNorm output of the reward model at the full stop; batch mean {mean.toFixed(3)}
          </div>
        </>
      }
      calc={
        <CalcPanel target={<>response {idx + 1}: <Tok>{words(ids)}</Tok></>}>
          {r.ended ? (
            <>
              <CalcLine label="score" tex={`r_\\phi = \\sum_j h_j\\,(w_r)_j + b_r = ${e.factors} ${b[0] < 0 ? "-" : "+"} ${tn(Math.abs(b[0]))}`} />
              <CalcLine label="" tex={`= ${tn(e.value)} ${b[0] < 0 ? "-" : "+"} ${tn(Math.abs(b[0]))} = ${boxed(r.rmScore)}`} />
            </>
          ) : (
            <CalcLine label="unfinished" tex={`\\text{no full stop within ${models.cfg.nCtx + 1} tokens} \\Rightarrow \\text{score} = ${trainer.settings.truncatedScore}`} />
          )}
        </CalcPanel>
      }
    >
      <table className="softmax-table score-table">
        <thead>
          <tr>
            <th>#</th>
            <th>sentence</th>
            <th>labeler (hidden)</th>
            <th>score</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {trace.rollouts.map((ro, k) => {
            const label = labelOf([...ro.prompt, ...ro.response].map((id) => models.cfg.vocab[id]));
            return (
              <tr key={k} className={k === idx ? "focus" : ""} onMouseEnter={() => setI(k)}>
                <td>{k + 1}</td>
                <td className="tok-cell">
                  <span className="muted-tok">{words(ro.prompt)}</span> {words(ro.response)}
                </td>
                <td>
                  <span className={`label-chip ${label}`}>{label}</span>
                </td>
                <td>{ro.score.toFixed(4)}</td>
                <td className="prob-bar-cell">
                  <div className="score-bar">
                    <div
                      className={ro.score >= 0 ? "score-fill pos" : "score-fill neg"}
                      style={
                        ro.score >= 0
                          ? { left: "50%", width: `${Math.min(1, ro.score / 2) * 50}%` }
                          : { right: "50%", width: `${Math.min(1, -ro.score / 2) * 50}%` }
                      }
                    />
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </StepLayout>
  );
}

export function PRewardsStep() {
  const { trace, trainer } = usePpo();
  const words = useWords();
  const [focus, setFocus] = useTokenFocus();
  const r = trace.rollouts[focus.i];
  const { t } = focus;
  const T = r.response.length;
  const beta = trainer.settings.beta;
  const last = t === T - 1;
  return (
    <StepLayout
      explain={
        <>
          <p>
            The score arrives only at the end, but the KL penalty is charged on every token:{" "}
            <Tex>{"r_t = -\\beta k_t"}</Tex>, and the last token also receives the score. Summed over a response, this
            is exactly the quantity inside InstructGPT's objective for that sample. In the numbers below the score
            dominates the last token of each response, and the penalties are the small amounts elsewhere.
          </p>
          <p>
            The penalty keeps the policy close to the reference. That stops it collapsing onto the single sentence
            the reward model likes best (in effect an entropy bonus), and keeps it among sentences like the ones the
            reward model was trained on (Stiennon et al., 2020). Here <Tex>{`\\beta = ${beta}`}</Tex>. InstructGPT
            used 0.02 on responses hundreds of tokens long.
          </p>
        </>
      }
      formula={
        <>
          <Tex display>{"r_t = -\\beta\\,k_t + [t = T-1]\\;r_\\phi(x, y)"}</Tex>
          <Tex display>{"\\sum_t r_t = r_\\phi(x,y) - \\beta\\log\\frac{\\pi_{\\theta_{\\text{old}}}(y\\mid x)}{\\pi_{\\text{ref}}(y\\mid x)}"}</Tex>
        </>
      }
      calc={
        <CalcPanel target={<>response {focus.i + 1}, token {t} (<Tok>{words([r.response[t]])}</Tok>)</>}>
          <CalcLine
            label="reward"
            tex={`r_{${t}} = -${beta}\\cdot${tp(r.kl[t])}${last ? ` + ${tp(r.score)}` : ""} = ${boxed(r.rewards[t])}`}
          />
          <CalcLine label="response total" tex={`\\textstyle\\sum_t r_t = ${tn(sum(r.rewards))}`} />
          {last && <CalcNote>This is the last token, so it also carries the score of the whole response.</CalcNote>}
        </CalcPanel>
      }
    >
      <TokenGrid
        value={(ro, _, k) => ro.rewards[k]}
        focus={focus}
        onFocus={setFocus}
        trailing={{ header: <Tex>{"\\sum_t r_t"}</Tex>, cell: (ro) => sum(ro.rewards).toFixed(4) }}
      />
    </StepLayout>
  );
}
