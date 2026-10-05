import { useMemo, useState } from "react";
import { CalcLine, CalcNote, CalcPanel } from "../../components/calc-panel";
import { EqRow, Op } from "../../components/layout";
import { MatrixView } from "../../components/matrix-view";
import { StepLayout } from "../../components/step-layout";
import { Tex } from "../../components/tex";
import { boxed, fmt, tg, tn, tp } from "../../lib/format";
import { softmaxRows } from "../../model/backward";
import { runForward } from "../../model/forward";
import { encode } from "../../model/tiny-gpt";
import type { ModelConfig, ModelParams } from "../../model/types";
import { usePpo } from "../../post-training/ppo-context";
import { evaluatePolicy, positionOf, type PolicyEvaluation } from "../../post-training/ppo";
import { useStep } from "../step-context";
import { Tok } from "../shared";
import { EpochTabs, TokenGrid, useTokenFocus, useWords } from "./shared";

function ClipPlot({ A, ratio, eps }: { A: number; ratio: number; eps: number }) {
  const W = 300;
  const H = 210;
  const pad = 30;
  const lo = Math.min(0.5, ratio - 0.1);
  const hi = Math.max(1.5, ratio + 0.1);
  const f = (r: number) => Math.min(r * A, Math.min(Math.max(r, 1 - eps), 1 + eps) * A);
  const ys = [lo * A, hi * A, 0];
  const yLo = Math.min(...ys);
  const yHi = Math.max(...ys);
  const sx = (r: number) => pad + ((r - lo) / (hi - lo)) * (W - 2 * pad);
  const sy = (y: number) => H - pad - ((y - yLo) / (yHi - yLo || 1)) * (H - 2 * pad);
  const line = (g: (r: number) => number) =>
    Array.from({ length: 121 }, (_, k) => lo + ((hi - lo) * k) / 120)
      .map((r, k) => `${k ? "L" : "M"}${sx(r).toFixed(1)},${sy(g(r)).toFixed(1)}`)
      .join(" ");
  return (
    <svg className="clip-plot" width={W} height={H}>
      <rect x={sx(1 - eps)} y={pad / 2} width={sx(1 + eps) - sx(1 - eps)} height={H - pad - pad / 2} className="clip-band" />
      <line x1={pad} x2={W - pad} y1={sy(0)} y2={sy(0)} className="axis" />
      <path d={line((r) => r * A)} className="clip-unclipped" />
      <path d={line(f)} className="clip-objective" />
      {[1 - eps, 1, 1 + eps].map((r) => (
        <text key={r} x={sx(r)} y={H - 10} textAnchor="middle" className="tick">
          {r.toFixed(1)}
        </text>
      ))}
      <circle cx={sx(ratio)} cy={sy(f(ratio))} r={6} className="gelu-focus" />
      <text x={W - pad} y={pad / 2 + 10} textAnchor="end" className="gelu-label">
        Â = {fmt(A, 3)}, ρ = {ratio.toFixed(4)}
      </text>
      <text x={W - pad} y={H - 10} textAnchor="end" className="tick">
        ρ
      </text>
    </svg>
  );
}

export function PRatioStep() {
  const { trace, trainer } = usePpo();
  const words = useWords();
  const [epoch, setEpoch] = useState(trainer.settings.epochs - 1);
  const e = Math.min(epoch, trace.epochs.length - 1);
  const [focus, setFocus] = useTokenFocus(() => 0);
  const r = trace.rollouts[focus.i];
  const { t } = focus;
  const tok = trace.epochs[e].policy.tokens[focus.i][t];
  const eps = trainer.settings.clip;
  const A = r.adv[t];
  const clippedRatio = Math.min(Math.max(tok.ratio, 1 - eps), 1 + eps);
  const nClipped = trace.epochs[e].policy.tokens.flat().filter((x) => !x.flows).length;
  return (
    <StepLayout
      explain={
        <>
          <p>
            The vanilla policy gradient raises <Tex>{"\\log\\pi_\\theta(a_t\\mid s_t)"}</Tex> in proportion to{" "}
            <Tex>{"\\hat A_t"}</Tex>. PPO gets more out of each batch by taking several gradient steps on it, but after
            the first step the policy is no longer the one that wrote the data. The probability ratio{" "}
            <Tex>{"\\rho_t = \\pi_\\theta(a_t\\mid s_t)/\\pi_{\\theta_{\\text{old}}}(a_t\\mid s_t)"}</Tex> corrects for
            that (importance sampling), and the clip keeps the new policy near the old one (Schulman et al., 2017,
            eq. 7). Once a token's ratio has risen past <Tex>{"1+\\epsilon"}</Tex> (positive advantage) or fallen
            below <Tex>{"1-\\epsilon"}</Tex> (negative advantage), the clipped term is the smaller one. That token's
            gradient is then zero, so there is no incentive to move it further. If an update made a token worse, the
            min keeps the unclipped term, so the mistake is still corrected.
          </p>
          <p>
            In epoch 1 <Tex>{"\\theta = \\theta_{\\text{old}}"}</Tex>, so every ratio is exactly 1 and PPO's first
            step is a plain policy-gradient step. Pick a later epoch to watch the ratios drift.{" "}
            {nClipped > 0
              ? `In epoch ${e + 1}, ${nClipped} token${nClipped === 1 ? " is" : "s are"} clipped (greyed out): they contribute no gradient.`
              : `No token is clipped in epoch ${e + 1}.`}
          </p>
        </>
      }
      formula={
        <>
          <Tex display>{"\\rho_t(\\theta) = \\exp\\big(\\log\\pi_\\theta(a_t\\mid s_t) - \\log\\pi_{\\theta_{\\text{old}}}(a_t\\mid s_t)\\big)"}</Tex>
          <Tex display>{"L^{\\text{CLIP}}(\\theta) = \\frac{1}{N}\\sum_t \\min\\big(\\rho_t\\hat A_t,\\ \\mathrm{clip}(\\rho_t, 1-\\epsilon, 1+\\epsilon)\\,\\hat A_t\\big)"}</Tex>
          <div className="formula-caption">
            <Tex>{`\\epsilon = ${eps}`}</Tex>; PPO maximises <Tex>{"L^{\\text{CLIP}}"}</Tex>
          </div>
        </>
      }
      calc={
        <CalcPanel target={<>epoch {e + 1}, response {focus.i + 1}, token {t} (<Tok>{words([r.response[t]])}</Tok>)</>}>
          <CalcLine label="ratio" tex={`\\rho = e^{${tn(tok.logp)} - ${tp(r.logpOld[t])}} = ${tn(tok.ratio)}`} />
          <CalcLine label="unclipped" tex={`\\rho\\hat A = ${tn(tok.ratio)}\\cdot${tp(A)} = ${tn(-tok.unclipped)}`} />
          <CalcLine label="clipped" tex={`\\mathrm{clip}(\\rho)\\,\\hat A = ${tn(clippedRatio)}\\cdot${tp(A)} = ${tn(-tok.clipped)}`} />
          <CalcLine label="objective" tex={`\\min(${tn(-tok.unclipped)},\\ ${tn(-tok.clipped)}) = ${boxed(-tok.loss)}`} />
          <CalcNote>
            {tok.flows
              ? "The unclipped term is the smaller (or they are equal), so this token's gradient flows."
              : `The clipped term is smaller: ρ has already moved past 1 ${A > 0 ? "+" : "−"} ε in the direction the advantage wants, so this token gets zero gradient.`}
          </CalcNote>
        </CalcPanel>
      }
    >
      <EpochTabs epoch={e} onChange={setEpoch} />
      <div className="ratio-layout">
        <TokenGrid
          value={(_, i, k) => trace.epochs[e].policy.tokens[i][k].ratio}
          dp={4}
          center={1}
          maxAbs={eps}
          focus={focus}
          onFocus={setFocus}
          dim={(_, i, k) => !trace.epochs[e].policy.tokens[i][k].flows}
        />
        <div>
          <div className="arcs-title">This token's term of the objective, as a function of ρ</div>
          <ClipPlot A={A} ratio={tok.ratio} eps={eps} />
          <p className="viz-caption clip-caption">
            Solid: min(ρÂ, clip(ρ)Â). Dashed: ρÂ. Shaded: [1 − ε, 1 + ε]. Outside the shaded band the solid line is flat on
            the side the advantage favours.
          </p>
        </div>
      </div>
    </StepLayout>
  );
}

export function PPolicyGradStep() {
  const { trace, trainer, rolloutIndex, models } = usePpo();
  const { goTo } = useStep();
  const words = useWords();
  const [epoch, setEpoch] = useState(0);
  const e = Math.min(epoch, trace.epochs.length - 1);
  const r = trace.rollouts[rolloutIndex];
  const T = r.response.length;
  const N = trace.rollouts.reduce((s, ro) => s + ro.response.length, 0);
  const toks = trace.epochs[e].policy.tokens[rolloutIndex];
  const P = toks.map((x) => x.probs);
  const dZ = r.response.map((_, t) => trace.epochs[e].policy.dLogits![rolloutIndex][positionOf(r, t)]);
  const [cell, setCell] = useState({ t: 0, v: r.response[0] });
  const t = Math.min(cell.t, T - 1);
  const v = cell.v;
  const a = r.response[t];
  const tok = toks[t];
  const vocab = models.cfg.vocab;
  const rowLabels = r.response.map((id, k) => (
    <span className="rl" key={k}>
      <span className="rl-pos">{k}</span>
      <span className="rl-tok">→ {vocab[id]}</span>
    </span>
  ));
  const chosen = r.response.map((id, k) => [k, id] as [number, number]);
  const hover = (k: number, u: number) => setCell({ t: k, v: u });
  return (
    <StepLayout
      explain={
        <>
          <p>
            As a loss to minimise, <Tex>{"L = -L^{\\text{CLIP}}"}</Tex>. For a token whose gradient flows,{" "}
            <Tex>{"\\partial L/\\partial\\log\\pi(a_t\\mid s_t) = -\\rho_t\\hat A_t/N"}</Tex>. For a clipped token it is 0. And
            because <Tex>{"\\log\\pi(a\\mid s) = z_a - \\log\\sum_v e^{z_v}"}</Tex>, the derivative with respect to
            logit v is <Tex>{"[v = a] - p_v"}</Tex>. So each row of the logit gradient is the chosen token's one-hot
            vector minus the probabilities, scaled by <Tex>{"-\\rho_t\\hat A_t/N"}</Tex>.
          </p>
          <p>
            Compare pretraining, where <Tex>{"\\delta Z = (P - Y)/T"}</Tex>. Pretraining is the special case{" "}
            <Tex>{"\\hat A_t = 1,\\ \\rho_t = 1"}</Tex>: it raises every token of the corpus. PPO raises tokens with
            positive advantage and lowers those with negative advantage. From here on backpropagation is exactly the
            pretraining chapter's, through the unembedding, every block and the embeddings. The next page adds a
            second gradient to the result.
          </p>
        </>
      }
      formula={
        <>
          <Tex display>{"\\frac{\\partial L}{\\partial z_{t,v}} = -\\frac{\\rho_t\\hat A_t}{N}\\big([v = a_t] - p_{t,v}\\big)"}</Tex>
          <div className="formula-caption">0 for a clipped token; N = {N} tokens in the batch</div>
        </>
      }
      deeper={
        <p>
          The policy shares nothing with the value model, so this gradient only reaches the policy's weights.
          Stiennon et al. tried sharing a network between the policy and the value function and found that value
          updates damaged the pretrained policy early in training.{" "}
          <button className="link-btn" onClick={() => goTo("t-dlogits")}>
            See the rest of the backward pass in the pretraining chapter
          </button>
          .
        </p>
      }
      calc={
        <CalcPanel target={<>epoch {e + 1}, response {rolloutIndex + 1}: logit of <Tok>{vocab[v]}</Tok> after <Tok>{words([...r.prompt, ...r.response.slice(0, t)])}</Tok></>}>
          {tok.flows ? (
            <CalcLine
              label="gradient"
              tex={`\\frac{\\partial L}{\\partial z} = -\\frac{${tn(tok.ratio)}\\cdot${tp(r.adv[t])}}{${N}}\\big(${v === a ? 1 : 0} - ${tn(tok.probs[v])}\\big) = ${boxed(dZ[t][v])}`}
            />
          ) : (
            <CalcLine label="clipped" tex={`\\rho_{${t}} = ${tn(tok.ratio)} \\text{ is outside the clip range on the advantage's side} \\Rightarrow \\frac{\\partial L}{\\partial z} = 0`} />
          )}
          <CalcNote>
            {v === a
              ? r.adv[t] > 0
                ? "The chosen token, with positive advantage: a negative gradient, so gradient descent raises its logit."
                : "The chosen token, with negative advantage: a positive gradient, so gradient descent lowers its logit."
              : `Not the chosen token (“${vocab[a]}”): it moves the opposite way to the chosen one, in proportion to its probability.`}
          </CalcNote>
        </CalcPanel>
      }
    >
      <div className="control-row">
        <EpochTabs epoch={e} onChange={setEpoch} />
        <span className="muted">
          Response {rolloutIndex + 1}: <Tok>{words(r.prompt)}</Tok> → <Tok>{words(r.response)}</Tok> (pick another at the top)
        </span>
      </div>
      <div className="stack-viz">
        <MatrixView
          data={P}
          label="P"
          note="policy probabilities at each state"
          rowLabels={rowLabels}
          colLabels={vocab.map((w) => <span key={w}>{w}</span>)}
          verticalColLabels
          size="medium"
          scale="sequential"
          maxAbs={1}
          highlight={{ cells: chosen, focus: [t, v] }}
          onHover={hover}
        />
        <div className="arch-down">↓ one-hot of the chosen token minus P, times −ρÂ/N (row by row)</div>
        <MatrixView
          data={dZ}
          label="\partial L/\partial Z"
          rowLabels={rowLabels}
          size="medium"
          autoScale
          highlight={{ cells: chosen, focus: [t, v] }}
          dimRow={(k) => !toks[k].flows}
          onHover={hover}
        />
      </div>
      <p className="viz-caption">
        Whitened advantages of this response: {r.adv.map((x) => x.toFixed(3)).join(", ")}. Greyed rows are clipped.{" "}
        {trainer.state.iteration === 0 && e === 0 ? "In epoch 1 no row can be clipped, because every ratio is 1." : ""}
      </p>
    </StepLayout>
  );
}

export function PPtxStep() {
  const { trace, trainer, models } = usePpo();
  const { model } = useStep();
  const words = useWords();
  const [epoch, setEpoch] = useState(0);
  const e = Math.min(epoch, trace.epochs.length - 1);
  const ep = trace.epochs[e];
  const gamma = trainer.settings.ptxCoef;
  const vocab = models.cfg.vocab;
  const [cell, setCell] = useState({ i: vocab.indexOf("the"), j: 0 });
  const hl = { focus: [cell.i, cell.j] as [number, number] };
  const hover = (i: number, j: number) => setCell({ i, j });
  const rowLabels = vocab.map((w, v) => (
    <span className="rl" key={v}>
      <span className="rl-pos">{v}</span>
      <span className="rl-tok">{w}</span>
    </span>
  ));
  const gPpo = ep.policy.grads!.wte;
  const gPtx = ep.ptx?.grads.wte.map((r) => r.map((x) => gamma * x));
  const norm = (m: number[][]) => Math.sqrt(m.flat().reduce((s, x) => s + x * x, 0));
  return (
    <StepLayout
      explain={
        <>
          <p>
            RLHF has a cost InstructGPT calls the <strong>alignment tax</strong>. As PPO pushes the policy toward
            what the reward model likes, it gets worse at everything nobody is rewarding: InstructGPT regressed on
            public NLP benchmarks. Their fix, <strong>PPO-ptx</strong>, adds the ordinary pretraining loss on a few
            pretraining sentences to every policy update, weighted by <Tex>{"\\gamma"}</Tex>. The models the
            paper calls InstructGPT are PPO-ptx models.
          </p>
          <p>
            Our tiny model pays the tax in a vivid form. It is nearly certain about most next tokens (“ate” is always
            followed by “the”). Those tokens give PPO almost no gradient, and the KL penalty only sees tokens that
            were sampled. Meanwhile every update to the shared weights nudges every logit a little, so within a few
            dozen iterations the logit of some never-sampled word after “ate” can creep up from{" "}
            <Tex>{"e^{-15}"}</Tex> until it does get sampled. The policy then suddenly writes things like{" "}
            <Tok>the cat ate cat sat on on</Tok>. The pretraining loss keeps exactly those predictions anchored. Here{" "}
            <Tex>{`\\gamma = ${gamma}`}</Tex>, with {trainer.settings.ptxBatch} corpus sentences per epoch. Set{" "}
            <Tex>{"\\gamma = 0"}</Tex> on the last page to watch the drift happen.
          </p>
        </>
      }
      formula={
        <>
          <Tex display>{"L^{\\text{ptx}}(\\theta) = -\\frac{1}{M}\\sum_i \\log\\pi_\\theta(t_{i+1}\\mid t_0,\\dots,t_i)"}</Tex>
          <Tex display>{"g = \\nabla_\\theta\\big(-L^{\\text{CLIP}}\\big) + \\gamma\\,\\nabla_\\theta L^{\\text{ptx}}"}</Tex>
          <div className="formula-caption">
            the pretraining chapter's loss, on M target tokens of the minibatch; InstructGPT eq. 2 has the same{" "}
            <Tex>{"\\gamma"}</Tex>
          </div>
        </>
      }
      calc={
        gPtx && (
          <CalcPanel target={<>gradient of <Tex>{"W_E"}</Tex> entry ({cell.i}, {cell.j}) (<Tok>{vocab[cell.i]}</Tok>), epoch {e + 1}</>}>
            <CalcLine
              label="total"
              tex={`g = ${tg(gPpo[cell.i][cell.j])} + ${gamma}\\cdot${tp(ep.ptx!.grads.wte[cell.i][cell.j], 6)} = ${boxed(ep.policyGrads.wte[cell.i][cell.j], 6)}`}
            />
            <CalcNote>
              Over the whole of <Tex>{"W_E"}</Tex>: ‖PPO part‖ = {norm(gPpo).toExponential(2)}, ‖γ × pretraining part‖ ={" "}
              {norm(gPtx).toExponential(2)}.
            </CalcNote>
          </CalcPanel>
        )
      }
    >
      <div className="control-row">
        <EpochTabs epoch={e} onChange={setEpoch} />
        {ep.ptx ? (
          <span className="muted">
            pretraining loss of this minibatch: {ep.ptx.loss.toFixed(4)} (the pretrained model's loss on the whole corpus is{" "}
            {model.training.roundedLoss.toFixed(3)})
          </span>
        ) : (
          <span className="muted">γ = 0: plain PPO, no pretraining term</span>
        )}
      </div>
      {ep.ptx && gPtx ? (
        <>
          <div className="arcs-title">This epoch's pretraining minibatch (drawn in proportion to corpus counts)</div>
          <ul className="ptx-batch">
            {ep.ptx.batch.map((b, k) => (
              <li key={k}>
                <code>{words([...b.ids, b.targets[b.targets.length - 1]])}</code>
              </li>
            ))}
          </ul>
          <h3 className="viz-title">
            The two gradients of <Tex>{"W_E"}</Tex>, and the sum the Adam step uses
          </h3>
          <EqRow>
            <MatrixView data={gPpo} label="\nabla(-L^{\text{CLIP}})" rowLabels={rowLabels} size="medium" autoScale highlight={hl} onHover={hover} />
            <Op tex="+" />
            <MatrixView data={gPtx} label="\gamma\,\nabla L^{\text{ptx}}" size="medium" autoScale highlight={hl} onHover={hover} />
            <Op tex="=" />
            <MatrixView data={ep.policyGrads.wte} label="g" size="medium" autoScale highlight={hl} onHover={hover} />
          </EqRow>
        </>
      ) : null}
    </StepLayout>
  );
}

export function PValueLossStep() {
  const { trace, trainer } = usePpo();
  const words = useWords();
  const [epoch, setEpoch] = useState(0);
  const e = Math.min(epoch, trace.epochs.length - 1);
  const [focus, setFocus] = useTokenFocus(() => 0);
  const r = trace.rollouts[focus.i];
  const { t } = focus;
  const N = trace.rollouts.reduce((s, ro) => s + ro.response.length, 0);
  const vt = trace.epochs[e].value.tokens;
  const tok = vt[focus.i][t];
  const R = r.returns[t];
  const ve = trainer.settings.valueClip;
  return (
    <StepLayout
      explain={
        <>
          <p>
            The value model regresses onto the returns <Tex>{"R_t"}</Tex> from GAE. Like the policy, it should not
            move too far on a single batch. So the error is computed twice, once for the new value and once for a
            clipped value <Tex>{`V_{\\text{old}} + \\mathrm{clip}(V - V_{\\text{old}}, -${ve}, ${ve})`}</Tex>, and the
            larger error is used (Ziegler et al.'s implementation; Huang et al., 2024, table 7). Once the clipped
            error is the larger one, the token stops contributing gradient.
          </p>
          <p>
            The gradient <Tex>{"\\partial L/\\partial V_t = (V_t - R_t)/N"}</Tex> enters at the scalar head and
            backpropagates through the value transformer, just as the cross-entropy gradient does in pretraining. The
            policy and the value model have separate weights and separate Adam optimisers, so this loss never touches
            the policy.
          </p>
        </>
      }
      formula={
        <>
          <Tex display>{"V^{\\text{clip}}_t = V^{\\text{old}}_t + \\mathrm{clip}\\big(V_t - V^{\\text{old}}_t, -\\epsilon_V, \\epsilon_V\\big)"}</Tex>
          <Tex display>{"L^V = \\frac{1}{N}\\sum_t \\tfrac12\\max\\big((V_t - R_t)^2,\\ (V^{\\text{clip}}_t - R_t)^2\\big)"}</Tex>
        </>
      }
      calc={
        <CalcPanel target={<>epoch {e + 1}, response {focus.i + 1}, before token {t} (<Tok>{words([r.response[t]])}</Tok>)</>}>
          <CalcLine label="values" tex={`V^{\\text{old}} = ${tn(r.values[t])},\\quad V = ${tn(tok.value)},\\quad R = ${tn(R)}`} />
          <CalcLine label="clipped value" tex={`V^{\\text{clip}} = ${tn(r.values[t])} + \\mathrm{clip}(${tn(tok.value - r.values[t])}, -${ve}, ${ve}) = ${tn(tok.clippedValue)}`} />
          <CalcLine
            label="loss term"
            tex={`\\tfrac12\\max(${tp(tok.value - R)}^2,\\ ${tp(tok.clippedValue - R)}^2) = ${boxed(tok.loss)}`}
          />
          <CalcLine label="gradient" tex={tok.flows ? `\\frac{\\partial L}{\\partial V} = \\frac{${tn(tok.value)} - ${tp(R)}}{${N}} = ${tg(tok.dValue)}` : `\\text{clipped error is larger} \\Rightarrow \\frac{\\partial L}{\\partial V} = 0`} />
        </CalcPanel>
      }
    >
      <div className="control-row">
        <EpochTabs epoch={e} onChange={setEpoch} />
        <span className="muted">
          value loss {trace.epochs[e].value.loss.toFixed(4)} · {(trace.epochs[e].value.clipFrac * 100).toFixed(0)}% of tokens clipped
        </span>
      </div>
      <h3 className="viz-title">
        Error <Tex>{"V_t - R_t"}</Tex> of every state in the batch (greyed: clipped)
      </h3>
      <TokenGrid
        value={(ro, i, k) => vt[i][k].value - ro.returns[k]}
        focus={focus}
        onFocus={setFocus}
        dim={(_, i, k) => !vt[i][k].flows}
      />
    </StepLayout>
  );
}

function RatioStrip({ ratios, eps }: { ratios: number[][]; eps: number }) {
  const W = 420;
  const rowH = 26;
  const pad = 70;
  const lo = Math.min(1 - 2 * eps, ...ratios.flat());
  const hi = Math.max(1 + 2 * eps, ...ratios.flat());
  const sx = (r: number) => pad + ((r - lo) / (hi - lo)) * (W - pad - 10);
  return (
    <svg className="ratio-strip" width={W} height={ratios.length * rowH + 24}>
      <rect x={sx(1 - eps)} y={0} width={sx(1 + eps) - sx(1 - eps)} height={ratios.length * rowH} className="clip-band" />
      <line x1={sx(1)} x2={sx(1)} y1={0} y2={ratios.length * rowH} className="axis" />
      {ratios.map((rs, e) => (
        <g key={e}>
          <text x={4} y={e * rowH + rowH / 2 + 4} className="tick">
            epoch {e + 1}
          </text>
          {rs.map((r, k) => (
            <circle key={k} cx={sx(r)} cy={e * rowH + rowH / 2} r={3.5} className="ratio-dot" />
          ))}
        </g>
      ))}
      {[1 - eps, 1, 1 + eps].map((r) => (
        <text key={r} x={sx(r)} y={ratios.length * rowH + 16} textAnchor="middle" className="tick">
          {r.toFixed(1)}
        </text>
      ))}
    </svg>
  );
}

export function PEpochsStep() {
  const { trace, trainer } = usePpo();
  const { goTo } = useStep();
  const { lr, valueLr, epochs, clip } = trainer.settings;
  const ratios = trace.epochs.map((ep) => ep.policy.tokens.flat().map((x) => x.ratio));
  return (
    <StepLayout
      explain={
        <>
          <p>
            Each epoch reruns both models on the whole batch, computes the losses and gradients of the previous
            pages (including a fresh pretraining minibatch), and takes one Adam step for each model. The learning
            rates are constant, {lr} for the policy and {valueLr} for the value model, with no weight decay and Adam's{" "}
            <Tex>{"\\epsilon = 10^{-5}"}</Tex> (Huang et al., 2024). Ziegler et al. (2019) used {epochs} epochs with
            one minibatch, as here. InstructGPT split each batch into 8 minibatches and made a single pass.
          </p>
          <p>
            The table shows the policy drifting away from <Tex>{"\\theta_{\\text{old}}"}</Tex> over the epochs. The
            ratios spread out, clipping starts to bite, and the approximate KL to the old policy grows. The usual
            diagnostic for that KL is <Tex>{"\\tfrac12\\,\\mathrm{mean}(\\log\\pi_\\theta - \\log\\pi_{\\theta_{\\text{old}}})^2"}</Tex>,
            measured before each epoch's update.
          </p>
        </>
      }
      formula={
        <>
          <Tex display>{"\\theta \\leftarrow \\theta - \\eta\\,\\mathrm{Adam}\\big(\\nabla_\\theta(-L^{\\text{CLIP}}) + \\gamma\\nabla_\\theta L^{\\text{ptx}}\\big)"}</Tex>
          <Tex display>{"\\psi \\leftarrow \\psi - \\eta_V\\,\\mathrm{Adam}\\big(\\nabla_\\psi L^V\\big)"}</Tex>
          <div className="formula-caption">repeated for {epochs} epochs, then θ_old ← θ</div>
        </>
      }
    >
      <div className="control-row">
        <button className="primary-btn" disabled={trainer.running} onClick={() => trainer.stepOnce(1)}>
          Apply this iteration (iteration {trainer.state.iteration} → {trainer.state.iteration + 1})
        </button>
        <button className="small-btn" onClick={() => goTo("p-after")}>
          See what it changed →
        </button>
      </div>
      <table className="softmax-table">
        <thead>
          <tr>
            <th>epoch</th>
            <th>policy loss</th>
            <th>tokens clipped</th>
            <th>approx. KL to π_old</th>
            <th>ratio range</th>
            <th>pretraining loss</th>
            <th>value loss</th>
            <th>values clipped</th>
          </tr>
        </thead>
        <tbody>
          {trace.epochs.map((ep, k) => (
            <tr key={k}>
              <td>{k + 1}</td>
              <td>{ep.policy.loss.toFixed(5)}</td>
              <td>{(ep.policy.clipFrac * 100).toFixed(1)}%</td>
              <td>{ep.policy.approxKl.toExponential(2)}</td>
              <td>
                [{Math.min(...ratios[k]).toFixed(3)}, {Math.max(...ratios[k]).toFixed(3)}]
              </td>
              <td>{ep.ptx ? ep.ptx.loss.toFixed(4) : "—"}</td>
              <td>{ep.value.loss.toFixed(4)}</td>
              <td>{(ep.value.clipFrac * 100).toFixed(0)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
      <h3 className="viz-title">Probability ratio of every token at the start of each epoch</h3>
      <RatioStrip ratios={ratios} eps={clip} />
      <p className="viz-caption">
        The clip does not cap the ratios themselves. Weights are shared across tokens, so a token whose own gradient
        is zero can still be pushed further by the others. It only removes the incentive to push a token further.
      </p>
    </StepLayout>
  );
}

const DECISION_POINTS = ["the", "the cat", "the dog", "the bird", "the cat sat on the", "the dog sat on the"];

function nextProbs(cfg: ModelConfig, params: ModelParams, ids: number[]): number[] {
  return softmaxRows([runForward({ config: cfg, params }, ids).logits])[0];
}

export function PAfterStep() {
  const { trace, trainer, models } = usePpo();
  const { goTo } = useStep();
  const vocab = models.cfg.vocab;
  const before = trace.before.policy.params;
  const after = trace.after.policy.params;
  const evals = useMemo(
    (): [PolicyEvaluation, PolicyEvaluation] | null =>
      trainer.running ? null : [evaluatePolicy(models, trainer.settings, before), evaluatePolicy(models, trainer.settings, after)],
    [models, trainer.settings, trainer.running, before, after],
  );
  const rows = useMemo(
    () =>
      DECISION_POINTS.map((text) => {
        const ids = encode(text);
        const pb = nextProbs(models.cfg, before, ids);
        const pa = nextProbs(models.cfg, after, ids);
        const toks = pb.map((_, v) => v).filter((v) => pb[v] >= 0.01 || pa[v] >= 0.01).sort((x, y) => pb[y] - pb[x]);
        return { text, toks, pb, pa };
      }),
    [models, before, after],
  );
  const beta = trainer.settings.beta;
  const delta = (x: number, y: number, dp = 4) => (
    <span className={y >= x ? "delta-up" : "delta-down"}>
      {y >= x ? "+" : "−"}
      {Math.abs(y - x).toFixed(dp)}
    </span>
  );
  return (
    <StepLayout
      explain={
        <>
          <p>
            Here is the policy before (<Tex>{"\\theta_{\\text{old}}"}</Tex>) and after the {trainer.settings.epochs} epochs
            (<Tex>{"\\theta"}</Tex>) at the points where it makes a real choice. The expected values on the right are
            exact: they enumerate every response with probability at least <Tex>{"10^{-4}"}</Tex>, which is
            possible for this tiny model but not for a real one.
          </p>
          <p>
            One iteration is a small step. Chasing becomes a little less likely, the expected score rises a little,
            and the policy moves slightly away from the reference. The batch had only{" "}
            {trace.rollouts.length} responses, so some changes are noise: a token that happened to be sampled with a
            positive advantage gets pushed up whatever its true merit. Averaged over many iterations, the noise
            cancels and the signal remains.
          </p>
        </>
      }
      formula={
        evals ? (
          <>
            <Tex display>{`\\mathbb{E}[r_\\phi]:\\ ${tn(evals[0].score)} \\to ${tn(evals[1].score)}`}</Tex>
            <Tex display>{`\\mathrm{KL}(\\pi\\,\\|\\,\\pi_{\\text{ref}}):\\ ${tn(evals[0].kl)} \\to ${tn(evals[1].kl)}`}</Tex>
            <Tex display>{`P(\\text{chasing}):\\ ${tn(evals[0].pChasing)} \\to ${tn(evals[1].pChasing)}`}</Tex>
          </>
        ) : (
          <div className="formula-caption">Pause training to compute the exact before/after values.</div>
        )
      }
      calc={
        evals && (
          <CalcPanel target="the objective, before and after (exact)">
            <CalcLine
              label="before"
              tex={`\\mathbb{E}[r_\\phi] - \\beta\\,\\mathrm{KL} = ${tn(evals[0].score)} - ${beta}\\cdot${tp(evals[0].kl)} = ${tn(evals[0].score - beta * evals[0].kl)}`}
            />
            <CalcLine label="after" tex={`${tn(evals[1].score)} - ${beta}\\cdot${tp(evals[1].kl)} = ${boxed(evals[1].score - beta * evals[1].kl)}`} />
          </CalcPanel>
        )
      }
    >
      <div className="control-row">
        <button className="primary-btn" disabled={trainer.running} onClick={() => trainer.stepOnce(1)}>
          Apply this iteration (iteration {trainer.state.iteration} → {trainer.state.iteration + 1})
        </button>
        <button
          className="small-btn"
          disabled={trainer.running}
          onClick={() => {
            trainer.stepOnce(1);
            goTo("p-rollout");
          }}
        >
          Apply and walk through the next iteration
        </button>
        <button className="small-btn" onClick={() => goTo("p-dynamics")}>
          Run hundreds of iterations →
        </button>
      </div>
      <table className="softmax-table after-table">
        <thead>
          <tr>
            <th>after</th>
            <th>next token</th>
            <th>p before</th>
            <th>p after</th>
            <th>change</th>
          </tr>
        </thead>
        <tbody>
          {rows.flatMap((row) =>
            row.toks.map((v, k) => (
              <tr key={`${row.text}-${v}`} className={k === 0 ? "group-start" : ""}>
                <td className="tok-cell">{k === 0 ? row.text : ""}</td>
                <td className={`tok-cell${vocab[v] === "chased" ? " chase-cell" : ""}`}>
                  <strong>{vocab[v]}</strong>
                </td>
                <td>{row.pb[v].toFixed(4)}</td>
                <td>{row.pa[v].toFixed(4)}</td>
                <td>{delta(row.pb[v], row.pa[v])}</td>
              </tr>
            )),
          )}
        </tbody>
      </table>
    </StepLayout>
  );
}
