import { useMemo } from "react";
import { Tabs } from "../../components/layout";
import { LineChart } from "../../components/line-chart";
import { StepLayout } from "../../components/step-layout";
import { Tex } from "../../components/tex";
import { usePpo } from "../../post-training/ppo-context";
import { evaluatePolicy, optimalPolicy, type PromptEvaluation } from "../../post-training/ppo";
import { PpoControls, useWords } from "./shared";

const BETAS = [1, 0.5, 0.2, 0.05, 0.02];
const GAMMAS = [0, 0.3, 1];

function TopResponses({ evaluation, words }: { evaluation: PromptEvaluation; words: (ids: number[]) => string }) {
  const Z = evaluation.responses.reduce((s, r) => s + r.prob, 0);
  const top = [...evaluation.responses].sort((a, b) => b.prob - a.prob).slice(0, 3);
  return (
    <div className="top-responses">
      {top.map((r) => (
        <div key={r.response.join(",")} className="top-response">
          <span className="top-prob">{(r.prob / Z).toFixed(2)}</span>
          <span className={`label-chip ${r.label}`}>{r.label}</span>
          <code>{words(r.response)}</code>
          <span className="top-score">r = {r.score.toFixed(2)}</span>
        </div>
      ))}
    </div>
  );
}

export function PDynamicsStep() {
  const { trainer, models } = usePpo();
  const words = useWords();
  const { settings, evals, running, state } = trainer;
  const reference = useMemo(() => evaluatePolicy(models, settings, models.reference), [models, settings]);
  const current = useMemo(
    () => (running ? null : evaluatePolicy(models, settings, state.policy.params)),
    [models, settings, running, state.policy.params],
  );
  const optimum = useMemo(() => optimalPolicy(models, settings, current ?? undefined), [models, settings, current]);
  const xMax = Math.max(settings.totalIterations, state.iteration);
  const pts = (f: (e: (typeof evals)[number]) => number) => evals.map((e) => ({ x: e.iteration, y: f(e) }));
  const scores = evals.map((e) => e.score);
  const kls = evals.map((e) => e.kl);
  return (
    <StepLayout
      explain={
        <>
          <p>
            Real RLHF repeats this loop for a long time (InstructGPT trained on 256,000 episodes). Here it runs live,
            with exactly the code of the previous pages: {trainer.pending.rollouts.length} responses and{" "}
            {settings.epochs} epochs per iteration. Within a few dozen iterations chasing all but disappears and the
            expected reward-model score rises. The labeler's true utility, which PPO never sees, rises with it from{" "}
            {reference.utility.toFixed(2)} to about 0. The KL from the reference settles at a level set by β.
          </p>
          <p>
            The dashed lines are the exact optimum of the objective's RL part. For a fixed reward model the best
            policy is <Tex>{"\\pi^*(y\\mid x) \\propto \\pi_{\\text{ref}}(y\\mid x)\\,e^{r_\\phi(x,y)/\\beta}"}</Tex>{" "}
            (Ziegler et al., 2019; the closed form DPO starts from), computed here by enumerating responses.{" "}
            <Tex>{"\\pi^*"}</Tex> keeps the reference's relative odds between responses that score alike, so the
            policy stops chasing without forgetting the rest of what it learned. PPO-ptx stays a little short of it,
            with slightly more chasing, because the pretraining term pulls toward the corpus, where a third of the
            sentences involve chasing. That is the price of the second regulariser.
          </p>
          <p>
            Now remove both regularisers: γ = 0 and β = 0.02 (InstructGPT's β, but for much longer responses).
            After a hundred or so iterations the policy collapses onto one sentence per prompt, such as{" "}
            <code>the dog ate the seed .</code> The pretrained model would almost never write these sentences, and the
            reward model, which never saw them, overrates them. The score climbs well past anything the corpus earns
            and the KL soars, but the true utility gains nothing. This is reward-model over-optimisation, or reward
            hacking (Gao et al., 2023). With γ = 0 alone, watch for sudden KL spikes: these are the drifting logits
            described on the pretraining-mix page.
          </p>
        </>
      }
      formula={
        <>
          <Tex display>{"\\pi^*(y\\mid x) = \\frac{1}{Z(x)}\\,\\pi_{\\text{ref}}(y\\mid x)\\,\\exp\\big(r_\\phi(x,y)/\\beta\\big)"}</Tex>
          <div className="formula-caption">
            maximises <Tex>{"\\mathbb{E}[r_\\phi] - \\beta\\,\\mathrm{KL}(\\pi\\,\\|\\,\\pi_{\\text{ref}})"}</Tex> (the RL part); at{" "}
            <Tex>{`\\beta = ${settings.beta}`}</Tex>: score {optimum.score.toFixed(3)}, KL {optimum.kl.toFixed(3)}
          </div>
        </>
      }
    >
      <PpoControls />
      <div className="control-row">
        <span className="muted">KL coefficient β:</span>
        <Tabs
          value={settings.beta}
          onChange={(beta) => {
            trainer.setSettings({ beta });
            trainer.reset();
          }}
          options={BETAS.map((b) => ({ value: b, label: String(b) }))}
        />
        <span className="muted">pretraining mix γ:</span>
        <Tabs
          value={settings.ptxCoef}
          onChange={(ptxCoef) => {
            trainer.setSettings({ ptxCoef });
            trainer.reset();
          }}
          options={GAMMAS.map((g) => ({ value: g, label: String(g) }))}
        />
        <span className="muted">(changing either restarts from the pretrained policy)</span>
      </div>
      <div className="ppo-charts">
        <LineChart
          title="expected score and true utility"
          series={[
            { label: "reward-model score", color: "#ea580c", points: pts((e) => e.score) },
            { label: "labeler's utility (hidden)", color: "#059669", points: pts((e) => e.utility), dashed: true },
          ]}
          guides={[{ y: optimum.score, label: `RL optimum ${optimum.score.toFixed(2)}`, color: "#ea580c" }]}
          xMax={xMax}
          yMin={Math.min(-1, ...evals.map((e) => e.utility))}
          yMax={Math.max(1, optimum.score, ...scores) + 0.1}
        />
        <LineChart
          title="KL(π ‖ π_ref), nats"
          series={[{ label: "KL to the reference", color: "#7c3aed", points: pts((e) => e.kl) }]}
          guides={[{ y: optimum.kl, label: `RL optimum ${optimum.kl.toFixed(2)}`, color: "#7c3aed" }]}
          xMax={xMax}
          yMin={0}
          yMax={Math.max(1, optimum.kl, ...kls) * 1.1}
        />
        <LineChart
          title="P(chasing)"
          series={[{ label: "probability of a chasing sentence", color: "#2563eb", points: pts((e) => e.pChasing) }]}
          guides={[{ y: optimum.pChasing, label: `RL optimum ${optimum.pChasing.toFixed(3)}`, color: "#2563eb" }]}
          xMax={xMax}
          yMin={0}
          yMax={0.4}
        />
      </div>
      <h3 className="viz-title">Most likely responses (probability, labeler's verdict, reward-model score)</h3>
      <table className="softmax-table policy-table">
        <thead>
          <tr>
            <th>prompt</th>
            <th>reference π_ref</th>
            <th>policy at iteration {state.iteration}</th>
            <th>optimum π* (β = {settings.beta})</th>
          </tr>
        </thead>
        <tbody>
          {models.prompts.map((prompt, i) => (
            <tr key={i}>
              <td className="tok-cell">{words(prompt)}</td>
              <td>
                <TopResponses evaluation={reference.prompts[i]} words={words} />
              </td>
              <td>{current ? <TopResponses evaluation={current.prompts[i]} words={words} /> : <span className="muted">pause to update</span>}</td>
              <td>
                <TopResponses evaluation={optimum.prompts[i]} words={words} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </StepLayout>
  );
}
