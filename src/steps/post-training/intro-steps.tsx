import { useMemo, useState } from "react";
import { CalcLine, CalcNote, CalcPanel } from "../../components/calc-panel";
import { BarList } from "../../components/charts";
import { StepLayout } from "../../components/step-layout";
import { Tex } from "../../components/tex";
import { boxed, fmt, tn } from "../../lib/format";
import { labelOf, rewardModel, rewardOf } from "../../model/reward-model";
import { encode } from "../../model/tiny-gpt";
import { usePpo } from "../../post-training/ppo-context";
import { useStep } from "../step-context";
import { Tok } from "../shared";
import { PpoControls, PpoDiagram } from "./shared";

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

export function POverviewStep() {
  const { trainer } = usePpo();
  const { cfg } = useStep();
  const k = trainer.state.iteration;
  return (
    <StepLayout
      explain={
        <>
          <p>
            <strong>Pretraining</strong> taught the model to imitate its corpus. Ask it to continue <Tok>the cat</Tok> and
            a third of the time the cat chases something, because that is how often it happens in the training
            sentences. <strong>Post-training</strong> changes what the model does rather than what it knows. Here the
            goal is a model that stops writing about chasing.
          </p>
          <p>
            The recipe is InstructGPT's (Ouyang et al., 2022), which built on Ziegler et al. (2019) and Stiennon et al.
            (2020): (1) supervised fine-tuning (SFT) on demonstrations, (2) a <strong>reward model</strong> trained on
            human comparisons, (3) reinforcement learning against that reward model with{" "}
            <strong>PPO</strong> (Schulman et al., 2017), plus a penalty on the KL divergence from the SFT model.
            The models the paper calls InstructGPT also mix the pretraining loss into every PPO update (“PPO-ptx”),
            and so do we. Our pretrained model already writes well-formed sentences, so we skip SFT and the pretrained model
            takes the SFT model's place.
          </p>
          <p>
            Four networks take part, listed below. This walkthrough performs one PPO iteration with every number
            exact, starting from the policy's current weights (iteration {k}).{" "}
            {k === 0
              ? "At iteration 0 the policy is still the pretrained model, which makes a few quantities trivially 0 or 1. Run a few iterations (buttons below, or the last page) to see one from later in training."
              : "Reset to see the very first iteration, where the policy is still the pretrained model."}
          </p>
        </>
      }
      formula={
        <>
          <Tex display>{"\\max_\\theta\\; \\mathbb{E}_{x\\sim D,\\; y\\sim\\pi_\\theta(\\cdot\\mid x)}\\Big[r_\\phi(x,y) - \\beta\\log\\frac{\\pi_\\theta(y\\mid x)}{\\pi_{\\text{ref}}(y\\mid x)}\\Big]"}</Tex>
          <Tex display>{"+\\;\\gamma\\,\\mathbb{E}_{t\\sim D_{\\text{pretrain}}}\\big[\\log\\pi_\\theta(t)\\big]"}</Tex>
          <div className="formula-caption">
            InstructGPT's objective (eq. 2): x a prompt, y a response, <Tex>{"r_\\phi"}</Tex> the reward model,{" "}
            <Tex>{`\\beta = ${trainer.settings.beta}`}</Tex>, <Tex>{`\\gamma = ${trainer.settings.ptxCoef}`}</Tex>
          </div>
        </>
      }
      deeper={
        <>
          <p>
            Two regularisers keep the policy from straying. The KL penalty sits inside the reward and keeps the
            responses close to the reference's. The pretraining term sits in the loss and keeps the model a good
            model of ordinary text. Direct Preference Optimization (DPO, week 5) optimises the first line of the
            objective without a reward model, sampling, or a value model, by turning its closed-form optimum into
            a classification loss on the comparisons.
          </p>
          <p>
            In RL terms each prompt starts an episode: the state is the text so far, an action is the next token
            (from {cfg.vocab.length} choices), and the episode ends with the last token. The only reward that
            reflects human preferences arrives at the very end.
          </p>
        </>
      }
    >
      <PpoControls />
      <table className="softmax-table models-table">
        <thead>
          <tr>
            <th>model</th>
            <th>role</th>
            <th>starts as</th>
            <th>during PPO</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>
              policy <Tex>{"\\pi_\\theta"}</Tex>
            </td>
            <td>writes the responses; the model being trained</td>
            <td>the pretrained model</td>
            <td>updated (clipped policy gradient + pretraining loss)</td>
          </tr>
          <tr>
            <td>
              reference <Tex>{"\\pi_{\\text{ref}}"}</Tex>
            </td>
            <td>the anchor for the KL penalty</td>
            <td>the pretrained model</td>
            <td>frozen</td>
          </tr>
          <tr>
            <td>
              reward model <Tex>{"r_\\phi"}</Tex>
            </td>
            <td>scores a finished response</td>
            <td>pretrained model + scalar head, trained on comparisons</td>
            <td>frozen</td>
          </tr>
          <tr>
            <td>
              value model <Tex>{"V_\\psi"}</Tex>
            </td>
            <td>predicts the reward still to come at each token</td>
            <td>the reward model</td>
            <td>updated (regression on returns)</td>
          </tr>
        </tbody>
      </table>
      <h3 className="viz-title">One PPO iteration (click any stage to jump to it)</h3>
      <PpoDiagram />
    </StepLayout>
  );
}

const EXTRA_SENTENCES = [
  "the bird sat on the rug .",
  "the dog ate the seed .",
  "the cat ate the bone .",
  "the cat sat on the mouse .",
  "the cat ate the ball .",
  "the cat sat the mouse .",
  "the .",
];

export function PRewardModelStep() {
  const { model } = useStep();
  const { data, labeler } = rewardModel;
  const corpus = model.training.corpus.map((c) => c.text);
  const scored = useMemo(
    () =>
      [...corpus, ...EXTRA_SENTENCES].map((text, k) => ({
        key: k,
        text,
        inCorpus: k < corpus.length,
        label: labelOf(text.split(" ")),
        score: rewardOf(rewardModel.config, rewardModel.params, encode(text)),
      })),
    [corpus],
  );
  const pairs = useMemo(
    () =>
      data.examples.map((e) => {
        const rw = rewardOf(rewardModel.config, rewardModel.params, encode(e.chosen));
        const rl = rewardOf(rewardModel.config, rewardModel.params, encode(e.rejected));
        return { ...e, rw, rl, p: sigmoid(rw - rl) };
      }),
    [data.examples],
  );
  const [focus, setFocus] = useState(0);
  const pair = pairs[focus];
  const sorted = (inCorpus: boolean) => scored.filter((s) => s.inCorpus === inCorpus).sort((a, b) => b.score - a.score);
  const bar = (s: (typeof scored)[number]) => ({
    key: s.key,
    label: (
      <span className="rm-label">
        <span className={`label-chip ${s.label}`}>{s.label}</span> {s.text}
      </span>
    ),
    value: s.score,
  });

  return (
    <StepLayout
      explain={
        <>
          <p>
            People find it easier to compare responses than to score them on an absolute scale. Stiennon et al.
            asked labelers which of two responses was better. InstructGPT had them rank 4 to 9 responses to the same
            prompt at once. Our labeler is simulated. Shown K = {labeler.k} sentences, it ranks them by sampling from a Plackett–Luce model over a
            hidden utility. Chasing scores {fmt(labeler.utility.chasing, 0)}. Nonsense scores {fmt(labeler.utility.nonsense, 0)}:
            that means ungrammatical, or an object that does not fit the verb (you sit on mats, rugs and
            branches, eat food, and chase animals or balls). Anything else scores {fmt(labeler.utility.fine, 0)}. It is noisy on purpose: it puts a fine sentence above a chasing
            one only <Tex>{`\\sigma(2) \\approx ${sigmoid(2).toFixed(2)}`}</Tex> of the time. Each ranking gives 6
            comparisons, and {data.rankings.toLocaleString()} rankings gave {data.trainComparisons.toLocaleString()}{" "}
            training comparisons.
          </p>
          <p>
            The reward model is the pretrained transformer with its unembedding replaced by a scalar head that reads
            the final hidden state of the last token (InstructGPT, appendix C). The Bradley–Terry model says a
            labeler prefers <Tex>{"y_w"}</Tex> to <Tex>{"y_l"}</Tex> with probability{" "}
            <Tex>{"\\sigma(r_w - r_l)"}</Tex>, and training maximises the likelihood of the labeler's choices. On
            held-out comparisons it agrees with the labeler {(data.valAccuracy * 100).toFixed(0)}% of the time, close
            to the best possible given the labeler's own noise. The loss only sees differences of rewards, so the bias
            is then shifted until the pretrained model's responses score 0 on average. A positive score means
            “better than the starting policy”.
          </p>
        </>
      }
      formula={
        <>
          <Tex display>{"P(y_w \\succ y_l \\mid x) = \\sigma\\big(r_\\phi(x,y_w) - r_\\phi(x,y_l)\\big)"}</Tex>
          <Tex display>{"\\mathcal{L}(\\phi) = -\\mathbb{E}\\,\\log\\sigma\\big(r_\\phi(x,y_w) - r_\\phi(x,y_l)\\big)"}</Tex>
          <Tex display>{"r_\\phi(x,y) = h_{\\text{last}}\\cdot w_r + b_r"}</Tex>
        </>
      }
      deeper={
        <p>
          A reward model is only trustworthy near its training data. The bottom chart scores sentences the
          pretrained model essentially never writes. Nonsense such as <Tok>the cat sat on the mouse .</Tok> scores
          very low, because the hotter samples showed the reward model what nonsense looks like. But fine sentences
          it never saw, such as <Tok>the dog ate the seed .</Tok>, score far above every sentence in the corpus,
          even though the labeler would call them merely fine. A policy optimised hard enough will find holes like
          these. The last page of this chapter shows it happening. The reward model also needs one more position
          than the policy: it reads whole 7-token sentences, while the policy only ever reads 6 tokens to write a
          7th. The extra position embedding started as a copy of the last one.
        </p>
      }
      calc={
        <CalcPanel target={`comparison ${focus + 1}: why the labeler's choice is likely under the reward model`}>
          <CalcLine label="chosen" tex={`r_\\phi(\\texttt{${pair.chosen}}) = ${tn(pair.rw)}`} />
          <CalcLine label="rejected" tex={`r_\\phi(\\texttt{${pair.rejected}}) = ${tn(pair.rl)}`} />
          <CalcLine label="preference" tex={`\\sigma(${tn(pair.rw)} - ${tn(pair.rl)}) = \\sigma(${tn(pair.rw - pair.rl)}) = ${boxed(pair.p)}`} />
          <CalcLine label="loss" tex={`-\\ln ${tn(pair.p)} = ${tn(-Math.log(pair.p))}`} />
          <CalcNote>
            Training lowered the average of this loss over all comparisons to {data.trainLoss.toFixed(3)} (held out:{" "}
            {data.valLoss.toFixed(3)}); always guessing 50/50 would give ln 2 = 0.693.
          </CalcNote>
        </CalcPanel>
      }
    >
      <div className="rm-layout">
        <div>
          <div className="arcs-title">Reward-model score of every corpus sentence (labeler's hidden judgement in the tag)</div>
          <BarList items={sorted(true).map(bar)} min={-2} max={1.5} dp={3} width={160} />
          <div className="arcs-title rm-gap">Sentences the pretrained model (almost) never writes</div>
          <BarList items={sorted(false).map(bar)} min={-4} max={3} dp={3} width={160} />
        </div>
        <div>
          <div className="arcs-title">Held-out comparisons (the labeler preferred the left one)</div>
          <table className="softmax-table pairs-table">
            <thead>
              <tr>
                <th>chosen</th>
                <th>rejected</th>
                <th>
                  <Tex>{"r_w"}</Tex>
                </th>
                <th>
                  <Tex>{"r_l"}</Tex>
                </th>
                <th>
                  <Tex>{"\\sigma(r_w - r_l)"}</Tex>
                </th>
              </tr>
            </thead>
            <tbody>
              {pairs.map((p, k) => (
                <tr key={k} className={k === focus ? "focus" : ""} onMouseEnter={() => setFocus(k)}>
                  <td className="tok-cell">{p.chosen}</td>
                  <td className="tok-cell">{p.rejected}</td>
                  <td>{p.rw.toFixed(3)}</td>
                  <td>{p.rl.toFixed(3)}</td>
                  <td className={p.p >= 0.5 ? "delta-up" : "delta-down"}>{p.p.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="viz-caption">
            Where <Tex>{"\\sigma(r_w - r_l) < 0.5"}</Tex> the reward model disagrees with that particular choice. Noisy labels
            make this unavoidable.
          </p>
        </div>
      </div>
    </StepLayout>
  );
}
