import { useMemo, useState } from "react";
import { LossCurve } from "../../components/loss-curve";
import { MatrixView } from "../../components/matrix-view";
import { StepLayout } from "../../components/step-layout";
import { Tex } from "../../components/tex";
import { HEAD_COLORS } from "../../lib/colors";
import { pca2 } from "../../lib/pca";
import { corpusNextCounts } from "../../model/corpus";
import { runForward } from "../../model/forward";
import type { ModelParams } from "../../model/types";
import { useTraining } from "../../training/training-context";
import { useStep } from "../step-context";
import { tokenColLabels } from "../shared";
import { TrainerControls } from "./trainer-controls";

const ATTENTION_PROMPT = "the cat sat on the";

function softmax(z: number[]): number[] {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

function EmbeddingScatter({ params }: { params: ModelParams }) {
  const { cfg } = useStep();
  const { coords, explained } = useMemo(() => pca2(params.wte), [params]);
  const W = 360;
  const H = 300;
  const pad = 26;
  const r = Math.max(1e-6, ...coords.flat().map(Math.abs));
  const sx = (x: number) => W / 2 + (x / r) * (W / 2 - pad);
  const sy = (y: number) => H / 2 - (y / r) * (H / 2 - pad);
  const groups: Record<string, string> = {
    cat: "#ea580c", dog: "#ea580c", bird: "#ea580c",
    sat: "#2563eb", ate: "#2563eb", chased: "#2563eb", on: "#2563eb",
    mat: "#059669", rug: "#059669", branch: "#059669",
    fish: "#7c3aed", bone: "#7c3aed", seed: "#7c3aed", mouse: "#7c3aed", ball: "#7c3aed", bug: "#7c3aed",
  };
  return (
    <svg className="pca-plot" width={W} height={H}>
      <line x1={pad} x2={W - pad} y1={H / 2} y2={H / 2} className="axis" />
      <line x1={W / 2} x2={W / 2} y1={pad} y2={H - pad} className="axis" />
      {coords.map(([x, y], v) => (
        <g key={v}>
          <circle cx={sx(x)} cy={sy(y)} r={4} fill={groups[cfg.vocab[v]] ?? "#6b7280"} />
          <text x={sx(x) + 6} y={sy(y) + 4} className="pca-label" fill={groups[cfg.vocab[v]] ?? "#374151"}>
            {cfg.vocab[v]}
          </text>
        </g>
      ))}
      <text x={pad} y={H - 6} className="legend">
        PC1 {(explained[0] * 100).toFixed(0)}% · PC2 {(explained[1] * 100).toFixed(0)}% of variance
      </text>
    </svg>
  );
}

export function TDynamicsStep() {
  const { trainer } = useTraining();
  const { cfg, model: shipped } = useStep();
  const [snapIndex, setSnapIndex] = useState<number | null>(null);
  const snaps = trainer.snapshots;
  const live = snapIndex === null || trainer.running;
  const selected = live ? { step: trainer.state.step, params: trainer.state.params } : snaps[Math.min(snapIndex, snaps.length - 1)];
  const model = useMemo(() => ({ config: cfg, params: selected.params }), [cfg, selected.params]);
  const prompts = useMemo(() => shipped.training.presetPrompts.slice(0, 7), [shipped]);

  const predictions = useMemo(
    () =>
      prompts.map((p) => {
        const ids = p.split(" ").map((w) => cfg.vocab.indexOf(w));
        const probs = softmax(runForward(model, ids).logits);
        const top = probs.map((_, v) => v).sort((a, b) => probs[b] - probs[a]).slice(0, 3);
        const { counts, total } = corpusNextCounts(shipped, ids);
        return { p, top, probs, data: [...counts.entries()].sort((a, b) => b[1] - a[1]).map(([v, c]) => `${cfg.vocab[v]} ${(c / total).toFixed(2)}`).join(", ") };
      }),
    [prompts, model, cfg, shipped],
  );

  const attention = useMemo(() => {
    const tr = runForward(model, ATTENTION_PROMPT.split(" ").map((w) => cfg.vocab.indexOf(w)));
    return tr.blocks.flatMap((b) => b.heads.map((h) => h.weights[h.weights.length - 1]));
  }, [model, cfg]);
  const headLabels = Array.from({ length: cfg.nLayers * cfg.nHeads }, (_, k) => (
    <span className="rl" key={k}>
      <span className="rl-tok" style={{ color: HEAD_COLORS[k % cfg.nHeads] }}>
        block {Math.floor(k / cfg.nHeads) + 1} head {(k % cfg.nHeads) + 1}
      </span>
    </span>
  ));

  return (
    <StepLayout
      explain={
        <>
          <p>
            Real pretraining repeats the step you just walked through millions of times. Here it takes{" "}
            {trainer.settings.totalSteps.toLocaleString()} steps of {trainer.settings.batchSize} sentences each, running
            live in your browser with exactly the code of the previous steps. Press train and watch the loss fall from
            about <Tex>{`\\ln ${cfg.vocab.length} = ${Math.log(cfg.vocab.length).toFixed(2)}`}</Tex> (random guessing)
            towards the best achievable value.
          </p>
          <p>
            Things to look for: the model first learns which words are common (unigram statistics), then what follows
            each word (bigrams), and only later uses attention to look back at the subject (so that{" "}
            <em>cat</em> leads to <em>mat</em> and <em>dog</em> to <em>rug</em>). Word embeddings of the same kind
            drift together. Drag the slider to inspect earlier snapshots, and use the walkthrough at any point to see
            the exact gradients of the current step.
          </p>
        </>
      }
    >
      <TrainerControls />
      <div className="control-row settings-row">
        <label>
          learning rate
          <input type="number" step={0.001} min={0.0001} value={trainer.settings.lr} disabled={trainer.running} onChange={(e) => trainer.setSettings({ lr: Number(e.target.value) || 0.01 })} />
        </label>
        <label>
          batch size
          <input type="number" min={1} max={36} value={trainer.settings.batchSize} disabled={trainer.running} onChange={(e) => trainer.setSettings({ batchSize: Math.max(1, Math.round(Number(e.target.value))) })} />
        </label>
        <label>
          total steps
          <input type="number" min={100} step={100} value={trainer.settings.totalSteps} disabled={trainer.running} onChange={(e) => trainer.setSettings({ totalSteps: Math.max(100, Math.round(Number(e.target.value))) })} />
        </label>
      </div>
      <div className="dynamics-top">
        <LossCurve
          corpus={trainer.corpusHistory}
          batch={trainer.batchHistory}
          optimal={shipped.training.optimalLoss}
          maxStep={trainer.settings.totalSteps}
          marker={live ? undefined : selected.step}
        />
        <div className="snapshot-control">
          <div className="arcs-title">Inspect the model at</div>
          <input
            type="range"
            min={0}
            max={snaps.length - 1}
            value={live ? snaps.length - 1 : Math.min(snapIndex!, snaps.length - 1)}
            disabled={trainer.running}
            onChange={(e) => {
              const k = Number(e.target.value);
              setSnapIndex(k === snaps.length - 1 && snaps[k].step === trainer.state.step ? null : k);
            }}
          />
          <div className="snapshot-label">
            step <strong>{selected.step}</strong> {live ? "(current)" : ""}
          </div>
          {!live && (
            <button className="small-btn" onClick={() => setSnapIndex(null)}>
              back to current
            </button>
          )}
        </div>
      </div>
      <div className="dynamics-grid">
        <div className="panel">
          <div className="arcs-title">What the model predicts next</div>
          <table className="pred-table">
            <thead>
              <tr>
                <th>prompt</th>
                <th>model (top 3)</th>
                <th>training data</th>
              </tr>
            </thead>
            <tbody>
              {predictions.map((row) => (
                <tr key={row.p}>
                  <td>
                    <code>{row.p}</code>
                  </td>
                  <td>
                    {row.top.map((v) => (
                      <span key={v} className="pred-chip">
                        {cfg.vocab[v]} <small>{row.probs[v].toFixed(2)}</small>
                      </span>
                    ))}
                  </td>
                  <td className="corpus-cell">{row.data}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="panel">
          <div className="arcs-title">
            Where the last token of “{ATTENTION_PROMPT}” attends, in every head
          </div>
          <MatrixView data={attention} rowLabels={headLabels} colLabels={tokenColLabels(ATTENTION_PROMPT.split(" "))} scale="sequential" maxAbs={1} showShape={false} />
          <p className="viz-caption">To predict mat vs rug the model must find the subject, “cat”, at position 1.</p>
        </div>
        <div className="panel">
          <div className="arcs-title">Word embeddings (rows of W_E), projected to 2-D</div>
          <EmbeddingScatter params={selected.params} />
        </div>
      </div>
    </StepLayout>
  );
}
