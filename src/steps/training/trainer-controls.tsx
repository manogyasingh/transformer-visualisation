import { useTraining } from "../../training/training-context";
import { lrAt } from "../../training/trainer";

export function TrainerControls({ compact = false }: { compact?: boolean }) {
  const { trainer } = useTraining();
  const { state, settings, running } = trainer;
  const loss = trainer.corpusHistory[trainer.corpusHistory.length - 1]?.loss;
  return (
    <div className="trainer-controls">
      <button className="primary-btn" onClick={running ? trainer.pause : trainer.start} disabled={!running && state.step >= settings.totalSteps}>
        {running ? "❚❚ Pause" : state.step === 0 ? "▶ Train from scratch" : "▶ Continue training"}
      </button>
      <button className="small-btn" disabled={running} onClick={() => trainer.stepOnce(1)}>
        +1 step
      </button>
      <button className="small-btn" disabled={running} onClick={() => trainer.stepOnce(10)}>
        +10 steps
      </button>
      <button className="small-btn" onClick={() => trainer.reset(settings.seed + 1)}>
        Re-initialise (new random weights)
      </button>
      <button className="small-btn" onClick={trainer.loadTrained}>
        Load the fully trained weights
      </button>
      {!compact && (
        <span className="trainer-readout">
          step <strong>{state.step}</strong> / {settings.totalSteps} · corpus loss <strong>{loss?.toFixed(4)}</strong> · next
          learning rate {lrAt(state.step + 1, settings).toExponential(2)}
          {trainer.origin === "trained" && " · started from the shipped trained weights"}
        </span>
      )}
    </div>
  );
}
