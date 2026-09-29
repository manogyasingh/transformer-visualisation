import { useState } from "react";
import type { ModelConfig } from "../model/types";
import type { StepDef } from "../steps/step-defs";

interface Group {
  key: string;
  title: string;
  collapsible: boolean;
  subs: { title?: string; steps: StepDef[] }[];
}

function groupSteps(steps: StepDef[], cfg: ModelConfig): Group[] {
  const groups: Group[] = [
    { key: "start", title: "Start", collapsible: false, subs: [{ steps: steps.filter((s) => s.group === "start") }] },
    { key: "input", title: "Input", collapsible: false, subs: [{ steps: steps.filter((s) => s.group === "input") }] },
  ];
  for (let l = 0; l < cfg.nLayers; l++) {
    const b = steps.filter((s) => s.layer === l);
    groups.push({
      key: `block-${l}`,
      title: `Block ${l + 1}`,
      collapsible: true,
      subs: [
        { title: "Attention", steps: b.filter((s) => s.sub === "attention") },
        { title: "MLP", steps: b.filter((s) => s.sub === "mlp") },
      ],
    });
  }
  groups.push({ key: "output", title: "Output", collapsible: false, subs: [{ steps: steps.filter((s) => s.group === "output") }] });
  return groups;
}

export function Sidebar({
  steps,
  current,
  onSelect,
  T,
  cfg,
}: {
  steps: StepDef[];
  current: number;
  onSelect: (i: number) => void;
  T: number;
  cfg: ModelConfig;
}) {
  const groups = groupSteps(steps, cfg);
  const [open, setOpen] = useState<Record<string, boolean>>({});
  const currentStep = steps[current];

  return (
    <nav className="sidebar">
      {groups.map((g) => {
        const containsCurrent = g.subs.some((s) => s.steps.includes(currentStep));
        const isOpen = !g.collapsible || containsCurrent || open[g.key];
        return (
          <div key={g.key} className={`nav-group${containsCurrent ? " current" : ""}`}>
            <button
              className="nav-group-title"
              onClick={() => g.collapsible && setOpen((o) => ({ ...o, [g.key]: !isOpen }))}
            >
              {g.collapsible && <span className="nav-caret">{isOpen ? "▾" : "▸"}</span>}
              {g.title}
              {g.collapsible && <span className="nav-group-meta">{g.subs.reduce((n, s) => n + s.steps.length, 0)} steps</span>}
            </button>
            {isOpen &&
              g.subs.map((sub, si) => (
                <div key={si} className="nav-sub">
                  {sub.title && <div className="nav-sub-title">{sub.title}</div>}
                  {sub.steps.map((s) => {
                    const i = steps.indexOf(s);
                    return (
                      <button
                        key={s.id}
                        className={`nav-step${i === current ? " active" : ""}${i < current ? " done" : ""}`}
                        onClick={() => onSelect(i)}
                      >
                        <span className="nav-num">{i}</span>
                        <span className="nav-label">{s.navLabel}</span>
                        <span className="nav-shape">{s.shape(T, cfg)}</span>
                      </button>
                    );
                  })}
                </div>
              ))}
          </div>
        );
      })}
    </nav>
  );
}
