import { useState } from "react";
import type { ModelConfig } from "../model/types";
import type { StepDef } from "../steps/step-defs";

interface Group {
  title: string;
  collapsible: boolean;
  subs: { title?: string; steps: StepDef[] }[];
}

/** Groups consecutive steps by section, and within a section by subsection. */
function groupSteps(steps: StepDef[]): Group[] {
  const groups: Group[] = [];
  for (const s of steps) {
    let g = groups[groups.length - 1];
    if (!g || g.title !== s.section) {
      g = { title: s.section, collapsible: false, subs: [] };
      groups.push(g);
    }
    let sub = g.subs[g.subs.length - 1];
    if (!sub || sub.title !== s.subsection) {
      sub = { title: s.subsection, steps: [] };
      g.subs.push(sub);
    }
    sub.steps.push(s);
    if (s.subsection) g.collapsible = true;
  }
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
  const groups = groupSteps(steps);
  const [open, setOpen] = useState<Record<string, boolean>>({});
  const currentStep = steps[current];

  return (
    <nav className="sidebar">
      {groups.map((g) => {
        const containsCurrent = g.subs.some((s) => s.steps.includes(currentStep));
        const isOpen = !g.collapsible || containsCurrent || open[g.title];
        return (
          <div key={g.title} className={`nav-group${containsCurrent ? " current" : ""}`}>
            <button
              className="nav-group-title"
              onClick={() => g.collapsible && setOpen((o) => ({ ...o, [g.title]: !isOpen }))}
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
