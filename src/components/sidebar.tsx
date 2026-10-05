import { useEffect, useRef, useState } from "react";
import type { ModelConfig } from "../model/types";
import type { ChapterDef, StepDef } from "../steps/step-defs";

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
  chapters,
  chapterIndex,
  positions,
  onSelectChapter,
  onSelectStep,
  T,
  cfg,
}: {
  chapters: ChapterDef[];
  chapterIndex: number;
  /** The step index each chapter is on, indexed like `chapters`. */
  positions: number[];
  onSelectChapter: (c: number) => void;
  onSelectStep: (i: number) => void;
  T: number;
  cfg: ModelConfig;
}) {
  const chapter = chapters[chapterIndex];
  const steps = chapter.steps;
  const current = positions[chapterIndex];
  const currentStep = steps[current];
  const groups = groupSteps(steps);
  const [open, setOpen] = useState<Record<string, boolean>>({});
  const activeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    activeRef.current?.scrollIntoView({ block: "nearest" });
  }, [currentStep]);

  return (
    <aside className="sidebar">
      <nav className="chapter-nav" aria-label="Chapters">
        <div className="sidebar-heading">Chapters</div>
        <div className="chapter-list">
          {chapters.map((c, ci) => (
            <button
              key={c.id}
              className={ci === chapterIndex ? "chapter-btn active" : "chapter-btn"}
              aria-current={ci === chapterIndex ? "true" : undefined}
              onClick={() => onSelectChapter(ci)}
            >
              <span className="chapter-num">{ci + 1}</span>
              <span className="chapter-title">{c.title}</span>
              <span className="chapter-progress" title={`step ${positions[ci]} of ${c.steps.length - 1}`}>
                {positions[ci]}/{c.steps.length - 1}
              </span>
              <span className="chapter-sub">{c.subtitle}</span>
            </button>
          ))}
        </div>
      </nav>
      <nav className="step-nav" aria-label={`${chapter.title} steps`}>
        {groups.map((g) => {
          const key = `${chapter.id}/${g.title}`;
          const containsCurrent = g.subs.some((s) => s.steps.includes(currentStep));
          const isOpen = !g.collapsible || containsCurrent || open[key];
          return (
            <div key={key} className={`nav-group${containsCurrent ? " current" : ""}`}>
              <button
                className="nav-group-title"
                onClick={() => g.collapsible && setOpen((o) => ({ ...o, [key]: !isOpen }))}
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
                          ref={i === current ? activeRef : undefined}
                          className={`nav-step${i === current ? " active" : ""}${i < current ? " done" : ""}`}
                          onClick={() => onSelectStep(i)}
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
    </aside>
  );
}
