import katex from "katex";
import "katex/dist/katex.min.css";
import { useMemo } from "react";

interface TexProps {
  children: string;
  display?: boolean;
  className?: string;
}

export function Tex({ children, display = false, className }: TexProps) {
  const html = useMemo(
    () =>
      katex.renderToString(children, {
        displayMode: display,
        throwOnError: false,
        strict: false,
      }),
    [children, display],
  );
  const cls = ["tex", display ? "tex-display" : "", className ?? ""].join(" ").trim();
  return <span className={cls} dangerouslySetInnerHTML={{ __html: html }} />;
}
