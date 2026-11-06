import type { ReactNode } from "react";

/** "Business / Brainforge / ABC Home & Commercial / Andi" from a record's home chain. */
export const homePathText = (home: readonly { title: string }[]) => home.map(h => h.title).join(" / ");

/** A home-path chip. When the path is too long it loses its start, not its end: the ellipsis comes first and the
 * last (most specific) segment stays visible. The full path is the tooltip, and screen readers read it whole (the
 * truncation is visual only). */
export function HomePathChip({ path, icon, className = "" }: { path: string; icon?: ReactNode; className?: string }) {
  return <span className={"chip chip-home" + (className ? " " + className : "")} title={path}>
    {icon}<span className="chip-text home-path"><bdi>{path}</bdi></span>
  </span>;
}
