/** Where a floating panel goes (ux.tsx `Popover floating`): below its anchor, or above when there is more room
 * there, clamped to the viewport; a bottom sheet on phone widths. Pure, so it can be tested without a layout. */
export const SHEET_MAX_WIDTH = 760, GUTTER = 16, GAP = 6;

export type Box = { top: number; left: number; bottom: number; right: number };
export type Size = { width: number; height: number };
export type Placement =
  | { mode: "sheet" }
  | { mode: "float"; top: number; left: number; maxHeight: number; above: boolean };

export function place(anchor: Box, panel: Size, viewport: Size): Placement {
  if (viewport.width <= SHEET_MAX_WIDTH) return { mode: "sheet" };
  const below = viewport.height - anchor.bottom - GAP - GUTTER;
  const aboveRoom = anchor.top - GAP - GUTTER;
  const above = panel.height > below && aboveRoom > below;
  const room = Math.max(above ? aboveRoom : below, 120);
  const height = Math.min(panel.height, room);
  const top = above ? Math.max(GUTTER, anchor.top - GAP - height) : anchor.bottom + GAP;
  const widest = viewport.width - GUTTER - panel.width;
  const left = widest < GUTTER ? GUTTER : Math.min(Math.max(anchor.left, GUTTER), widest);
  return { mode: "float", top, left, maxHeight: room, above };
}
