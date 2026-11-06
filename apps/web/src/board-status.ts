// Board columns by status. Backlog and Open both mean "not started yet", so the board shows one Open column: a
// backlog card (a Linear Backlog issue, say) sits there with its status and source chips. Only a drop on another
// column changes its status, so Linear keeps "Backlog" until the card really moves.

/** The column a status meaning belongs to: backlog shares Open's. */
export const boardMeaning = (meaning: string | null | undefined) => (meaning === "backlog" ? "open" : meaning ?? "");

/** The status a card takes when dropped on a meaning column: the type's status with that meaning, or for Open, its
 * Backlog status when the type has no Open one. Undefined when the type has neither. */
export function statusForColumn<S extends { id: string; meaning: string }>(statuses: readonly S[], column: string) {
  return (statuses.find(s => s.meaning === column) ?? (column === "open" ? statuses.find(s => s.meaning === "backlog") : undefined))?.id;
}

/** A type's workflow statuses as board columns: backlog statuses fold into the first Open status (when the type has
 * one), and `column` maps a record's status id to the column it shows in. */
export function boardColumns<S extends { id: string; meaning: string }>(statuses: readonly S[]) {
  const open = statuses.find(s => s.meaning === "open");
  const columns = open ? statuses.filter(s => s.meaning !== "backlog") : [...statuses];
  const column = (statusId: string | null | undefined) =>
    open && statuses.find(s => s.id === statusId)?.meaning === "backlog" ? open.id : statusId ?? "";
  return { columns, column };
}
