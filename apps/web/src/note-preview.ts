import type { NoteRecord } from "./Notes";

/** The preview line for a note card. Saved items often have no body, so fall back to the
 * source line they were saved from, then to how many items a source saved; empty otherwise. */
export function notePreview(note: Pick<NoteRecord, "excerpt" | "sources" | "saved_entries">) {
  const body = note.excerpt.trim();
  if (body) return body;
  const source = note.sources?.find((s) => s.evidence.trim());
  if (source) return "From " + source.title + ": " + source.evidence;
  const entries = note.saved_entries?.length ?? 0;
  return entries ? (entries === 1 ? "1 saved item" : entries + " saved items") : "";
}
