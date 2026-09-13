import { useState } from "react";
import { z } from "zod";
import { X } from "lucide-react";
import { useEditor } from "./editor-control";
import { useDialogFocus } from "./components";
import type { Memory } from "./types";

export function MemoryEditor({
  memory,
  busy,
  mutate,
  onClose,
}: {
  memory: Memory;
  busy: boolean;
  mutate: (tool: string, args: unknown, message: string) => Promise<unknown>;
  onClose: () => void;
}) {
  useDialogFocus();
  const [content, setContent] = useState(memory.content);
  async function save() {
    if (!content.trim()) return;
    const result = await mutate(
      "memory.correct",
      { memory_id: memory.id, content },
      "Memory corrected",
    );
    if (result) onClose();
    return result;
  }
  useEditor({
    kind: "memory",
    record_id: memory.id,
    dirty: content !== memory.content,
    busy,
    schema: z.object({ content: z.string().min(1).max(20000) }),
    values: { content },
    patch: (v) => setContent(v.content as string),
    save,
    close: onClose,
  });
  return (
    <div className="modal-backdrop">
      <form
        className="dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="memory-editor-title"
        onSubmit={(e) => {
          e.preventDefault();
          void save();
        }}
      >
        <div className="dialog-heading">
          <h2 id="memory-editor-title">Correct memory</h2>
          <button
            type="button"
            className="icon-button"
            aria-label="Close memory editor"
            onClick={onClose}
          >
            <X size={18} />
          </button>
        </div>
        <textarea
          aria-label="Memory correction"
          required
          maxLength={20000}
          rows={5}
          value={content}
          onChange={(e) => setContent(e.target.value)}
        />
        <button className="primary" disabled={busy || !content.trim()}>
          Save correction
        </button>
      </form>
    </div>
  );
}
