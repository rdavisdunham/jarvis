import {
  createContext,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { z } from "zod";

export type EditorKind =
  | "record"
  | "task"
  | "reminder"
  | "note"
  | "goal"
  | "project"
  | "area"
  | "space"
  | "actor"
  | "event"
  | "google_event"
  | "bulk"
  | "memory";
export type EditorSummary = {
  mode: "detail" | "edit";
  auto_save?: boolean;
  kind: EditorKind;
  record_id: string | null;
  dirty: boolean;
  busy: boolean;
  fields: string[];
};
type Editor = {
  mode?: "detail" | "edit";
  auto_save?: boolean;
  beforeLeave?: () => Promise<void>;
  discard?: () => void;
  kind: EditorKind;
  record_id?: string | null;
  dirty: boolean;
  busy?: boolean;
  schema: z.ZodObject;
  values: Record<string, unknown>;
  patch: (values: Record<string, unknown>) => unknown | Promise<unknown>;
  save?: () => Promise<unknown>;
  close: () => void;
};
type EditorAction = {
  operation?: "read" | "patch" | "save" | "close" | "discard";
  changes?: Record<string, unknown>;
};
type Bridge = {
  current: React.MutableRefObject<Editor | null>;
  summary: EditorSummary | null;
  publish: (editor: Editor | null) => void;
};
const Context = createContext<Bridge | null>(null);
const settled = () =>
  new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
export const choice = (values: string[]) =>
  z.enum([...new Set(values)] as [string, ...string[]]);
export const nullableId = (values: string[]) =>
  choice(["", ...values]).nullable();
export const textField = (max = 10000) => z.string().max(max);
export const tagsField = z.array(z.string().max(40)).max(20);

export function EditorProvider({ children }: { children: ReactNode }) {
  const current = useRef<Editor | null>(null);
  const [summary, setSummary] = useState<EditorSummary | null>(null);
  const publish = (editor: Editor | null) => {
    current.current = editor;
    const next = editor
      ? {
          mode: editor.mode ?? "edit",
          auto_save: !!editor.auto_save,
          kind: editor.kind,
          record_id: editor.record_id ?? null,
          dirty: editor.dirty,
          busy: !!editor.busy,
          fields: Object.keys(editor.schema.shape),
        }
      : null;
    setSummary((previous) =>
      JSON.stringify(previous) === JSON.stringify(next) ? previous : next,
    );
  };
  return (
    <Context.Provider value={{ current, summary, publish }}>
      {children}
    </Context.Provider>
  );
}

export function useEditor(editor: Editor) {
  const bridge = useContext(Context);
  const latest = useRef(editor);
  latest.current = editor;
  useEffect(() => {
    bridge?.publish(editor);
  });
  useEffect(
    () => () => {
      if (bridge?.current.current === latest.current) bridge.publish(null);
    },
    [],
  );
}

export function useEditorBridge() {
  const bridge = useContext(Context);
  return {
    summary: bridge?.summary ?? null,
    current: () => bridge?.current.current ?? null,
    async act(action: EditorAction) {
      const editor = bridge?.current.current;
      if (!editor)
        throw new Error(
          "No editor is open. Open the requested record or a new form first.",
        );
      const operation = action.operation ?? "read";
      if (operation === "read")
        return {
          outcome: editor.mode === "detail" ? "detail" : "draft",
          mode: editor.mode ?? "edit",
          auto_save: !!editor.auto_save,
          editor: editor.kind,
          record_id: editor.record_id ?? null,
          dirty: editor.dirty,
          busy: !!editor.busy,
          fields: z.toJSONSchema(editor.schema.partial()),
          values: editor.values,
          saved: editor.mode === "detail" && !editor.dirty,
          note: editor.auto_save
            ? "Inline fields save immediately through normal commands. Patch returns a saved receipt; no separate save is needed."
            : editor.mode === "detail"
              ? editor.kind === "note" ? "Saved note details. ui_editor patch starts an unsaved draft; ui_editor save persists it." : "Saved record details. Use ui_form for this record to open an editable draft."
              : "These are device-local draft values. Reading or filling them does not save.",
        };
      if (editor.busy)
        throw new Error(
          "This editor is still loading or saving. Wait for it to finish.",
        );
      if (operation === "patch") {
        const parsed = editor.schema
          .partial()
          .strict()
          .safeParse(action.changes ?? {});
        if (!parsed.success)
          throw new Error(
            parsed.error.issues
              .map((i) => i.path.join(".") + ": " + i.message)
              .join("; ")
              .slice(0, 500),
          );
        if (!Object.keys(parsed.data).length)
          throw new Error("Provide at least one field to fill.");
        // Validate the entire patch before setting any field.
        const result = await editor.patch(parsed.data);
        if (editor.auto_save) {
          await settled();
          return { outcome: "saved", saved: true, editor: editor.kind, result, command_id: (result as {__command_id?: string} | null)?.__command_id };
        }
        await settled();
        return {
          outcome: "draft_updated",
          saved: false,
          fields: Object.keys(parsed.data),
          note: "Draft filled. It has not been saved.",
        };
      }
      if (operation === "save" && editor.auto_save) {
        await editor.beforeLeave?.();
        return {
          outcome: "saved",
          saved: true,
          note: "Inline changes are saved; no separate form save is needed.",
        };
      }
      if (operation === "save") {
        if (!editor.save)
          throw new Error(
            "Saving this editor is unavailable in its current state.",
          );
        const result = await editor.save();
        if (!result)
          throw new Error(
            "Save did not complete. The draft is still open; inspect the error before retrying.",
          );
        await settled();
        if (
          typeof result === "object" &&
          result !== null &&
          "outcome" in result
        )
          return { editor: editor.kind, ...result };
        return { outcome: "saved", saved: true, editor: editor.kind, result, command_id: (result as {__command_id?: string} | null)?.__command_id };
      }
      if (operation === "close" && editor.auto_save)
        await editor.beforeLeave?.();
      if (operation === "close" && editor.dirty && !editor.auto_save)
        throw new Error(
          "This editor has unsaved changes. Save them, or ask the owner before discarding.",
        );
      if (operation === "discard" && editor.discard) editor.discard();
      else editor.close();
      await settled();
      return {
        outcome: operation === "discard" ? "discarded" : "closed",
        saved: false,
      };
    },
  };
}
