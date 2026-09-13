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
  kind: EditorKind;
  record_id: string | null;
  dirty: boolean;
  busy: boolean;
  fields: string[];
};
type Editor = {
  kind: EditorKind;
  record_id?: string | null;
  dirty: boolean;
  busy?: boolean;
  schema: z.ZodObject;
  values: Record<string, unknown>;
  patch: (values: Record<string, unknown>) => void;
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
          outcome: "draft",
          editor: editor.kind,
          record_id: editor.record_id ?? null,
          dirty: editor.dirty,
          busy: !!editor.busy,
          fields: z.toJSONSchema(editor.schema.partial()),
          values: editor.values,
          saved: false,
          note: "These are device-local draft values. Reading or filling them does not save.",
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
        editor.patch(parsed.data);
        await settled();
        return {
          outcome: "draft_updated",
          saved: false,
          fields: Object.keys(parsed.data),
          note: "Draft filled. It has not been saved.",
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
        return { outcome: "saved", saved: true, editor: editor.kind, result };
      }
      if (operation === "close" && editor.dirty)
        throw new Error(
          "This editor has unsaved changes. Save them, or ask the owner before discarding.",
        );
      editor.close();
      await settled();
      return {
        outcome: operation === "discard" ? "discarded" : "closed",
        saved: false,
      };
    },
  };
}
