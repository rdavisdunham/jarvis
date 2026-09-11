import { useMemo, useRef, type ReactNode } from "react";
import {
  CopilotKitContext,
  CopilotKitCoreReact,
} from "@copilotkit/react-core/v2/context";
import {
  useCopilotKit,
  useFrontendTool,
  useAgentContext,
} from "@copilotkit/react-core/v2/headless";
import { z } from "zod";
import type { UIAction, UIContext } from "./types";

const views = [
  "today",
  "inbox",
  "week",
  "all",
  "reminders",
  "calendar",
  "memory",
  "notifications",
  "settings",
] as const;
const actionSchema = z.object({
  id: z.string().max(150),
  kind: z
    .enum(["show", "chat", "search", "filter", "form", "calendar"])
    .default("show"),
  view: z.enum(views).optional(),
  entity_id: z.string().nullable().optional(),
  mode: z.enum(["open", "close", "auto"]).optional(),
  query: z.string().max(300).optional(),
  status: z
    .enum([
      "all",
      "open",
      "in_progress",
      "waiting",
      "deferred",
      "completed",
      "cancelled",
    ])
    .optional(),
  project: z.string().max(200).optional(),
  date: z
    .string()
    .regex(/^\d{4}-\d{2}-\d{2}$/)
    .optional(),
  work_kind: z.enum(["all", "task", "reminder"]).optional(),
  form: z.enum(["task", "reminder"]).optional(),
});

// The existing authenticated chat/voice transports deliver tool calls. CopilotKit owns
// the browser registry and handler lifecycle; no second model runtime or cloud service.
export function SiteCopilot({ children }: { children: ReactNode }) {
  const copilotkit = useMemo(() => new CopilotKitCoreReact({}), []);
  const value = useMemo(
    () => ({ copilotkit, executingToolCallIds: new Set<string>() }),
    [copilotkit],
  );
  return (
    <CopilotKitContext.Provider value={value}>
      {children}
    </CopilotKitContext.Provider>
  );
}
export function useSiteControl(
  context: UIContext,
  apply: (action: UIAction) => Promise<void>,
  enabled: boolean,
) {
  const { copilotkit } = useCopilotKit();
  const applyRef = useRef(apply);
  applyRef.current = apply;
  useAgentContext({
    description:
      "Eridani's current page, selected records, filters and chat visibility",
    value: { ...context },
  });
  useFrontendTool(
    {
      name: "eri_site_control",
      description:
        "Open pages and records, control chat, search and filter tasks, and open task/reminder forms.",
      parameters: actionSchema,
      available: enabled,
      handler: async (args) => {
        await applyRef.current(args as UIAction);
        return { status: "displayed" };
      },
    },
    [enabled],
  );
  return async (action: UIAction) => {
    const tool = copilotkit.getTool({ toolName: "eri_site_control" });
    if (!enabled || !tool?.handler)
      throw new Error("Site controls are unavailable.");
    const args = actionSchema.parse(action);
    // These calls arrive over our server-owned voice/text transport, not an AG-UI run.
    await tool.handler(args, {
      toolCall: {
        id: action.id,
        type: "function",
        function: { name: tool.name, arguments: JSON.stringify(args) },
      },
    });
  };
}
