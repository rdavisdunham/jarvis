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
import { actionSchema } from "./site-actions";
import type { UIAction, UIContext } from "./types";
import { EditorProvider } from "./editor-control";

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
      <EditorProvider>{children}</EditorProvider>
    </CopilotKitContext.Provider>
  );
}
export function useSiteControl(
  context: UIContext,
  apply: (action: UIAction) => Promise<Record<string, unknown> | void>,
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
        "Use typed page, chat, filter, layout, editor and device-preference actions. Draft patches do not save; server commands and remote receipts establish saved outcomes.",
      parameters: actionSchema,
      available: enabled,
      handler: async (args) => {
        const data = await applyRef.current(args as UIAction);
        return { status: "displayed", data };
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
    return await tool.handler(args, {
      toolCall: {
        id: action.id,
        type: "function",
        function: { name: tool.name, arguments: JSON.stringify(args) },
      },
    });
  };
}
