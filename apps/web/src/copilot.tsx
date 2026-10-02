import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useRef,
  type ReactNode,
} from "react";
import { actionSchema } from "./site-actions";
import type { UIAction, UIContext } from "./types";
import { EditorProvider } from "./editor-control";

// The existing authenticated chat/voice transports deliver tool calls. This is only the
// browser-side registry of typed handlers; there is no second model runtime or cloud
// service. (It replaced a CopilotKit registry that pulled ~700 KB of unused runtime.)
export type SiteTool = {
  name: string;
  description: string;
  available: boolean;
  handler: (args: UIAction) => Promise<unknown>;
};
export type SiteToolRegistry = {
  register: (tool: SiteTool) => () => void;
  get: (name: string) => SiteTool | undefined;
};
export function createToolRegistry(): SiteToolRegistry {
  const tools = new Map<string, SiteTool>();
  return {
    register(tool) {
      tools.set(tool.name, tool);
      return () => {
        if (tools.get(tool.name) === tool) tools.delete(tool.name);
      };
    },
    get: (name) => tools.get(name),
  };
}
const ToolRegistryContext = createContext<SiteToolRegistry | null>(null);
export function SiteCopilot({ children }: { children: ReactNode }) {
  const registry = useMemo(createToolRegistry, []);
  return (
    <ToolRegistryContext.Provider value={registry}>
      <EditorProvider>{children}</EditorProvider>
    </ToolRegistryContext.Provider>
  );
}
const SITE_CONTROL = "eri_site_control";
export function useSiteControl(
  context: UIContext,
  apply: (action: UIAction) => Promise<Record<string, unknown> | void>,
  enabled: boolean,
) {
  const registry = useContext(ToolRegistryContext);
  if (!registry) throw new Error("useSiteControl must be used inside SiteCopilot.");
  const contextRef = useRef(context);
  contextRef.current = context;
  const applyRef = useRef(apply);
  applyRef.current = apply;
  useEffect(
    () =>
      registry.register({
        name: SITE_CONTROL,
        description:
          "Use typed page, chat, filter, layout, editor and device-preference actions. Draft patches do not save; server commands and remote receipts establish saved outcomes.",
        available: enabled,
        handler: async (args) => {
          const data = await applyRef.current(args);
          await new Promise<void>((resolve) =>
            requestAnimationFrame(() => requestAnimationFrame(() => resolve())),
          );
          const observed = contextRef.current;
          return {
            status: "displayed",
            data: {
              ...data,
              observed: {
                view: observed.view,
                layout: observed.layout,
                visible_ids: observed.visible_ids,
                reported_visible_count: observed.visible_ids.length,
                query: observed.query,
                assignee: observed.assignee,
              },
            },
          };
        },
      }),
    [registry, enabled],
  );
  return async (action: UIAction) => {
    const tool = registry.get(SITE_CONTROL);
    if (!enabled || !tool?.available)
      throw new Error("Site controls are unavailable.");
    // These calls arrive over our server-owned voice/text transport.
    return await tool.handler(actionSchema.parse(action) as UIAction);
  };
}
