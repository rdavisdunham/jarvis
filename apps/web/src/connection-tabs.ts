// Settings → Connections tabs. Kept apart from settings-sections.ts, whose ids the backend drift guard reads as the
// settings sections. Deep link: `?view=settings&section=integrations&connection=<id>`, or the shorter
// `?view=settings&section=<id>`; Google's OAuth return (`google=…`) opens the Google tab.
export const connectionTabs = [
  {id: "agents", label: "Agent keys", description: "Keys that let other assistants work in this workspace."},
  {id: "remotea", label: "remotea", description: "Your remotea server, its self-check and each target's readiness."},
  {id: "linear", label: "Linear", description: "Linear connection, sync scope and team mapping."},
  {id: "google", label: "Google", description: "Google sign-in and Calendar."},
  {id: "meetings", label: "Meetings", description: "Meeting sources and meeting memory."},
] as const;
export type ConnectionTab = typeof connectionTabs[number]["id"];
export const isConnectionTab = (value: string | null | undefined): value is ConnectionTab =>
  connectionTabs.some(tab => tab.id === value);

/** The Connections tab a settings URL asks for: `connection`, a tab id in `section`, or Google's OAuth return. */
export function readConnectionTab(params: URLSearchParams): ConnectionTab {
  const connection = params.get("connection"), section = params.get("section");
  if (isConnectionTab(connection)) return connection;
  if (isConnectionTab(section)) return section;
  return params.has("google") ? "google" : "agents";
}
