export const settingsSections = [
  {id: "profile", label: "Profile & display", description: "Your name and how the planner looks."},
  {id: "organization", label: "Organization", description: "How Eri learns your filing habits and work hours."},
  {id: "notifications", label: "Notifications", description: "Reminders, quiet hours and delivery on this device."},
  {id: "voice", label: "Voice", description: "Eri’s voice and wake-word listening."},
  {id: "integrations", label: "Connections", description: "Google, Linear and access for other agents."},
  {id: "privacy", label: "Privacy & memory", description: "Conversation history and personal memory."},
  {id: "sharing", label: "Sharing", description: "Workspaces, invitations and permissions."},
  {id: "system", label: "System", description: "Models, usage and data exports."},
] as const;
export type SettingsSection = typeof settingsSections[number]["id"];
export function readSettingsSection(value: string | null): SettingsSection {
  return settingsSections.find(section => section.id === value)?.id ?? "profile";
}
