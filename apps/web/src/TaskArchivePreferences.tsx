import { useEffect, useState } from "react";
import { SettingRow, SettingsGroup } from "./SettingsLayout";
import { useStructureActions } from "./structure-actions";

export function TaskArchivePreferences({ enabled, canEdit, shared, onChanged }: {
  enabled: boolean;
  canEdit: boolean;
  shared: boolean;
  onChanged: () => void;
}) {
  const [saved, setSaved] = useState(enabled);
  const { run, busy, error } = useStructureActions();
  useEffect(() => setSaved(enabled), [enabled]);

  const change = async (value: boolean) => {
    if (!canEdit || busy) return;
    try {
      const preferences = await run<{ auto_archive_finished_tasks: boolean }>("settings.update", {
        auto_archive_finished_tasks: value,
      });
      setSaved(preferences.auto_archive_finished_tasks);
      onChanged();
    } catch {
      // Keep the last saved value visible; the command hook exposes the error.
    }
  };

  return <SettingsGroup title="Finished tasks" description={shared
    ? "This policy applies to tasks in this shared workspace. Only its owner can change it."
    : "This policy applies to tasks in your personal workspace."}>
    <SettingRow as="label" label="Automatically archive finished tasks"
      hint="File completed and canceled tasks away after 24 hours. Their outcome, notes, and history stay available in Archived.">
      <input type="checkbox" checked={saved} disabled={busy || !canEdit}
        aria-label="Automatically archive finished tasks"
        onChange={event => void change(event.target.checked)} />
    </SettingRow>
    <p className="footnote">Tasks and Organization hide finished work immediately; use Show completed and canceled to see it.
      Today keeps your completed and canceled items below unfinished work for the day.</p>
    {!saved && <p className="footnote">Automatic archiving is paused. The display filters still apply. Turning it on again files eligible older tasks on the next background pass.</p>}
    {error && <p role="alert">{error}</p>}
  </SettingsGroup>;
}
