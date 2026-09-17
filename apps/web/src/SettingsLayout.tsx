import type { ReactNode } from "react";
import { Tabs } from "./ux";
import { settingsSections, type SettingsSection } from "./settings-sections";
export function SettingsLayout({section, onChange, children}: {
  section: SettingsSection; onChange: (section: SettingsSection) => void; children: ReactNode;
}) {
  const selected = settingsSections.find(item => item.id === section)!;
  return <div className="settings-container"><div className="settings-layout">
    <div className="settings-navigation"><Tabs id="settings-tab" label="Settings sections"
      orientation="vertical" className="settings-tabs" panel="settings-panel" value={section}
      onChange={onChange} items={settingsSections}/></div>
    <label className="settings-mobile-nav">Settings section
      <select value={section} onChange={event => onChange(event.target.value as SettingsSection)}>
        {settingsSections.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}
      </select>
    </label>
    <div id="settings-panel" className="settings-content" role="region" aria-label={selected.label}>
      <p className="settings-intro">{selected.description}</p>{children}
    </div>
  </div></div>;
}
