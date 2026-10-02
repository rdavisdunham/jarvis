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

/** A settings section: a panel with a section title, optional description and trailing action. */
export function SettingsGroup({title, description, action, icon, className = "", children, labelledBy}: {
  title: ReactNode; description?: ReactNode; action?: ReactNode; icon?: ReactNode; className?: string; children?: ReactNode; labelledBy?: string;
}) {
  return <section className={"settings-panel " + className} aria-labelledby={labelledBy}>
    <header className="settings-panel-head"><div><h2 id={labelledBy}>{icon}{title}</h2>{description && <p>{description}</p>}</div>{action}</header>
    {children}
  </section>;
}

/** One field row: label and hint on the left, the control on the right (stacked when narrow). */
export function SettingRow({label, hint, children, as = "div", className = "", labelId}: {
  label: ReactNode; hint?: ReactNode; children?: ReactNode; as?: "div" | "label"; className?: string; labelId?: string;
}) {
  const Tag = as;
  return <Tag className={"setting-row " + className}>
    <span className="setting-text"><strong className="setting-label" id={labelId}>{label}</strong>{hint && <small className="setting-hint">{hint}</small>}</span>
    {children !== undefined && <span className="setting-control">{children}</span>}
  </Tag>;
}
