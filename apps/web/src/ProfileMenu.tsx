import { useEffect, useId, useRef, useState } from "react";
import { Brain, ChevronUp, LogOut, Settings2 } from "lucide-react";
import type { View } from "./types";

export function ProfileMenu({ name, view, personal, navigationOpen, onNavigate, onLogout }: {
  name: string; view: View; personal: boolean; navigationOpen: boolean;
  onNavigate: (view: "memory" | "settings") => void;
  onLogout: () => Promise<void>;
}) {
  const [open, setOpen] = useState(false), [busy, setBusy] = useState(false), [error, setError] = useState("");
  const root = useRef<HTMLDivElement>(null), trigger = useRef<HTMLButtonElement>(null);
  const menu = useRef<HTMLDivElement>(null), firstFocus = useRef<"first" | "last">("first");
  const id = useId();
  function close(restoreFocus = false) {
    setOpen(false);
    if (restoreFocus) trigger.current?.focus({ preventScroll: true });
  }
  useEffect(() => { setOpen(false); }, [view, personal, navigationOpen]);
  useEffect(() => {
    if (!open) return;
    const items = menu.current?.querySelectorAll<HTMLButtonElement>('[role="menuitem"]');
    (firstFocus.current === "last" ? items?.[items.length - 1] : items?.[0])?.focus({ preventScroll: true });
    const outside = (event: PointerEvent) => {
      if (!root.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", outside);
    return () => document.removeEventListener("pointerdown", outside);
  }, [open]);
  function navigate(target: "memory" | "settings") {
    close(true);
    onNavigate(target);
  }
  return <div className="profile-menu" ref={root} onBlur={event => {
    if (!event.currentTarget.contains(event.relatedTarget as Node | null)) close();
  }} onKeyDown={event => {
    if (open && event.key === "Escape") { event.preventDefault(); event.stopPropagation(); close(true); }
  }}>
    <button type="button" ref={trigger} className={"profile-trigger" + (open ? " open" : "")}
      aria-label={"Profile menu for " + name} aria-haspopup="menu" aria-expanded={open} aria-controls={open ? id : undefined}
      onClick={() => { firstFocus.current = "first"; setError(""); setOpen(!open); }}
      onKeyDown={event => {
        if (event.key === "ArrowDown" || event.key === "ArrowUp") {
          event.preventDefault(); firstFocus.current = event.key === "ArrowUp" ? "last" : "first";
          setError(""); setOpen(true);
        }
      }}>
      <span className="avatar" aria-hidden="true">{name.trim().charAt(0).toUpperCase() || "?"}</span>
      <span className="profile-identity"><strong>{name}</strong><span><i className="status-dot"/>Connected</span></span>
      <ChevronUp size={16} className="profile-chevron" aria-hidden="true"/>
    </button>
    {open && <div ref={menu} id={id} className="profile-dropdown" role="menu" aria-label="Profile" aria-busy={busy}
      onKeyDown={event => {
        const items = Array.from(menu.current?.querySelectorAll<HTMLButtonElement>('[role="menuitem"]:not(:disabled)') ?? []);
        const index = items.indexOf(document.activeElement as HTMLButtonElement);
        const next = event.key === "ArrowDown" ? (index + 1) % items.length
          : event.key === "ArrowUp" ? (index + items.length - 1) % items.length
          : event.key === "Home" ? 0 : event.key === "End" ? items.length - 1 : null;
        if (next !== null) { event.preventDefault(); items[next]?.focus(); }
        else if (event.key === "Tab") close();
      }}>
      {personal && <button type="button" role="menuitem" tabIndex={-1} disabled={busy}
        aria-current={view === "memory" ? "page" : undefined} onClick={() => navigate("memory")}><Brain size={17}/>Memory</button>}
      <button type="button" role="menuitem" tabIndex={-1} disabled={busy}
        aria-current={view === "settings" ? "page" : undefined} onClick={() => navigate("settings")}><Settings2 size={17}/>Settings</button>
      <div className="profile-menu-divider" role="separator"/>
      <button type="button" role="menuitem" tabIndex={-1} disabled={busy} onClick={async () => {
        setBusy(true); setError("");
        try { await onLogout(); close(); }
        catch (e) { setError((e as Error).message); }
        finally { setBusy(false); }
      }}><LogOut size={17}/>{busy ? "Logging out…" : "Log out"}</button>
      {error && <p role="alert" className="profile-menu-error">{error}</p>}
    </div>}
  </div>;
}
