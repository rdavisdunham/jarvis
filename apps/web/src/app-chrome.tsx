import { Bell } from "lucide-react";

/** The header bell. Its name always starts with "Notifications" and carries the unread count. */
export function NotificationBell({ unread, current, onOpen }: { unread: number; current: boolean; onOpen: () => void }) {
  const name = "Notifications" + (unread ? ", " + unread + " unread" : "");
  return <button type="button" className={"icon-button " + (unread ? "has-notice" : "")} aria-label={name} title={name}
    aria-current={current ? "page" : undefined} onClick={onOpen}>
    <Bell size={19} aria-hidden="true"/>
    {unread > 0 && <i aria-hidden="true"/>}
  </button>;
}

/** The page footer's status line. When Dani's work needs attention it is a link into Activity. */
export function FooterStatus({ text, onAttention }: { text: string; onAttention?: () => void }) {
  return onAttention
    ? <button type="button" className="footer-status-link" onClick={onAttention}>{text}</button>
    : <span>{text}</span>;
}
