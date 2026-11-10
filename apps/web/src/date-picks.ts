import { shiftDate } from "./workspace";
import { dayOffset, monthDay } from "./work-views";

/** 0 = Sunday … 6 = Saturday, for a YYYY-MM-DD day. */
export function weekday(day: string) {
  return new Date(day + "T12:00:00Z").getUTCDay();
}
/** The coming Saturday, or `today` itself on a Saturday or Sunday. */
export function weekendFrom(today: string) {
  const dow = weekday(today);
  return dow === 0 || dow === 6 ? today : shiftDate(today, 6 - dow);
}
/** Next week's Monday (a week from today on a Monday). */
export function nextMonday(today: string) {
  const dow = weekday(today);
  return shiftDate(today, dow === 0 ? 1 : 8 - dow);
}
export function weekdayName(day: string, style: "short" | "long" = "short") {
  return new Date(day + "T12:00:00Z").toLocaleDateString("en-US", { timeZone: "UTC", weekday: style });
}
export type QuickPick = { label: string; date: string; hint: string };
export function quickPicks(today: string): QuickPick[] {
  const tomorrow = shiftDate(today, 1), weekend = weekendFrom(today), monday = nextMonday(today);
  return [
    { label: "Today", date: today, hint: weekdayName(today) },
    { label: "Tomorrow", date: tomorrow, hint: weekdayName(tomorrow) },
    { label: "This weekend", date: weekend, hint: weekdayName(weekend) },
    { label: "Next week", date: monday, hint: monthDay(monday) },
  ];
}
/** Row chip for a do date: "Do today", "Do tomorrow", "Do Tue" within the coming week, else "Do Oct 5". */
export function doLabel(date: string, today: string) {
  const offset = dayOffset(today, date);
  if (offset === 0) return "Do today";
  if (offset === 1) return "Do tomorrow";
  if (offset === -1) return "Do yesterday";
  if (offset > 1 && offset < 7) return "Do " + weekdayName(date);
  return "Do " + monthDay(date) + (date.slice(0, 4) !== today.slice(0, 4) ? ", " + date.slice(0, 4) : "");
}
export function monthTitle(day: string) {
  return new Date(day.slice(0, 7) + "-01T12:00:00Z").toLocaleDateString("en-US", { timeZone: "UTC", month: "long", year: "numeric" });
}
/** The same day of the month `amount` months away, clamped to that month's last day. */
export function shiftMonthKeepDay(day: string, amount: number) {
  const [y, m, d] = day.split("-").map(Number);
  const last = new Date(Date.UTC(y, m - 1 + amount + 1, 0)).getUTCDate();
  return new Date(Date.UTC(y, m - 1 + amount, Math.min(d, last))).toISOString().slice(0, 10);
}
