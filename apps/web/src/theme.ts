// Light / Dark / System preference. Applied from main.tsx before the first render so the page
// never flashes the wrong theme (the CSP forbids an inline <head> script).
export type ThemePreference = "light" | "dark" | "system";
export const THEME_KEY = "eridani-theme";
const META = { light: "#f7f7fa", dark: "#121219" } as const;

export function readTheme(): ThemePreference {
  try {
    const value = localStorage.getItem(THEME_KEY);
    return value === "light" || value === "dark" ? value : "system";
  } catch {
    return "system";
  }
}
function systemDark() {
  return typeof matchMedia === "function" && matchMedia("(prefers-color-scheme: dark)").matches;
}
export function applyTheme(preference: ThemePreference = readTheme()) {
  const root = document.documentElement;
  if (preference === "system") delete root.dataset.theme;
  else root.dataset.theme = preference;
  const dark = preference === "dark" || (preference === "system" && systemDark());
  document.querySelector('meta[name="theme-color"]')?.setAttribute("content", dark ? META.dark : META.light);
}
export function saveTheme(preference: ThemePreference) {
  try {
    if (preference === "system") localStorage.removeItem(THEME_KEY);
    else localStorage.setItem(THEME_KEY, preference);
  } catch {
    // Storage can be unavailable (private mode); the choice still applies to this page.
  }
  applyTheme(preference);
}
/** Keep the browser chrome colour in step when the system theme changes. */
export function watchSystemTheme() {
  if (typeof matchMedia !== "function") return;
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (readTheme() === "system") applyTheme("system");
  });
}
