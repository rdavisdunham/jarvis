// Product, protocol and saved-state names remain Eridani/eri-*.
export const ASSISTANT_NAME = "Dani";
// Browser ASR spellings for DAN-ee and DAH-nee; no fuzzy name matching.
export const SPOKEN_NAMES = ["dani", "danny", "danni", "donny", "donnie"] as const;
const prefixes = ["hey", "hi", "okay", "ok"];
const wakePattern = new RegExp(
  `^(?:(?:${prefixes.join("|")})[,\\s]+)?(?:${SPOKEN_NAMES.join("|")})(?=$|[.!?,\\s])(?:[.!?,\\s]*)(.*)$`, "is",
);
export function wakeRequest(text: string): string | null {
  const match = text.trim().match(wakePattern);
  return match ? match[1].trim() : null;
}
export function isPartialWakePhrase(text: string): boolean {
  const words = text.toLowerCase().replace(/[^a-z ]/g, " ").replace(/\s+/g, " ").trim();
  return ["", ...prefixes.map(prefix => `${prefix} `)]
    .some(prefix => SPOKEN_NAMES.some(name => `${prefix}${name}`.startsWith(words)));
}
export const WAKE_HINTS = SPOKEN_NAMES.flatMap(name => [name, `hey ${name}`]);
