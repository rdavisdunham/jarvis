/**
 * The Report a bug screenshot: the visible page drawn by modern-screenshot, loaded on demand so the
 * library stays out of the main chunk. Downscaled to MAX_WIDTH and re-encoded as JPEG until it fits
 * MAX_BYTES. Elements marked `data-bug-report-ignore` (the button itself) are left out.
 */
export const MAX_WIDTH = 1600;
export const MAX_BYTES = 1_500_000;
const QUALITIES = [0.82, 0.7, 0.55, 0.4];
const TIMEOUT_MS = 10_000;

/** Decoded size of a base64 data URL. */
export const dataUrlBytes = (url: string) => Math.floor(((url.length - url.indexOf(",") - 1) * 3) / 4);

export async function captureScreenshot(): Promise<string> {
  const { domToCanvas } = await import("modern-screenshot");
  const width = window.innerWidth, height = window.innerHeight;
  const scale = Math.min(window.devicePixelRatio || 1, MAX_WIDTH / width);
  const background = getComputedStyle(document.body).backgroundColor;
  const canvas = await Promise.race([
    domToCanvas(document.body, {
      width, height, scale, backgroundColor: background, timeout: 5_000,
      filter: (node) => !(node instanceof HTMLElement && node.hasAttribute("data-bug-report-ignore")),
      features: { restoreScrollPosition: true },
    }),
    new Promise<never>((_, reject) => setTimeout(() => reject(new Error("timed out")), TIMEOUT_MS)),
  ]);
  for (const quality of QUALITIES) {
    const url = canvas.toDataURL("image/jpeg", quality);
    if (url.startsWith("data:image/jpeg") && dataUrlBytes(url) <= MAX_BYTES) return url;
  }
  throw new Error("the screenshot was larger than 1.5 MB");
}
