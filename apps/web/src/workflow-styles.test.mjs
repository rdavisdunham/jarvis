import { readFileSync } from "node:fs";
import { expect, it } from "vitest";
it("disables thinking animation with reduced motion and reserves stable activity space", () => {
  const css = readFileSync(new URL("./remotea.css", import.meta.url), "utf8");
  expect(css).toMatch(/@media\s*\(prefers-reduced-motion:\s*reduce\)\s*\{[^}]*animation:\s*none/);
  expect(css).toMatch(/\.run-activity\s*\{[^}]*min-height:\s*28px/);
});
