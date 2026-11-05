import { readFileSync } from "node:fs";
import { expect, it } from "vitest";
it("disables thinking animation with reduced motion and reserves stable activity space", () => {
  const css = readFileSync(new URL("./remotea.css", import.meta.url), "utf8");
  expect(css).toMatch(/@media\s*\(prefers-reduced-motion:\s*reduce\)\s*\{[^}]*animation:\s*none/);
  expect(css).toMatch(/\.run-activity\s*\{[^}]*min-height:\s*28px/);
});
it("keeps the run stages calm: no pulse or rotation under reduced motion, 44px toggles, no all-caps labels", () => {
  const css = readFileSync(new URL("./run-stages.css", import.meta.url), "utf8");
  const reduced = css.match(/@media\s*\(prefers-reduced-motion:\s*reduce\)\s*\{([\s\S]*?)\n\}/)?.[1] ?? "";
  expect(reduced).toMatch(/run-step-marker::after\s*\{[^}]*animation:\s*none/);
  expect(reduced).toMatch(/run-stage-chevron\s*\{[^}]*transition:\s*none/);
  expect(css).toMatch(/\.run-stage-toggle\s*\{[^}]*min-height:\s*44px/);
  expect(css).not.toMatch(/text-transform:\s*uppercase/);
  // The orbit stays in the progress ring: no gradient fills anywhere else in the stages.
  expect(css).not.toMatch(/gradient\(/);
});
