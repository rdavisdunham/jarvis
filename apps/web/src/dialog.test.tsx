import { expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { Dialog, trapTarget } from "./ux";
import { createToolRegistry } from "./copilot";

it("renders one modal shell with dialog semantics on the inner element", () => {
  const html = renderToStaticMarkup(
    <Dialog as="form" backdropClassName="activity-backdrop" className="dialog" aria-labelledby="t" onBackdrop={() => {}}>
      <h2 id="t">Title</h2>
    </Dialog>,
  );
  expect(html).toMatch(/^<div class="modal-backdrop activity-backdrop"><form role="dialog" aria-modal="true" class="dialog" aria-labelledby="t">/);
  expect(html.match(/role="dialog"/g)).toHaveLength(1);
});
it("defaults to a section element", () => {
  expect(renderToStaticMarkup(<Dialog aria-label="Details">x</Dialog>)).toContain('<section role="dialog" aria-modal="true" aria-label="Details">x</section>');
});
it("wraps Tab inside the dialog and only pulls idle focus in", () => {
  const items = ["first", "middle", "last"];
  expect(trapTarget(items, "last", false, true, false)).toBe("first");
  expect(trapTarget(items, "first", true, true, false)).toBe("last");
  expect(trapTarget(items, "middle", false, true, false)).toBeNull();
  // Focus on the page body (e.g. after a backdrop click) returns into the dialog.
  expect(trapTarget(items, null, false, false, true)).toBe("first");
  expect(trapTarget(items, null, true, false, true)).toBe("last");
  // Focus in the conversation panel is left alone so chat stays usable beside a dialog.
  expect(trapTarget(items, "chat-input", false, false, false)).toBeNull();
  expect(trapTarget([], null, false, false, true)).toBeNull();
});
it("site tool registry registers, replaces and unregisters handlers", async () => {
  const registry = createToolRegistry();
  const first = { name: "eri_site_control", description: "", available: true, handler: async () => 1 };
  const second = { ...first, handler: async () => 2 };
  const dropFirst = registry.register(first);
  const dropSecond = registry.register(second);
  dropFirst();
  expect(await registry.get("eri_site_control")?.handler({} as never)).toBe(2);
  dropSecond();
  expect(registry.get("eri_site_control")).toBeUndefined();
});
