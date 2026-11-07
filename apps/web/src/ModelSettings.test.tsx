// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SettingsPanel } from "./components";
import type { Bootstrap } from "./types";

let element: HTMLDivElement;
let root: Root;
beforeEach(() => {
  (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
  element = document.createElement("div");
  document.body.append(element);
  root = createRoot(element);
});
afterEach(async () => {
  await act(async () => root.unmount());
  element.remove();
});

const option = (id: string, label: string, model: string, reasoning_effort: string | null) =>
  ({ id, label, model, reasoning_effort, provider: "openai", available: true }) as Bootstrap["agent_options"][number];

function boot(overrides: Partial<Bootstrap> = {}): Bootstrap {
  return {
    agent_profile: "luna",
    agent_reasoning: "low",
    agent_provider: "openai",
    agent_model: "gpt-5.6-luna",
    chat_profile: "terra-6.1",
    chat_model: "gpt-6.1-terra",
    chat_reasoning: "medium",
    agent_options: [
      option("luna", "GPT-5.6 Luna · reasoning", "gpt-5.6-luna", "low"),
      option("terra-6.1", "GPT-6.1 Terra · reasoning", "gpt-6.1-terra", "medium"),
    ],
    preferences: { agent_profile: "luna", chat_profile: "terra-6.1", monthly_budget_usd: 20 },
    budget: { tracking_enabled: false, budget_mode: "disabled" },
    ...overrides,
  } as unknown as Bootstrap;
}

const select = (label: string) => element.querySelector<HTMLSelectElement>(`select[aria-label="${label}"]`)!;

describe("model settings", () => {
  it("shows typed chat on Terra and background work on Luna, and saves each separately", async () => {
    const onSave = vi.fn().mockResolvedValue(undefined);
    await act(async () =>
      root.render(<SettingsPanel section="system" boot={boot()} busy={false} onSave={onSave} onPush={vi.fn()} />));
    expect(select("Typed chat model").value).toBe("terra-6.1");
    expect(select("Background model").value).toBe("luna");
    expect(element.textContent).toContain("Terra uses medium reasoning effort");

    await act(async () => {
      const chat = select("Typed chat model");
      chat.value = "luna";
      chat.dispatchEvent(new Event("change", { bubbles: true }));
    });
    expect(onSave).toHaveBeenLastCalledWith({ chat_profile: "luna" });
    await act(async () => {
      const background = select("Background model");
      background.value = "terra-6.1";
      background.dispatchEvent(new Event("change", { bubbles: true }));
    });
    expect(onSave).toHaveBeenLastCalledWith({ agent_profile: "terra-6.1" });
    expect(onSave).toHaveBeenCalledTimes(2);
  });

  it("falls back to Terra for an older server without a typed-chat field", async () => {
    const older = boot({ chat_profile: undefined, chat_model: undefined, chat_reasoning: undefined });
    older.preferences = { ...older.preferences, chat_profile: undefined };
    await act(async () =>
      root.render(<SettingsPanel section="system" boot={older} busy onSave={vi.fn()} onPush={vi.fn()} />));
    expect(select("Typed chat model").value).toBe("terra-6.1");
    expect(select("Typed chat model").disabled).toBe(true);
  });
});
