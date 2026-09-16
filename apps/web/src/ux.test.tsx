import { expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { Tabs, tabDestination, humanLabel, priorityLabels } from "./ux";
import { readRecordLink, recordLink } from "./record-links";
it("supports wraparound and Home/End without consuming typing keys", () => {
  expect(tabDestination("ArrowLeft",0,6)).toBe(5);
  expect(tabDestination("ArrowRight",5,6)).toBe(0);
  expect(tabDestination("Home",3,6)).toBe(0);
  expect(tabDestination("End",3,6)).toBe(5);
  expect(tabDestination("Enter",3,6)).toBeNull();
});
it("associates a single active tab with its panel and roves the tab stop", () => {
  const html = renderToStaticMarkup(<Tabs id="settings" label="Settings" panel="panel" value="voice" onChange={() => {}} items={[{id:"profile",label:"Profile"},{id:"voice",label:"Voice"}]}/>);
  expect(html).toContain('id="settings-voice"');expect(html.match(/tabindex="0"/g)).toHaveLength(1);
  expect(html).toContain('aria-controls="panel" aria-selected="true" tabindex="0"');
});
it("keeps workspace identity in record links and rejects malformed targets", () => {
  const id = "e158639b-9d0e-4208-8b15-57d6428e9fd1", workspace = "0d572eb5-1630-46f8-bb2a-e68d1e6e3bc2";
  const link = recordLink("https://app.example.test",{kind:"task",id},workspace);
  expect(readRecordLink(new URL(link).search)).toEqual({kind:"task",id,workspace});
  expect(readRecordLink("?record=task:../../../admin")).toBeNull();
  expect(readRecordLink(`?record=task:${id}&workspace=https://evil.test`)).toBeNull();
  expect(readRecordLink(`?record=unknown:${id}`)).toBeNull();
});
it("uses shared human status and priority labels", () => {expect(humanLabel("in_progress")).toBe("In progress");expect(humanLabel("owner")).toBe("Me");expect(priorityLabels).toEqual(["None","Low","Medium","High"]);});
