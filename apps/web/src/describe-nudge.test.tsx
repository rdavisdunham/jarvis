// Clients and projects without the owner's description ask for one on their card: overviews, placement by meaning
// and Dani build on it (docs/ORGANIZATION_KNOWLEDGE_PLAN.md).
import { describe, expect, it, vi } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";

vi.mock("./Remotea", () => ({ RunTask: () => null }));
vi.mock("./Linear", () => ({ LinearTask: () => null, LinearSettings: () => null }));

import { RecordCard } from "./RecordCard";
import type { CustomRecord, Schema } from "./structure-types";

const type = (id: string, name: string) => ({
  id, name, plural: name + "s", description: "", capabilities: [], parent_types: [], archived: false,
  statuses: [{ id: "active", name: "Active", meaning: "open" }], fields: [],
});
const schema = { revision: 1, relationships: [], understandings: [], types: [type("client", "Client"), type("note", "Note")] } as unknown as Schema;
const row = (extra: Partial<CustomRecord> = {}): CustomRecord => ({
  id: "66666666-6666-4666-8666-666666666666", revision: 1, schema_revision: 1, type_id: "client", type_name: "Client",
  title: "ABC", body: "", values: {}, inherited: {}, parent_id: null, status_id: "active", archived: false,
  task_id: null, note_id: null, capabilities: [], home: [], links: [], local_notes: "", ...extra,
} as CustomRecord);
const card = (initial: CustomRecord, canEdit = true) => renderToStaticMarkup(
  <RecordCard schema={schema} initial={initial} choices={[]} canEdit={canEdit} onClose={() => {}} onChanged={async () => {}} onOpen={() => {}}/>);

describe("description nudge", () => {
  it("asks an undescribed client for its description", () => {
    const html = card(row());
    expect(html).toContain("Describe this client.");
    expect(html).toContain("what your work covers");
  });
  it("stays quiet once described, for viewers, and on other record types", () => {
    expect(card(row({ body: "Home services; we run their call analytics." }))).not.toContain("Describe this");
    expect(card(row(), false)).not.toContain("Describe this");
    expect(card(row({ type_id: "note", type_name: "Note" }))).not.toContain("Describe this");
  });
});
