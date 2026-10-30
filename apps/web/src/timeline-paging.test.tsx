import { afterEach, describe, expect, it, vi } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { postAction } from "./api";
import { Bubble, RunDrawer } from "./Remotea";
import {
  anchoredScrollTop, earlierLine, mergeDetail, nextOffset, scrollAnchor, timeline, type RunCommand, type RunDetail,
} from "./remotea-model";

const RUN = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa";
type TurnRow = RunDetail["turns"][number];
const at = (minute: number) => `2026-10-30T10:${String(minute).padStart(2, "0")}:00Z`;
const turn = (seq: number, values: Partial<TurnRow> = {}): TurnRow => ({
  id: "t" + seq, seq, text: `answer ${seq}`, clipped: false, truncated: false, generation: 1, at: at(seq), prompt: "", ...values,
});
const command = (minute: number, values: Partial<RunCommand> = {}): RunCommand => ({
  id: "c" + minute, op_id: "o" + minute, kind: "input", status: "acked", generation: 1, mode: "queue", text: `message ${minute}`,
  outcome: null, created_at: at(minute), updated_at: at(minute), ...values,
});
const detail = (values: Partial<RunDetail> = {}): RunDetail => ({
  id: RUN, task: { id: "task", title: "Paged task", status: "in_progress" }, state: "executing", state_revision: 3,
  state_reason: "", reason_text: "", state_since: at(0), generation: 1, plan_revision: 1, tool: "claude", model: "",
  branch: "task/x", base_branch: "main", worktree: "", effective_permission_mode: "", turn_count: 1, turn_cap: 15,
  started_at: null, finished_at: null, stop_requested_at: null, stop_mode: "", surf: "surf-1",
  host: { id: "h", name: "desk", connected: true }, structured_answers: true, session: null, created_at: at(0),
  history: [{ at: at(0), from: "queued", to: "spawning" }], turns: [], commands: [], questions: [], plan: null,
  more: { turns: false, commands: false, offset: 0, limit: 100, remaining: { turns: 0, commands: 0 } },
  retention: { expires_at: null, expired: false },
  ...values,
});

afterEach(() => vi.unstubAllGlobals());

describe("mergeDetail: an older page folded into the drawer's run", () => {
  const newest = detail({
    turns: [turn(5), turn(6), turn(7)], commands: [command(6), command(8, { status: "executed" })],
    more: { turns: true, commands: true, offset: 0, limit: 3, remaining: { turns: 4, commands: 1 } },
  });
  // The older page overlaps (t5, c6): two turns arrived between the first load and Load earlier.
  const older = detail({
    state: "spawning", state_revision: 1,
    turns: [turn(3), turn(4), turn(5, { text: "stale copy" })], commands: [command(2), command(6, { status: "queued" })],
    more: { turns: true, commands: false, offset: 3, limit: 3, remaining: { turns: 1, commands: 0 } },
  });

  it("orders turns and commands by time and keeps one row per id, the newer copy winning", () => {
    const merged = mergeDetail(newest, older);
    expect(merged.turns.map(t => t.id)).toEqual(["t3", "t4", "t5", "t6", "t7"]);
    expect(merged.turns.find(t => t.id === "t5")!.text).toBe("answer 5");
    expect(merged.commands.map(c => [c.id, c.status])).toEqual([["c2", "acked"], ["c6", "acked"], ["c8", "executed"]]);
    // Everything that is not paged comes from the current run, never from the older page's stale copy.
    expect(merged).toMatchObject({ state: "executing", state_revision: 3 });
    expect(merged.history).toBe(newest.history);
  });

  it("takes `more` from the oldest page, so the next Load earlier asks for the page after it", () => {
    const merged = mergeDetail(newest, older);
    expect(merged.more).toEqual(older.more);
    expect(nextOffset(newest)).toBe(3);
    expect(nextOffset(merged)).toBe(6);
    // A poll replaces the newest page; merging the kept older pages back in keeps the oldest `more`.
    const polled = detail({ turns: [turn(6), turn(7), turn(8)], more: { turns: true, commands: true, offset: 0, limit: 3 } });
    expect(mergeDetail(polled, merged).more).toEqual(older.more);
    expect(mergeDetail(polled, merged).turns.map(t => t.seq)).toEqual([3, 4, 5, 6, 7, 8]);
    const last = detail({ turns: [turn(1), turn(2)], more: { turns: false, commands: false, offset: 6, limit: 3 } });
    expect(nextOffset(mergeDetail(merged, last))).toBeNull();
  });

  it("is idempotent and ignores a page for another run", () => {
    const once = mergeDetail(newest, older);
    expect(mergeDetail(once, older).turns).toEqual(once.turns);
    expect(mergeDetail(newest, { ...older, id: "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb" })).toBe(newest);
  });

  it("keeps timeline ids stable, so the anchored entry is found again after the merge", () => {
    const before = timeline(newest), after = timeline(mergeDetail(newest, older));
    expect(new Set(before.map(e => e.id)).size).toBe(before.length);
    expect(new Set(after.map(e => e.id)).size).toBe(after.length);
    for (const entry of before) expect(after.map(e => e.id)).toContain(entry.id);
    expect(after.findIndex(e => e.id === "turn:t5")).toBeGreaterThan(after.findIndex(e => e.id === "turn:t3"));
  });

  it("anchors the scroll to the first entry on screen and restores it after older rows land above", () => {
    // The log's top edge is at 100; the first row is scrolled off, the second is partly visible.
    const rows = [{ id: "history:0", top: 20, bottom: 90 }, { id: "turn:t5", top: 92, bottom: 140 }, { id: "turn:t6", top: 140, bottom: 200 }];
    const anchor = scrollAnchor(rows, 100)!;
    expect(anchor).toEqual({ id: "turn:t5", offset: -8 });
    // After the merge two older turns sit above it, 120px further down the content; scrolling by that keeps it put.
    expect(anchoredScrollTop(400, anchor, 112)).toBe(520);
    expect(anchoredScrollTop(400, anchor, -8)).toBe(400);  // Nothing landed above: no jump.
    expect(scrollAnchor([], 0)).toBeNull();
  });
});

describe("what the drawer says about missing text", () => {
  it("says how many older rows are paged out, when the server counted them", () => {
    expect(earlierLine({ turns: true, commands: true, offset: 0, limit: 100, remaining: { turns: 37, commands: 1 } }))
      .toBe("37 older turns and 1 older command not shown yet");
    expect(earlierLine({ turns: true, commands: false, offset: 0, limit: 100 })).toBe("Older turns and commands are not shown yet");
    expect(earlierLine({ turns: false, commands: false, offset: 0, limit: 100 })).toBeNull();
    expect(earlierLine(undefined)).toBeNull();
  });

  it("shows Full response and Clipped on a clipped turn, and Truncated by host on a truncated one", () => {
    const [clipped, truncated, plain] = timeline(detail({ turns: [
      turn(1, { text: "x".repeat(4000), clipped: true }), turn(2, { text: "cut", truncated: true }), turn(3, { text: "fine" })] }))
      .filter(e => e.kind === "turn");
    const clippedHtml = renderToStaticMarkup(<Bubble entry={clipped} runId={RUN}/>);
    expect(clippedHtml).toContain(">Clipped<");
    expect(clippedHtml).toMatch(/<button[^>]*>.*Full response<\/button>/);
    expect(clippedHtml).not.toContain("Truncated by host");
    expect(clippedHtml).toContain('data-entry="turn:t1"');
    const truncatedHtml = renderToStaticMarkup(<Bubble entry={truncated} runId={RUN}/>);
    expect(truncatedHtml).toContain(">Truncated by host<");
    expect(truncatedHtml).not.toContain("Full response");
    expect(truncatedHtml).not.toContain(">Clipped<");
    expect(truncatedHtml).not.toContain("[truncated]");
    const plainHtml = renderToStaticMarkup(<Bubble entry={plain} runId={RUN}/>);
    expect(plainHtml).not.toMatch(/Clipped|Truncated by host|Full response|Expired/);
  });

  it("offers Load earlier with the count when a page remains, and says when retention deleted the run's turns", () => {
    const paged = renderToStaticMarkup(<RunDrawer id={RUN} onClose={() => {}} preloaded={detail({
      turns: [turn(5)], more: { turns: true, commands: false, offset: 0, limit: 100, remaining: { turns: 3, commands: 0 } } })}/>);
    expect(paged).toContain(">Paged out<");
    expect(paged).toContain("3 older turns not shown yet");
    expect(paged).toMatch(/<button[^>]*>.*Load earlier<\/button>/);
    expect(paged).not.toContain(">Expired<");
    const expired = renderToStaticMarkup(<RunDrawer id={RUN} onClose={() => {}} preloaded={detail({
      state: "completed", finished_at: "2026-07-01T10:00:00Z", retention: { expires_at: "2026-09-29T10:00:00Z", expired: true } })}/>);
    expect(expired).toContain(">Expired<");
    expect(expired).toContain("turns and commands were deleted");
    expect(expired).not.toContain("Load earlier");
  });
});

describe("run actions retry with one action_id", () => {
  it("resends a lost message with the same action_id and the reviewed revision", async () => {
    const bodies: Record<string, unknown>[] = [];
    let calls = 0;
    vi.stubGlobal("fetch", async (_url: string, init: RequestInit) => {
      bodies.push(JSON.parse(String(init.body)));
      if (++calls === 1) throw new TypeError("network down");
      return new Response(JSON.stringify({ ok: true }), { status: 200 });
    });
    await postAction(`/remotea/runs/${RUN}/message`, { text: "hi", mode: "steer", expected_revision: 3 }, { delayMs: 0 });
    expect(bodies).toHaveLength(2);
    expect(bodies[0]).toEqual(bodies[1]);
    expect(bodies[0]).toMatchObject({ text: "hi", mode: "steer", expected_revision: 3, action_id: expect.any(String) });
  });
});
