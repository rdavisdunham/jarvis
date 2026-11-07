import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { ActivityPanel, attentionFirst, WorkCard, type WorkItem } from "./Activity";
import { FooterStatus, NotificationBell } from "./app-chrome";

const base: WorkItem = { id:"work-1",parent_id:null,conversation_id:"chat-1",request:"Um, actually, make that call tomorrow please.",
  status:"succeeded",revision:1,message:"Updated.",actions:[],cancel_requested:false,seen:false,
  created_at:"2026-09-16T15:00:00Z",updated_at:"2026-09-16T15:00:00Z",can_continue:false };
const render = (item: WorkItem) => renderToStaticMarkup(<WorkCard item={item} onRefresh={async()=>{}} onOpen={async()=>{}}/>);
describe("confirmed action cards",()=>{
  it("opens custom records and renders nested values readably",()=>{
    const html=render({...base,actions:[{id:"change-custom",command_id:"work-1:0",kind:"record",entity_id:"custom-1",title:"Docs",operation:"updated",fields:{values:{before:null,after:{Client:"ABC"}}},can_revert:true,revert_reason:"",reverted:false}]});
    expect(html).toContain("Edit");expect(html).toContain("Client: ABC");expect(html).not.toContain("[object Object]");
  });
  it("keeps the answered question inside one completed card",()=>{
    const html=render({...base,clarification_history:[{question:"What time?",answer:"9 a.m."}]});
    expect(html).toContain("Clarification history"); expect(html).toContain("What time?");
    expect(html).toContain("9 a.m."); expect(html).toContain("Completed");
    expect(html).not.toContain("Waiting for you");
    expect(html.match(/data-work-id=/g)).toHaveLength(1);
  });
  it("leads with the saved change, puts speech in closed details, and has no review step",()=>{
    const html=render({...base,actions:[{id:"change-1",command_id:"work-1:0",kind:"task",entity_id:"task-1",title:"Call Alex",
      summary:"Updated task: Call Alex",operation:"updated",fields:{due_date:{before:null,after:"2026-09-17"}},can_revert:true,revert_reason:"",reverted:false}]});
    expect(html).toContain("Updated task: Call Alex"); expect(html).toContain("due date: 2026-09-17");
    expect(html).toContain("Edit"); expect(html).toContain("Revert");
    expect(html).toContain('<details class="work-original"><summary>Original request</summary>');
    expect(html).not.toContain("Mark reviewed"); expect(html).not.toContain("Dismiss notification");
    expect(html.indexOf("Updated task:")).toBeLessThan(html.indexOf(base.request));
  });
  it("only offers notification dismissal on failures, without approving or retrying them",()=>{
    const html=render({...base,status:"failed",message:"The connection failed.",can_continue:true});
    expect(html).toContain("Dismiss notification"); expect(html).toContain("Retry");
    expect(html).not.toContain("Mark reviewed");
    expect(render({...base,status:"needs_input",message:"Which task?",can_continue:true})).not.toContain("Dismiss notification");
  });
  it("distinguishes dependency waits from user clarification",()=>{
    const html=render({...base,status:"queued",waiting:true,related_request_id:"earlier",message:"Waiting for earlier related work."});
    expect(html).toContain("Waiting for related work"); expect(html).toContain("Follow-up to earlier work");
    expect(html).not.toContain("Waiting for you");
  });
});


describe("compact conversation markers", () => {
  it("keeps Edit/Revert and the outcome, with the audit details minimized", () => {
    const item={...base, actions:[{id:"change",command_id:"work-1:0",kind:"task",entity_id:"task-1",title:"Call Alex",operation:"updated",summary:"Updated task: Call Alex",fields:{due_date:{before:null,after:"2026-09-18"}},can_revert:true,revert_reason:"",reverted:false}]};
    const html=renderToStaticMarkup(<WorkCard item={item} compact onRefresh={async()=>{}} onOpen={async()=>{}}/>);
    expect(html).toContain("Updated task: Call Alex");expect(html).toContain("Edit");expect(html).toContain("Revert");
    expect(html).toContain('aria-expanded="false"');expect(html).toContain("2026-09-18");
    expect(html).not.toContain("Original request");expect(html).not.toContain(base.request);expect(html).not.toContain("<dl>");
  });
});

describe("work that needs attention", () => {
  const done = { ...base, id: "done-1" }, done2 = { ...base, id: "done-2" };
  const asking = { ...base, id: "ask-1", status: "needs_input", message: "Which task?" };
  const failed = { ...base, id: "fail-1", status: "failed", message: "The connection failed." };
  const order = (html: string) => [...html.matchAll(/data-work-id="([^"]+)"/g)].map(m => m[1]);
  const panel = (items: WorkItem[], focus?: "attention") => renderToStaticMarkup(
    <ActivityPanel focus={focus} items={items} error="" onClose={() => {}} onRefresh={async () => {}} onOpen={async () => {}}/>);

  it("lists requests that need the user above completed ones", () => {
    expect(attentionFirst([done, asking, done2, failed]).map(i => i.id)).toEqual(["ask-1", "fail-1", "done-1", "done-2"]);
    const html = panel([done, asking, done2, failed]);
    expect(order(html)).toEqual(["ask-1", "fail-1", "done-1", "done-2"]);
    expect(html).toContain('aria-pressed="false"');
  });

  it("opens on just those requests from the status line, with a toggle back to all", () => {
    const html = panel([done, asking, failed], "attention");
    expect(order(html)).toEqual(["ask-1", "fail-1"]);
    expect(html).toContain('aria-pressed="true"');
    expect(html).toContain("2 need attention");
    // Nothing left to attend to: show everything rather than an empty drawer.
    expect(order(panel([done], "attention"))).toEqual(["done-1"]);
  });
});

describe("app chrome", () => {
  it("names the notification bell, with the unread count when there is one", () => {
    const quiet = renderToStaticMarkup(<NotificationBell unread={0} current={false} onOpen={() => {}}/>);
    expect(quiet).toContain('aria-label="Notifications"');
    const unread = renderToStaticMarkup(<NotificationBell unread={3} current onOpen={() => {}}/>);
    expect(unread).toContain('aria-label="Notifications, 3 unread"');
    expect(unread).toContain('aria-current="page"');
  });

  it("makes the needs-attention status line a button and leaves other statuses as text", () => {
    expect(renderToStaticMarkup(<FooterStatus text="Dani has work that needs attention" onAttention={() => {}}/>))
      .toBe('<button type="button" class="footer-status-link">Dani has work that needs attention</button>');
    expect(renderToStaticMarkup(<FooterStatus text="Connected. Changes save automatically."/>)).toBe("<span>Connected. Changes save automatically.</span>");
  });
});
