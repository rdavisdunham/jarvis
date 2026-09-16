import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { BotSettings, permissionScopes } from "./BotSettings";
import { WorkCard, type WorkItem } from "./Activity";

describe("connected agents", () => {
  it("exposes only selected grants and keeps paid queued work opt-in", () => {
    expect(permissionScopes({ tasks:"write", organization:"read", notes:"none" }, false)).toEqual(["tasks:write", "organization:read"]);
    expect(permissionScopes({ tasks:"read", notes:"none" }, true)).toEqual(["tasks:read", "work:run"]);
  });
  it("identifies the selected workspace", () => {
    const html = renderToStaticMarkup(<BotSettings workspace="Team launch"/>);
    expect(html).toContain("Team launch"); expect(html).toContain("Connected agents");
  });
  it("attributes saved actions to the initiating agent", () => {
    const item: WorkItem = { id:"w",parent_id:null,conversation_id:"c",request:"task.create",status:"succeeded",
      revision:1,message:"Saved.",actions:[],children:[],cancel_requested:false,seen:false,
      created_at:"2026-09-16T15:00:00Z",updated_at:"2026-09-16T15:00:00Z",can_continue:false,
      actor:{type:"bot",id:"b",name:"Codex"} };
    const html = renderToStaticMarkup(<WorkCard item={item} onRefresh={async()=>{}} onOpen={async()=>{}}/>);
    expect(html).toContain("Codex"); expect(html).toContain("connected agent");
  });
});
