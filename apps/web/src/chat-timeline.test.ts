import {describe, it, expect} from "vitest";
import {chatTimeline, mergeWorkReplies, showChatCard} from "./chat-timeline";
import type {WorkItem} from "./Activity";
import type {ChatMessage} from "./types";
const stamp=(minute:number)=>`2026-09-17T15:${String(minute).padStart(2,"0")}:00Z`;
const work=(id:string, minute:number, changes=true):WorkItem=>({id,parent_id:null,conversation_id:"chat",request:`Add ${id}`,status:"succeeded",revision:1,message:"Done",actions:changes?[{id:"change-"+id,command_id:id+":0",kind:"task",entity_id:id,title:id,operation:"created",fields:{},can_revert:true,revert_reason:"",reverted:false}]:[],children:[],cancel_requested:false,seen:false,created_at:stamp(minute),updated_at:stamp(minute),can_continue:false});
const msg=(id:string,minute:number,content="Add "+id):ChatMessage=>({id,role:"user",content,created_at:stamp(minute)});
const ids=(entries:ReturnType<typeof chatTimeline>)=>entries.map(e=>e.kind==="message"?e.message.id:"card:"+e.work.id);
describe("conversation action timeline",()=>{
 it("does not create cards for navigation, ordinary responses or unknown queued intent",()=>{
  expect(showChatCard(work("show",1,false))).toBe(false);
  expect(showChatCard({...work("show",1,false),status:"running"})).toBe(false);
  expect(showChatCard({...work("show",1,false),status:"failed",navigation_only:true})).toBe(false);
  expect(showChatCard({...work("edit",1,false),status:"needs_input"})).toBe(true);
 });
 it("keeps parallel requests by their original messages even if they complete in reverse order",()=>{
  const anchors=new Map();const messages=[msg("one",1),msg("two",2)];
  chatTimeline(messages,[{...work("one",1,false),status:"running"},{...work("two",2,false),status:"running"}],anchors);
  expect(ids(chatTimeline(messages,[work("two",2),{...work("one",1,false),status:"running"}],anchors))).toEqual(["one","two","card:two"]);
  expect(ids(chatTimeline([...messages,msg("three",3)],[work("two",2),{...work("one",1),updated_at:stamp(4)}],anchors))).toEqual(["one","card:one","two","card:two","three"]);
 });
 it("waits for history to load before fixing an anchor",()=>{
  const anchors=new Map();chatTimeline([], [work("one",1)], anchors);
  expect(ids(chatTimeline([msg("one",1)],[work("one",1)],anchors))).toEqual(["one","card:one"]);
 });
 it("anchors reloaded source IDs and resolved clarification at the original request",()=>{
  const messages=[{...msg("source-uuid",1),native_id:"work:one:user"},msg("answer",3,"Tomorrow"),msg("next",4)];
  const anchors=new Map();chatTimeline(messages,[{...work("one",1,false),status:"needs_input"}],anchors);
  expect(ids(chatTimeline(messages,[{...work("one",1),updated_at:stamp(5)}],anchors))).toEqual(["source-uuid","card:one","answer","next"]);
 });
 it("places voice receipts beside matching speech, even if later speech began before queuing",()=>{
  const messages=[msg("voice-one",1,"Please add one"),msg("voice-two",2,"Now add two")];
  expect(ids(chatTimeline(messages,[{...work("one",3),request:"Please add one",voice_session_id:"voice"}],new Map()))).toEqual(["voice-one","card:one","voice-two"]);
 });
 it("keeps navigation replies as ordinary messages and does not duplicate saved or spoken replies",()=>{
  const item={...work("show",1,false),response_native_id:"work:show:assistant:1",finished_at:stamp(2),message:"Here is your calendar."};
  const messages=mergeWorkReplies([msg("show",1)], [item]);
  expect(messages[1].content).toBe("Here is your calendar.");expect(ids(chatTimeline(messages,[item],new Map()))).not.toContain("card:show");
  expect(mergeWorkReplies(messages,[item])).toBe(messages);
  expect(mergeWorkReplies([msg("show",1)],[{...item,voice_session_id:"live"}])).toHaveLength(1);
 });
});
