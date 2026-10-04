export type SchemaField = { library_id?:string|null; id:string; name:string; description:string; kind:string; options:{id:string;name:string}[]; target_types:string[]; multiple:boolean; inherit:boolean; visible:boolean; archived:boolean; binding:string|null };
export type SchemaStatus = { id:string; name:string; meaning:string };
export type SchemaType = { id:string; name:string; plural:string; description:string; capabilities:string[]; parent_types:string[]; fields:SchemaField[]; statuses:SchemaStatus[]; opens_as?:OpensAs; review?:{enabled:boolean;every:"1w"|"2w"|"1m"|"3m"|"6m"|"1y"}; archived:boolean };
export type SchemaRelation = {behavior?:"related"|"blocks";id:string;name:string;description:string;source_types:string[];target_types:string[];cardinality:string;archived:boolean};
export type Schema = {field_library?:SchemaField[];type_layout?:{type_id:string;parent_type_id:string|null}[];revision:number;types:SchemaType[];relationships:SchemaRelation[];understandings:{definition_id:string;status:string;understanding:Record<string,unknown>;questions:unknown[]}[]};
export type CustomRecord = {last_reviewed_at?:string|null;next_review_at?:string|null;review_paused?:boolean;review_due?:boolean;review_every?:"1w"|"2w"|"1m"|"3m"|"6m"|"1y"|null;opens_as?:"container"|"item";blockers?:{id:string;title:string;status:string}[];contents?:ContentsSummary;quick_done?:number;quick_total?:number;is_quick_list?:boolean;quick_list_parent_id?:string|null;source?:import("./SourceDetails").SourceInfo;sort_order?:number;local_notes?:string;id:string;revision:number;schema_revision:number;type_id:string;type_name:string;title:string;body:string;values:Record<string,unknown>;inherited:Record<string,string>;parent_id:string|null;status_id:string|null;status_meaning?:string;archived:boolean;task_id:string|null;note_id:string|null;capabilities:string[];home:{id:string;title:string;type_id:string}[];links:{id:string;source_id:string;target_id:string;relationship_id:string}[]};
export type Proposal={applied_at?:string|null;id:string;schema_revision:number;definition:Pick<Schema,"types"|"relationships"|"field_library"|"type_layout">;impact:{review_changes?:{type_id:string;name:string;every:string|null;every_label:string|null;records:number}[];affected_count:number;affected_records:{id:string;title:string}[];issues:ProposalIssue[];blocking_count:number};expires_at:string};
/** Placement issues name the record and its current home so the editor can list them. */
export type ProposalIssue={message:string;record_id?:string;link_id?:string;kind?:"placement";type_id?:string;title?:string;archived?:boolean;home?:{id:string;title:string;type_id:string}};
export const meanings=["backlog","open","in_progress","waiting","deferred","completed","cancelled"];
export const fieldKinds=["text","long_text","number","boolean","date","datetime","select","multiselect","relation"];
export const describe=(value:string)=>value.replaceAll("_"," ");
export const emptyField=():SchemaField=>({id:crypto.randomUUID(),name:"",description:"",kind:"text",options:[],target_types:[],multiple:false,inherit:false,visible:true,archived:false,binding:null});

export type ContentsSummary={direct:number;descendants:number;work_total:number;work_done:number;work_open:number;ready_to_complete:boolean;child_start:string|null;child_end:string|null;blockers:{id:string;title:string;status:string}[]};
export type OpensAs="auto"|"container"|"item";
/** Containers open their Browse contents; items open their detail card. The server resolves "auto". */
export const openTarget=(r:Pick<CustomRecord,"opens_as"|"type_id">,schema?:Pick<Schema,"types">):"contents"|"details"=>{
  const mode=r.opens_as??schema?.types.find(t=>t.id===r.type_id)?.opens_as;
  return mode==="container"?"contents":"details";
};
