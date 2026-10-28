"""Resumable account setup. Schema changes still use the existing preview/apply guard."""
from copy import deepcopy
from typing import Literal
from pydantic import Field, model_validator
from sqlalchemy import select
from .domain import Args, DomainError, emit, preferences, zone
from .models import OwnerSettings, SharedWorkspace, now
from .structure_schema import Definition

class Save(Args):
    expected_revision: int = Field(ge=0)
    status: Literal["in_progress","skipped","completed"] = "in_progress"
    step: Literal["profile","organization","preview","done"] = "profile"
    preferred_name: str | None = Field(default=None,min_length=1,max_length=100)
    timezone: str | None = Field(default=None,max_length=100)
    purpose: str | None = Field(default=None,max_length=2000)
    classification_name: str | None = Field(default=None,max_length=80)
    description: str | None = Field(default=None,max_length=5000)
    examples: str | None = Field(default=None,max_length=2000)
    design_notes: str | None = Field(default=None,max_length=12000, description="Saved user answers about types, fields, homes and relationships. Examples remain examples, not instructions to create records.")

class Preview(Args):
    expected_revision: int = Field(ge=0)
    schema_revision: int | None = Field(default=None, ge=1)
    definition: Definition | None = None
    status_mappings: dict[str, dict[str, str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def complete_definition(self):
        if (self.definition is None) != (self.schema_revision is None):
            raise ValueError("A full organization proposal needs both definition and schema_revision.")
        if self.status_mappings and self.definition is None:
            raise ValueError("Status mappings belong to a full organization proposal.")
        return self

COMMANDS={"onboarding.save":Save,"onboarding.preview":Preview}

def state(db,owner):
    if db.get(SharedWorkspace,owner):raise DomainError("PERSONAL_WORKSPACE","Set up your own account in Personal; shared structure belongs to its workspace owner.",403)
    settings=db.get(OwnerSettings,owner)
    saved=(settings.values if settings else {}).get("onboarding",{})
    prefs=preferences(db,owner)
    return {"revision":0,"status":"not_started","step":"profile","preferred_name":prefs["preferred_name"],"timezone":prefs["timezone"],"purpose":"","classification_name":"","description":"","examples":"","design_notes":"",**saved,"preferred_name":prefs["preferred_name"],"timezone":prefs["timezone"]}

def store(db,owner,data):
    row=db.get(OwnerSettings,owner)
    if not row:row=OwnerSettings(owner_id=owner,values={});db.add(row)
    row.values={**row.values,"onboarding":data}
    emit(db,owner,"settings.changed",owner)
    return data

def mutate(db,owner,tool,args,command_id):
    current=state(db,owner)
    if current["revision"]!=args.expected_revision:raise DomainError("REVISION_CONFLICT","Setup changed on another device. Reload your progress.",409)
    if tool=="onboarding.save":
        changes=args.model_dump(exclude_unset=True,exclude={"expected_revision"})
        if changes.get("timezone"):zone(changes["timezone"])
        if changes.get("status")=="completed":
            from .structure_models import StructureProposal
            proposal=db.get(StructureProposal,current.get("proposal_id")) if current.get("proposal_id") else None
            if not proposal or proposal.owner_id!=owner or not proposal.applied_at:raise DomainError("PREVIEW_REQUIRED","Preview and confirm your organization before completing setup, or skip for now.",409)
        # A changed answer invalidates its preview, even if an older preview was applied.
        if any(k in changes and changes[k]!=current.get(k) for k in ("purpose","classification_name","description","examples","design_notes")):
            if changes.get("status")=="completed":raise DomainError("PREVIEW_REQUIRED","Preview the revised answers before completing setup.",409)
            expire_proposal(db, owner, current.get("proposal_id"))
            current.pop("proposal_id",None)
        current.update(changes);current["revision"]+=1
        from .domain import SettingsUpdate, mutate as domain_mutate
        profile={k:changes[k] for k in ("preferred_name","timezone") if changes.get(k)}
        if profile:domain_mutate(db,owner,"settings.update",SettingsUpdate(**profile),command_id)
        return store(db,owner,current)
    from .structure import ensure, preview, assert_schema
    from .structure_schema import Preview as SchemaPreview
    schema=ensure(db,owner)
    name=current["classification_name"].strip()
    if args.definition is not None:
        assert_schema(schema, args.schema_revision)
        preserve_definitions(schema.definition, args.definition)
        proposal_args = SchemaPreview(expected_revision=args.schema_revision,
            definition=args.definition, status_mappings=args.status_mappings)
    else:
        definition=deepcopy(schema.definition)
        if name:
            if len(current["description"].strip())<20 or not current["examples"].strip():
                raise DomainError("CLARIFICATION_REQUIRED","What belongs in this group, and what is one example? Describe it in at least 20 characters.")
            kind={"id":"setup-group","name":name,"plural":name,"description":current["description"]+" Examples: "+current["examples"],"capabilities":[],"parent_types":[],"fields":[],"statuses":[],"archived":False}
            existing=next((t for t in definition["types"] if t["id"]=="setup-group"),None)
            if existing:existing.update(name=name,plural=name,description=kind["description"])
            else:definition["types"].append(kind)
            for t in definition["types"]:
                if t["id"] in {"task","note","project"} and "setup-group" not in t["parent_types"]:t["parent_types"].append("setup-group")
        proposal_args = SchemaPreview(expected_revision=schema.revision, definition=definition)
    proposal=preview(db,owner,proposal_args,command_id)
    expire_proposal(db, owner, current.get("proposal_id"))
    current.update(status="in_progress",step="preview",proposal_id=proposal["id"],revision=current["revision"]+1)
    store(db,owner,current)
    result={"progress":current,"proposal":proposal,
        "note":"This previews type definitions, fields and allowed homes, not actual named containers or tasks. Confirm separately before applying. Existing records are retained."}
    if args.definition is None:
        result["example"]=[name,"Example project","First task"] if name else ["Project","First task"]
    return result


def preserve_definitions(before, proposed):
    """Setup extends/refines the current design; it must not silently erase it."""
    old_types={t["id"]:t for t in before["types"]}
    new_types={t.id:t for t in proposed.types}
    missing=set(old_types)-set(new_types)
    for identity, old in old_types.items():
        if identity in new_types:
            missing.update("field:"+identity+":"+field for field in
                {f["id"] for f in old["fields"]}-{f.id for f in new_types[identity].fields})
    missing.update("relationship:"+identity for identity in
        {r["id"] for r in before["relationships"]}-{r.id for r in proposed.relationships})
    # Omitted library is preserved by the ordinary structural preview adapter.
    if "field_library" in proposed.model_fields_set:
        missing.update("library:"+identity for identity in
            {f["id"] for f in before.get("field_library",[])}-{f.id for f in proposed.field_library})
    if missing:
        raise DomainError("SETUP_PRESERVE_DEFINITIONS",
            "Keep existing types, fields and relationships in the setup proposal; archive those you no longer want.",
            data={"missing":sorted(missing)})


def expire_proposal(db, owner, identity):
    """A revised interview answer/proposal cannot leave an older preview executable."""
    from .structure_models import StructureProposal
    previous = db.get(StructureProposal, identity) if identity else None
    if previous and previous.owner_id == owner and not previous.applied_at:
        previous.expires_at = now()
