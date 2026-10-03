"""Compact lists reuse Task IDs, revisions, alerts, history and search indexes."""
from datetime import date
from typing import Literal
from pydantic import Field
from sqlalchemy import select
from .domain import Args, DomainError, TaskCreate, TaskUpdate, check_revision, emit, owned, serial
from .models import Task, now

class Item(Args):
    title: str = Field(min_length=1,max_length=500)
    section: str = Field(default="",max_length=120)

class Create(Args):
    title: str = Field(min_length=1,max_length=500)
    items: list[Item] = Field(default_factory=list,max_length=100)
    due_date: date | None = None
    due_time: str | None = None
    due_timezone: str | None = None
    planned_date: date | None = None
    deadline_alert: Literal["off","on"] = "off"

class Change(Args):
    list_id: str
    expected_revision: int = Field(ge=1)

class ItemChange(Change):
    operation: Literal["add","edit","reorder"]
    item_id: str | None = None
    expected_item_revision: int | None = Field(default=None,ge=1)
    title: str | None = Field(default=None,min_length=1,max_length=500)
    section: str | None = Field(default=None,max_length=120)
    completed: bool | None = None
    order: list[str] = Field(default_factory=list,max_length=100)

class Promote(Change):
    project_id: str | None = None

COMMANDS={"quicklist.create":Create,"quicklist.item":ItemChange,"quicklist.promote":Promote}

def children(db, owner, identity):
    return list(db.scalars(select(Task).where(Task.owner_id==owner,Task.parent_task_id==identity,Task.archived.is_(False)).order_by(Task.quick_order,Task.created_at,Task.id)))

def read(db,owner,identity):
    row=owned(db,Task,identity,owner)
    if not row.is_quick_list:
        raise DomainError("NOT_QUICK_LIST","This is now a normal task; open its task details.",409)
    items=children(db,owner,identity)
    return {**serial(row),"items":[serial(x) for x in items],"done":sum(x.status=="completed" for x in items),"total":len(items)}

def listing(db,owner,query=""):
    from .text_normalize import sql_filter
    q=select(Task).where(Task.owner_id==owner,Task.is_quick_list.is_(True),Task.archived.is_(False))
    if query:
        matches=select(Task.parent_task_id).where(Task.owner_id==owner,sql_filter(query,Task.title))
        q=q.where(sql_filter(query,Task.title) | Task.id.in_(matches))
    rows=list(db.scalars(q.order_by(Task.updated_at.desc()).limit(100)))
    return {"items":[read(db,owner,r.id) for r in rows],"limit":100}

def core(db,owner,tool,args,command_id):
    from .domain import mutate
    from .structure import observe_core
    # Use canonical task commands and index observation inside the enclosing atomic receipt.
    from .access import command_access
    values=args.model_dump(exclude_unset=True)
    command_access(db,owner,tool,values)
    args=type(args).model_validate(values)
    result=mutate(db,owner,tool,args,command_id)
    observe_core(db,owner,tool,result,command_id,arguments=args.model_dump(exclude_unset=True))
    return result

def mutate(db,owner,tool,args,command_id):
    if tool=="quicklist.create":
        values=args.model_dump(exclude={"items"})
        root=core(db,owner,"task.create",TaskCreate(**values),command_id)
        row=owned(db,Task,root["id"],owner)
        row.is_quick_list=True
        for i,item in enumerate(args.items):
            child=core(db,owner,"task.create",TaskCreate(title=item.title,parent_task_id=row.id,deadline_alert="off"),command_id)
            task=owned(db,Task,child["id"],owner)
            task.quick_section=item.section;task.quick_order=i
        db.flush()
        return read(db,owner,row.id)
    row=owned(db,Task,args.list_id,owner,lock=True)
    check_revision(row,args.expected_revision)
    if not row.is_quick_list or row.archived:
        raise DomainError("NOT_QUICK_LIST","Open an active Quick list.",409)
    items=children(db,owner,row.id)
    if tool=="quicklist.promote":
        # The container becomes a normal parent task. Children are the same Task rows.
        for task in [row,*items]:
            patch={"is_quick_list":False} if task.id==row.id else {}
            if args.project_id is not None:patch["project_id"]=args.project_id
            if patch:core(db,owner,"task.update",TaskUpdate(task_id=task.id,expected_revision=task.revision,**patch),command_id)
        db.flush()
        return {"task":serial(row),"task_ids":[x.id for x in items],"promoted":True,"note":"Same IDs and history; the list is now a normal parent task and its subtasks."}
    if args.operation=="add":
        if not args.title:raise DomainError("INVALID_ARGUMENT","Give the new item a title.")
        if len(items)>=100:raise DomainError("LIST_FULL","Quick lists hold up to 100 items. Promote it to normal tasks for larger work.")
        child=core(db,owner,"task.create",TaskCreate(title=args.title,parent_task_id=row.id,deadline_alert="off"),command_id)
        item=owned(db,Task,child["id"],owner)
        item.quick_section=args.section or "";item.quick_order=max([x.quick_order for x in items],default=-1)+1
    elif args.operation=="edit":
        item=next((t for t in items if t.id==args.item_id),None)
        if not item or not args.expected_item_revision:raise DomainError("INVALID_ARGUMENT","Read the list and use the item's ID and revision.")
        patch={}
        if args.title is not None:patch["title"]=args.title
        if args.section is not None:patch["quick_section"]=args.section
        if args.completed is not None:patch["status"]="completed" if args.completed else "open"
        if not patch:raise DomainError("INVALID_ARGUMENT","Provide an item change.")
        core(db,owner,"task.update",TaskUpdate(task_id=item.id,expected_revision=args.expected_item_revision,**patch),command_id)
    else:
        if len(args.order)!=len(items) or set(args.order)!={t.id for t in items}:raise DomainError("REVISION_CONFLICT","The list changed. Read it again before reordering.",409)
        for i,identity in enumerate(args.order):
            task=next(t for t in items if t.id==identity)
            if task.quick_order!=i:core(db,owner,"task.update",TaskUpdate(task_id=task.id,expected_revision=task.revision,quick_order=i),command_id)
    row.revision+=1;row.updated_at=now();emit(db,owner,"task.changed",row.id,row.revision)
    db.flush()
    return read(db,owner,row.id)
