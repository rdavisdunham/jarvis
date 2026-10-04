"""Reusable templates: saved starting structures per type, kept outside the schema.

A template holds default values, a description outline and optional child records.
Creating from a template (``record.instantiate``) follows the record_contents pattern:
plan with a hash, apply re-plans and compares, every record goes through
``structure.mutate("record.create")`` under one command, and grouped Revert archives
the created subtree only while nothing has changed since. Created records are
independent copies; ``provenance.template`` is a reference, never a live link.
"""

from collections import Counter
from sqlalchemy import select, func
from .domain import DomainError, check_revision, emit, owned, encode
from .models import now
from .structure_models import RecordTemplate, StructureLink, StructureRecord
from .structure_schema import InstantiatePlan, TEMPLATE_DEPTH, TEMPLATE_NODES

# Templates never carry deadlines, schedules, assignees or a workflow position.
EXCLUDED_BINDINGS = {"due_date", "due_time", "due_timezone", "planned_date", "start_date", "target_date", "assignee"}
EXCLUDED_KINDS = {"date", "datetime", "relation"}
MAX_TEMPLATES = 200


def _types(schema):
    return {t["id"]: t for t in schema.definition["types"]}


def _active(schema, type_id):
    t = _types(schema).get(type_id)
    return t if t and not t["archived"] else None


def portable(t, values):
    """Values worth copying into a template: no dates, assignees, status or record links."""
    from .structure import BINDING_DEFAULTS

    fields = {f["id"]: f for f in t["fields"] if not f["archived"]}
    return {
        k: v for k, v in values.items()
        if k in fields and v is not None and BINDING_DEFAULTS.get(fields[k].get("binding")) != v
        and fields[k].get("binding") not in EXCLUDED_BINDINGS and fields[k]["kind"] not in EXCLUDED_KINDS
    }


def check_tree(db, owner, schema, type_id, payload, *, outdated=False):
    """Validate a payload against the current schema; instantiate re-runs this."""
    from .structure import validate_values

    def fail(message):
        if outdated:
            raise DomainError(
                "TEMPLATE_OUTDATED",
                "This template no longer fits your structure: " + message + " Edit the template, then try again.",
                409,
            )
        raise DomainError("INVALID_TEMPLATE", message)

    def values(t, supplied, title):
        fields = {f["id"]: f for f in t["fields"]}
        for key in supplied:
            f = fields.get(key)
            if f and (f.get("binding") in EXCLUDED_BINDINGS or f["kind"] in {"date", "datetime"}):
                fail(f"Templates don't carry dates or assignees; remove {f['name']} from “{title}”.")
        try:
            validate_values(db, owner, t, supplied)
        except DomainError as exc:
            fail(f"a value on “{title}” no longer fits {t['name']} ({exc.message})")

    root = _active(schema, type_id)
    if not root:
        fail("its type is no longer active.")
    values(root, payload.get("values", {}), "the record")

    def visit(parent, nodes):
        for node in nodes:
            t = _active(schema, node["type_id"])
            if not t:
                fail(f"“{node['title']}” uses a type that is no longer active.")
            if parent["id"] not in t["parent_types"]:
                fail(f"{t['plural']} can no longer live inside {parent['plural']} (“{node['title']}”).")
            values(t, node.get("values", {}), node["title"])
            visit(t, node.get("children", []))

    visit(root, payload.get("children", []))
    return root


def count_nodes(nodes):
    return sum(1 + count_nodes(n.get("children", [])) for n in nodes)


def summary(schema, type_id, payload):
    """One line for stamps and lists: "A project with 4 tasks: Contract, Access, Kickoff, Discovery"."""
    types = _types(schema)
    t = types.get(type_id, {"name": "record"})
    name = t["name"].lower()
    lead = ("An " if name[:1] in "aeiou" else "A ") + name
    children = payload.get("children", [])
    if not children:
        return lead + (" with a description outline" if payload.get("body") else "")
    counts = Counter(c["type_id"] for c in children)
    parts = []
    for key, n in counts.items():
        ct = types.get(key, {"name": "record", "plural": "records"})
        parts.append(f"{n} {(ct['name'] if n == 1 else ct['plural']).lower()}")
    titles = [c["title"] for c in children[:4]]
    more = f" and {len(children) - 4} more" if len(children) > 4 else ""
    return f"{lead} with {' and '.join(parts)}: {', '.join(titles)}{more}"


def public(row, schema):
    t = _types(schema).get(row.type_id)
    data = encode({k: getattr(row, k) for k in (
        "id", "type_id", "name", "description", "payload", "revision", "archived", "created_by", "created_at", "updated_at")})
    data.update(
        type_name=t["name"] if t else row.type_id,
        summary=summary(schema, row.type_id, row.payload),
        record_count=1 + count_nodes(row.payload.get("children", [])),
    )
    return data


def listing(db, owner, *, type_id=None, archived=False):
    from .structure import ensure

    schema = ensure(db, owner)
    q = select(RecordTemplate).where(RecordTemplate.owner_id == owner, RecordTemplate.archived == archived)
    if type_id:
        q = q.where(RecordTemplate.type_id == type_id)
    rows = db.scalars(q.order_by(func.lower(RecordTemplate.name), RecordTemplate.id).limit(MAX_TEMPLATES))
    return {"items": [public(r, schema) for r in rows], "schema_revision": schema.revision}


def get(db, owner, identity):
    from .structure import ensure

    return public(owned(db, RecordTemplate, identity, owner), ensure(db, owner))


def snapshot(db, owner, root, include_children):
    """Copy a record (and optionally its active subtree) into a payload without operational values."""
    from .record_contents import rows_under
    from .structure import data, ensure, reconcile_core

    schema = ensure(db, owner)
    types = _types(schema)

    def portable_of(row):
        reconcile_core(db, row, schema)
        current = data(db, row, schema)
        own = {k: v for k, v in current["values"].items() if k not in current["inherited"]}
        return current, portable(types[row.type_id], own)

    current, values = portable_of(root)
    payload = {"values": values, "body": current["body"], "children": []}
    if not include_children:
        return payload
    rows = [r for r in rows_under(db, owner, root.id) if not r.archived and _active(schema, r.type_id)]
    by_parent = {}
    for r in sorted(rows, key=lambda r: (r.sort_order, r.id)):
        by_parent.setdefault(r.parent_id, []).append(r)
    total = 0

    def build(parent_id, depth):
        nonlocal total
        nodes = []
        for row in by_parent.get(parent_id, []):
            if depth > TEMPLATE_DEPTH:
                raise DomainError("TEMPLATE_TOO_LARGE", f"Templates keep at most {TEMPLATE_DEPTH} levels of contents. Save it without its contents, or flatten it first.")
            total += 1
            if total > TEMPLATE_NODES:
                raise DomainError("TEMPLATE_TOO_LARGE", f"Templates keep at most {TEMPLATE_NODES} records inside. Save it without its contents, or trim it first.")
            cur, vals = portable_of(row)
            nodes.append({"type_id": row.type_id, "title": cur["title"], "body": cur["body"], "values": vals,
                          "children": build(row.id, depth + 1)})
        return nodes

    payload["children"] = build(root.id, 1)
    return payload


def mutate(db, owner, tool, args, command_id):
    from .access import actor
    from .structure import ensure

    schema = ensure(db, owner)
    if tool in {"template.create", "template.capture"}:
        active = db.scalar(select(func.count()).select_from(RecordTemplate).where(
            RecordTemplate.owner_id == owner, RecordTemplate.archived.is_(False)))
        if active >= MAX_TEMPLATES:
            raise DomainError("TEMPLATE_LIMIT", f"This workspace has {MAX_TEMPLATES} templates. Archive one first.")
        if tool == "template.capture":
            root = owned(db, StructureRecord, args.record_id, owner)
            if root.archived:
                raise DomainError("ARCHIVED_RECORD", "Restore this record before saving it as a template.")
            type_id, payload = root.type_id, snapshot(db, owner, root, args.include_children)
        else:
            type_id, payload = args.type_id, args.payload.model_dump()
        check_tree(db, owner, schema, type_id, payload)
        row = RecordTemplate(owner_id=owner, type_id=type_id, name=args.name, description=args.description,
                             payload=payload, created_by=actor(db, owner))
        db.add(row)
    else:
        row = owned(db, RecordTemplate, args.template_id, owner, lock=True)
        check_revision(row, args.expected_revision)
        if tool == "template.archive":
            row.archived = args.archived
        else:
            if args.payload is not None:
                payload = args.payload.model_dump()
                check_tree(db, owner, schema, row.type_id, payload)
                row.payload = payload
            for key in ("name", "description"):
                if key in args.model_fields_set and getattr(args, key) is not None:
                    setattr(row, key, getattr(args, key))
        row.revision += 1
        row.updated_at = now()
    db.flush()
    emit(db, owner, "template.changed", row.id, row.revision)
    return public(row, schema)


def plan(db, owner, args):
    from . import structure

    schema = structure.ensure(db, owner)
    structure.assert_schema(schema, args.schema_revision)
    template = owned(db, RecordTemplate, args.template_id, owner)
    if template.archived:
        raise DomainError("ARCHIVED_TEMPLATE", "This template is archived. Restore it before using it.")
    payload = template.payload
    t = check_tree(db, owner, schema, template.type_id, payload, outdated=True)
    chain = structure.validate_parent(db, owner, None, t, args.parent_id) if args.parent_id else []
    types = _types(schema)

    def node(type_id, title, children):
        return {"type_id": type_id, "type_name": types[type_id]["name"], "title": title,
                "children": [node(c["type_id"], c["title"], c.get("children", [])) for c in children]}

    title = args.title or template.name
    tree = node(template.type_id, title, payload.get("children", []))

    def kinds(n):
        return Counter({n["type_id"]: 1}) + sum((kinds(c) for c in n["children"]), Counter())

    stamp = structure.fingerprint({
        "request": args.model_dump(include=set(InstantiatePlan.model_fields)),
        "schema": schema.revision,
        "template": (template.id, template.revision),
        "home": [(r.id, r.revision, r.parent_id, r.archived) for r in chain],
    })
    return {
        "preview_hash": stamp,
        "template_id": template.id,
        "template_revision": template.revision,
        "name": template.name,
        "title": title,
        "parent_id": args.parent_id,
        "home": [{"id": r.id, "title": r.title, "type_id": r.type_id} for r in reversed(chain)],
        "tree": tree,
        "record_count": 1 + count_nodes(payload.get("children", [])),
        "types": dict(kinds(tree)),
        "summary": summary(schema, template.type_id, payload),
        "source_effect": "Creates independent records. Later template edits never change them.",
        "schema_revision": schema.revision,
    }


def apply(db, owner, args, command_id):
    from . import structure
    from .record_contents import guard
    from .structure_schema import RecordCreate

    preview = plan(db, owner, args)
    if preview["preview_hash"] != args.preview_hash:
        raise DomainError("STALE_TEMPLATE", "The template or its home changed. Review a fresh preview.", 409)
    schema = structure.ensure(db, owner)
    template = owned(db, RecordTemplate, args.template_id, owner)
    provenance = {"id": template.id, "revision": template.revision}
    created = []

    def make(type_id, title, body, values, parent_id, root=False):
        # Child records are template structure, not the owner's own filing evidence.
        made = structure.mutate(db, owner, "record.create", RecordCreate(
            type_id=type_id, title=title, body=body, values=values, parent_id=parent_id,
            schema_revision=schema.revision), command_id if root else command_id + ":template")
        row = db.get(StructureRecord, made["id"])
        row.provenance = {**row.provenance, "template": provenance}
        created.append(row.id)
        return row

    payload = template.payload
    root = make(template.type_id, preview["title"], payload.get("body", ""), payload.get("values", {}), args.parent_id, True)

    def visit(parent, nodes):
        for n in nodes:
            visit(make(n["type_id"], n["title"], n.get("body", ""), n.get("values", {}), parent.id), n.get("children", []))

    visit(root, payload.get("children", []))
    db.flush()
    return {
        **preview,
        "id": root.id,
        "record": structure.data(db, root, schema),
        "changed_ids": created,
        "applied": True,
        "restore_guard": guard(db, owner, created),
    }


def undo_problem(db, owner, ids):
    """Why the created subtree cannot be archived safely, or None."""
    rows = [db.get(StructureRecord, i) for i in ids]
    if any(r is None or r.owner_id != owner for r in rows):
        return "These records are no longer available."
    if all(r.archived for r in rows):
        return "These records are already archived."
    linked = db.scalar(select(StructureLink.id).where(
        StructureLink.owner_id == owner,
        StructureLink.source_id.in_(ids) | StructureLink.target_id.in_(ids)).limit(1))
    if linked:
        return "Other records are linked to these records. Review those links before reverting."
    return None


def restore(db, owner, receipt, rows, command_id):
    """Archive every record a template created, deepest first, after the group guard passed."""
    from uuid import NAMESPACE_URL, uuid5
    from .domain import execute
    from .structure import ensure, parent_chain

    ids = receipt.result["data"]["changed_ids"]
    reason = undo_problem(db, owner, ids)
    if reason or any(r.reverted_by for r in rows):
        raise DomainError("REVISION_CONFLICT", reason or "Already reverted.", 409)
    schema = ensure(db, owner)
    order = sorted(ids, key=lambda i: -len(parent_chain(db, owner, db.get(StructureRecord, i).parent_id)))
    for n, identity in enumerate(order):
        current = db.get(StructureRecord, identity)
        if current.archived:
            continue
        child = str(uuid5(NAMESPACE_URL, command_id + ":" + str(n)))
        execute(db, owner, child, "record.update", {"record_id": identity, "expected_revision": current.revision,
                                                     "schema_revision": schema.revision, "archived": True})
    for row in rows:
        row.reverted_by = command_id
    return {"restored": True, "changed_count": len(order), "archived_ids": order}
