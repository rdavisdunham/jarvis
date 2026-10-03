"""Reusable workspace definitions; attachment IDs and saved values remain stable."""
from copy import deepcopy
from hashlib import sha256
import json

SEMANTIC = ("name", "description", "kind", "options", "target_types", "multiple", "binding")
# Default types keep these presentation defaults only while their name and behaviors are unedited.
DEFAULT_OPENS_AS = {
    "space": ("Space", (), "container"), "area": ("Area", (), "container"),
    "client": ("Client", (), "container"), "project": ("Project", ("timeline", "work"), "container"),
    "task": ("Task", ("work",), "item"), "note": ("Note", ("content",), "item"),
}


def default_opens_as(t):
    name, caps, mode = DEFAULT_OPENS_AS.get(t["id"], (None, (), "auto"))
    return mode if t.get("name") == name and tuple(sorted(t.get("capabilities", []))) == caps else "auto"


def upgrade(definition):
    """Lossless upgrade of legacy inline fields. Names alone never merge fields."""
    result = deepcopy(definition)
    library = list(result.get("field_library", []))
    by_id = {f["id"]: f for f in library}
    matches = {}
    for shared in library:
        signature = json.dumps({k: shared.get(k) for k in SEMANTIC}, sort_keys=True)
        matches[signature] = shared["id"]
    for t in result["types"]:
        for f in t["fields"]:
            if f.get("library_id") in by_id:
                continue
            signature = json.dumps({k: f.get(k) for k in SEMANTIC}, sort_keys=True)
            # Only operational definitions are deduplicated automatically.
            # A custom inline field retains its own identity even when labels match.
            identity = matches.get(signature) if f.get("binding") else None
            if not identity:
                identity = "field_" + sha256((signature + ("" if f.get("binding") else ":" + t["id"] + ":" + f["id"])).encode()).hexdigest()[:24]
                shared = {**deepcopy(f), "id": identity, "library_id": None,
                          "inherit": False, "visible": True, "archived": False}
                if identity not in by_id:
                    library.append(shared)
                    by_id[identity] = shared
                matches[signature] = identity
            f["library_id"] = identity
    result["field_library"] = library
    result.setdefault("type_layout", [])
    for t in result["types"]:
        if "opens_as" not in t:
            t["opens_as"] = default_opens_as(t)
    return result


def prepare(incoming, current, supplied, unset_opens_as=()):
    """Older clients may edit inline fields without erasing the reusable library."""
    result = deepcopy(incoming)
    prior = {t["id"]: t.get("opens_as") for t in current["types"]}
    for t in result["types"]:
        if t["id"] in unset_opens_as:
            t.pop("opens_as", None)
            if prior.get(t["id"]):
                t["opens_as"] = prior[t["id"]]
    if "type_layout" not in supplied:
        result["type_layout"] = deepcopy(current.get("type_layout", []))
    if "field_library" not in supplied:
        result["field_library"] = deepcopy(current.get("field_library", []))
        old = {(t["id"], f["id"]): f for t in current["types"] for f in t["fields"]}
        for t in result["types"]:
            for f in t["fields"]:
                prior = old.get((t["id"], f["id"]))
                if prior and any(f.get(k) != prior.get(k) for k in SEMANTIC):
                    # Legacy edits affect this attachment only, not other types.
                    f["library_id"] = None
                elif prior:
                    f["library_id"] = prior.get("library_id")
    return upgrade(result)
