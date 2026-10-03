"""Reusable workspace definitions; attachment IDs and saved values remain stable."""
from copy import deepcopy
from hashlib import sha256
import json

SEMANTIC = ("name", "description", "kind", "options", "target_types", "multiple", "binding")


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
                library.append(shared)
                by_id[identity] = shared
                matches[signature] = identity
            f["library_id"] = identity
    result["field_library"] = library
    result.setdefault("type_layout", [])
    return result


def prepare(incoming, current, supplied):
    """Older clients may edit inline fields without erasing the reusable library."""
    result = deepcopy(incoming)
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
