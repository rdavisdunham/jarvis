"""Deterministic plan-scale corpus for the context platform (plan_final.md section 11, corpus scenario).

`generate(spec)` builds every row in memory from one seed; `load(db, corpus)` bulk-inserts them; `queries(corpus,
n)` draws benchmark queries. Same spec, same rows: IDs, text, timestamps and embeddings are all seeded.

Shape at the default spec: 500 meeting sources over two years for three owners (60/25/15%), about 50,000
segments (a tenth of the sources have a second revision), chunks of 20 segments and assertions for 10,000
embedded rows in all, a few hundred entity IDs, and realistic history:

- deleted sources, most purged as `sources.purge_source` leaves them (blank text, no embedding, quote-less
  evidence) and the rest still waiting for the purge job (content intact; reads must exclude them);
- assertions retracted because their only support was deleted, retracted by a human correction (with the
  decision), retracted when a source edit removed their passage, superseded by a later assertion about the same
  subject (closed interval, `supersedes_id`, version history), contested, shadow and future-effective;
- independent corroboration from a second meeting of the same series, so some assertions survive a deletion;
- entity links on sources, assertions and segments, a few rejected or unresolved, and speaker actors.

Embeddings are synthetic 512-dimension unit vectors clustered by topic and meeting series (no API calls),
recorded under `memory_learning.EMBEDDING_MODEL` so the retriever's semantic leg uses them. Text is invented and
is data, never instructions.
"""

import hashlib
import math
import random
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from uuid import UUID

from jarvis import memory_learning
from jarvis.context.models import (
    DIMENSIONS,
    ContextAssertion,
    ContextAssertionVersion,
    ContextChunk,
    ContextDecision,
    ContextEntityLink,
    ContextEvidence,
    ContextOwnerState,
    ContextSegment,
    ContextSource,
    ContextSourceRevision,
)
from jarvis.context.schema import Participant, Segment, SourceEnvelope
from sqlalchemy import insert

NOW = datetime(2026, 11, 1, 12, tzinfo=UTC)
TABLES = (ContextSource, ContextSourceRevision, ContextSegment, ContextChunk, ContextDecision, ContextAssertion,
          ContextAssertionVersion, ContextEvidence, ContextEntityLink, ContextOwnerState)
KINDS = {"fact": 25, "decision": 15, "commitment": 20, "risk": 8, "open_question": 7, "rationale": 8,
         "preference": 5, "procedure": 5, "relationship": 7}
PREDICATES = {"commitment": "due_date", "decision": "decided", "risk": "risk", "open_question": "asks",
              "rationale": "because", "preference": "prefers", "procedure": "procedure", "relationship": "works_with",
              "fact": "states"}
LEADS = {"decision": "Decision:", "commitment": "Action:", "risk": "Risk:", "open_question": "Open question:",
         "rationale": "Because", "preference": "Prefers", "procedure": "Process:", "relationship": "Works with",
         "fact": ""}
FIRST = ("Ada", "Bo", "Cy", "Dee", "Eli", "Fay", "Gus", "Hal", "Ivy", "Jo", "Kai", "Lin", "Mo", "Nia", "Oz", "Pia",
         "Quin", "Ravi", "Sol", "Tia", "Uma", "Vic", "Wes", "Xan", "Yara", "Zed")
LAST = ("Abbot", "Brook", "Crane", "Dale", "Ekker", "Frost", "Grove", "Hollis", "Imari", "Jensen", "Kowal", "Lark",
        "Moreau", "Nakai", "Orsa", "Pryce", "Quill", "Rowe", "Saito", "Tamm", "Ulric", "Voss", "Wren", "Yates")
TOPIC_WORDS = (
    "pilot", "budget", "clinic", "rollout", "vendor", "contract", "launch", "hiring", "roadmap", "pricing",
    "migration", "postgres", "security", "audit", "onboarding", "training", "survey", "dashboard", "invoice",
    "renewal", "schedule", "staffing", "forecast", "checklist", "portal", "billing", "compliance",
    "encryption", "backup", "outage", "incident", "latency", "capacity", "warehouse", "freight", "customs",
    "tariff", "shipment", "inventory", "procurement", "supplier", "quote", "discount", "margin", "revenue",
    "churn", "retention", "cohort", "funnel", "pipeline", "campaign", "webinar", "newsletter", "branding",
    "logo", "website", "accessibility", "localization", "translation", "release", "beta", "testflight",
    "android", "ios", "crash", "analytics", "experiment", "pricingpage", "checkout", "refund", "subscription",
    "trial", "contractor", "designer", "prototype", "wireframe", "research", "interview", "persona",
    "journey", "sprint", "backlog", "estimate", "velocity", "standup", "retrospective", "hiringplan", "offer",
    "recruiter", "benefits", "payroll", "pension", "insurance", "lease", "office", "relocation", "furniture",
    "laptop", "licence", "firewall", "vpn", "sso", "okta", "identity", "permissions", "gdpr", "dpa",
    "archive", "export", "import", "api", "sdk", "webhook", "integration", "partner", "reseller", "channel",
    "territory", "quota", "commission", "board", "investor", "runway", "fundraising", "valuation",
    "diligence", "legal", "patent", "trademark", "grant", "sustainability", "carbon", "emissions", "solar",
    "battery", "fleet", "route", "dispatch", "scheduling", "triage", "dental", "patient", "appointment",
    "referral", "pharmacy", "formulary", "claims", "reimbursement"
)
FILLER = ("we", "should", "need", "to", "the", "next", "week", "before", "after", "review", "check", "agree",
          "confirm", "update", "plan", "with", "team", "for", "on", "first", "then", "still", "maybe", "really",
          "draft", "send", "share", "numbers", "status", "today", "monday", "friday", "quarter", "client", "call")


@dataclass(frozen=True)
class CorpusSpec:
    sources: int = 500
    segments: int = 50_000  # across all revisions, deleted ones included
    embedded: int = 10_000  # chunks plus assertions
    segments_per_chunk: int = 20
    owners: tuple[str, ...] = ("bench-owner-a", "bench-owner-b", "bench-owner-c")
    owner_weights: tuple[float, ...] = (0.6, 0.25, 0.15)
    entities_per_owner: int = 150
    topics: int = 40
    sources_per_series: int = 8
    revised: float = 0.10  # sources with a second revision
    deleted: float = 0.06  # sources deleted
    pending_purge: float = 0.2  # share of deleted sources whose purge has not run yet
    corroborated: float = 0.15  # assertions with support from a second meeting
    superseded: float = 0.12
    corrected: float = 0.03  # retracted by a human "wrong" decision
    contested: float = 0.02
    shadow: float = 0.01
    future: float = 0.01  # effective after `now`
    seed: int = 1104
    now: datetime = NOW
    history_days: int = 730


@dataclass(frozen=True)
class Series:
    owner: str
    topic: int
    title: str
    centroid: array
    project: str
    client: str | None
    actors: tuple[str, ...]
    tasks: tuple[str, ...]


@dataclass
class Corpus:
    spec: CorpusSpec
    rows: dict[type, list[dict]]
    topic_words: list[tuple[str, ...]]
    series: list[Series]
    actor_names: dict[str, str]
    owner_history: dict[str, tuple[datetime, datetime]] = field(default_factory=dict)

    def counts(self) -> dict[str, int]:
        return {model.__tablename__: len(rows) for model, rows in self.rows.items()}

    def digest(self) -> str:
        """A fingerprint of every row, for determinism checks."""
        h = hashlib.sha256()
        for model in TABLES:
            for row in self.rows[model]:
                h.update(repr(sorted((k, v.tobytes() if isinstance(v, array) else v)
                                     for k, v in row.items())).encode())
        return h.hexdigest()


@dataclass(frozen=True)
class BenchQuery:
    owner: str
    text: str
    vector: array
    entity: tuple[str, str]  # (entity_type, entity_id) for scoped runs
    known_at: datetime


# ---------------------------------------------------------------- vectors


def _unit(values) -> array:
    norm = math.sqrt(sum(x * x for x in values)) or 1.0
    return array("f", (x / norm for x in values))


def _noisy(rng: random.Random, centre, spread: float) -> array:
    """`centre` plus isotropic noise of expected length `spread`, normalized: cosine to centre ~ 1/sqrt(1+spread^2)."""
    sigma = spread / math.sqrt(DIMENSIONS)
    return _unit([c + rng.gauss(0.0, sigma) for c in centre])


def _random_unit(rng: random.Random) -> array:
    return _unit([rng.gauss(0.0, 1.0) for _ in range(DIMENSIONS)])


# ---------------------------------------------------------------- generation


class _Builder:
    def __init__(self, spec: CorpusSpec):
        self.spec, self.rng = spec, random.Random(spec.seed)
        self.rows: dict[type, list[dict]] = {model: [] for model in TABLES}
        self.model = memory_learning.EMBEDDING_MODEL

    def uid(self) -> str:
        return str(UUID(int=self.rng.getrandbits(128), version=4))

    def sentence(self, words: tuple[str, ...], names: list[str], n: int | None = None) -> str:
        rng = self.rng
        n = n or rng.randint(8, 22)
        out = []
        for _ in range(n):
            roll = rng.random()
            out.append(rng.choice(words) if roll < 0.5 else rng.choice(names) if roll < 0.58
                       else str(rng.randint(2, 400)) if roll < 0.63 else rng.choice(FILLER))
        return out[0].capitalize() + " " + " ".join(out[1:]) + "."

    # -------------------------------------------------- owners, entities and series

    def entities(self, owner: str) -> dict[str, list[str]]:
        n = self.spec.entities_per_owner
        shares = {"actor": 0.45, "project": 0.2, "client": 0.1, "task": 0.25}
        return {kind: [f"{kind}-{owner[-1]}-{i:03d}-{self.uid()[:8]}" for i in range(max(1, round(n * share)))]
                for kind, share in shares.items()}

    def build(self) -> Corpus:
        spec, rng = self.spec, self.rng
        topic_words = [tuple(rng.sample(TOPIC_WORDS, 12)) for _ in range(spec.topics)]
        topic_centroids = [_random_unit(rng) for _ in range(spec.topics)]
        actor_names: dict[str, str] = {}
        series: list[Series] = []
        per_owner = _apportion(spec.sources, spec.owner_weights)
        sources_by_owner: dict[str, list[tuple[Series, datetime]]] = {}
        for owner, count in zip(spec.owners, per_owner, strict=True):
            ents = self.entities(owner)
            for actor in ents["actor"]:
                actor_names[actor] = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
            owned = []
            for s in range(max(1, math.ceil(count / spec.sources_per_series))):
                topic = rng.randrange(spec.topics)
                owned.append(Series(
                    owner=owner, topic=topic, title=f"{topic_words[topic][0].capitalize()} sync {s + 1}",
                    centroid=_noisy(rng, topic_centroids[topic], 0.8),
                    project=ents["project"][s % len(ents["project"])],
                    client=rng.choice(ents["client"]) if rng.random() < 0.5 else None,
                    actors=tuple(rng.sample(ents["actor"], min(len(ents["actor"]), rng.randint(3, 8)))),
                    tasks=tuple(rng.sample(ents["task"], min(len(ents["task"]), 3)))))
            series += owned
            start = spec.now - timedelta(days=spec.history_days)
            meetings = sorted(((owned[i % len(owned)], start + timedelta(minutes=rng.randrange(spec.history_days
                                                                                              * 1440 - 1440)))
                               for i in range(count)), key=lambda m: m[1])
            sources_by_owner[owner] = meetings
        self.topic_words = topic_words
        self.actor_names = actor_names
        meetings = [(owner, s, at) for owner in spec.owners for s, at in sources_by_owner[owner]]
        self.sources(meetings)
        corpus = Corpus(spec=spec, rows=self.rows, topic_words=topic_words, series=series, actor_names=actor_names)
        for owner in spec.owners:
            times = [at for _, at in sources_by_owner[owner]]
            corpus.owner_history[owner] = (min(times), max(times)) if times else (spec.now, spec.now)
        return corpus

    # -------------------------------------------------- sources, revisions, segments, chunks

    def sources(self, meetings: list[tuple[str, Series, datetime]]) -> None:
        spec, rng = self.spec, self.rng
        n = len(meetings)
        revised = set(rng.sample(range(n), round(n * spec.revised)))
        deleted = set(rng.sample(range(n), round(n * spec.deleted)))
        pending = set(rng.sample(sorted(deleted), round(len(deleted) * spec.pending_purge)))
        revisions_total = n + len(revised)
        sizes = _apportion(spec.segments, [rng.uniform(0.6, 1.4) for _ in range(revisions_total)])
        chunks_total = sum(math.ceil(s / spec.segments_per_chunk) for s in sizes)
        assertions_total = max(0, spec.embedded - chunks_total)
        current_sizes = []
        size_iter = iter(sizes)
        self.meetings = []
        for i, (owner, series, occurred) in enumerate(meetings):
            source_id = self.uid()
            received = occurred + timedelta(minutes=rng.randint(35, 180) if rng.random() > 0.05
                                            else rng.randint(2, 6) * 1440)
            source = {"id": source_id, "owner_id": owner, "connector": rng.choice(("granola", "granola", "upload",
                                                                                   "api")),
                      "external_id": f"ext-{self.uid()[:12]}", "kind": "meeting",
                      "title": f"{series.title} #{i + 1}", "current_revision": 2 if i in revised else 1,
                      "status": "ready", "availability": "available", "deleted_at": None,
                      "upstream_owner_email": None, "upstream_owner_name": None, "created_at": received,
                      "updated_at": received}
            revs = []
            for revision in (1, 2) if i in revised else (1,):
                at = received + timedelta(days=rng.randint(1, 3)) * (revision - 1)
                revs.append(self.revision(owner, series, source_id, revision, occurred, at, next(size_iter),
                                          previous=revs[-1] if revs else None))
            current = revs[-1]
            current_sizes.append(len(current["segments"]))
            meeting = {"index": i, "owner": owner, "series": series, "source": source, "revisions": revs,
                       "current": current, "occurred": occurred, "received": received, "deleted": None}
            if i in deleted:
                gone = min(spec.now - timedelta(hours=1), current["at"] + timedelta(days=rng.randint(1, 60)))
                meeting["deleted"] = {"at": gone, "purged": i not in pending}
            self.meetings.append(meeting)
            self.rows[ContextSource].append(source)
            self.link(owner, "source", source_id, "project", series.project, "about", "exact", at=received)
            if series.client:
                self.link(owner, "source", source_id, "client", series.client, "about", "inferred", at=received)
        per_meeting = _apportion(assertions_total, current_sizes)
        for meeting, count in zip(self.meetings, per_meeting, strict=True):
            self.assertions(meeting, count)
        self.history()
        for meeting in self.meetings:
            if meeting["deleted"]:
                self.delete(meeting)
        for owner in spec.owners:
            seq = sum(1 for d in self.rows[ContextDecision] if d["owner_id"] == owner)
            self.rows[ContextOwnerState].append({"owner_id": owner, "decision_seq": seq, "created_at": spec.now,
                                                 "updated_at": spec.now})

    def revision(self, owner, series: Series, source_id, revision, occurred, at, size, previous) -> dict:
        rng, spec = self.rng, self.spec
        revision_id = self.uid()
        names = [self.actor_names[a].split()[0] for a in series.actors]
        words = self.topic_words[series.topic]
        self.rows[ContextSourceRevision].append({
            "id": revision_id, "owner_id": owner, "source_id": source_id, "revision": revision,
            "content_hash": hashlib.sha256(f"{revision_id}".encode()).hexdigest(), "change_kind": "content",
            "occurred_at": occurred, "ended_at": occurred + timedelta(minutes=rng.choice((30, 45, 60))),
            "timezone": "UTC", "participants": [{"name": self.actor_names[a]} for a in series.actors],
            "payload": {"title": series.title}, "summary": None,
            "received_at": at, "created_at": at, "upstream_updated_at": at})
        segments, offset = [], 0
        old = previous["segments"] if previous else []
        for seq in range(size):
            speaker = rng.choice(series.actors)
            text = (old[seq]["text"] if seq < len(old) and rng.random() < 0.8
                    else self.sentence(words, names))
            linked = rng.random() < 0.7
            row = {"id": self.uid(), "owner_id": owner, "revision_id": revision_id, "seq": seq,
                   "kind": "transcript" if rng.random() > 0.03 else "private_note",
                   "speaker_label": self.actor_names[speaker].split()[0],
                   "speaker_attribution": "other", "speaker_actor_id": speaker if linked else None,
                   "speaker_link_status": rng.choice(("exact", "inferred")) if linked else "unresolved",
                   "text": text, "start_ms": seq * 9_000, "end_ms": seq * 9_000 + 8_000, "char_start": offset,
                   "char_end": offset + len(text), "created_at": at}
            offset += len(text) + 1
            segments.append(row)
            if rng.random() < 0.05:
                kind, target = (("task", rng.choice(series.tasks)) if rng.random() < 0.5
                                else ("project", series.project))
                self.link(owner, "segment", row["id"], kind, target, "mentioned", "inferred", at=at)
        self.rows[ContextSegment] += segments
        chunks = []
        for c, start in enumerate(range(0, size, spec.segments_per_chunk)):
            part = segments[start:start + spec.segments_per_chunk]
            chunks.append({"id": self.uid(), "owner_id": owner, "revision_id": revision_id, "seq": c,
                           "segment_start": part[0]["seq"], "segment_end": part[-1]["seq"],
                           "token_count": sum(len(s["text"].split()) for s in part),
                           "text": "\n".join(s["text"] for s in part),
                           "embedding": _noisy(rng, series.centroid, 0.9), "embedding_model": self.model,
                           "embedded_at": at, "created_at": at, "updated_at": at})
        self.rows[ContextChunk] += chunks
        return {"id": revision_id, "revision": revision, "at": at, "segments": segments, "chunks": chunks}

    def link(self, owner, subject_type, subject_id, target_type, target_id, role, status, *, at, assertion_id=None):
        self.rows[ContextEntityLink].append({
            "id": self.uid(), "owner_id": owner, "assertion_id": assertion_id, "subject_type": subject_type,
            "subject_id": subject_id, "target_type": target_type if status != "unresolved" else None,
            "target_id": target_id if status != "unresolved" else None,
            "mention": self.actor_names.get(target_id, target_id)[:500], "role": role, "status": status,
            "score": 1.0 if status == "exact" else round(self.rng.uniform(0.6, 0.95), 3), "created_at": at,
            "updated_at": at})

    # -------------------------------------------------- assertions and evidence

    def assertions(self, meeting: dict, count: int) -> None:
        rng, owner, series = self.rng, meeting["owner"], meeting["series"]
        revs = meeting["revisions"]
        meeting["assertions"] = []
        for _ in range(count):
            # In an edited source, a fifth of the claims cite a passage the edit removed (retracted below).
            removed = len(revs) > 1 and rng.random() < 0.2
            revision = revs[0] if removed else revs[-1]
            segment = rng.choice(revision["segments"])
            kind = rng.choices(list(KINDS), weights=list(KINDS.values()))[0]
            roll = rng.random()
            subject = (("project", series.project) if roll < 0.5
                       else ("actor", segment["speaker_actor_id"] or rng.choice(series.actors)) if roll < 0.75
                       else ("client", series.client) if roll < 0.85 and series.client
                       else ("task", rng.choice(series.tasks)))
            words = segment["text"].rstrip(".").split()
            body = " ".join(words[: rng.randint(min(6, len(words)), len(words))])
            statement = f"{LEADS[kind]} {body}.".strip()
            valid_from = meeting["occurred"]
            if rng.random() < self.spec.future:
                valid_from = self.spec.now + timedelta(days=rng.randint(3, 60))
            assertion_id, at = self.uid(), revision["at"]
            commitment = kind == "commitment"
            row = {"id": assertion_id, "owner_id": owner, "kind": kind, "statement": statement,
                   "subject_type": subject[0], "subject_id": subject[1],
                   "object_type": "actor" if kind == "relationship" else None,
                   "object_id": rng.choice(series.actors) if kind == "relationship" else None,
                   "predicate": PREDICATES[kind], "key": f"{subject[1]}:{rng.randrange(4)}" if commitment else None,
                   "modality": "asserted", "explicit": True, "valid_from": valid_from, "valid_to": None,
                   "valid_precision": "day", "lifecycle": "current", "current_version": 1, "supersedes_id": None,
                   "confidence": round(rng.uniform(0.5, 0.99), 2),
                   "responsible": [{"entity_type": "actor", "entity_id": rng.choice(series.actors)}] if commitment
                   else [],
                   "due_at": valid_from + timedelta(days=rng.randint(1, 30)) if commitment else None,
                   "due_precision": "day" if commitment else None,
                   "embedding": _noisy(rng, series.centroid, 1.0), "embedding_model": self.model,
                   "embedded_at": at, "created_at": at, "updated_at": at}
            self.rows[ContextAssertion].append(row)
            meeting["assertions"].append(row)
            self.evidence(meeting, row, revision, segment)
            self.link(owner, "assertion", assertion_id, subject[0], subject[1], "subject", "exact", at=at,
                      assertion_id=assertion_id)
            extra = rng.random()
            if extra < 0.3:
                target = (("actor", rng.choice(series.actors)) if rng.random() < 0.6
                          else ("task", rng.choice(series.tasks)))
                status = "inferred" if extra < 0.24 else "rejected" if extra < 0.27 else "unresolved"
                self.link(owner, "assertion", assertion_id, *target, "mentioned", status, at=at,
                          assertion_id=assertion_id)
            row["_meeting"], row["_removed"] = meeting["index"], removed

    def evidence(self, meeting, assertion, revision, segment, *, role="support") -> None:
        self.rows[ContextEvidence].append({
            "id": self.uid(), "owner_id": meeting["owner"], "assertion_id": assertion["id"], "assertion_version": 1,
            "source_id": meeting["source"]["id"], "revision_id": revision["id"], "segment_id": segment["id"],
            "quote": segment["text"], "char_start": 0, "char_end": len(segment["text"]), "role": role,
            "origin": "independent", "derivation": {}, "created_at": revision["at"]})

    # -------------------------------------------------- lifecycle history

    def history(self) -> None:
        spec, rng = self.spec, self.rng
        by_series = defaultdict(list)
        for meeting in self.meetings:
            by_series[id(meeting["series"])].append(meeting)
        rows = self.rows[ContextAssertion]
        # Corroboration from another meeting of the same series.
        for row in rows:
            if rng.random() < spec.corroborated:
                peers = [m for m in by_series[id(self.meetings[row["_meeting"]]["series"])]
                         if m["index"] != row["_meeting"]]
                if peers:
                    other = rng.choice(peers)
                    self.evidence(other, row, other["current"], rng.choice(other["current"]["segments"]))
        support = defaultdict(set)
        for e in self.rows[ContextEvidence]:
            support[e["assertion_id"]].add(e["source_id"])
        gone = {m["source"]["id"] for m in self.meetings if m["deleted"] and m["deleted"]["purged"]}
        deleted_at = {m["source"]["id"]: m["deleted"]["at"] for m in self.meetings if m["deleted"]}
        open_rows = []
        for row in rows:
            meeting = self.meetings[row["_meeting"]]
            if row["_removed"]:
                self.change(row, meeting["revisions"][-1]["at"], lifecycle="retracted")
            elif support[row["id"]] <= gone:  # every supporting source deleted and purged
                self.change(row, max(deleted_at[s] for s in support[row["id"]]), lifecycle="retracted")
            else:
                open_rows.append(row)
        # Supersession: a later assertion about the same subject, kind and predicate from a later meeting.
        groups = defaultdict(list)
        for row in open_rows:
            if row["valid_from"] <= spec.now:
                groups[(row["owner_id"], row["subject_type"], row["subject_id"], row["kind"])].append(row)
        pairs = [(a, b) for group in groups.values() for a, b in pairwise(sorted(group, key=_order))
                 if a["_meeting"] != b["_meeting"] and a["valid_from"] < b["valid_from"]]
        rng.shuffle(pairs)
        target, used = round(len(rows) * spec.superseded), set()
        for a, b in pairs:
            if target <= 0:
                break
            if a["id"] in used or b["id"] in used:
                continue
            used |= {a["id"], b["id"]}
            b["supersedes_id"] = a["id"]
            self.change(a, b["created_at"], lifecycle="superseded", valid_to=b["valid_from"])
            target -= 1
        rest = [r for r in open_rows if r["id"] not in used]
        rng.shuffle(rest)
        n_corrected, n_contested = round(len(rows) * spec.corrected), round(len(rows) * spec.contested)
        n_shadow = round(len(rows) * spec.shadow)
        for row in rest[:n_corrected]:
            decision = self.decision(row, row["created_at"] + timedelta(days=rng.randint(1, 20)))
            self.change(row, decision["recorded_at"], lifecycle="retracted", decision_id=decision["id"])
        for row in rest[n_corrected:n_corrected + n_contested]:
            meeting = self.meetings[row["_meeting"]]
            self.evidence(meeting, row, meeting["current"], rng.choice(meeting["current"]["segments"]),
                          role="contradiction")
            self.change(row, row["created_at"] + timedelta(hours=1), lifecycle="contested")
        for row in rest[n_corrected + n_contested:n_corrected + n_contested + n_shadow]:
            row["lifecycle"] = "shadow"
        for row in rows:
            self.versions(row)
            del row["_meeting"], row["_removed"]
        # Successors (never themselves predecessors) insert after every predecessor they point at.
        rows.sort(key=lambda r: (r["supersedes_id"] is not None, r["created_at"], r["id"]))

    def change(self, row: dict, at: datetime, decision_id: str | None = None, **changes) -> None:
        row.setdefault("_changes", []).append((min(at, self.spec.now), changes, decision_id))

    def decision(self, row: dict, at: datetime) -> dict:
        at = min(at, self.spec.now)
        decision = {"id": self.uid(), "owner_id": row["owner_id"], "seq": 0, "target_type": "assertion",
                    "target_id": row["id"], "expected_revision": 1, "operation": "wrong",
                    "reason": "Corrected in review", "origin": "human", "principal": "owner",
                    "before": {"lifecycle": "current"}, "after": {"lifecycle": "retracted"}, "effective_at": at,
                    "recorded_at": at, "created_at": at}
        self.rows[ContextDecision].append(decision)
        return decision

    def versions(self, row: dict) -> None:
        state = {"statement": row["statement"], "lifecycle": row["lifecycle"], "valid_from": row["valid_from"],
                 "valid_to": None, "valid_precision": row["valid_precision"]}
        recorded, decision_id, version = row["created_at"], None, 1
        for at, changes, next_decision in sorted(row.pop("_changes", []), key=lambda c: c[0]):
            at = max(at, recorded)
            self.rows[ContextAssertionVersion].append(self._version(row, version, state, recorded, at, decision_id))
            state, recorded, decision_id, version = {**state, **changes}, at, next_decision, version + 1
        self.rows[ContextAssertionVersion].append(self._version(row, version, state, recorded, None, decision_id))
        row.update(lifecycle=state["lifecycle"], valid_to=state["valid_to"], current_version=version,
                   updated_at=recorded)

    def _version(self, row, version, state, recorded_from, recorded_to, decision_id) -> dict:
        return {"id": self.uid(), "owner_id": row["owner_id"], "assertion_id": row["id"], "version": version,
                **state, "recorded_from": recorded_from, "recorded_to": recorded_to, "decision_id": decision_id,
                "created_at": recorded_from}

    def delete(self, meeting: dict) -> None:
        """Deleted source; when purged, content is blanked as `sources.purge_source` leaves it."""
        source, gone = meeting["source"], meeting["deleted"]
        source.update(status="deleted", deleted_at=gone["at"], updated_at=gone["at"])
        if not gone["purged"]:
            return
        source.update(title="", upstream_owner_email=None, upstream_owner_name=None)
        revision_ids = {r["id"] for r in meeting["revisions"]}
        for revision in self.rows[ContextSourceRevision]:
            if revision["id"] in revision_ids:
                revision.update(payload={"purged": True}, participants=[], summary=None)
        for revision in meeting["revisions"]:
            for segment in revision["segments"]:
                segment.update(text="", speaker_label=None)
            for chunk in revision["chunks"]:
                chunk.update(text="", embedding=None, embedding_model=None, embedded_at=None)
        for e in self.rows[ContextEvidence]:
            if e["source_id"] == source["id"]:
                e.update(quote="", char_start=None, char_end=None, segment_id=None,
                         derivation={"source_deleted": True})
        for link in self.rows[ContextEntityLink]:
            if link["subject_type"] == "source" and link["subject_id"] == source["id"]:
                link["mention"] = ""


def _order(row: dict):
    return row["valid_from"], row["created_at"], row["id"]


def _apportion(total: int, weights) -> list[int]:
    """Integer shares of `total` proportional to `weights` (largest remainder), summing exactly to `total`."""
    weights = list(weights)
    scale = sum(weights)
    raw = [total * w / scale for w in weights]
    out = [math.floor(x) for x in raw]
    for i in sorted(range(len(raw)), key=lambda i: out[i] - raw[i])[: total - sum(out)]:
        out[i] += 1
    return out


def generate(spec: CorpusSpec | None = None) -> Corpus:
    corpus = _Builder(spec or CorpusSpec()).build()
    seqs = Counter()
    corpus.rows[ContextDecision].sort(key=lambda d: (d["recorded_at"], d["id"]))
    for decision in corpus.rows[ContextDecision]:
        seqs[decision["owner_id"]] += 1
        decision["seq"] = seqs[decision["owner_id"]]
    return corpus


# ---------------------------------------------------------------- loading and queries


def load(db, corpus: Corpus, *, batch: int = 2_000) -> dict[str, int]:
    """Insert every row in foreign-key order inside the caller's transaction; returns row counts per table."""
    for model in TABLES:
        rows = corpus.rows[model]
        for start in range(0, len(rows), batch):
            part = [{k: (v.tolist() if isinstance(v, array) else v) for k, v in row.items()}
                    for row in rows[start:start + batch]]
            if part:
                db.execute(insert(model), part)
    return corpus.counts()


def queries(corpus: Corpus, n: int, *, seed: int = 7) -> list[BenchQuery]:
    """Benchmark queries: owners by corpus weight, two or three words of one of the owner's series topics (one
    query in ten adds a word no meeting uses), a query vector near that series, and a scope entity from the series
    (project, actor or client) plus a `known_at` instant inside the owner's history."""
    rng, spec = random.Random(seed), corpus.spec
    by_owner = defaultdict(list)
    for s in corpus.series:
        by_owner[s.owner].append(s)
    owners = [o for o in spec.owners if by_owner[o]]
    weights = [w for o, w in zip(spec.owners, spec.owner_weights, strict=True) if by_owner[o]]
    out = []
    for _ in range(n):
        owner = rng.choices(owners, weights=weights)[0]
        series = rng.choice(by_owner[owner])
        words = rng.sample(corpus.topic_words[series.topic], rng.randint(2, 3))
        if rng.random() < 0.1:
            words.append("zyxquorble")
        roll = rng.random()
        entity = (("project", series.project) if roll < 0.6 else ("client", series.client) if roll < 0.75
                  and series.client else ("actor", rng.choice(series.actors)))
        first, last = corpus.owner_history[owner]
        known_at = first + (last - first) * rng.uniform(0.3, 0.9)
        out.append(BenchQuery(owner=owner, text=" ".join(words), vector=_noisy(rng, series.centroid, 1.2),
                              entity=entity, known_at=known_at))
    return out


# ---------------------------------------------------------------- one synthetic meeting

WORDS_PER_MINUTE = 150  # conversational speech: about 9,000 words in an hour
TASKS = ("send the revised budget", "draft the rollout checklist", "book the training rooms", "share the vendor shortlist",
         "update the migration plan", "review the clinic contract", "write up the pricing options",
         "schedule the security review")
DECISIONS = ("we keep the pilot on the current vendor", "we run training on Thursdays",
             "we move reporting into the dashboard", "we start with the northern clinics",
             "we freeze scope until the audit", "we pay the supplier quarterly")


def meeting(minutes: int = 60, *, seed: int = 60, owner_email: str = "owner@example.com") -> SourceEnvelope:
    """A synthetic transcript of `minutes` of speech from four participants (the first is the note owner): utterances
    of 6-40 words at WORDS_PER_MINUTE (0.4 s between turns), with a commitment ("I'll <task> by <ISO date>.") about every five minutes and
    a decision ("Decision: ...") about every ten, which both the rule-based double and a model would pick up."""
    rng = random.Random(seed)
    at = NOW - timedelta(days=1)
    people = [("Alex Morgan", owner_email, "owner"), ("Priya Natarajan", None, "other"),
              ("Marco Rossi", None, "other"), ("Lena Weber", None, "other")]
    words = rng.sample(TOPIC_WORDS, 16)
    names = [p[0].split()[0] for p in people]
    builder = _Builder(CorpusSpec(seed=seed))
    segments, ms, seq = [], 0, 0
    while ms < minutes * 60_000:
        name, _, attribution = people[0] if rng.random() < 0.3 else rng.choice(people[1:])
        n = rng.randint(6, 40)
        text = " ".join(builder.sentence(words, names, max(4, n // 2)) for _ in range(2 if n > 20 else 1))
        if seq % 37 == 11:
            due = (at + timedelta(days=rng.randint(2, 21))).date().isoformat()
            text += f" I'll {rng.choice(TASKS)} by {due}."
        elif seq % 73 == 29:
            text = f"Decision: {rng.choice(DECISIONS)}. " + text
        count = len(text.split())
        duration = round(count * 60_000 / WORDS_PER_MINUTE)
        segments.append(Segment(seq=seq, speaker_label=name.split()[0], attribution=attribution, text=text,
                                start_ms=ms, end_ms=ms + duration))
        ms, seq = ms + duration + 400, seq + 1
    return SourceEnvelope(
        connector="api", external_id=f"synthetic-{minutes}m-{seed}", title="Pilot rollout working session",
        occurred_at=at, ended_at=at + timedelta(minutes=minutes), timezone="America/New_York",
        participants=[Participant(name=n, email=e, role="organizer" if a == "owner" else "attendee",
                                  speaker_label=n.split()[0]) for n, e, a in people],
        segments=segments, external_revision="rev-1", upstream_owner_email=owner_email,
        upstream_owner_name=people[0][0])
