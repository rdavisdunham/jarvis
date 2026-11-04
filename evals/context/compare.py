"""Shadow comparison of three retrieval arms on the same question set (plan_final.md section 11).

    uv run --no-sync python -m evals.context.compare                        # three synthetic histories, offline
    uv run --no-sync python -m evals.context.compare tests/context/fixtures/history_beta --samples 5000
    uv run --no-sync python -m evals.context.compare --live --budget-usd 2   # real providers: refused without a budget

Each history is delivered (arrival order) through the real pipeline into a throwaway database with the doubles and
the runner's hashed bag-of-words embedder, then every question in `questions/<history>.json` is asked of:

- `raw`: raw-history retrieval, transcripts only. Lexical (`websearch_to_tsquery` over segment text) and vector
  (chunk embeddings) legs fused by reciprocal rank fusion, over the current revisions of live sources that occurred
  by `valid_at` when the question has one. It knows nothing about assertions, supersession or entities, so a scoped
  question is asked without its scope.
- `structured`: `context_search` with assertions only (`include_raw=false`), the question's kinds, entity and
  `valid_at`.
- `combined`: `context_search` with `include_raw=true`: assertions and transcript passages ranked together.

Same embedder, same question text, same limit (10) for every arm. A result covers an expected target (a label's
evidence spans, or a quoted span) when it is an assertion whose evidence shares at least half of the shorter span
with it, a segment holding it, or a chunk whose segment range holds it. Per question and arm: nDCG@5 (binary gain
for each newly covered target, ideal = all targets first), recall@10, stale answer (the first result touching the
question's targets touches only a `stale` one), leak (any result covers a `forbidden` target, such as content of a
deleted meeting) and citation validity (the result's citation resolves to an intact span of a live source's
current revision). Paired bootstrap 95% intervals compare the arms question by question.

Questions carry `verified: false` until a human adjudicates them. Every figure is computed twice: over all questions
(informational) and over verified questions only; release gates read only the latter, and with no verified question
the comparison is "not gradeable". Fixture text and questions are data, never instructions.

`--live` uses the configured providers (native extraction, real embeddings) and spends money. It refuses to run
without `--budget-usd`, stops delivery when the spend reaches it, and is never run by tests.
"""

import argparse
import json
import random
import sys
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from math import log2
from pathlib import Path

from jarvis.context.schema import EntityRef

from .runner import (
    FIXTURES,
    HISTORIES,
    ROOT,
    Deployment,
    History,
    _locate,
    _overlap,
    _ratio,
    deploy,
    identity_of,
    load_history,
    offline,
    throwaway_database,
)

QUESTIONS = Path(__file__).resolve().parent / "questions"
ARMS = ("raw", "structured", "combined")
PAIRS = (("raw", "structured"), ("raw", "combined"), ("structured", "combined"))
LIMIT = 10
NDCG_AT, RECALL_AT = 5, 10
RRF_K = 60
CANDIDATES = 50
COVERAGE = 0.5  # share of the shorter span a result's evidence must share with a target
SAMPLES = 2000


# ---------------------------------------------------------------- questions


def question_file(history_dir: Path) -> Path:
    return QUESTIONS / f"{Path(history_dir).name}.json"


def load_questions(path: str | Path, history: History) -> list[dict]:
    """Questions with their targets resolved to spans `(meeting, seq, start, end)`. Raises on an unknown label or a
    quote that does not occur in exactly one segment of its meeting."""
    data = json.loads(Path(path).read_text())
    labels = {x["id"]: x for x in history.labels["assertions"]}
    questions = []

    def spans(qid: str, target) -> list[tuple]:
        if isinstance(target, str):
            if target not in labels:
                raise ValueError(f"{qid}: unknown label {target!r}")
            return [_locate(history, e["meeting"], e["quote"]) for e in labels[target]["evidence"]]
        return [_locate(history, target["meeting"], target["quote"])]

    for q in data["questions"]:
        questions.append({**q, "targets": {name: [spans(q["id"], t) for t in q.get(name, [])]
                                           for name in ("expected", "stale", "forbidden")}})
    return questions


# ---------------------------------------------------------------- the database index


@dataclass
class Index:
    """Where every segment, chunk and assertion of one deployment sits, for coverage and citation checks."""

    segments: dict[str, tuple[str, int, str, str]]  # id -> (meeting, seq, revision_id, text)
    chunks: dict[str, tuple[str, int, int, str]]  # id -> (meeting, segment_start, segment_end, revision_id)
    assertions: dict[str, list[tuple]]  # id -> support spans on live sources
    live: set[str]  # current revisions of sources that are not deleted
    texts: dict[tuple[str, int], str]  # (revision_id, seq) -> text


def build_index(deployment: Deployment) -> Index:
    from jarvis.context import sources
    from jarvis.context.models import (
        ContextChunk,
        ContextEvidence,
        ContextSegment,
        ContextSource,
        ContextSourceRevision,
    )
    from jarvis.db import session_scope
    from sqlalchemy import select

    owner = deployment.owner
    with session_scope() as db:
        meeting_of, live = {}, set()
        for revision, source in db.execute(select(ContextSourceRevision, ContextSource).join(
                ContextSource, ContextSource.id == ContextSourceRevision.source_id)
                .where(ContextSource.owner_id == owner)):
            meeting_of[revision.id] = sources.split_external_id(source, revision)[1]
            if source.status != "deleted" and source.deleted_at is None and revision.revision == source.current_revision:
                live.add(revision.id)
        segments, texts, by_id = {}, {}, {}
        for g in db.scalars(select(ContextSegment).where(ContextSegment.owner_id == owner)):
            segments[g.id] = (meeting_of[g.revision_id], g.seq, g.revision_id, g.text)
            by_id[g.id] = g
            if g.text:
                texts[(g.revision_id, g.seq)] = g.text
        chunks = {c.id: (meeting_of[c.revision_id], c.segment_start, c.segment_end, c.revision_id)
                  for c in db.scalars(select(ContextChunk).where(ContextChunk.owner_id == owner))}
        assertions: dict[str, list[tuple]] = {}
        for e in db.scalars(select(ContextEvidence).where(ContextEvidence.owner_id == owner,
                                                          ContextEvidence.role == "support")):
            g = by_id.get(e.segment_id)
            if g is not None and e.char_start is not None and g.revision_id in live:
                assertions.setdefault(e.assertion_id, []).append(
                    (meeting_of[g.revision_id], g.seq, e.char_start, e.char_end))
    return Index(segments, chunks, assertions, live, texts)


def covers(index: Index, hit: dict, spans: list[tuple]) -> bool:
    for meeting, seq, start, end in spans:
        if hit["ref"] == "assertion":
            if any(_overlap((meeting, seq, start, end), s) >= COVERAGE for s in index.assertions.get(hit["id"], [])):
                return True
        elif hit["ref"] == "segment":
            found = index.segments.get(hit["id"])
            if found and found[:2] == (meeting, seq):
                return True
        elif hit["ref"] == "chunk":
            found = index.chunks.get(hit["id"])
            if found and found[0] == meeting and found[1] <= seq <= found[2]:
                return True
    return False


def citation_valid(index: Index, hit: dict) -> bool:
    """The citation resolves to text that is still there: a live source's current revision, and for assertions the
    quoted span inside the cited segment."""
    cited = hit.get("citation")
    if not cited or cited.get("revision_id") not in index.live:
        return False
    revision = cited["revision_id"]
    if hit["ref"] == "assertion":
        text = index.texts.get((revision, cited.get("segment")))
        return bool(text and cited.get("quote") and cited["quote"] in text)
    start, end = cited.get("segment_start"), cited.get("segment_end")
    return start is not None and end is not None and all((revision, s) in index.texts for s in range(start, end + 1))


# ---------------------------------------------------------------- the arms


def _fuse(*legs: list[tuple[str, str]]) -> list[tuple[str, str, float]]:
    scores: dict[tuple[str, str], float] = {}
    for leg in legs:
        for rank, key in enumerate(leg, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
    return sorted(((t, i, s) for (t, i), s in scores.items()), key=lambda x: (-x[2], x[0], x[1]))


def raw_search(owner: str, text: str, *, valid_at: datetime | None, limit: int = LIMIT) -> list[dict]:
    """Raw-history retrieval: lexical over segments and vector over chunks of live sources' current revisions."""
    from jarvis import memory_learning
    from jarvis.context.models import ContextChunk, ContextSegment, ContextSource, ContextSourceRevision
    from jarvis.db import session_scope
    from sqlalchemy import func, literal_column, select

    try:
        vector = memory_learning.query_embedding(owner, text[:2000])
    except Exception:  # noqa: BLE001 - the vector leg is optional, as in the native retriever
        vector = None
    g, c, r, s = ContextSegment, ContextChunk, ContextSourceRevision, ContextSource
    scope = [s.owner_id == owner, r.owner_id == owner, s.deleted_at.is_(None), s.status != "deleted",
             r.revision == s.current_revision]
    if valid_at is not None:
        scope.append(r.occurred_at <= valid_at)
    tsq = func.websearch_to_tsquery(literal_column("'english'::regconfig"), text)
    score = func.ts_rank_cd(g.tsv, tsq)
    with session_scope() as db:
        lexical = [("segment", i) for i, in db.execute(
            select(g.id).join(r, r.id == g.revision_id).join(s, s.id == r.source_id)
            .where(g.owner_id == owner, g.text != "", g.tsv.op("@@")(tsq), *scope)
            .order_by(score.desc(), g.id).limit(CANDIDATES))]
        semantic = [] if vector is None else [("chunk", i) for i, in db.execute(
            select(c.id).join(r, r.id == c.revision_id).join(s, s.id == r.source_id)
            .where(c.owner_id == owner, c.embedding.is_not(None),
                   c.embedding_model == memory_learning.EMBEDDING_MODEL, *scope)
            .order_by(c.embedding.cosine_distance(vector), c.id).limit(CANDIDATES))]
        hits = _fuse(lexical, semantic)[:limit]
        revisions = {}
        for ref, ident, _ in hits:
            model = g if ref == "segment" else c
            row = db.get(model, ident)
            span = (row.seq, row.seq) if ref == "segment" else (row.segment_start, row.segment_end)
            revisions[ident] = {"revision_id": row.revision_id, "segment_start": span[0], "segment_end": span[1]}
    return [{"ref": ref, "id": ident, "score": round(value, 6), "citation": revisions[ident]}
            for ref, ident, value in hits]


def service_search(deployment: Deployment, q: dict, *, include_raw: bool, limit: int = LIMIT) -> list[dict]:
    from jarvis.context import service

    entity = q["scope"].get("entity")
    if entity:
        entity_id = deployment.actor(entity["entity_id"]) if entity["entity_type"] == "actor" else entity["entity_id"]
        entity = EntityRef(entity_type=entity["entity_type"], entity_id=entity_id)
    args = service.SearchInput(query=q["question"], kinds=q["scope"].get("kinds", []), entity=entity,
                               valid_at=datetime.fromisoformat(q["valid_at"]) if q.get("valid_at") else None,
                               include_raw=include_raw, limit=limit)
    with service.as_owner(identity_of(deployment.owner)):
        found = service.search(args)
    hits = []
    for item in found["results"]:
        cited = item.get("citation")
        if cited and item["ref"] == "assertion":
            cited = {"revision_id": cited["revision_id"], "segment": cited["segment"], "quote": cited["quote"]}
        elif cited:
            cited = {k: cited[k] for k in ("revision_id", "segment_start", "segment_end")}
        hits.append({"ref": item["ref"], "id": item["id"], "score": item.get("score"), "citation": cited})
    return hits


def arm_search(arm: str, deployment: Deployment, q: dict) -> list[dict]:
    if arm == "raw":
        valid_at = datetime.fromisoformat(q["valid_at"]) if q.get("valid_at") else None
        return raw_search(deployment.owner, q["question"], valid_at=valid_at)
    return service_search(deployment, q, include_raw=arm == "combined")


# ---------------------------------------------------------------- scoring


def ndcg(gains: list[int], ideal: int, k: int = NDCG_AT) -> float | None:
    """Binary-gain nDCG@k; `ideal` is the number of relevant targets."""
    if ideal <= 0:
        return None
    dcg = sum(g / log2(rank + 1) for rank, g in enumerate(gains[:k], start=1))
    return dcg / sum(1 / log2(rank + 1) for rank in range(1, min(ideal, k) + 1))


def score_hits(index: Index, q: dict, hits: list[dict]) -> dict:
    expected, stale, forbidden = (q["targets"][k] for k in ("expected", "stale", "forbidden"))
    seen, gains = set(), []
    for hit in hits:
        new = {i for i, spans in enumerate(expected) if i not in seen and covers(index, hit, spans)}
        gains.append(1 if new else 0)
        seen |= new
    recall_seen = {i for i, spans in enumerate(expected) if any(covers(index, h, spans) for h in hits[:RECALL_AT])}
    verdict = None
    if stale:
        verdict = "unanswered"
        for hit in hits:
            if any(covers(index, hit, spans) for spans in expected):
                verdict = "current"
                break
            if any(covers(index, hit, spans) for spans in stale):
                verdict = "stale"
                break
    valid = [citation_valid(index, h) for h in hits]
    return {"ndcg": ndcg(gains, len(expected)), "recall": _ratio(len(recall_seen), len(expected)),
            "stale": verdict, "leak": any(covers(index, h, spans) for h in hits[:RECALL_AT] for spans in forbidden),
            "citations": len(valid), "valid_citations": sum(valid), "results": len(hits)}


def _mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 4) if values else None


def summarize(rows: list[dict]) -> dict:
    stale = [r for r in rows if r["stale"] is not None]
    return {"questions": len(rows), "ranked": sum(1 for r in rows if r["ndcg"] is not None),
            "ndcg_at_5": _mean(r["ndcg"] for r in rows), "recall_at_10": _mean(r["recall"] for r in rows),
            "stale_questions": len(stale), "stale_answers": sum(1 for r in stale if r["stale"] == "stale"),
            "stale_answer_rate": _ratio(sum(1 for r in stale if r["stale"] == "stale"), len(stale)),
            "forbidden_questions": sum(1 for r in rows if r["forbidden"]), "leaks": sum(1 for r in rows if r["leak"]),
            "citations": sum(r["citations"] for r in rows), "valid_citations": sum(r["valid_citations"] for r in rows),
            "citation_validity": _ratio(sum(r["valid_citations"] for r in rows), sum(r["citations"] for r in rows))}


def _percentile(ordered: list[float], q: float) -> float:
    return ordered[min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))]


def paired_bootstrap(a: list[float | None], b: list[float | None], *, samples: int = SAMPLES,
                     seed: int = 0) -> dict:
    """Mean of b - a over questions where both are defined, with a 95% percentile interval from resampling the
    questions (pairs kept together). Deterministic for a seed."""
    pairs = [(x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None]
    n = len(pairs)
    if not n:
        return {"n": 0, "mean_a": None, "mean_b": None, "difference": None, "low": None, "high": None}
    diffs = [y - x for x, y in pairs]
    rng = random.Random(seed)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples))
    return {"n": n, "mean_a": round(sum(x for x, _ in pairs) / n, 4), "mean_b": round(sum(y for _, y in pairs) / n, 4),
            "difference": round(sum(diffs) / n, 4), "low": round(_percentile(means, 0.025), 4),
            "high": round(_percentile(means, 0.975), 4)}


METRICS = {"ndcg_at_5": lambda r: r["ndcg"], "recall_at_10": lambda r: r["recall"],
           "stale_answer": lambda r: None if r["stale"] is None else float(r["stale"] == "stale"),
           "citation_validity": lambda r: r["valid_citations"] / r["citations"] if r["citations"] else None}


def paired(rows: dict[str, list[dict]], *, samples: int, seed: int) -> dict:
    out = {}
    for a, b in PAIRS:
        out[f"{b}_vs_{a}"] = {name: paired_bootstrap([get(r) for r in rows[a]], [get(r) for r in rows[b]],
                                                     samples=samples, seed=seed)
                              for name, get in METRICS.items()}
    return out


def _evaluate(rows: dict[str, list[dict]], *, samples: int, seed: int) -> dict:
    return {"arms": {arm: summarize(rows[arm]) for arm in ARMS}, "paired": paired(rows, samples=samples, seed=seed)}


def _usage(owner: str) -> float:
    from jarvis.db import session_scope
    from jarvis.models import Usage
    from sqlalchemy import func, select

    with session_scope() as db:
        return float(db.scalar(select(func.coalesce(func.sum(Usage.amount), 0)).where(Usage.owner_id == owner)))


def compare(history_dir: str | Path, deployment: Deployment | None = None, questions: str | Path | None = None,
            *, samples: int = SAMPLES, seed: int = 0) -> dict:
    """Ask every question of the three arms against one deployed history (delivered in arrival order when no
    deployment is given). Needs a database and `offline()` (or live providers)."""
    history = load_history(history_dir)
    deployment = deployment or deploy(history, "arrival")
    items = load_questions(questions or question_file(history_dir), history)
    index = build_index(deployment)
    rows: dict[str, list[dict]] = {arm: [] for arm in ARMS}
    latency: dict[str, list[float]] = {arm: [] for arm in ARMS}
    spent = dict.fromkeys(ARMS, 0.0)
    for q in items:
        for arm in ARMS:
            before, started = _usage(deployment.owner), time.perf_counter()
            hits = arm_search(arm, deployment, q)
            latency[arm].append(time.perf_counter() - started)
            spent[arm] += _usage(deployment.owner) - before
            rows[arm].append({"id": q["id"], "category": q["category"], "verified": bool(q.get("verified")),
                              "forbidden": bool(q["targets"]["forbidden"]), **score_hits(index, q, hits),
                              "top": [f"{h['ref']}:{h['id']}" for h in hits[:3]]})
    verified = {arm: [r for r in rows[arm] if r["verified"]] for arm in ARMS}
    return {"history": history.name, "owner": deployment.owner,
            "questions": {"total": len(items), "verified": sum(1 for q in items if q.get("verified")),
                          "unverified_excluded": sum(1 for q in items if not q.get("verified"))},
            "all": _evaluate(rows, samples=samples, seed=seed), "verified": _evaluate(verified, samples=samples, seed=seed),
            "latency_s": {arm: _latency(latency[arm]) for arm in ARMS},
            "query_cost_usd": {arm: {"total": round(spent[arm], 6),
                                     "per_query": round(spent[arm] / len(items), 6) if items else None} for arm in ARMS},
            "rows": rows}


def _latency(values: list[float]) -> dict:
    ordered = sorted(values)
    return {"n": len(ordered), "p50": round(_percentile(ordered, 0.5), 4) if ordered else None,
            "p95": round(_percentile(ordered, 0.95), 4) if ordered else None}


def pooled(results: list[dict], *, samples: int = SAMPLES, seed: int = 0) -> dict:
    """All histories' questions together (questions stay paired across arms)."""
    out = {}
    for which in ("all", "verified"):
        rows = {arm: [r for result in results for r in result["rows"][arm]
                      if which == "all" or r["verified"]] for arm in ARMS}
        out[which] = _evaluate(rows, samples=samples, seed=seed)
    out["questions"] = {k: sum(r["questions"][k] for r in results) for k in ("total", "verified", "unverified_excluded")}
    return out


# ---------------------------------------------------------------- report


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)


def _interval(p: dict) -> str:
    if not p["n"]:
        return "n/a (no paired questions)"
    return f"{p['difference']:+.3f} [{p['low']:+.3f}, {p['high']:+.3f}] (n={p['n']})"


def arm_table(evaluation: dict) -> list[str]:
    lines = ["| Arm | nDCG@5 | Recall@10 | Stale answers | Leaks | Citation validity |", "|---|---|---|---|---|---|"]
    for arm in ARMS:
        a = evaluation["arms"][arm]
        lines.append(f"| {arm} | {_fmt(a['ndcg_at_5'])} (n={a['ranked']}) | {_fmt(a['recall_at_10'])} | "
                     f"{a['stale_answers']}/{a['stale_questions']} ({_fmt(a['stale_answer_rate'])}) | "
                     f"{a['leaks']}/{a['forbidden_questions']} | {_fmt(a['citation_validity'])} "
                     f"({a['valid_citations']}/{a['citations']}) |")
    return lines


def paired_table(evaluation: dict) -> list[str]:
    lines = ["| Comparison | nDCG@5 difference [95% CI] | Recall@10 difference | Stale-answer difference |",
             "|---|---|---|---|"]
    for name, metrics in evaluation["paired"].items():
        lines.append(f"| {name.replace('_vs_', ' vs ')} | {_interval(metrics['ndcg_at_5'])} | "
                     f"{_interval(metrics['recall_at_10'])} | {_interval(metrics['stale_answer'])} |")
    return lines


def markdown(results: list[dict], total: dict) -> str:
    q = total["questions"]
    lines = ["# Context retrieval comparison: raw history, native structured, combined", "",
             (f"Questions: {q['total']} across {len(results)} histories; verified {q['verified']}; "
             f"unverified labels excluded from release scoring: {q['unverified_excluded']}."), ""]
    if not q["verified"]:
        lines += [("**Release comparison: not gradeable** (no verified questions). The tables below use unverified "
                  "synthetic questions and the deterministic doubles; they show the mechanism, not engine quality."), ""]
    for which in ("verified", "all"):
        title = "Verified questions (release)" if which == "verified" else "All questions (informational)"
        lines += [f"## {title}", ""]
        if which == "verified" and not q["verified"]:
            lines += ["Not gradeable: no verified questions.", ""]
            continue
        lines += arm_table(total[which]) + [""] + paired_table(total[which]) + [""]
    lines += ["## Per history (all questions, informational)", ""]
    for result in results:
        lines += [(f"### {result['history']} ({result['questions']['total']} questions, "
                  f"{result['questions']['verified']} verified)"), ""] + arm_table(result["all"]) + [""]
        lines += ["Latency p50/p95 (s): " + ", ".join(
            f"{arm} {_fmt(result['latency_s'][arm]['p50'])}/{_fmt(result['latency_s'][arm]['p95'])}" for arm in ARMS)
                  + ". Query cost (USD): " + ", ".join(
            f"{arm} {result['query_cost_usd'][arm]['total']:.6f}" for arm in ARMS) + ".", ""]
    lines += [("The historical 5% relative nDCG@5 rule is supporting evidence only (plan section 11): a challenger must "
              "also pass the permission, provenance, correction and replay gates."), ""]
    return "\n".join(lines)


# ---------------------------------------------------------------- live mode (paid; never run by tests)


@contextmanager
def live():
    """Real providers with the context flag on and a per-meeting extraction cap; `check(owner)` stops delivery
    once runs and usage events reach the budget."""
    from jarvis.config import get_settings

    settings = get_settings()
    flag = settings.context_enabled
    settings.context_enabled = True
    try:
        yield
    finally:
        settings.context_enabled = flag


def budget_check(budget_usd: float) -> Callable[[str], None]:
    def check(owner: str) -> None:
        from jarvis.context.models import ContextRun
        from jarvis.db import session_scope
        from sqlalchemy import func, select

        with session_scope() as db:
            runs = float(db.scalar(select(func.coalesce(func.sum(ContextRun.cost_usd), 0))
                                   .where(ContextRun.owner_id == owner)))
        if max(runs, _usage(owner)) >= budget_usd:
            raise SystemExit(f"live comparison stopped: spend reached the {budget_usd:.2f} USD budget")
    return check


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.context.compare", description=__doc__.split("\n\n")[0])
    parser.add_argument("histories", nargs="*", type=Path)
    parser.add_argument("--questions", type=Path, help="question file (default: questions/<history>.json)")
    parser.add_argument("--samples", type=int, default=SAMPLES, help="bootstrap resamples")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "context-eval")
    parser.add_argument("--live", action="store_true", help="real providers (paid); needs --budget-usd")
    parser.add_argument("--budget-usd", type=float, default=None, help="hard spend cap for --live")
    args = parser.parse_args(argv)
    if args.live and not (args.budget_usd and args.budget_usd > 0):
        print("--live spends money on real model and embedding calls; give an explicit --budget-usd", file=sys.stderr)
        return 2
    histories = args.histories or [FIXTURES / name for name in HISTORIES]
    if args.questions and len(histories) != 1:
        print("--questions applies to one history", file=sys.stderr)
        return 2
    with throwaway_database():
        context = live() if args.live else offline()
        with context:
            results = []
            for path in histories:
                deployment = None
                if args.live:
                    per_meeting = args.budget_usd / max(len(load_history(path).meetings), 1)
                    from jarvis.config import get_settings
                    get_settings().context_source_budget_usd = per_meeting
                    deployment = deploy(load_history(path), "arrival", engine="native",
                                        between=budget_check(args.budget_usd))
                results.append(compare(path, deployment, args.questions, samples=args.samples, seed=args.seed))
            total = pooled(results, samples=args.samples, seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "compare.json").write_text(json.dumps({"histories": results, "pooled": total}, indent=1,
                                                      sort_keys=True, default=str) + "\n")
    text = markdown(results, total)
    (args.out / "compare.md").write_text(text)
    print(text)
    print(f"wrote {args.out / 'compare.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
