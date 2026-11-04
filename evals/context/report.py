"""The release report: every plan_final.md section 11 gate in one markdown file.

    uv run --no-sync python -m evals.context.report                                    # synthetic, offline
    uv run --no-sync python -m pytest -q tests/context --junitxml outputs/context-eval/pytest.xml
    uv run --no-sync python -m evals.context.report --junit outputs/context-eval/pytest.xml \\
        --series outputs/context-series/weekly --series outputs/context-series/client

For each synthetic history it runs the runner's database mode (the real pipeline with the doubles in a throwaway
database: three orders, redelivery, ten maintenance runs) and the three-arm comparison, then grades:

- mechanical gates from a pytest JUnit file (`--junit`; without it they are "not run") plus the evaluation's own
  probes (no task written by ingestion or maintenance, nothing created by unchanged runs, no disclosure to another
  owner, full coverage, valid citations);
- critical temporal and permission scenarios, which fail the release whatever the averages say;
- quality targets on verified labels only, with sample counts and 95% intervals: synthetic labels are unverified,
  so the targets stay "not gradeable" until adjudicated series (`--series`, from `worksheet import`) supply
  verified verdicts;
- cost by meeting, stage and query, from `context_runs`, budget usage events and the comparison's queries;
- performance from the retrieval benchmark's JSON (`--perf`, written by `evals.context.bench`): every scope's p95
  at scale 1 under 2000 ms and no leaks; larger scales reported, not gated; without the file "not run".

Outputs `report.md` and `report.json` under `--out`. Nothing calls a model.
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from . import gates
from .compare import ARMS, arm_table, compare, paired_table, pooled, service_search
from .gates import Critical, Quality
from .runner import (
    FIXTURES,
    HISTORIES,
    ROOT,
    Deployment,
    db_run,
    load_history,
    offline,
    source_ids,
    throwaway_database,
)
from .worksheet import score as score_series

PLATFORM_ARMS = ("structured", "combined")  # the raw arm is the baseline, not the product


# ---------------------------------------------------------------- collection


def costs(deployment: Deployment, comparison: dict) -> dict:
    """Cost and tokens by stage and by meeting (`context_runs`), budget usage events by cost feature, and the
    comparison's per-query cost and latency."""
    from jarvis.context.models import ContextRun
    from jarvis.db import session_scope
    from jarvis.models import Usage
    from sqlalchemy import select

    owner = deployment.owner
    meeting = {v: k for k, v in source_ids(owner).items()}
    stage, by_meeting = defaultdict(lambda: defaultdict(float)), defaultdict(lambda: defaultdict(float))
    features = defaultdict(lambda: defaultdict(float))
    with session_scope() as db:
        for run in db.scalars(select(ContextRun).where(ContextRun.owner_id == owner)):
            for bucket in (stage[run.stage], *([by_meeting[meeting.get(run.source_id, run.source_id)]]
                                               if run.source_id else [])):
                bucket["runs"] += 1
                bucket["input_tokens"] += run.input_tokens or 0
                bucket["output_tokens"] += run.output_tokens or 0
                bucket["cost_usd"] += float(run.cost_usd or 0)
        for usage in db.scalars(select(Usage).where(Usage.owner_id == owner)):
            feature = (usage.tokens or {}).get("feature") or "unlabelled"
            features[feature]["events"] += 1
            features[feature]["cost_usd"] += float(usage.amount)

    def plain(groups):
        return {k: {m: round(v, 6) for m, v in sorted(g.items())} for k, g in sorted(groups.items())}
    return {"by_stage": plain(stage), "by_meeting": plain(by_meeting), "usage_by_feature": plain(features),
            "by_query": comparison["query_cost_usd"], "latency_s": comparison["latency_s"],
            "total_usd": round(sum(g["cost_usd"] for g in stage.values()), 6)}


def cross_owner_probe(history_dir: Path, deployment: Deployment) -> dict:
    """Every question asked by an owner with no context at all, through the platform arms: any result is a
    disclosure of another owner's data."""
    from .compare import load_questions, question_file

    stranger = Deployment(deployment.history, f"probe-{deployment.tag}", deployment.tag, "none", 0)
    asked, disclosed = 0, []
    for q in load_questions(question_file(history_dir), load_history(history_dir)):
        for arm in PLATFORM_ARMS:
            hits = service_search(stranger, q, include_raw=arm == "combined")
            asked += 1
            disclosed += [f"{q['id']}/{arm}: {h['ref']}:{h['id']}" for h in hits]
    return {"asked": asked, "disclosed": disclosed}


def collect(histories: list[Path], *, samples: int, seed: int, maintenance_runs: int) -> dict:
    """Run everything inside a throwaway database with every model call refused."""
    runs, comparisons, cost, probes = [], [], {}, {}
    with throwaway_database(), offline():
        for path in histories:
            report, deployment = db_run(path, "arrival", seed, maintenance_runs=maintenance_runs)
            comparison = compare(path, deployment, samples=samples, seed=seed)
            runs.append(report.to_dict())
            comparisons.append(comparison)
            cost[report.history] = costs(deployment, comparison)
            probes[report.history] = {"cross_owner": cross_owner_probe(path, deployment)}
    return {"runs": runs, "compare": comparisons, "pooled": pooled(comparisons, samples=samples, seed=seed),
            "costs": cost, "probes": probes}


# ---------------------------------------------------------------- grading


def _scores(run: dict, verified: bool) -> dict:
    return run["scores"]["verified" if verified else "all"]


def quality(data: dict, series: list[dict]) -> tuple[list[Quality], list[Quality]]:
    """(targets, reported metrics). Synthetic runs feed both sides (verified labels only on the verified side);
    adjudicated series feed the verified side."""
    targets = {
        "entity_link_precision": Quality("entity_link_precision", "Precision of automatically applied entity links",
                                         0.95),
        "preselected_precision": Quality("preselected_precision", "Precision of preselected proposals", 0.95,
                                         note="own-action proposals; graded once a series adjudicates them"),
        "commitment_recall": Quality("commitment_recall", "Recall of labelled actionable commitments", 0.90),
        "decision_rationale_recall": Quality("decision_rationale_recall",
                                             "Recall of labelled decisions and rationale", 0.90),
    }
    reported = {
        "attribution_errors": Quality("attribution_errors", "Attribution errors (share of matched commitments)",
                                      None, lower_is_better=True),
        "fabricated_actions": Quality("fabricated_actions",
                                      "Fabricated actions (hypothetical or negated statements made commitments or "
                                      "proposals, per labelled statement)", None, lower_is_better=True),
        "stale_answer_rate": Quality("stale_answer_rate", "Stale-answer rate (as-of questions and comparison "
                                     "questions with a stale answer, platform arms)", None, lower_is_better=True),
        "supersession_errors": Quality("supersession_errors", "Supersession errors (expected changes missed, over "
                                       "three orders)", None, lower_is_better=True),
        "human_overturn_rate": Quality("human_overturn_rate", "Human overturn rate (adjudicated items marked wrong, "
                                       "changed or must-not)", None, lower_is_better=True,
                                       note="from adjudicated series only"),
        "repeated_question_rate": Quality("repeated_question_rate", "Repeated-question rate (Questions created by "
                                          "unchanged maintenance runs, per open Question)", None, lower_is_better=True),
    }
    for run in data["runs"]:
        for verified, side in ((True, "verified"), (False, "unverified")):
            s = _scores(run, verified)
            links, props = s["entity_links"] or {}, s["proposals"] or {}
            getattr(targets["entity_link_precision"], side).add(links.get("correct"), links.get("auto_applied"))
            getattr(targets["preselected_precision"], side).add(
                props.get("correct"), (props.get("correct") or 0) + (props.get("wrong") or 0))
            kinds = s["per_kind"]
            c = kinds.get("commitment", {})
            getattr(targets["commitment_recall"], side).add(c.get("found"), c.get("labels"))
            for kind in ("decision", "rationale"):
                k = kinds.get(kind, {})
                getattr(targets["decision_rationale_recall"], side).add(k.get("found"), k.get("labels"))
            a = s["commitments"]["attribution"]
            getattr(reported["attribution_errors"], side).add(a.get("wrong", 0), sum(a.values()))
            f = s["fabricated_actions"]
            getattr(reported["fabricated_actions"], side).add(f["total"], f["labels"])
            q = s["as_of"]
            getattr(reported["stale_answer_rate"], side).add(q["stale"], q["questions"])
            changes = run["labels"]["verified_by_section"]["changes"] if verified else run["labels"]["changes"]
            errors = sum(v["verified" if verified else "all"]
                         for v in run["stability"]["supersession_errors_by_order"].values())
            getattr(reported["supersession_errors"], side).add(errors, changes * 3 if changes else 0)
        kept = run["stability"].get("maintenance") or {}
        open_questions = len((kept.get("settled") or {}).get("questions", []))
        reported["repeated_question_rate"].unverified.add((kept.get("created") or {}).get("questions", 0),
                                                          open_questions)
    for comparison in data["compare"]:
        for verified, side in ((True, "verified"), (False, "unverified")):
            arms = comparison["verified" if verified else "all"]["arms"]
            for arm in PLATFORM_ARMS:
                getattr(reported["stale_answer_rate"], side).add(arms[arm]["stale_answers"], arms[arm]["stale_questions"])
    for s in series:
        targets["entity_link_precision"].verified.add(s["links"]["correct"], s["links"]["correct"] + s["links"]["wrong"])
        targets["preselected_precision"].verified.add(s["preselected"]["correct"],
                                                      s["preselected"]["correct"] + s["preselected"]["wrong"])
        c = s["per_kind"].get("commitment", {})
        targets["commitment_recall"].verified.add(c.get("found"), c.get("labelled"))
        for kind in ("decision", "rationale"):
            k = s["per_kind"].get(kind, {})
            targets["decision_rationale_recall"].verified.add(k.get("found"), k.get("labelled"))
        reported["attribution_errors"].verified.add(s["attribution"]["errors"], s["attribution"]["judged"])
        extracted = sum(k.get("judged_extracted", 0) for k in s["per_kind"].values())
        reported["fabricated_actions"].verified.add(s["fabricated_actions"], extracted)
        reported["human_overturn_rate"].verified.add(s["overturn"]["overturned"], s["overturn"]["judged"])
    return list(targets.values()), list(reported.values())


def critical(data: dict) -> list[Critical]:
    temporal = Critical("temporal", "Temporal scenarios: as-of answers, supersession under three orders, stale "
                        "answers from the platform arms", "temporal")
    permission = Critical("permission", "Permission scenarios: deleted meetings' content, disclosure to another "
                          "owner", "permission")
    for run in data["runs"]:
        name = run["history"]
        for verified in (True, False):
            q = _scores(run, verified)["as_of"]
            stale = [f"{name}: as-of {r['id']} answered stale" for r in q["results"] if r["verdict"] == "stale"]
            if verified:
                temporal.failures += stale
                temporal.checked += q["questions"]
            else:
                temporal.warnings += [x for x in stale if x not in temporal.failures]
        for order, errors in run["stability"]["supersession_errors_by_order"].items():
            if errors["verified"]:
                temporal.failures.append(f"{name}: {errors['verified']} verified supersession errors ({order})")
            elif errors["all"]:
                temporal.warnings.append(f"{name}: {errors['all']} supersession errors on unverified labels ({order})")
        temporal.checked += run["labels"]["verified_by_section"]["changes"] * 3
        leaks = _scores(run, False)["deletion_leaks"]
        permission.failures += [f"{name}: still supported only by a deleted meeting: {x}" for x in leaks]
        permission.checked += 1
    for comparison in data["compare"]:
        name = comparison["history"]
        for arm in PLATFORM_ARMS:
            for row in comparison["rows"][arm]:
                if row["stale"] == "stale":
                    (temporal.failures if row["verified"] else temporal.warnings).append(
                        f"{name}: {row['id']} ({arm}) answered with a superseded value")
                temporal.checked += row["verified"] and row["stale"] is not None
                if row["leak"]:
                    permission.failures.append(f"{name}: {row['id']} ({arm}) returned a deleted meeting's content")
                permission.checked += row["forbidden"]
    for name, probe in data["probes"].items():
        permission.failures += [f"{name}: disclosed to another owner: {x}" for x in probe["cross_owner"]["disclosed"]]
        permission.checked += probe["cross_owner"]["asked"]
    return [temporal, permission]


def probes(data: dict) -> dict[str, list[dict]]:
    out = defaultdict(list)
    for run in data["runs"]:
        name, kept, p = run["history"], run["stability"].get("maintenance") or {}, run["processing"]
        settled = kept.get("settled") or {}
        out["unapproved_task_writes"].append({"name": f"{name}: tasks written by ingestion and maintenance",
                                              "ok": settled.get("tasks", 0) == 0 and not kept["created"]["tasks"],
                                              "value": settled.get("tasks", 0)})
        out["duplicate_task_writes"].append({"name": f"{name}: redelivery and {kept.get('runs')} unchanged maintenance "
                                                     "runs create nothing", "ok": bool(kept.get("no_change")),
                                             "value": kept.get("created")})
        out["processing_accounting"].append({"name": f"{name}: coverage of live segments and ready sources",
                                             "ok": p["covered_segments"] == p["segments"]
                                             and set(p["sources"]) <= {"ready", "deleted"},
                                             "value": f"{p['covered_segments']}/{p['segments']} {p['sources']}"})
    for name, probe in data["probes"].items():
        out["cross_owner_disclosure"].append({"name": f"{name}: {probe['cross_owner']['asked']} questions from an "
                                                      "owner without context", "ok": not probe["cross_owner"]["disclosed"],
                                              "value": len(probe["cross_owner"]["disclosed"])})
    for comparison in data["compare"]:
        for arm in PLATFORM_ARMS:
            a = comparison["all"]["arms"][arm]
            out["surviving_provenance"].append({"name": f"{comparison['history']}: {arm} citations resolve to intact "
                                                        "spans of live sources", "ok": a["valid_citations"] == a["citations"],
                                                "value": f"{a['valid_citations']}/{a['citations']}"})
    return out


def performance(data: dict, perf: list[dict] | None) -> dict:
    """The benchmark gate, plus the synthetic histories' p95 for information (a tiny corpus: never gated)."""
    p95 = max((c["latency_s"][arm]["p95"] or 0) for c in data["compare"] for arm in PLATFORM_ARMS) \
        if data["compare"] else None
    return gates.grade_performance(perf) | {"synthetic_p95_s": p95}


def grade(data: dict, outcomes: dict | None, series: list[dict], perf: list[dict] | None = None) -> dict:
    found = probes(data)
    mechanical = [gates.grade_mechanical(g, outcomes, found.get(g.key, [])) for g in gates.MECHANICAL]
    mechanical.append(performance(data, perf))
    targets, reported = quality(data, series)
    graded_targets = [gates.grade_quality(q) for q in targets]
    graded_reported = [gates.grade_quality(q) for q in reported]
    graded_critical = [gates.grade_critical(c) for c in critical(data)]
    labels = sum(r["labels"]["total"] - r["labels"]["verified"] for r in data["runs"])
    questions = data["pooled"]["questions"]["unverified_excluded"]
    return {"verdict": gates.verdict(mechanical, graded_targets, graded_critical), "mechanical": mechanical,
            "critical": graded_critical, "targets": graded_targets, "reported": graded_reported,
            "unverified_excluded": {"labels": labels, "questions": questions, "total": labels + questions},
            "verified_included": {"labels": sum(r["labels"]["verified"] for r in data["runs"]),
                                  "questions": data["pooled"]["questions"]["verified"], "series": len(series)}}


# ---------------------------------------------------------------- markdown


def _rate(x: dict) -> str:
    if not x["n"]:
        return "n/a (n=0)"
    interval = f" [{x['interval'][0]:.3f}, {x['interval'][1]:.3f}]" if x.get("interval") else ""
    return f"{x['rate']:.3f}{interval} ({x['successes']}/{x['n']})"


def _status(status: str) -> str:
    return status.upper()


def performance_section(perf: dict) -> list[str]:
    lines = ["## Performance", "", f"{perf['title']}: **{_status(perf['status'])}**"
             + (f" ({perf['pending']})" if perf["pending"] else "") + f". Gated at scale "
             f"{gates.PERF_GATED_SCALE:g} only: every scope's p95 under {gates.PERF_P95_MS:.0f} ms and no leaks; a "
             "leak at any scale fails.", ""]
    for run in perf["scales"]:
        corpus = ", ".join(f"{k} {v}" for k, v in run["corpus"].items()) or "not given"
        lines += [f"### Scale {run['scale']:g} ({'gated' if run['gated'] else 'reported, not gated'})", "",
                  f"Corpus: {corpus}. Leaks: {run['leaks']}.", "", ("| Scope | p50 (ms) | p95 (ms) | Queries | Under "
                  f"{gates.PERF_P95_MS:.0f} ms |"), "|---|---|---|---|---|"]
        lines += [f"| {name} | {s.get('p50_ms', 'n/a')} | {s['p95_ms']} | {s.get('queries', 'n/a')} | "
                  f"{'yes' if s['p95_ms'] < gates.PERF_P95_MS else 'no'} |" for name, s in sorted(run["scopes"].items())]
        lines.append("")
    if perf["synthetic_p95_s"] is not None:
        lines += [(f"Synthetic histories (tiny corpus, informational): worst platform-arm p95 "
                  f"{perf['synthetic_p95_s']} s."), ""]
    return lines


def markdown(data: dict, graded: dict, *, junit: Path | None, series: list[dict]) -> str:
    excluded = graded["unverified_excluded"]
    included = graded["verified_included"]
    lines = [
        "# Context platform release report (plan section 11)", "",
        f"Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}. Verdict: **{graded['verdict']}**.", "",
        (f"**Unverified labels excluded: {excluded['total']}** ({excluded['labels']} labels, {excluded['questions']} "
        f"questions). Verified included: {included['labels']} labels, {included['questions']} questions, "
        f"{included['series']} adjudicated series."), "",
        f"Synthetic histories: {', '.join(r['history'] for r in data['runs'])}, delivered through the real pipeline "
        "(doubles, hashed bag-of-words embeddings, no model calls) in a throwaway database. Test results: "
        + (f"`{junit}`." if junit else "none supplied (`--junit`), so suite-level gates are NOT RUN."), "",
        ("Rule: critical temporal and permission failures fail the release whatever the averages; a target with no "
        "verified sample is NOT GRADEABLE, never a pass."), "",
        "## Mechanical gates", "", "| Gate | Status | Tests passed | Evaluation probes | Not yet covered |",
        "|---|---|---|---|---|"]
    for g in graded["mechanical"]:
        passed = sum(1 for v in g["tests"].values() if v == "passed")
        held = sum(1 for p in g["probes"] if p["ok"])
        lines.append(f"| {g['title']} | {_status(g['status'])} | {passed}/{len(g['tests'])} | "
                     f"{held}/{len(g['probes'])} held | {g['pending'] or ''} |")
    lines += ["", "## Critical scenarios (averages cannot override)", "",
              "| Scenario | Status | Checked (verified or mechanical) | Failures | Warnings on unverified labels |",
              "|---|---|---|---|---|"]
    for c in graded["critical"]:
        lines.append(f"| {c['title']} | {_status(c['status'])} | {c['checked']} | {len(c['failures'])} | "
                     f"{len(c['warnings'])} |")
    for c in graded["critical"]:
        if c["failures"] or c["warnings"]:
            lines += ["", f"{c['title']}:"] + [f"- FAILURE {x}" for x in c["failures"]] + \
                [f"- warning (unverified, adjudicate first) {x}" for x in c["warnings"]]
    lines += ["", "## Quality targets (verified labels only)", "",
              "| Metric | Target | Status | Verified estimate [95% CI] (successes/n) | Unverified, informational | Note |",
              "|---|---|---|---|---|---|"]
    for q in graded["targets"]:
        lines.append(f"| {q['title']} | >= {q['target']:.2f} | {_status(q['status'])} | {_rate(q['verified'])} | "
                     f"{_rate(q['unverified'])} | {q['note']} |")
    lines += ["", "## Reported metrics", "", "| Metric | Verified [95% CI] | Unverified, informational | Note |",
              "|---|---|---|---|"]
    for q in graded["reported"]:
        lines.append(f"| {q['title']} | {_rate(q['verified'])} | {_rate(q['unverified'])} | {q['note']} |")
    minutes = [s["correction_minutes"] for s in series if s["correction_minutes"]["meetings"]]
    lines += ["", "Correction minutes per meeting: " + (", ".join(
        f"{s['series']} {s['correction_minutes']['per_meeting']}" for s in series if s["correction_minutes"]["meetings"])
        if minutes else "n/a (recorded in adjudicated worksheets)") + ". Pending-review age: n/a until review and "
        "publication (D1/D2) record dispositions.", ""]
    pooled_data, q = data["pooled"], data["pooled"]["questions"]
    lines += ["## Shadow comparison: raw history vs native structured vs combined", "",
              (f"{q['total']} questions, {q['verified']} verified, unverified excluded from release scoring: "
              f"{q['unverified_excluded']}."), ""]
    if q["verified"]:
        lines += ["Verified questions:", ""] + arm_table(pooled_data["verified"]) + [""] + \
            paired_table(pooled_data["verified"]) + [""]
    else:
        lines += ["Release comparison: NOT GRADEABLE (no verified questions). Informational, all questions:", ""]
    lines += arm_table(pooled_data["all"]) + [""] + paired_table(pooled_data["all"]) + ["",
              ("On these tiny histories one chunk usually holds a whole meeting, so the raw arm's recall@10 is close "
              "to trivially high; the doubles extract only decisions and commitments, so the structured arm cannot "
              "answer fact, risk or rationale questions. Neither says anything about the native engine."), ""]
    lines += performance_section(graded["mechanical"][-1])
    lines += ["## Cost by meeting, stage and query", ""]
    for name, cost in data["costs"].items():
        lines += [f"### {name} (total {cost['total_usd']:.6f} USD)", "", ("| Stage | Runs | Input tokens | Output tokens "
                  "| Cost (USD) |"), "|---|---|---|---|---|"]
        lines += [f"| {k} | {int(v.get('runs', 0))} | {int(v.get('input_tokens', 0))} | {int(v.get('output_tokens', 0))} "
                  f"| {v.get('cost_usd', 0):.6f} |" for k, v in cost["by_stage"].items()]
        lines += ["", "| Meeting | Runs | Input tokens | Output tokens | Cost (USD) |", "|---|---|---|---|---|"]
        lines += [f"| {k} | {int(v.get('runs', 0))} | {int(v.get('input_tokens', 0))} | {int(v.get('output_tokens', 0))} "
                  f"| {v.get('cost_usd', 0):.6f} |" for k, v in cost["by_meeting"].items()]
        lines += ["", "Per query (USD, total / per query): " + ", ".join(
            f"{arm} {cost['by_query'][arm]['total']:.6f} / {cost['by_query'][arm]['per_query'] or 0:.6f}" for arm in ARMS)
                  + ". Budget usage events by feature: " + (", ".join(
                      f"{k} {int(v['events'])} events {v['cost_usd']:.6f}" for k, v in cost["usage_by_feature"].items())
                      or "none") + ".", ""]
    lines += ["## Tests covering the mechanical gates", ""]
    for g in graded["mechanical"]:
        if g["tests"]:
            lines += [f"- {g['title']}:"] + [f"  - `{node}`: {state}" for node, state in g["tests"].items()]
    lines += ["", "## Per-history runs", ""]
    for run in data["runs"]:
        lines += [(f"- {run['history']}: labels {run['labels']['total']} ({run['labels']['verified']} verified); "
                  f"owner `{run['processing']['owner']}`; maintenance created "
                  f"{run['stability']['maintenance']['created']}.")]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.context.report", description=__doc__.split("\n\n")[0])
    parser.add_argument("histories", nargs="*", type=Path)
    parser.add_argument("--junit", type=Path, help="pytest --junitxml output for the suite-level gates")
    parser.add_argument("--series", type=Path, action="append", default=[],
                        help="an adjudicated series directory (worksheet import); repeatable")
    parser.add_argument("--perf", type=Path, help="retrieval benchmark JSON from evals.context.bench")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--maintenance-runs", type=int, default=10)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "context-eval")
    args = parser.parse_args(argv)
    histories = args.histories or [FIXTURES / name for name in HISTORIES]
    try:
        perf = gates.load_perf(args.perf) if args.perf else None
    except (OSError, ValueError) as exc:
        print(f"--perf: {exc}", file=sys.stderr)
        return 2
    data = collect(histories, samples=args.samples, seed=args.seed, maintenance_runs=args.maintenance_runs)
    outcomes = gates.junit_outcomes(args.junit) if args.junit else None
    series = [score_series(path) for path in args.series]
    graded = grade(data, outcomes, series, perf)
    text = markdown(data, graded, junit=args.junit, series=series)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.md").write_text(text)
    (args.out / "report.json").write_text(json.dumps({"graded": graded, "series": series, "data": data}, indent=1,
                                                     sort_keys=True, default=str) + "\n")
    print(text)
    print(f"wrote {args.out / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
