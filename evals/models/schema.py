"""Case-file schema for the backend model eval (scripts/model_eval.py, docs/COST_TRACKING.md "Model battle royale").

One JSON file per job type in evals/models/cases/. Each case is fixed input, what a correct answer must show
(`expect`), an optional rubric for the grader model, and `stub`: a correct answer in the production response
shape, which the deterministic fake provider replays (`stub_wrong`, when present, is a plausible wrong answer it
replays for its simulated misses). `validate_file` checks the file against CaseFile, each case's `expect` against
its job's model and each stub against the production response model, so a case can never drift from what the job
really returns. schema.json is CaseFile's JSON schema (tests/test_model_eval.py keeps it current).
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CASES = HERE / "cases"
SCHEMA = HERE / "schema.json"
JOBS = ("session_summary", "routing_review", "eri_chat", "run_judge", "run_summary")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Import(Strict):
    """Cases reused from another eval file (the run judge reuses evals/agent_control)."""
    path: str
    exclude_kinds: list[str] = Field(default_factory=list)


class Case(Strict):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    tags: list[str] = Field(default_factory=list)
    input: dict
    expect: dict
    rubric: str | None = None
    stub: dict
    stub_wrong: dict | None = None


class CaseFile(Strict):
    version: Literal[1]
    job: Literal[JOBS]
    description: str
    prompt_source: str  # Where the production prompt and request come from.
    grader: Literal["text", "structured", "tool_call", "verdict"]
    rubric: str | None = None  # Job-level rubric; a case's own rubric is appended.
    config: dict = Field(default_factory=dict)
    imports: list[Import] = Field(default_factory=list)
    cases: list[Case] = Field(default_factory=list)


# ---------------------------------------------------------------- per-job expectations


class TextExpect(Strict):
    max_chars: int = Field(gt=0)
    must_mention: list[list[str]] = Field(default_factory=list)  # Each group: at least one term, any case.
    must_not_mention: list[str] = Field(default_factory=list)


class RoutingTarget(Strict):
    type_id: str
    parent_id: str | None
    phrase_any: list[str] = Field(min_length=1)  # The candidate phrase contains one of these (word match).


class RoutingExpect(Strict):
    must_find: list[RoutingTarget] = Field(default_factory=list)
    max_invalid: int = 0  # Candidates with an invented home, no support, or contradicting evidence.
    forbid_phrases: list[str] = Field(default_factory=list)  # Generic phrases a good review never proposes.


class ToolExpect(Strict):
    tools_any: list[str]  # The first tool call is one of these; empty means answer without a tool.
    args_contains: dict[str, str] = Field(default_factory=dict)  # Argument value contains text, any case.
    forbid_tools: list[str] = Field(default_factory=list)


class VerdictExpect(Strict):
    expect: Literal["met", "not_met", "needs_you", "not_met_or_needs_you"]


class ToolAnswer(Strict):
    """An Eri chat stub: the first tool call, or a text answer with no tool."""
    tool: str | None = None
    arguments: dict = Field(default_factory=dict)
    text: str | None = None


class TextAnswer(Strict):
    summary: str


EXPECT = {"session_summary": TextExpect, "run_summary": TextExpect, "routing_review": RoutingExpect,
          "eri_chat": ToolExpect, "run_judge": VerdictExpect}


def answer_model(job):
    """The production response model a stub must validate against."""
    if job == "session_summary":
        from jarvis.agent_sessions import Summary
        return Summary
    if job == "routing_review":
        from jarvis.routing import Candidates
        return Candidates
    if job == "run_judge":
        from jarvis.agent_judge import Verdict
        return Verdict
    if job == "run_summary":
        return TextAnswer
    return ToolAnswer


def imported(entry):
    data = json.loads((ROOT / entry.path).read_text())
    out = []
    for case in data["cases"]:
        if case.get("kind") in entry.exclude_kinds:
            continue
        good, wrong = judge_answers(case["stub"], case["expect"])
        out.append({"id": case["id"], "tags": [case.get("category", "")], "input": {"evidence": case["evidence"]},
                    "expect": {"expect": case["expect"]}, "rubric": None, "stub": good, "stub_wrong": wrong})
    return out


def judge_answers(stub, expect):
    """(correct, wrong) model answers for an agent_control case. Its own stub is a plausible answer that is
    sometimes deliberately fooled (to show the hard rules catching it): a fooled stub becomes the wrong answer and
    the correct one says the labelled verdict; a correct stub keeps its place and the wrong answer flips it."""
    target = "not_met" if expect == "not_met_or_needs_you" else expect
    right = stub["verdict"] in ("not_met", "needs_you") if expect == "not_met_or_needs_you" \
        else stub["verdict"] == expect
    status = {"met": "met", "not_met": "not_met", "needs_you": "unclear"}

    def saying(verdict):
        return {**stub, "verdict": verdict, "gap": "" if verdict == "met" else "Finish the criteria still open.",
                "criteria": [{**c, "status": status[verdict]} for c in stub.get("criteria", [])]}

    if not right:
        return saying(target), stub
    return stub, saying("not_met" if stub["verdict"] == "met" else "met")


def validate_file(path):
    """The parsed file with imported cases resolved; raises on any schema, expect or stub mismatch."""
    data = CaseFile.model_validate_json(Path(path).read_text())
    cases = [c.model_dump() for c in data.cases] + [c for entry in data.imports for c in imported(entry)]
    ids = [c["id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate case ids")
    if not cases:
        raise ValueError(f"{path}: no cases")
    answer = answer_model(data.job)
    for case in cases:
        try:
            EXPECT[data.job].model_validate(case["expect"])
            answer.model_validate(case["stub"])
            if case.get("stub_wrong") is not None:
                answer.model_validate(case["stub_wrong"])
        except Exception as exc:
            raise ValueError(f"{path}: case {case['id']}: {exc}") from exc
    return data, cases


def load(jobs=JOBS):
    """{job: (CaseFile, [case dicts])} for every case file of the selected jobs."""
    out = {}
    for path in sorted(CASES.glob("*.json")):
        data, cases = validate_file(path)
        if data.job in jobs:
            out[data.job] = (data, cases)
    return out


def schema_text():
    return json.dumps(CaseFile.model_json_schema(), indent=2) + "\n"


if __name__ == "__main__":
    SCHEMA.write_text(schema_text())
    print(f"wrote {SCHEMA.relative_to(ROOT)}")
