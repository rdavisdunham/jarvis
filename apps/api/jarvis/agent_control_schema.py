"""Voice commands for agent runs (`control.*`): the same command service the remotea page calls.

Only present while remotea is enabled (tool_policy.remotea_tools); external bots never get them
(no bot scope maps to `control.*`). Each command resolves its run from run_id, task_id or a task
name (navigation_find's matcher) and calls agent_control: there is no second control path.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Target(Input):
    """Which run: its id, its task's id, or the task's name / Linear identifier (exactly one, or none
    when exactly one run is active)."""

    run_id: str | None = Field(default=None, max_length=36)
    task_id: str | None = Field(default=None, max_length=36)
    task: str | None = Field(default=None, min_length=1, max_length=300)

    @model_validator(mode="after")
    def one_target(self):
        if sum(bool(v) for v in (self.run_id, self.task_id, self.task)) > 1:
            raise ValueError("Give one of run_id, task_id or task.")
        return self


class ControlStart(Input):
    task_id: str | None = Field(default=None, max_length=36)
    task: str | None = Field(default=None, min_length=1, max_length=300)
    harness: Literal["claude", "codex"] = "claude"

    @model_validator(mode="after")
    def one_task(self):
        if bool(self.task_id) == bool(self.task):
            raise ValueError("Give task_id or task (its name or Linear identifier).")
        return self


class ControlMessage(Target):
    text: str = Field(min_length=1, max_length=4000)
    mode: Literal["queue", "steer"] = "queue"


class ControlStop(Target):
    mode: Literal["interrupt", "kill"] = "interrupt"


class SpokenAnswer(Input):
    item_id: str = Field(min_length=1, max_length=64)
    choice_ids: list[str] = Field(default_factory=list, max_length=32)
    text: str = Field(default="", max_length=4000)


class ControlAnswer(Input):
    """Answer an agent's question (question_id + answers) or its planning interview (interview_id + responses)."""

    question_id: str | None = Field(default=None, max_length=36)
    answers: list[SpokenAnswer] = Field(default_factory=list, max_length=16)
    interview_id: str | None = Field(default=None, max_length=36)
    responses: dict[str, str | list[str]] = Field(default_factory=dict, max_length=12)

    @model_validator(mode="after")
    def one_kind(self):
        if bool(self.question_id) == bool(self.interview_id):
            raise ValueError("Give question_id (with answers) or interview_id (with responses).")
        if self.question_id and not self.answers:
            raise ValueError("Answer at least one item.")
        if self.interview_id and not self.responses:
            raise ValueError("Answer the interview's questions.")
        return self


class ControlConfirm(Input):
    """Confirm or decline a pending control_start / control_message (stage F, two-phase control)."""

    # Omitted: the newest pending action of this account (the one the owner is answering).
    pending_id: str | None = Field(default=None, min_length=1, max_length=36)
    decision: Literal["confirm", "decline"] = "confirm"


COMMANDS = {
    "control.start": ControlStart,
    "control.message": ControlMessage,
    "control.confirm": ControlConfirm,
    "control.stop": ControlStop,
    "control.answer": ControlAnswer,
}
