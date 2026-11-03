# Durable Eridani–Remotea runs

Eridani saves the task's questions, plans, responses, and command receipts independently of an open browser or a live terminal prompt. A run can wait for its owner without a response deadline.

## Using a run

The header names the workflow stage: starting, interviewing, planning, reviewing the plan, working, checking results, or ready for review. Whether the host is connected, the run is paused, or the agent needs an answer is shown separately.

The activity line above the message input describes current model or tool activity. Thinking and deliberating use a subtle animation; known tool activity uses a specific label. Waiting for a decision and losing the connection are different from active processing. Reduced-motion settings disable decorative animation.

One decision card presents interviews, plans, execution permissions, proposed starts, and final acceptance. Its buttons reflect the decision: answer questions, approve a plan, request changes, allow an action once, or deny it. Complete redacted permission details are expandable. Typing **Approved** acts on the one unambiguous approval in the selected run; it does not choose interview answers or create a permanent permission grant.

The latest plan remains available after follow-up messages, with prior revisions retained. Plan approval authorizes its agreed scope; changed scope, criteria, target, or permissions requires a new decision. Sending a message does not approve a plan.

**Pause** preserves the conversation, workspace, and decisions. **Resume** continues after reconciling any interrupted work. **Cancel** ends execution without discarding the work or its history. A pause remains a pause after reconnecting.

## Saving and delivery

The browser preserves unsent drafts and action identities. When Eridani can be reached, drafts and role preferences synchronize across devices. When the host is offline, Eridani keeps messages and answers for delivery to the same valid request after reconnection.

An API response confirms that Eridani saved an action. A separate receipt records host delivery. The interface distinguishes queued, saved, delivered, and uncertain outcomes. A lost acknowledgement is reconciled before retrying; an uncertain action is never blindly repeated. Reusing an action ID with different content is rejected.

Submitting a plan saves a version and returns a receipt. Managed submissions use a stable `submission_id` UUID; `task_plan(submission_id=...)` retrieves its receipt if the first response is lost. Plan visibility does not depend on the agent's final chat message.

## Managed harnesses

New managed runs use structured Codex or Claude controls. Remotea displays their transcript, while Eridani controls messages and decisions. **Take over in terminal** explicitly stops managed control before continuing the same conversation in its native terminal. A session has one controller at a time.

Planner and reviewer profiles restrict repository changes and permit the necessary task coordination tools. Worker profiles use supported automatic review within the approved workspace and scope. Publishing interview questions, submitting a plan, and reporting progress do not require redundant permission to communicate with Eridani. Additional execution permissions remain separate decisions.

Model and effort choices are specific to planner, reviewer, and worker. Options come from the connected host's installed harness capabilities; GPT-6.1 Sol appears when available there. **Model default** uses that harness's default. An unsupported remembered choice is shown for correction instead of silently replaced.

New runs have no default duration or turn cap. Explicit limits remain visible when configured. Network requests and individual tools still have bounded deadlines; repeated failures or uncertain recovery pause the run with an explanation. The default delegated coding endpoint is an open pull request; merging is a separate action.

## Recovery and compatibility

Saved decisions outlive human delays and browser reconnects. A provider process restart can invalidate its original callback, so Remotea restores the conversation and binds the saved decision only to a verified recreation of the same pending action. Changed actions require a new decision. Submitted actions with uncertain effects require reconciliation rather than replay.

Pending runs, their decisions, and unresolved command identities are protected from routine retention cleanup. This does not recover manually deleted provider history or revoked access; those conditions are surfaced without pretending the agent is still working.

The API, host, and desktop negotiate managed-session capabilities. The initial verified Linux transports are Codex 0.166.x and Claude Code 2.1.312–2.1.317. Other installed versions require adapter verification before managed execution; their existing native sessions remain available. Legacy terminal sessions are labeled explicitly and retain their provider's native approval limits. No Windows or macOS local Eridani control is enabled by this work.

Terminal takeover is exclusive and persistent: Eridani records that the terminal has control and disables automatic continuation for that conversation. Opening the same run on another device does not take control back.

## Incident that motivated this change

For “Thank you in voice mode ends the chat prematurely,” the first native permission authorized `task_interview`; the later one authorized `submit_plan`, rather than execution of the proposed plan. The unanswered submission permission prevented a saved plan revision. A retry's `task_plan` request completed promptly, but its surrounding execution stalled. The new profiles remove redundant coordination approvals, and separate request, persistence, and delivery receipts identify which boundary has stalled.

## Verification

Regression coverage must include delayed responses, offline submission, reloads, two-device responses, duplicate receipts, superseded plans, pause/resume, provider restarts, uncertain effects, and model/effort preservation. Harness transport checks use isolated fixtures without paid model calls. Cross-repository protocol and browser acceptance checks run against synthetic databases and temporary host configurations; they must never use production credentials or restart live sessions.
