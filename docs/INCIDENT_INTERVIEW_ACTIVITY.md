# Interview planning activity investigation

Run `634e8657-2ddf-40f8-9450-c9a76affc185`; planner conversation `01a19fab-2422-7203-b5f6-bd083d157db1`.

Read-only inspection used the local Codex transcript and Remotea's persisted command/outbox metadata. No live message, approval, run mutation, restart or configuration change was performed. Times below are UTC on 2026-11-03.

| Time | Evidence |
| --- | --- |
| 04:32:03 | The 12-question `task_interview` completed and returned an open interview receipt. |
| 05:03:49 | The interview-answer input was accepted and the provider immediately began planning turn 2. |
| 05:03:56–05:07:45 | Tool results show code/documentation research and creation of a stable submission UUID. |
| 05:08:33 | Last host activity report before a long output gap; a Codex reasoning item was incorrectly reported as tool activity. |
| 05:12:58 | Remotea accepted the user's check-in and queued it behind the still-active planning turn. |
| 05:13:05 | The model finished composing a 30,419-character PRD draft. It then shortened it to fit the 28,000-character limit. |
| 05:14:30–05:14:34 | `submit_plan` completed successfully in about four seconds: revision 1, nine steps, thirteen criteria and `full` verification. |
| 05:14:37 | Planning turn 2 ended. About 0.2 seconds later, the queued check-in began turn 3. |
| 05:14:40 | The agent answered that planning was complete and it was waiting for approval. |

The check-in did not unblock plan submission: persistence preceded the check-in's provider turn. No native follow-up question, automatic native answer or “operator already answered” exchange was found in the inspected transcript or host question events. The planner's managed decision journal contained no native decisions. The normal interview-answer prompt did repeat the saved interview responses as input data.

The long interval was genuine planning/draft composition, not a demonstrated hung MCP call. The roughly five-minute activity gap exceeded the UI's 90-second freshness threshold, which is sufficient to explain “Activity unconfirmed”; no historical screenshot was available to verify the exact render.

## Local fixes and verification

- Report Codex reasoning as thinking and response composition as writing a response, rather than claiming a tool is running. Native questions explicitly report that they are waiting for an answer or approval.
- Keep the actual activity timestamp. Stale output now says “No recent output · last reported …”; offline, explicit pause and human-decision states retain priority. No heartbeat or synthetic progress event was added.
- A separate regression reproduced a real queue race: if turn completion arrived before `send()` returned its receipt, the completion's pump attempt was guarded and the next input could remain queued. Retry the pump after releasing that guard and avoid reporting thinking after an already-completed turn. This race was reproduced in a controlled test, not observed in this run.
- A native-follow-up regression confirms that a new question after an interview response is surfaced as a new decision and receives no automatic answer copied from the interview.

Changes remain local and un-deployed. Existing paused/uncertain-delivery guards and stable operation IDs remain in place.
