# Linear → Eridani cancel sync review (2026-11-02)

The question: why did a Linear issue that was Canceled in Linear stay open in Eridani? This review checks every Linear issue currently synced to Eridani.

## How the review was done

- **Read-only.** Nothing was changed in Linear or Eridani by hand.
- **Eridani:** all 251 tasks were read through the Eridani records tools, 219 unarchived and 32 archived. 46 of them are linked to Linear, and all 46 are unarchived.
- **Linear:** every issue assigned to the owner was read, including archived ones, plus the three linked issues that are no longer assigned (read by id).
- **Comparison:** each link's Linear state type was checked against the task's Eridani status, using the sync mapping: started → in_progress, completed → completed, canceled/duplicate → cancelled, backlog → backlog, anything else → open. The last applied snapshot (`external.state`, `remote_updated_at`) was also checked against Linear's current `updatedAt`.

## Result

| Team | Linked tasks | Sync state |
|---|---|---|
| Team A (ABCD) | 40 | all `synced`, all match Linear exactly |
| Team B (LIV) | 6 | LIV-43, LIV-44, LIV-51 `synced` and matching; LIV-32, LIV-37, LIV-70 `out_of_scope` and **mismatched** |

**The spelling is not the problem.** Linear's built-in state is named "Canceled", type `canceled`, and it maps to Eridani `cancelled` correctly. ABCD-37 synced as cancelled. Before this change it was not archived, because a cancel did not archive the task back then.

### Mismatches

| Issue | Linear now | Eridani status | Last snapshot | Link |
|---|---|---|---|---|
| LIV-32 | Canceled (updated 2026-09-29) | open | Todo, 2026-09-18 | out_of_scope |
| LIV-37 | Canceled (updated 2026-09-29) | in_progress | In Progress, 2026-09-23 | out_of_scope |
| LIV-70 | Canceled (updated 2026-09-30) | backlog | Backlog, 2026-09-23 | out_of_scope |

### Cause

The connection syncs only issues assigned to the owner ("only mine"). All three issues were cancelled **and unassigned at the same moment** (2026-09-29 13:52 UTC).

On the next sync, each issue no longer matched "only mine". `process_sync` therefore marked the link `out_of_scope` and skipped `apply_remote`, so the Canceled state was never applied to the task. From then on, nothing ever updated those tasks again.

Other causes were ruled out:
- **Held changes:** no link was pending, in conflict or unconfirmed.
- **Timestamp guard:** the `older()` guard did not reject any snapshot.
- **Mapping:** the state mapping is correct.

### The issue the operator remembered

The operator's Eridani issue log has the bot's account of the incident. The task was "In Progress" locally, Canceled in Linear, and "out of scope for sync". That fits **LIV-37**.

The bot then tried to cancel the task in Eridani and hit two more bugs:

1. **A local cancel was refused as a conflict.** `require_editable` treated `out_of_scope` like a conflict and returned "Review the Linear difference before making another change", even though there was nothing to review.
2. **The error blamed Google.** The bot then called `linear.resolve` without a valid review token. The token is decrypted with the shared `google_auth.unseal`, whose error always says "Reconnect Google in Settings." The Linear connection was never broken. The same misleading message would also appear if the stored Linear API key couldn't be decrypted.

## Fix (this branch)

1. **Out-of-scope issues still finish.** When an issue leaves scope and its last state is Canceled, Duplicate or Done, that state is applied before the link is marked `out_of_scope`. This does not happen while the link holds a conflict or an unconfirmed write.
2. **Cancel archives.** A Linear Canceled or Duplicate issue makes the task `cancelled` **and archived**. Reopening the issue in Linear brings the task back, unless the user archived it in Eridani. Archiving completed tasks is left for later.
3. **Repair on the next full sync, no script needed.** A full sync, at most once a day, re-reads every known link by id, including out-of-scope ones. LIV-32, LIV-37 and LIV-70 become cancelled and archived on the first full sync after deploy, and ABCD-37 gets archived at the same time.
4. **A cancel stops runs.** A cancel from Linear or from the user stops an active agent run, the same way a completion does.
5. **Out-of-scope tasks can be finished here.** Cancelling or completing an out-of-scope task in Eridani now applies locally and doesn't write to Linear. Other edits to shared fields on such a task now say the issue is outside the sync scope, and point to Unlink.
6. **Linear errors name Linear.** A missing or invalid `linear.resolve` token now returns `LINEAR_STALE` ("Refresh the Linear comparison."). A Linear key that can't be decrypted now returns `LINEAR_RECONNECT` ("Reconnect Linear in Settings.").
7. **Held cancels are shown.** A cancel or completion that arrives while the link holds a change is still not applied. It now appears on the task's Linear badge and detail as "Cancelled in Linear: review to apply".

## Expected state after deploy and the next full sync

| Issue | After |
|---|---|
| LIV-32, LIV-37, LIV-70 | cancelled, archived, link stays `out_of_scope` |
| ABCD-37 | cancelled, archived |
| The other 42 links | unchanged (already matching) |

No mismatch is left for the operator to fix by hand. If a task still differs after that sync, its badge will show a held change ("review to apply") or a sync state other than `synced`/`out_of_scope`.
