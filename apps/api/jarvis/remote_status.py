"""Factual remote-job status; suggested polling is not a scheduled follow-up."""


def retry_metadata(status, result=None):
    active = status in {"queued", "running", "retrying"}
    attempts = (result or {}).get("attempts", 0)
    attempts = attempts if isinstance(attempts, int) else 0
    return {
        "retry_active": active,
        "poll_after_seconds": min(60, max(5, 2 ** min(6, max(0, attempts)))) if active else None,
        "completion_notification_scheduled": False,
        "assistant_followup_scheduled": False,
        "polling_guidance": (
            "The server owns the queued work/retries. This delay is a polling recommendation, "
            "not an exact retry time. Report pending once; do not repeatedly poll immediately. "
            "No assistant follow-up or completion notification has been scheduled."
            if active
            else "No automatic retry is active. Report the recorded result without promising monitoring."
        ),
    }
