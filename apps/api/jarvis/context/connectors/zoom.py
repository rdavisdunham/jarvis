"""Zoom cloud-recording transcripts: a documented stub, not implemented in the pilot.

Today Zoom transcripts arrive as exports through the generic `api` path (`scripts/bridge_submit.py` or any
client calling `PUT /api/v1/external/sources/zoom/{meeting_uuid}`). A future in-app adapter would:

- Authenticate with a Server-to-Server OAuth app (scope `cloud_recording:read:list_user_recordings`), credentials
  sealed in `ContextConnection.credential_ciphertext` like Granola's key.
- Poll `GET /v2/users/me/recordings?from=YYYY-MM-DD&to=YYYY-MM-DD` (30-day windows, `next_page_token`), keeping a
  date watermark in `context_cursors` and advancing it only after a full successful scan.
- For each meeting, download the `recording_files` entry with `file_type == "TRANSCRIPT"` (WebVTT) using its
  `download_url` and the bearer token.
- Map to `SourceEnvelope`: `connector="zoom"`, `external_id` = meeting `uuid` (each occurrence of a recurring
  meeting has its own), `title` = `topic`, `occurred_at` = `start_time` (UTC), `ended_at` = start plus
  `duration` minutes, `timezone` = the meeting's `timezone`, one `Segment` per VTT cue with `start_ms` and
  `end_ms` from the cue timings and `speaker_label` from the `Name: text` prefix (no attribution unless the
  speaker's name matches the account owner exactly), participants from `GET /v2/past_meetings/{uuid}/participants`,
  `external_revision` = the transcript file's `id` plus `recording_end`, and `upstream_owner_email` = `host_email`.
- Import only meetings whose `host_email` is the connected account, and treat 404 for a recording as
  ambiguous (retention or permissions), never as deletion.
"""


def sync(db, owner, connection):
    raise NotImplementedError("The Zoom adapter is not implemented; submit Zoom VTT exports through the api path.")


def to_envelope(meeting, vtt_text):
    raise NotImplementedError("The Zoom adapter is not implemented; submit Zoom VTT exports through the api path.")
