# Connect Google to Eridani

Google sign-in and Calendar sync are implemented. The owner has configured the
Cloud client and confirmed real Calendar sync. The next optional step is
**Enable Calendar editing** in Settings (step 4 below). Earlier setup steps are
retained for recovery or a new deployment.

## 1. Create the Google client

Open [Google Cloud credentials](https://console.cloud.google.com/apis/credentials)
and use a project for Eridani. Enable the **Google Calendar API** in that project.

In Google Auth Platform, configure the app branding and audience. For an external
app in Testing, add your own Google account as a test user. Use your actual account
for the support/contact addresses.

Create an OAuth client with application type **Web application**. Add this exact
authorized redirect URI:

```text
https://davispc.tail957c2.ts.net:9443/api/v1/auth/google/callback
```

The scheme, hostname, port and path must match. This is a server authorization-code
flow; it does not need a browser JavaScript origin or a public inbound webhook.
Your browser must have access to the Tailnet when Google redirects it back.
[Google's web-server setup guide](https://developers.google.com/identity/protocols/oauth2/web-server)

The connection asks first for OpenID identity/email; Calendar is a separate grant
using `https://www.googleapis.com/auth/calendar.readonly`. Editing is an additional,
optional grant using `https://www.googleapis.com/auth/calendar.events`. Read-only
sync works without it. If Google Console rejects the Tailnet hostname or requires domain ownership
verification for your chosen publishing configuration, stop at that specific
message and configure an owner-controlled HTTPS hostname; do not expose Jarvis
publicly as a workaround.

## 2. Save credentials locally

The two empty rows have been added to the ignored repository `.env`:

```dotenv
JARVIS_GOOGLE_CLIENT_ID=
JARVIS_GOOGLE_CLIENT_SECRET=
```

Paste the values from your OAuth client there and save. Keep the secret out of chat,
screenshots and Git. The separate integration encryption key was generated in
ignored `.env.upgrade`; it should not be replaced when adding these credentials.

Recreate the API and worker so they read the updated environment. From a terminal
in the repository, using Docker Desktop:

```text
docker compose --env-file .env.upgrade -f compose.upgrade.yml up -d --no-deps api worker
```

## 3. Link your account

1. Open Eridani using your existing pairing session.
2. In **Settings → Google**, choose **Link Google account**.
3. Choose your account in Google and return to Eridani.
4. Choose **Connect Calendar** and grant the read-only Calendar permission.
5. After the first sync, select the calendars Eri should use. The primary calendar
   is selected initially; additional calendars start off.
6. Open Calendar. Google events appear alongside tasks and reminders.
   **Find an open time** asks Google for current free/busy availability.
7. Verify Google sign-in on a second tab/device before changing any pairing setup.
   Pairing remains a recovery route in this release.

Only the Google account linked from an existing authenticated owner session can
sign in. Eridani matches Google's stable account subject, not merely an email
address. [Google OpenID Connect](https://developers.google.com/identity/openid-connect/openid-connect)

## 4. Enable event editing

1. In **Settings → Google**, choose **Enable Calendar editing**.
2. Approve the additional Calendar permission in Google and return to Eridani.
3. Select a calendar marked **Can edit**. Your Google account must have writer
   or owner access to that calendar.
4. In Calendar, choose **New event**, or open an existing event and choose
   **Edit event** / **Delete event**. Eri can use the same operations by request.

Use the existing OAuth client and callback; no new credentials are needed.
If you maintain the scope list in Google Auth Platform → Data Access, include
`https://www.googleapis.com/auth/calendar.events` alongside the existing read scope.
The app requests the extra scope only when you enable editing. Denying that
additional permission preserves the existing read connection.

Choose Month, Week or Day above the calendar. A single date tap selects its agenda;
double-tap opens Day. **Open day** provides the same action without a gesture.
The chosen view is remembered on the current browser.

Recurring-event changes offer **This occurrence** or **Entire series**. All-day
forms use inclusive first/last dates. Google stores the end date exclusively.
Guest invitations and special Google events stay in Google's own editor in this
release; the event dialog provides a Google link.

**Recent calendar changes** tracks write results. A queued change is not yet saved.
If an outcome is unconfirmed, check Google before creating a replacement; repeat
status checks reuse the original receipt. Concurrent Google edits require reopening
the event instead of overwriting a newer version.
[Google event creation](https://developers.google.com/workspace/calendar/api/guides/create-events),
[conditional updates](https://developers.google.com/workspace/calendar/api/guides/version-resources)

## Connection maintenance

Google refresh tokens for external apps in Testing generally expire after seven
days when Calendar scopes are included. Reconnect Calendar when requested, or use
the appropriate publishing configuration for your personal app once ready.
[Google token expiration guidance](https://developers.google.com/identity/protocols/oauth2#expiration)

**Disconnect Calendar** removes cached Google events and attempts to revoke the
grant; the linked sign-in identity stays. If Google's revocation service cannot be
reached, the UI says the local disconnect is complete and points to your Google
account connections to finish revocation. **Unlink Google sign-in** also removes
the identity and invalidates sessions created through Google. Pairing sessions remain.

Encrypted database backups include encrypted refresh tokens. Keep the integration
key with your other recovery secrets; its local recovery file is
`.runtime/integration-key`. If that key is unavailable after a restore, reconnect
Calendar to replace the credentials. The app can still recover ordinary tasks,
notes and memories without the Google grant.

## Validation boundary

Automated tests use synthetic Google responses in a disposable database. They cover
the real browser consent redirects, source selection, read/write forms, duplicate
save retries, recurring targets, permissions, disconnect and unlink. Mobile checks
also verify scroll stability across the normal 30-second refresh and all three
views. The owner has separately confirmed real read sync. Real write consent and
the first owner-created event still require the step above; the tests did not
create fixtures on the owner's calendar.

The sign-in button uses Google's unmodified neutral rectangular PNG from the
[official branding assets](https://developers.google.com/identity/branding-guidelines),
stored at apps/web/public/google-sign-in.png and displayed at its original aspect ratio.


## Local appointments and work blocks

**Calendar → New event** creates a local Eridani appointment. Keep the Google copy
set to “Eridani only” or choose a writable selected calendar. **Reserve time for
this task** in task details creates a work block. Neither action changes task
deadlines or reminder times.

A published entry shows its Google state. Pending means the local record is saved
but Google has not confirmed it. Subsequent edits update the linked copy with
conditional writes. If Google changes independently, compare the copies and choose
which version to keep, or unlink them. Unlinking keeps both records; cancelling
a linked local entry also deletes its Google copy. The task itself remains.

Google-origin events still open their own editor, including occurrence/series
selection for repeats. Guests and special event types remain managed in Google.
Descriptions, meeting links, organizer, guest responses and attachment links now
sync locally; the first refresh after schema 0010 backfills those details.
