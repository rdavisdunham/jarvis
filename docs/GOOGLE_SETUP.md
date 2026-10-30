# Connect Google to Eridani

Google sign-in and Calendar read/write sync are deployed on Railway at
**https://app.eridani.app**. The September 16 cloud cutover preserved the owner's
Google identity and grant; Calendar synchronized successfully from Railway.
Invited accounts are supported; public self-service registration is not enabled.

The production client already exists. See [production publishing and verification](#production-publishing-and-google-verification)
for moving beyond Google's Testing audience. The setup steps below also describe
recovery or a future separate development environment.

## 1. Create the Google client

Open [Google Cloud credentials](https://console.cloud.google.com/apis/credentials)
and use a project for Eridani. Enable the **Google Calendar API** in that project.

In Google Auth Platform, configure the app branding and audience. For an external
app in Testing, add your own Google account as a test user. Use your actual account
for the support/contact addresses.

Create an OAuth client with application type **Web application**. Add this exact
authorized redirect URI:

```text
https://app.eridani.app/api/v1/auth/google/callback
```

The scheme, hostname, port and path must match. This is a server authorization-code
flow; it does not need a browser JavaScript origin or a public inbound webhook.
The cloud callback is public HTTPS; it does not require Tailnet access.
[Google's web-server setup guide](https://developers.google.com/identity/protocols/oauth2/web-server)

The connection asks first for OpenID identity/email; Calendar is a separate grant
using `https://www.googleapis.com/auth/calendar.readonly`. Editing is an additional,
optional grant using `https://www.googleapis.com/auth/calendar.events`. Read-only
sync works without it. The production authorized domain is `eridani.app`; domain
ownership verification for review uses Google Search Console.

## 2. Maintain production credentials

The existing `JARVIS_GOOGLE_CLIENT_ID`, `JARVIS_GOOGLE_CLIENT_SECRET` and
`JARVIS_INTEGRATION_ENCRYPTION_KEY` are configured in Railway's API and worker.
Keep the encryption key unchanged so stored Google grants remain readable.
Do not paste secrets into chat, screenshots or Git. If credentials must change,
update the affected Railway variables and redeploy both services through the
[cloud runbook](CLOUD_MIGRATION.md). Do not restart the retired local production
stack or create a new OAuth client just to publish the existing app.

## 3. Sign in and connect Calendar

1. Open https://app.eridani.app and choose **Sign in with Google**.
2. Use the already linked owner account or an email invited through **Settings → Sharing**.
3. In **Settings → Google**, choose **Connect Calendar** if it is not connected.
4. Grant Calendar read access, then select the calendars Eri should use.
5. Open Calendar; synced events appear with Eridani's own scheduled records.
6. Optionally enable Calendar editing below. Login and Calendar consent remain separate.

The cloud app disables PIN login. Google subjects are verified, and new accounts
require a valid invitation bound to their verified email. Creating an invitation
does not send email; share the app address with the person yourself.
[Google OpenID Connect](https://developers.google.com/identity/openid-connect/openid-connect)

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
Events with guests and special Google events keep their title, time and description
in Google's own editor (`blocked_event` in `google_writes.py`); the event dialog
provides a Google link.

### Answering invitations (RSVP)

When the connected account is a guest, Eridani offers **Yes / Maybe / No** in the
event dialog, on the event's row in the Calendar agenda, and through Eri ("I'm not
going to the 3pm": `calendar_list` → `calendar_event_read` → `calendar_respond`).
It needs Calendar editing (`calendar.events`) and a calendar marked **Can edit**.

- Only the account's own `attendees[self].responseStatus` is written, with a PATCH
  that sets `attendeesOmitted` and an `If-Match` ETag. Title, time, description and
  other guests are untouched; a Google edit in between fails as a conflict.
- A response shows **Sending to Google…**, then **Saved** or the conflict/failure
  message, like other source edits. **Revert** restores the previous answer, from
  the row, the dialog or the agent Activity receipt, and refuses if Google changed
  the event since.
- **Organizer notification:** responses use `sendUpdates=all`, so the organizer is
  told, as when answering in Google Calendar. The dialog has a checkbox to answer
  quietly (`sendUpdates=none`); Eri's `calendar_respond` takes `notify=false`.
  The agenda row always notifies.
- **Declined events are dimmed, not hidden.** They stay on the calendar struck
  through so the answer can be changed or reverted, but they are not busy time:
  they never show as deadline conflicts. Google's free/busy (availability) already
  ignores them. `calendar_list` reports `my_response` and `declined` for Eri.
- Cached events gain the `my_response` field on the first sync after this change:
  each calendar resyncs fully once (`DETAILS_VERSION` in `google_calendar.py`).

**Recent calendar changes** tracks write results. A queued change is not yet saved.
If an outcome is unconfirmed, check Google before creating a replacement; repeat
status checks reuse the original receipt. Concurrent Google edits require reopening
the event instead of overwriting a newer version.
[Google event creation](https://developers.google.com/workspace/calendar/api/guides/create-events),
[conditional updates](https://developers.google.com/workspace/calendar/api/guides/version-resources)

## Production publishing and Google verification

Checked against Google's current guidance on September 16, 2026. Publishing the
OAuth audience and completing verification are separate actions.

1. In **Google Cloud Console → Google Auth Platform → Audience**, keep the audience
   External if people outside one Google Workspace organization will use it.
   Select **Publish app** to change Testing to In production. Calendar consent
   granted in Testing has a seven-day lifetime, including its refresh token;
   reconnect Calendar after switching if an existing test grant expires. Production
   does not make every token permanent or remove unverified-app warnings by itself.
   [Audience and publishing rules](https://support.google.com/cloud/answer/15549945)
2. Build a public explanatory homepage at `https://eridani.app`, with accessible
   privacy and terms pages and a monitored support contact. The current login card
   alone is insufficient for Google's homepage requirement. Disclose actual data
   use, including any Calendar data sent to AI providers to fulfill user requests.
   Verify `eridani.app` ownership in **Google Search Console** (the DNS TXT record
   can be added in Cloudflare), and list it under authorized domains.
   [Branding and domain requirements](https://support.google.com/cloud/answer/15549049)
3. In **Branding**, provide Eridani's name, logo, support/developer emails and those
   public URLs. Complete **Verify Branding**, then **Publish branding** when ready.
   Current Google guidance requires published branding before data-access review.
4. In **Data Access**, declare only the scopes used by the application:
   `openid`, `https://www.googleapis.com/auth/userinfo.email`,
   `https://www.googleapis.com/auth/calendar.readonly`, and
   `https://www.googleapis.com/auth/calendar.events`. Login uses the first two;
   Calendar read and optional event editing request the others separately. Review
   the console's scope classifications and least-privilege justification.
5. In **Verification Center**, submit the Calendar data-access review. Explain
   calendar display/availability and user-directed event create/update/delete.
   Supply an unlisted demonstration video showing the actual English sign-in and
   consent flow and how each permission is used. Provide reviewer access through
   the app's invitation mechanism if requested. Google may follow up by email.
   [Sensitive-scope verification process](https://developers.google.com/identity/protocols/oauth2/production-readiness/sensitive-scope-verification)

Google documents a personal-use exception for only you or a few people you know.
That can allow limited use without full review, while warnings and the unverified
user cap still apply. For a broader Eridani launch and a clean consent experience,
complete branding and Calendar verification. Switching Google's audience to
production does not disable Eridani's invitation requirement.

## Connection maintenance

Google refresh tokens for external apps in Testing generally expire after seven
days when Calendar scopes are included. Reconnect Calendar when requested, or use
the appropriate publishing configuration for your personal app once ready.
[Google token expiration guidance](https://developers.google.com/identity/protocols/oauth2#expiration)

### How Eridani keeps the grant

- The refresh token, and the short-lived access token cached beside it, are stored
  only in `google_identities.credentials`, Fernet-encrypted with
  `JARVIS_INTEGRATION_ENCRYPTION_KEY`. Neither reaches the browser, the API or logs.
- **Reconnect keeps the grant.** Google often omits a new refresh token when you
  consent again; Eridani then keeps the stored one and its age. A grant Google
  already rejected is not reused: that reconnect must return a new token.
- **One refresh at a time.** Every Google call takes the account's
  `google-token:<owner>` lock, reuses the cached access token until five minutes
  before it expires, and otherwise refreshes once and saves the result (including a
  rotated refresh token) before anyone else reads it. Sync, writes, event reads and
  availability share it, so concurrent work cannot lose a token.
- **Revocation.** When Google answers `invalid_grant` (expired or revoked), the
  connection is marked **needs reconnect** once. Later sync, writes and reads fail
  locally instead of retrying Google, and Settings → Google shows what happened
  with a **Reconnect Google Calendar** button that asks for the same permissions
  (editing included). Transient token-endpoint errors stay temporary and retry.
- **Connection health** in Settings → Google (and `grant` in
  `GET /api/v1/integrations/google`) shows how long ago access was granted, when it
  was last refreshed, and whether the app looks like it is in Testing. Google does
  not report publishing status, so this is inferred: "likely" when the previous
  grant died after about seven days, "not limited" once the current grant has
  refreshed for more than seven days, otherwise "unknown". Grants stored before
  this change show no age until the next reconnect.

### Stopping the weekly reconnect: steps only the owner can do

Nothing in code can extend a Testing grant past seven days. In Google Cloud Console,
for the existing Eridani project and OAuth client:

1. **Google Auth Platform → Audience:** confirm the user type is External and the
   status is Testing. Select **Publish app** to move it to **In production**. The
   personal-use exception above applies while the app is unverified; Google may show
   an unverified-app warning at consent.
2. **Data Access:** check the declared scopes are exactly the four listed in step 4
   of the publishing section. Do not add scopes.
3. If Google requires it for production with Calendar scopes, complete branding and
   the sensitive-scope verification in the publishing section above.
4. In Eridani, **Settings → Google → Reconnect Calendar** (or **Reconnect editing**
   if editing is on) once after publishing, so the grant is issued under the
   production audience. Grants issued while in Testing keep their seven-day expiry.
5. After eight days, check **Connection health** reads "Not limited to 7 days".

Reauthorization can still be needed after this: revoking access in your Google
account, six months without use, Google's limit
on live refresh tokens per client, security reviews, or scope changes all end a
grant. Eridani then asks to reconnect instead of failing silently.

**Disconnect Calendar** removes cached Google events and attempts to revoke the
grant; the linked sign-in identity stays. If Google's revocation service cannot be
reached, the UI says the local disconnect is complete and points to your Google
account connections to finish revocation. **Unlink Google sign-in** also removes
the identity and invalidates sessions created through Google. Public cloud login
has no PIN fallback; the app protects accounts from removing their sole login.

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
