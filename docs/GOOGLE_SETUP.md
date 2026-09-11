# Connect Google to Eridani

The app integration is implemented. The remaining one-time setup is a Google Cloud
OAuth client, followed by linking your account in Settings. No Google client
credentials were present in this checkout when this batch was built.

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
using `https://www.googleapis.com/auth/calendar.readonly`. It cannot edit Google
events. If Google Console rejects the Tailnet hostname or requires domain ownership
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

Automated tests replace Google consent/token/calendar responses in an isolated
database. They exercise the actual browser state/cookie redirects, selected calendars,
event overlays, availability, disconnect and unlink. Token verification also has
local signed-JWT tests through Google's verifier. These tests do not establish that
your real Cloud project, consent settings or Calendar permissions are configured.
Complete the account-link and first-sync checks above after saving the real client.

The sign-in button uses Google's unmodified neutral rectangular PNG from the
[official branding assets](https://developers.google.com/identity/branding-guidelines),
stored at apps/web/public/google-sign-in.png and displayed at its original aspect ratio.

The Codex browser helper failed to start during this batch, so no Cloud Console
client was created automatically. The local integration and credential slots are
ready for the setup above.
