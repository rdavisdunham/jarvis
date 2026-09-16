# External-agent release validation — September 16, 2026

## Verified

- The full backend regression run passed 576 tests with one existing skip. The final focused run passed 77 tests, including resumed receipt filtering and preventing direct commands from being requeued as model jobs.
- Frontend regression run: 116 passed, one existing Realtime skip. Production TypeScript/Vite build passed; existing third-party annotation and bundle-size warnings remain.
- New integration tests exercise hashed credentials, one-time token delivery, scope filtering, expiry, revocation, viewer downgrades, membership removal, cross-workspace isolation, origin checks, rate limits and separation from browser-cookie authentication.
- Concurrent duplicate creates produce one record and one Activity item. Two edits using the same revision produce one saved edit and one conflict. Mismatched retry inputs fail without a partial change.
- Direct bot actions appear in Activity with the bot name. The initiating account can Edit or Revert them; unrelated later fields survive guarded undo. Other bots cannot revert or inspect the first bot's work.
- A queued bot cannot load private tools or personal memories. Revocation between two planned writes preserves the first committed task and blocks the second. Clarification replies are idempotent and account/bot scoped.
- API and MCP write retries share the same receipt. The actual official MCP 2.2.0 client connected to a synthetic HTTP server, discovered scoped tools, created/read a task, retried without duplication and reverted it.
- A real browser exercised desktop and 390px mobile Settings, one-time key creation, API-created task, Activity attribution, Edit, Revert and key revocation. Eight screenshots, no JavaScript errors or horizontal overflow. The Settings `section` URL now selects the correct tab.
- PostgreSQL migration `0013 → 0014 → 0013 → 0014` preserved a synthetic existing task. An empty database also migrated to head and served the browser checks. Migration is additive: one credentials table and a nullable work attribution column.
- Offline R2 setup checking makes no network/database calls and reports variable names without exposing credentials. Existing encrypted backup tests still pass.

Evidence from disposable local verification is kept in ignored `.runtime/external-agents/`; no user task or production credential was used. Temporary credentials are revoked and the temporary database is removed when the browser server stops.

## Deployment and remaining acceptance

The web/API and worker deploy together from main using their existing serialized pre-deploy migration. Release completion requires both deployments to be successful, public readiness to return 200, anonymous bot/API/MCP requests to be rejected, and the production bundle to contain the connected-agent settings. The final release report records those live checks.

R2 activation and its real remote restore remain deferred until the owner supplies credentials. Native Railway PITR stays active. MCP OAuth-only clients and signed push webhooks remain TODOs; Bearer-header MCP clients and polling synchronization are supported now. The owner's real-device voice acceptance from the prior batch remains open.
