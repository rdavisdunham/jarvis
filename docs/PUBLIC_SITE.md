# Public Eridani site

The planner and API remain at https://app.eridani.app on Railway. Public product,
privacy, terms and help pages are served at https://eridani.app and
https://www.eridani.app by the `eridani-public` Cloudflare static-assets Worker.
It has no application/database credentials or runtime bindings. The same public
pages are available at `/welcome`, `/privacy`, `/terms` and `/support` on the app
origin. Google OAuth continues to use the app origin callback.

## Publish

From the repository root, build the frontend and publish the resulting assets:

```sh
cd apps/web
npm ci
npm run build
cd ../..
npm exec --yes --package=wrangler@4.132.0 -- wrangler deploy --config deploy/public/wrangler.jsonc --dry-run
npm exec --yes --package=wrangler@4.132.0 -- wrangler deploy --config deploy/public/wrangler.jsonc
```

Wrangler authentication stays outside the repository. Custom-domain routes manage
apex/www DNS and TLS; they do not change the app subdomain. `_headers` applies the
public site's restrictive browser policy, including disabling microphone access.
Railway serves the planner with its own API security headers and microphone policy.
The public entry lazily loads no planner code. Deploying public assets does not
restart a user's voice session.

Railway deploys API/worker changes from GitHub main; public assets currently require
the explicit command above. A later CI setup can use a scoped deployment token.
Do not put an interactive OAuth token in GitHub secrets or source control.

## Verify and recover

Check apex/www HTTPS, all information links, mobile layout and the Sign in link
back to `app.eridani.app`. Confirm the Google button reaches Google's consent flow
without broadening registration. An invited account still must accept the specific
workspace invitation. No invitation email is sent automatically.

Use `wrangler deployments list --config deploy/public/wrangler.jsonc` to identify
versions and `wrangler rollback --config deploy/public/wrangler.jsonc` for an asset
rollback. API/worker rollback stays in Railway. The new database migration is
additive; do not downgrade it as a shortcut while jobs or action receipts exist.

Publishing a homepage and policy pages does not complete Google's verification.
The app owner must submit/review the consent configuration and any requested
Calendar scope demonstration through Google Cloud Console.